"""
Step 2 - Leak-safe label construction
HBF WADSIH Hackathon 2026 - Proactive Retention Challenge

Design decisions (locked in after Step 1 EDA):
  - Anchor date is TYPE-SPECIFIC:
      * System Lapsed        -> POLICY_EFFECTIVE_CEASE_DATE
        (avoids the ~78-day median administrative lag between real lapse
         and HBF processing it, which the EDA showed is systematic, not noise)
      * All other types (incl. blank/unclassified) -> POLICY_PROCESSED_CEASE_DATE
        (matches the dictionary's own description of this field as closer to
         "when the member decision was made"; gap to effective date is only
         ~1-2 weeks for these types, so the choice barely matters for them)
  - The expensive step (finding each policy-month's NEXT cancellation event)
    is computed once. Labels for every prediction window are then derived
    from that single result, so adding/removing windows is cheap.
  - Exclusion rule (leak-safety): a policy-month can only be labelled 0
    ("no cancellation") if we've actually observed the full window without
    running out of data. If INFORMATION_DATE + window months extends past
    the last month in the spine, we don't know what happens in the unseen
    tail, so that row is dropped rather than guessed at. A confirmed
    positive (cancellation actually observed within the window) is always
    kept, regardless of the cutoff.

MEMORY NOTE: this machine has ~3.8GB RAM and no swap, and the spine is
12.9M rows across 630k policies. Two things keep this script inside that
budget:
  1. Only policies that appear at least once in Cancellations can ever get
     a future-cancellation match, so we only asof-join that slice (~13%
     of policies) instead of the full spine.
  2. POLICY_HASH is cast to 'category' so each 64-char hash is stored once
     per unique policy, not once per policy-month row.

Run from inside the OneDrive_2_9-19-2026 folder:
    python3 scripts/02_build_label.py

Requires: pandas, pyarrow
"""

import gc
import os
import pandas as pd

pd.set_option("display.width", 160)

SPINE_PATH = "DATA_Monthly_Active_Policy_Spline/HBF_2026_WADSIH_CHALLENGE-DATA-Monthly_Active_Policy_Spine.parquet"
CANCELLATIONS_PATH = "DATA_Cancellations/Cancellations.parquet"
OUT_DIR = "DATA_derived"
WINDOWS_MONTHS = [1, 2, 3, 4, 5, 6, 9]


# ---------------------------------------------------------------------------
def load_spine_ids(path=SPINE_PATH):
    spine = pd.read_parquet(path, columns=["POLICY_HASH", "INFORMATION_DATE"])
    spine["INFORMATION_DATE"] = pd.to_datetime(spine["INFORMATION_DATE"])
    spine["POLICY_HASH"] = spine["POLICY_HASH"].astype("category")
    return spine


def load_cancellations_typed(path=CANCELLATIONS_PATH):
    canc = pd.read_parquet(path)
    for col in ["POLICY_PROCESSED_CEASE_DATE", "POLICY_EFFECTIVE_CEASE_DATE"]:
        canc[col] = pd.to_datetime(canc[col], errors="coerce")

    is_system_lapsed = canc["CANCELLATION_REASON_CLASSIFICATION"] == "System Lapsed"
    canc["CANCEL_EVENT_DATE"] = canc["POLICY_PROCESSED_CEASE_DATE"].where(
        ~is_system_lapsed, canc["POLICY_EFFECTIVE_CEASE_DATE"]
    )
    canc["POLICY_HASH"] = canc["POLICY_HASH"].astype("category")
    return canc[["POLICY_HASH", "CANCEL_EVENT_DATE", "CANCELLATION_REASON_CLASSIFICATION", "CANCELLATIONREASONTYPE"]]


# ---------------------------------------------------------------------------
def compute_next_cancel_event(spine_ids, canc):
    """For every policy-month row, find the EARLIEST cancellation event
    (by our type-specific anchor date) that occurs strictly AFTER that
    row's INFORMATION_DATE, for the same policy. Only policies that ever
    appear in `canc` can possibly match, so we split the spine and only
    run the (expensive) asof-join on that slice."""

    canc_policies = set(canc["POLICY_HASH"].unique())

    print(f"  policies with >=1 cancellation record: {len(canc_policies):,} / {spine_ids['POLICY_HASH'].nunique():,}", flush=True)

    could_match = spine_ids["POLICY_HASH"].isin(canc_policies)

    with_canc = spine_ids.loc[could_match].sort_values("INFORMATION_DATE").reset_index(drop=True)
    without_canc = spine_ids.loc[~could_match].reset_index(drop=True)
    del spine_ids
    gc.collect()

    # merge_asof requires matching categorical dtypes on the join key, and
    # the two frames' category sets differ (each only saw its own slice of
    # POLICY_HASH values) -- fall back to plain strings for the small
    # subset that actually goes through the join.
    with_canc["POLICY_HASH"] = with_canc["POLICY_HASH"].astype(str)

    print(f"  rows needing the join: {len(with_canc):,} | rows filled instantly (no cancellation ever): {len(without_canc):,}", flush=True)

    right = (
        canc.dropna(subset=["CANCEL_EVENT_DATE"])
        .sort_values("CANCEL_EVENT_DATE")
        .reset_index(drop=True)
    )
    right["POLICY_HASH"] = right["POLICY_HASH"].astype(str)

    matched = pd.merge_asof(
        with_canc,
        right,
        left_on="INFORMATION_DATE",
        right_on="CANCEL_EVENT_DATE",
        by="POLICY_HASH",
        direction="forward",
        allow_exact_matches=False,  # strictly AFTER, not on the same day
    )
    del with_canc, right
    gc.collect()

    without_canc["CANCEL_EVENT_DATE"] = pd.NaT
    without_canc["CANCELLATION_REASON_CLASSIFICATION"] = pd.NA
    without_canc["CANCELLATIONREASONTYPE"] = pd.NA

    merged = pd.concat([matched, without_canc], ignore_index=True)
    del matched, without_canc
    gc.collect()

    merged["days_to_event"] = (merged["CANCEL_EVENT_DATE"] - merged["INFORMATION_DATE"]).dt.days
    return merged


# ---------------------------------------------------------------------------
def build_label_for_window(merged, window_months, spine_max_date):
    cutoff = merged["INFORMATION_DATE"] + pd.DateOffset(months=window_months)

    has_event_in_window = merged["CANCEL_EVENT_DATE"].notna() & (merged["CANCEL_EVENT_DATE"] <= cutoff)
    label = has_event_in_window.astype("int8")

    # leak-safety: only trust a "0" if we actually observed the full window
    window_fully_observed = cutoff <= spine_max_date
    keep = has_event_in_window | window_fully_observed
    dropped = int((~keep).sum())

    out = pd.DataFrame({
        "POLICY_HASH": merged.loc[keep, "POLICY_HASH"].values,
        "INFORMATION_DATE": merged.loc[keep, "INFORMATION_DATE"].values,
        "label": label.loc[keep].values,
        "CANCELLATION_REASON_CLASSIFICATION": merged.loc[keep, "CANCELLATION_REASON_CLASSIFICATION"].values,
        "CANCELLATIONREASONTYPE": merged.loc[keep, "CANCELLATIONREASONTYPE"].values,
    })
    return out, dropped


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading spine IDs...", flush=True)
    spine_ids = load_spine_ids()
    spine_max_date = spine_ids["INFORMATION_DATE"].max()
    print(f"  {len(spine_ids):,} policy-months, max INFORMATION_DATE = {spine_max_date.date()}", flush=True)

    print("Loading + typing cancellations...", flush=True)
    canc = load_cancellations_typed()

    print("Finding next cancellation event per policy-month...", flush=True)
    merged = compute_next_cancel_event(spine_ids, canc)
    del canc
    gc.collect()
    print(f"  done. {merged['CANCEL_EVENT_DATE'].notna().sum():,} / {len(merged):,} rows have SOME future cancellation on record", flush=True)

    print("\n" + "=" * 80)
    print(f"{'window':>8} | {'rows_kept':>12} | {'rows_dropped':>12} | {'positive_rate':>14}")
    print("=" * 80)
    for n in WINDOWS_MONTHS:
        labelled, dropped = build_label_for_window(merged, n, spine_max_date)
        pos_rate = labelled["label"].mean()
        print(f"{n:>6}m | {len(labelled):>12,} | {dropped:>12,} | {pos_rate:>13.3%}", flush=True)

        out_path = f"{OUT_DIR}/label_window_{n}m.parquet"
        labelled.to_parquet(out_path, index=False)
        del labelled
        gc.collect()

    print("\nSaved one parquet per window to", OUT_DIR)


if __name__ == "__main__":
    main()
