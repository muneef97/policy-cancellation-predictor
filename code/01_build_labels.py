"""
Step 1 - The target: will this policy cancel within the next 3 months?

For every policy-month in the monthly spine:
  1. find the policy's next cancellation after that month-end,
  2. label 1 if it falls within 3 months, and record how many months ahead (1, 2 or 3),
     the cancellation type and the detailed reason,
  3. drop the last 3 months of the extract, where a full 3-month window can't be observed yet,
  4. assign each month to train / gap / validation / test in date order.

Cancellation date used:
  System Lapsed  -> POLICY_EFFECTIVE_CEASE_DATE (HBF processes lapses about 78 days after cover ends;
                    the effective date is when the member actually stopped paying)
  all other types -> POLICY_PROCESSED_CEASE_DATE (closest to when the member made the decision)

Writes WORK/labels.parquet
"""
import gc
import os

import pandas as pd

import config as C

READING_FIXES = {  # spelling variants in the detailed reason
    "Request from another fund": "Request received from other fund",
    "Moving to partner’s policy": "Moving to partner's policy",
    "Change in circumstances": "Change in circumstance",
}


def next_cancellation(spine, canc):
    """For each policy-month, the earliest cancellation strictly after INFORMATION_DATE.
    Only policies that ever cancel can match, so only that slice goes through the as-of join."""
    could = spine["POLICY_HASH"].isin(set(canc["POLICY_HASH"]))
    left = spine[could].sort_values("INFORMATION_DATE").reset_index(drop=True)
    left["POLICY_HASH"] = left["POLICY_HASH"].astype(str)
    rest = spine[~could].reset_index(drop=True)
    right = canc.dropna(subset=["EVENT_DATE"]).sort_values("EVENT_DATE").reset_index(drop=True)
    matched = pd.merge_asof(left, right, left_on="INFORMATION_DATE", right_on="EVENT_DATE",
                            by="POLICY_HASH", direction="forward", allow_exact_matches=False)
    for c in ["EVENT_DATE", "ctype", "reason"]:
        rest[c] = pd.NaT if c == "EVENT_DATE" else pd.NA
    return pd.concat([matched, rest], ignore_index=True)


def assign_splits(months):
    """Train / gap / validation / gap / test, in date order."""
    n = len(months)
    n_train, n_val = round(n * C.TRAIN_SHARE), round(n * C.VAL_SHARE)
    train, val, test = months[:n_train], months[n_train:n_train + n_val], months[n_train + n_val:]
    split = {m: "train" for m in train[:-C.GAP_MONTHS]}
    split.update({m: "gap" for m in train[-C.GAP_MONTHS:]})
    split.update({m: "val" for m in val[:-C.GAP_MONTHS]})
    split.update({m: "gap" for m in val[-C.GAP_MONTHS:]})
    split.update({m: "test" for m in test})
    return split


def main():
    os.makedirs(C.WORK, exist_ok=True)
    spine = pd.read_parquet(C.SPINE, columns=["POLICY_HASH", "INFORMATION_DATE"])
    spine["INFORMATION_DATE"] = pd.to_datetime(spine["INFORMATION_DATE"])
    spine["POLICY_HASH"] = spine["POLICY_HASH"].astype("category")   # each hash stored once: saves memory
    last_month = spine["INFORMATION_DATE"].max()

    canc = pd.read_parquet(C.CANCELLATIONS)
    for c in ["POLICY_PROCESSED_CEASE_DATE", "POLICY_EFFECTIVE_CEASE_DATE"]:
        canc[c] = pd.to_datetime(canc[c], errors="coerce")
    lapsed = canc["CANCELLATION_REASON_CLASSIFICATION"] == "System Lapsed"
    canc = pd.DataFrame({
        "POLICY_HASH": canc["POLICY_HASH"].astype(str),
        "EVENT_DATE": canc["POLICY_PROCESSED_CEASE_DATE"].where(~lapsed, canc["POLICY_EFFECTIVE_CEASE_DATE"]),
        "ctype": canc["CANCELLATION_REASON_CLASSIFICATION"],
        "reason": canc["CANCELLATIONREASONTYPE"].replace(READING_FIXES),
    })

    df = next_cancellation(spine, canc)
    del spine, canc; gc.collect()

    # keep only months whose full 3-month window is observed
    df = df[df["INFORMATION_DATE"] <= last_month - pd.DateOffset(months=C.HORIZON_MONTHS)].copy()
    ahead = pd.Series(0, index=df.index, dtype="int8")
    for m in range(C.HORIZON_MONTHS, 0, -1):     # smallest window that contains the event
        ahead[df["EVENT_DATE"].notna() & (df["EVENT_DATE"] <= df["INFORMATION_DATE"] + pd.DateOffset(months=m))] = m
    df["months_ahead"] = ahead
    df["label"] = (ahead > 0).astype("int8")
    df.loc[df["label"] == 0, ["ctype", "reason"]] = None   # type/reason only describe cancellations in the window

    months = sorted(df["INFORMATION_DATE"].unique())
    df["split"] = df["INFORMATION_DATE"].map(assign_splits(months))
    out = df[["POLICY_HASH", "INFORMATION_DATE", "label", "months_ahead", "ctype", "reason", "split"]]
    out.to_parquet(C.LABELS, index=False)

    s = out.groupby("split").agg(first=("INFORMATION_DATE", "min"), last=("INFORMATION_DATE", "max"),
                                 months=("INFORMATION_DATE", "nunique"), rows=("label", "size"), leavers=("label", "sum"))
    s["cancel_rate"] = (s["leavers"] / s["rows"]).round(4)
    print(f"spine ends {last_month.date()}; labelled months {len(months)}")
    print(s.loc[["train", "gap", "val", "test"]].to_string())
    print("saved", C.LABELS)


if __name__ == "__main__":
    main()
