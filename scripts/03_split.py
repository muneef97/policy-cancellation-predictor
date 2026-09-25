"""
Step 3 - Train / validation / test split
HBF WADSIH Hackathon 2026 - Proactive Retention Challenge

Design decisions (locked in with the user):
  - CHRONOLOGICAL split, not random row split: train on earlier months,
    validate/test on later months. This matches production reality -- the
    model only ever predicts forward from what's known today, so evaluating
    on the future relative to training is the honest test.

  - "Degenerate tail" months are dropped entirely, from every split. Step 2
    correctly keeps only CONFIRMED positives in the trailing N months of the
    spine's range (a negative there can't be verified without a full window
    of future data) -- but that makes those specific months 100% positive,
    which is useless/misleading for both training and evaluation. So here we
    drop every row dated after (spine_max_date - N months), leaving exactly
    the months where the true label (0 or 1) is fully trustworthy.

  - EMBARGO: a fixed 1 month gap is left between train/val and val/test.
    A full embargo equal to the window length was the original plan, but is
    mathematically impossible for the larger windows (it needs 2*N months of
    pure buffer, which exceeds the total usable months once N gets to 6-9).
    A small fixed embargo instead buffers the single worst adjacent-month
    overlap (where a label's forward-looking window spills across the split
    boundary) without breaking the split for any window.

  - Split proportions (70% / 15% / 15%) are applied to the count of USABLE
    MONTHS, not rows, so a chronological boundary always falls cleanly on a
    month boundary -- no month's data is ever split across two sets.

Run from inside the OneDrive_2_9-19-2026 folder:
    python3 scripts/03_split.py

Requires: pandas, pyarrow
"""

import gc
import pandas as pd

LABEL_DIR = "DATA_derived"
SPINE_PATH = "DATA_Monthly_Active_Policy_Spline/HBF_2026_WADSIH_CHALLENGE-DATA-Monthly_Active_Policy_Spine.parquet"
WINDOWS_MONTHS = [1, 2, 3, 4, 5, 6, 9]
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
EMBARGO_MONTHS = 1


def get_spine_max_date():
    df = pd.read_parquet(SPINE_PATH, columns=["INFORMATION_DATE"])
    return pd.to_datetime(df["INFORMATION_DATE"]).max()


def compute_month_assignment(months, train_frac, val_frac, embargo):
    """months: sorted ascending list of unique pd.Timestamp values.
    Returns {month: 'train'|'val'|'test'|'embargo'}."""
    n = len(months)
    train_n = round(n * train_frac)
    val_n = round(n * val_frac)
    # whatever's left goes to test, so the three always sum to n exactly
    test_n = n - train_n - val_n

    train_raw = months[:train_n]
    val_raw = months[train_n:train_n + val_n]
    test_raw = months[train_n + val_n:]

    assignment = {}

    train_final = train_raw[:-embargo] if len(train_raw) > embargo else []
    for m in train_raw:
        assignment[m] = "train" if m in train_final else "embargo"

    val_final = val_raw[:-embargo] if len(val_raw) > embargo else []
    for m in val_raw:
        assignment[m] = "val" if m in val_final else "embargo"

    for m in test_raw:
        assignment[m] = "test"

    return assignment


def process_window(n, spine_max_date):
    path = f"{LABEL_DIR}/label_window_{n}m.parquet"
    df = pd.read_parquet(path)
    df["INFORMATION_DATE"] = pd.to_datetime(df["INFORMATION_DATE"])

    fully_observable_cutoff = spine_max_date - pd.DateOffset(months=n)
    is_degenerate_tail = df["INFORMATION_DATE"] > fully_observable_cutoff
    n_degenerate = int(is_degenerate_tail.sum())
    df = df.loc[~is_degenerate_tail].copy()

    months = sorted(df["INFORMATION_DATE"].unique())
    assignment = compute_month_assignment(months, TRAIN_FRAC, VAL_FRAC, EMBARGO_MONTHS)
    df["split"] = df["INFORMATION_DATE"].map(assignment)

    summary = df.groupby("split").agg(
        months=("INFORMATION_DATE", "nunique"),
        rows=("label", "size"),
        positives=("label", "sum"),
    )
    summary["pos_rate"] = summary["positives"] / summary["rows"]
    summary = summary.reindex(["train", "embargo", "val", "test"])

    print(f"\n{'='*70}\nWindow = {n} month(s)")
    print(f"  degenerate tail rows dropped (insufficient future data): {n_degenerate:,}")
    print(f"  usable months remaining: {len(months)}")
    print(summary.to_string())

    out_path = f"{LABEL_DIR}/label_window_{n}m_split.parquet"
    df.to_parquet(out_path, index=False)

    del df
    gc.collect()
    return summary


def main():
    print("Reading spine max date...")
    spine_max_date = get_spine_max_date()
    print("Spine max INFORMATION_DATE:", spine_max_date.date())

    for n in WINDOWS_MONTHS:
        process_window(n, spine_max_date)

    print("\nSaved *_split.parquet (adds a 'split' column: train/embargo/val/test) for every window.")


if __name__ == "__main__":
    main()
