"""
Step 1 - Exploratory Data Analysis
HBF WADSIH Hackathon 2026 - Proactive Retention Challenge

Run from inside the OneDrive_2_9-19-2026 folder:
    python3 scripts/01_eda.py

Requires: pandas, pyarrow  (pip install pandas pyarrow)
"""

import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

CANCELLATIONS_PATH = "DATA_Cancellations/Cancellations.parquet"
SPINE_PATH = "DATA_Monthly_Active_Policy_Spline/HBF_2026_WADSIH_CHALLENGE-DATA-Monthly_Active_Policy_Spine.parquet"


# ---------------------------------------------------------------------------
# 1. CANCELLATIONS TABLE — the target/outcome dataset
# ---------------------------------------------------------------------------
def load_cancellations(path=CANCELLATIONS_PATH):
    df = pd.read_parquet(path)
    for col in ["POLICY_PROCESSED_CEASE_DATE", "POLICY_EFFECTIVE_CEASE_DATE"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    # gap tells us how far apart "HBF found out" is from "cover actually ended"
    df["gap_days"] = (df["POLICY_EFFECTIVE_CEASE_DATE"] - df["POLICY_PROCESSED_CEASE_DATE"]).dt.days
    df["processed_month"] = df["POLICY_PROCESSED_CEASE_DATE"].dt.to_period("M")
    return df


def eda_cancellations(df):
    print("=" * 80)
    print("CANCELLATIONS TABLE")
    print("=" * 80)
    print("Shape:", df.shape)
    print("\nNull counts:\n", df.isnull().sum())

    print("\nCANCELLATION_REASON_CLASSIFICATION distribution:")
    print(df["CANCELLATION_REASON_CLASSIFICATION"].value_counts(dropna=False))

    print("\nTop granular reasons (CANCELLATIONREASONTYPE):")
    print(df["CANCELLATIONREASONTYPE"].value_counts(dropna=False).head(20))

    print("\nMonthly cancellation counts by classification:")
    print(pd.crosstab(df["processed_month"], df["CANCELLATION_REASON_CLASSIFICATION"]))

    print("\nGap (effective - processed, in days) by classification:")
    print(df.groupby("CANCELLATION_REASON_CLASSIFICATION")["gap_days"].describe())

    dup_counts = df["POLICY_HASH"].value_counts()
    print("\nUnique policies:", dup_counts.shape[0])
    print("Policies with >1 cancellation record (repeat cancel/reinstate cycles):", (dup_counts > 1).sum())


# ---------------------------------------------------------------------------
# 2. SPINE TABLE — the core policy-month predictor dataset
# ---------------------------------------------------------------------------
def load_spine_light(path=SPINE_PATH):
    # only pulling the two ID columns here — full spine load happens in the
    # feature-engineering step, not during EDA
    spine = pd.read_parquet(path, columns=["INFORMATION_DATE", "POLICY_HASH"])
    spine["INFORMATION_DATE"] = pd.to_datetime(spine["INFORMATION_DATE"])
    return spine


def eda_spine(spine):
    print("=" * 80)
    print("SPINE TABLE")
    print("=" * 80)
    print("Rows:", len(spine))
    print("Unique policies:", spine["POLICY_HASH"].nunique())
    print("INFORMATION_DATE range:", spine["INFORMATION_DATE"].min(), "-", spine["INFORMATION_DATE"].max())
    print("Distinct months:", spine["INFORMATION_DATE"].nunique())


# ---------------------------------------------------------------------------
# 3. CROSS-CHECK — do the oldest backdated System Lapsed records actually
#    threaten the label, or are they filtered out naturally by the join?
# ---------------------------------------------------------------------------
def cross_check_stale_lapses(canc, spine):
    print("=" * 80)
    print("STALE SYSTEM LAPSED CHECK")
    print("=" * 80)
    last_active = spine.groupby("POLICY_HASH")["INFORMATION_DATE"].max()

    extreme = canc[
        (canc["gap_days"] < -365) & (canc["CANCELLATION_REASON_CLASSIFICATION"] == "System Lapsed")
    ].copy()
    extreme = extreme.merge(last_active.rename("last_spine_month"), left_on="POLICY_HASH", right_index=True, how="left")

    print("Extreme-gap (>1yr) System Lapsed records:", len(extreme))
    print("Of those, how many even appear in the spine during 2024-2026:", extreme["last_spine_month"].notna().sum())
    print("(the rest ceased before the spine window starts, so they can never be")
    print(" matched as a 'future' cancellation relative to any in-window INFORMATION_DATE)")


if __name__ == "__main__":
    canc = load_cancellations()
    eda_cancellations(canc)

    spine = load_spine_light()
    eda_spine(spine)

    cross_check_stale_lapses(canc, spine)
