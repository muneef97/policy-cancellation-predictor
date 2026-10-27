"""
Step 2 - Features, one row per policy-month, using only information known at that month-end.

  claims  rolling 3, 6 and 12-month sums of claim counts, benefits, out-of-pocket costs, hospital
          claims, reversals and no-benefit claims, plus months since the last claim.
          Claim months are first laid onto the full monthly spine (zero where there was no
          claim), so a quiet month counts as a month rather than being skipped.
  survey  the most recent survey answers as at each month (5% of policies ever respond).
  spine   this month's policy details, plus: complaint this month, arrears flag from 2 months ago
          (HBF cautions that this month's flag already triggers its own process), premium change
          over 3 months, negative-sentiment calls in the last 6 months.
  join    the three blocks combined into WORK/features.parquet (same row count as the spine).

Run all blocks:   python3 02_build_features.py
Run one block:    python3 02_build_features.py --step claims
"""
import argparse
import gc
import os

import numpy as np
import pandas as pd

import config as C
from common import duckdb_conn

ROLLING = [3, 6, 12]
CLAIM_COLS = ["TOTAL_CLAIM_COUNT", "TOTAL_BENEFIT_PAID", "TOTAL_OUT_OF_POCKET",
              "HOSPITAL_CLAIM_COUNT", "REVERSAL_FLAG", "NO_BENEFIT_PAID_FLAG"]
SPINE_COLS = ["POLICY_OWNER_AGE_BAND", "SCALE", "STATE_CODE", "CHANNEL_ACQUISITION", "TENURE_BAND",
              "ACTIVE_ADULT_COUNT", "ACTIVE_CHILD_COUNT", "YOUNGEST_DEPENDANT_AGE_BAND", "INCOME_TIER",
              "PRODUCT_HOLDINGS", "HOSPITAL_TIER", "EXTRAS_TIER", "HOSPITAL_EXCESS_AMT", "LHC_LOADING_RATE",
              "PAYMENT_METHOD_DESC", "PAYMENT_FREQUENCY_DESC", "ARREARS_FLAG", "MONTHLY_PREMIUM",
              "MARKETING_OPTIN", "CALL_NEGATIVE_SENTIMENT", "COMPLAINT_PER_MONTH", "RESOLUTION_TIME"]
ROWS_PER_BATCH = 1_200_000


def spine_ids():
    s = pd.read_parquet(C.SPINE, columns=["POLICY_HASH", "INFORMATION_DATE"])
    s["INFORMATION_DATE"] = pd.to_datetime(s["INFORMATION_DATE"])
    s["POLICY_HASH"] = s["POLICY_HASH"].astype(str)
    return s


def build_claims():
    import pyarrow as pa
    import pyarrow.parquet as pq
    ids = spine_ids()
    claims = pd.read_parquet(C.CLAIMS, columns=["POLICY_HASH", "INFORMATION_DATE"] + CLAIM_COLS)
    claims["INFORMATION_DATE"] = pd.to_datetime(claims["INFORMATION_DATE"])
    claimers = pd.unique(claims["POLICY_HASH"])
    has_claims = ids["POLICY_HASH"].isin(set(claimers))
    never = ids[~has_claims].reset_index(drop=True)
    batches = np.array_split(claimers, max(1, round(int(has_claims.sum()) / ROWS_PER_BATCH)))
    roll_names = [f"{c}_ROLL{w}M" for c in CLAIM_COLS for w in ROLLING]
    writer = None
    for i, pols in enumerate(batches):            # policy batches keep memory bounded
        pols = set(pols)
        dense = ids[ids["POLICY_HASH"].isin(pols)].merge(
            claims[claims["POLICY_HASH"].isin(pols)], on=["POLICY_HASH", "INFORMATION_DATE"], how="left")
        dense = dense.sort_values(["POLICY_HASH", "INFORMATION_DATE"]).reset_index(drop=True)
        claimed = dense["TOTAL_CLAIM_COUNT"].notna() & (dense["TOTAL_CLAIM_COUNT"] > 0)
        for c in CLAIM_COLS:
            dense[c] = dense[c].fillna(0).astype("float32")
        code = dense["POLICY_HASH"].astype("category").cat.codes.to_numpy()
        cum = dense.groupby(code, sort=False)[CLAIM_COLS].cumsum().astype("float32")
        out = pd.DataFrame({"POLICY_HASH": dense["POLICY_HASH"], "INFORMATION_DATE": dense["INFORMATION_DATE"]})
        for w in ROLLING:      # trailing w-month sum = cumulative sum minus cumulative sum w months earlier
            roll = (cum.to_numpy() - cum.groupby(code, sort=False).shift(w).fillna(0).to_numpy()).astype("float32")
            for j, c in enumerate(CLAIM_COLS):
                out[f"{c}_ROLL{w}M"] = roll[:, j]
        out = out[["POLICY_HASH", "INFORMATION_DATE"] + roll_names]
        last_claim = dense["INFORMATION_DATE"].where(claimed).groupby(code).ffill()
        out["MONTHS_SINCE_LAST_CLAIM"] = ((dense["INFORMATION_DATE"] - last_claim).dt.days / 30.44).round() \
            .astype("float32").fillna(-1.0)       # -1 = no claim yet
        table = pa.Table.from_pandas(out, preserve_index=False)
        writer = writer or pq.ParquetWriter(C.FEATURES_CLAIMS, table.schema)
        writer.write_table(table)
        print(f"  claims batch {i + 1}/{len(batches)}: {len(out):,} rows", flush=True)
        del dense, cum, out, table; gc.collect()
    for c in roll_names:                          # policies that never claimed
        never[c] = np.float32(0.0)
    never["MONTHS_SINCE_LAST_CLAIM"] = np.float32(-1.0)
    writer.write_table(pa.Table.from_pandas(never[["POLICY_HASH", "INFORMATION_DATE"] + roll_names +
                                                  ["MONTHS_SINCE_LAST_CLAIM"]], preserve_index=False, schema=writer.schema))
    writer.close()
    print("saved", C.FEATURES_CLAIMS)


def build_survey():
    ids = spine_ids()
    survey = pd.read_parquet(C.SURVEY, columns=["POLICY_HASH", "INFORMATION_DATE", "OVERALL_EXPERIENCE_RATING_MEAN",
                                                "NPS_RATING_MEAN", "SERVICE_RATING_MEAN"])
    survey["INFORMATION_DATE"] = pd.to_datetime(survey["INFORMATION_DATE"])
    responded = ids["POLICY_HASH"].isin(set(survey["POLICY_HASH"]))
    left = ids[responded].sort_values("INFORMATION_DATE").reset_index(drop=True)
    never = ids[~responded].reset_index(drop=True)
    right = survey.rename(columns={"INFORMATION_DATE": "SURVEY_DATE"}).sort_values("SURVEY_DATE").reset_index(drop=True)
    m = pd.merge_asof(left, right, left_on="INFORMATION_DATE", right_on="SURVEY_DATE", by="POLICY_HASH",
                      direction="backward", allow_exact_matches=True)   # latest survey at or before this month
    has = m["OVERALL_EXPERIENCE_RATING_MEAN"].notna()
    out = pd.DataFrame({
        "POLICY_HASH": m["POLICY_HASH"], "INFORMATION_DATE": m["INFORMATION_DATE"],
        "MONTHS_SINCE_LAST_SURVEY": ((m["INFORMATION_DATE"] - m["SURVEY_DATE"]).dt.days / 30.44).round()
                                    .astype("float32").where(has, -1.0),
        "LAST_OVERALL_EXPERIENCE_RATING": m["OVERALL_EXPERIENCE_RATING_MEAN"].astype("float32"),
        "LAST_NPS_RATING": m["NPS_RATING_MEAN"].astype("float32"),
        "LAST_SERVICE_RATING": m["SERVICE_RATING_MEAN"].astype("float32"),
    })
    out["RECENT_LOW_RATING_FLAG"] = (out["LAST_OVERALL_EXPERIENCE_RATING"] <= 3).astype("float32").where(has, 0.0)
    out["HAS_EVER_SURVEYED"] = has.astype("int8")
    nan = np.float32("nan")
    never = never.assign(MONTHS_SINCE_LAST_SURVEY=np.float32(-1.0), LAST_OVERALL_EXPERIENCE_RATING=nan,
                         LAST_NPS_RATING=nan, LAST_SERVICE_RATING=nan, RECENT_LOW_RATING_FLAG=np.float32(0.0),
                         HAS_EVER_SURVEYED=np.int8(0))
    pd.concat([out, never], ignore_index=True).to_parquet(C.FEATURES_SURVEY, index=False)
    print("saved", C.FEATURES_SURVEY)


def build_spine():
    cols = ",\n            ".join(SPINE_COLS)
    con = duckdb_conn()
    con.execute(f"""
    COPY (
        SELECT POLICY_HASH, INFORMATION_DATE,
            {cols},
            CASE WHEN COMPLAINT_PER_MONTH IS NOT NULL THEN 1 ELSE 0 END AS HAD_COMPLAINT_FLAG,
            LAG(CASE WHEN ARREARS_FLAG = 'Y' THEN 1 WHEN ARREARS_FLAG = 'N' THEN 0 END, 2)
                OVER (PARTITION BY POLICY_HASH ORDER BY INFORMATION_DATE) AS ARREARS_FLAG_LAG2M,
            MONTHLY_PREMIUM - LAG(MONTHLY_PREMIUM, 3)
                OVER (PARTITION BY POLICY_HASH ORDER BY INFORMATION_DATE) AS PREMIUM_CHANGE_3M,
            SUM(CASE WHEN CALL_NEGATIVE_SENTIMENT = 'Y' THEN 1 ELSE 0 END)
                OVER (PARTITION BY POLICY_HASH ORDER BY INFORMATION_DATE
                      ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS NEGATIVE_SENTIMENT_CALLS_LAST_6M
        FROM read_parquet('{C.SPINE}')
    ) TO '{C.FEATURES_SPINE}' (FORMAT PARQUET)""")
    con.close()
    print("saved", C.FEATURES_SPINE)


def join():
    con = duckdb_conn()
    con.execute(f"""
    COPY (
        SELECT s.*, c.* EXCLUDE (POLICY_HASH, INFORMATION_DATE), v.* EXCLUDE (POLICY_HASH, INFORMATION_DATE)
        FROM read_parquet('{C.FEATURES_SPINE}') s
        LEFT JOIN read_parquet('{C.FEATURES_CLAIMS}') c USING (POLICY_HASH, INFORMATION_DATE)
        LEFT JOIN read_parquet('{C.FEATURES_SURVEY}') v USING (POLICY_HASH, INFORMATION_DATE)
    ) TO '{C.FEATURES}' (FORMAT PARQUET)""")
    n_spine, n_out, miss_c, miss_s = con.execute(f"""
        SELECT (SELECT COUNT(*) FROM read_parquet('{C.FEATURES_SPINE}')), COUNT(*),
               SUM(CASE WHEN TOTAL_CLAIM_COUNT_ROLL3M IS NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN HAS_EVER_SURVEYED IS NULL THEN 1 ELSE 0 END)
        FROM read_parquet('{C.FEATURES}')""").fetchone()
    con.close()
    assert n_out == n_spine and miss_c == 0 and miss_s == 0, "feature join lost or duplicated rows"
    print(f"saved {C.FEATURES}: {n_out:,} rows, every row matched")


STEPS = {"claims": build_claims, "survey": build_survey, "spine": build_spine, "join": join}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=list(STEPS))
    a = ap.parse_args()
    os.makedirs(C.WORK, exist_ok=True)
    for name, fn in STEPS.items():
        if a.step in (None, name):
            print(f"== {name}", flush=True)
            fn()
