"""
Step 4 - Feature engineering
HBF WADSIH Hackathon 2026 - Proactive Retention Challenge

Builds ONE feature table keyed by (POLICY_HASH, INFORMATION_DATE), independent
of prediction window -- features only ever use information available AT OR
BEFORE each row's own INFORMATION_DATE, so the same feature table can be
joined to any of the 7 label windows from Step 2/3 without rebuilding it.

Run in separate steps (each checkpoints to its own parquet, so a crash in one
block doesn't lose the others -- this machine has only 3.8GB RAM and no swap,
so treat every block as a potential OOM risk until proven otherwise):

    python3 scripts/04_build_features.py --step claims
    python3 scripts/04_build_features.py --step survey
    python3 scripts/04_build_features.py --step spine
    python3 scripts/04_build_features.py --step join

Design notes:
  - Claims: only ~64% of policy-months have any claim row at all (the table
    only records months WITH activity). A naive rolling window over just the
    sparse claim rows would be WRONG -- it would silently skip zero-claim
    months as if they didn't exist, compressing a real 6-month gap into what
    looks like "2 claims ago". So claim months are reindexed to the FULL
    monthly spine (zero-filled where there's no activity) before rolling,
    which is the only way to get a calendar-correct trailing sum.
  - Claims covers 523,687 / 630,369 policies (83%) -- unlike Cancellations in
    Step 2/3, we can't shortcut by skipping "most policies". The reindexed
    frame is genuinely large, so everything here uses float32 and drops
    intermediate columns as soon as they're no longer needed.
  - ARREARS_FLAG is lagged by 2 months (not used at its current-month value)
    per the data dictionary's own caution: current-month arrears is already
    the trigger for HBF's existing reactive process, so using it live risks
    the model just re-detecting an intervention that's already underway.
  - Survey is very sparse (5% of policies ever respond) -- carried forward as
    "most recent known value as of this row", not rolled/summed.

Requires: pandas, pyarrow
"""

import argparse
import gc
import os
import numpy as np
import pandas as pd

SPINE_PATH = "DATA_Monthly_Active_Policy_Spline/HBF_2026_WADSIH_CHALLENGE-DATA-Monthly_Active_Policy_Spine.parquet"
CLAIMS_PATH = "DATA_Claims/Claims.parquet"
SURVEY_PATH = "DATA_Channel Survey Feedback/Channel_Survey_Feedback.parquet"
OUT_DIR = "DATA_derived"

ROLLING_WINDOWS = [3, 6, 12]
CLAIMS_COLS = [
    "TOTAL_CLAIM_COUNT", "TOTAL_BENEFIT_PAID", "TOTAL_OUT_OF_POCKET",
    "HOSPITAL_CLAIM_COUNT", "REVERSAL_FLAG", "NO_BENEFIT_PAID_FLAG",
]

SPINE_SNAPSHOT_COLS = [
    "POLICY_OWNER_AGE_BAND", "SCALE", "STATE_CODE", "CHANNEL_ACQUISITION",
    "TENURE_BAND", "ACTIVE_ADULT_COUNT", "ACTIVE_CHILD_COUNT",
    "YOUNGEST_DEPENDANT_AGE_BAND", "INCOME_TIER", "PRODUCT_HOLDINGS",
    "HOSPITAL_TIER", "EXTRAS_TIER", "HOSPITAL_EXCESS_AMT", "LHC_LOADING_RATE",
    "PAYMENT_METHOD_DESC", "PAYMENT_FREQUENCY_DESC", "ARREARS_FLAG",
    "MONTHLY_PREMIUM", "MARKETING_OPTIN", "CALL_NEGATIVE_SENTIMENT",
    "COMPLAINT_PER_MONTH", "RESOLUTION_TIME",
]


def load_spine_ids():
    spine = pd.read_parquet(SPINE_PATH, columns=["POLICY_HASH", "INFORMATION_DATE"])
    spine["INFORMATION_DATE"] = pd.to_datetime(spine["INFORMATION_DATE"])
    spine["POLICY_HASH"] = spine["POLICY_HASH"].astype(str)
    return spine


# ===========================================================================
# BLOCK: CLAIMS rolling features
# ===========================================================================
def build_claims_features():
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np

    TARGET_ROWS_PER_BATCH = 1_200_000  # keeps peak memory well under the 3.8GB ceiling

    print("Loading spine IDs...", flush=True)
    spine_ids = load_spine_ids()

    print("Loading claims (curated columns only)...", flush=True)
    claims = pd.read_parquet(CLAIMS_PATH, columns=["POLICY_HASH", "INFORMATION_DATE"] + CLAIMS_COLS)
    claims["INFORMATION_DATE"] = pd.to_datetime(claims["INFORMATION_DATE"])

    claims_policies = pd.unique(claims["POLICY_HASH"])
    total_policies = spine_ids["POLICY_HASH"].nunique()
    print(f"  policies with >=1 claim ever: {len(claims_policies):,} / {total_policies:,}", flush=True)

    could_match = spine_ids["POLICY_HASH"].isin(set(claims_policies))
    without_ids = spine_ids.loc[~could_match].reset_index(drop=True)
    print(f"  rows with zero claims ever (no computation needed): {len(without_ids):,}", flush=True)

    # how many policy-batches we need, given how many rows each one carries on average
    rows_needing_join = int(could_match.sum())
    avg_rows_per_policy = rows_needing_join / len(claims_policies)
    n_batches = max(1, round((rows_needing_join / TARGET_ROWS_PER_BATCH)))
    policy_batches = np.array_split(claims_policies, n_batches)
    print(f"  processing {rows_needing_join:,} rows in {n_batches} batches (~{TARGET_ROWS_PER_BATCH:,} rows/batch target)", flush=True)

    out_path = f"{OUT_DIR}/features_claims.parquet"
    writer = None

    for i, batch_policies in enumerate(policy_batches):
        batch_policy_set = set(batch_policies)
        batch_ids = spine_ids.loc[spine_ids["POLICY_HASH"].isin(batch_policy_set)].reset_index(drop=True)
        batch_claims = claims.loc[claims["POLICY_HASH"].isin(batch_policy_set)]

        dense = batch_ids.merge(batch_claims, on=["POLICY_HASH", "INFORMATION_DATE"], how="left")
        del batch_ids, batch_claims
        dense["HAD_CLAIM_THIS_MONTH"] = dense["TOTAL_CLAIM_COUNT"].notna() & (dense["TOTAL_CLAIM_COUNT"] > 0)
        for c in CLAIMS_COLS:
            dense[c] = dense[c].fillna(0).astype("float32")

        dense = dense.sort_values(["POLICY_HASH", "INFORMATION_DATE"]).reset_index(drop=True)
        dense["POLICY_HASH"] = dense["POLICY_HASH"].astype("category")
        policy_codes = dense["POLICY_HASH"].cat.codes.to_numpy()

        cum = dense.groupby(policy_codes, sort=False)[CLAIMS_COLS].cumsum().astype("float32")
        roll_cols = {}
        for w in ROLLING_WINDOWS:
            shifted = cum.groupby(policy_codes, sort=False).shift(w).fillna(0)
            roll = (cum.to_numpy() - shifted.to_numpy()).astype("float32")
            for j, c in enumerate(CLAIMS_COLS):
                roll_cols[f"{c}_ROLL{w}M"] = roll[:, j]
            del shifted
        del cum

        # NOTE: cummax() does NOT forward-fill through NaT (it only prevents a
        # null from resetting the running max -- the null position itself stays
        # null), so it silently fails to carry a claim date into later empty
        # months. Verified this against real data and it was wrong. ffill() is
        # what's actually needed here, and is safe now that batches are ~1.1M
        # rows instead of the full 11.46M.
        last_claim_date = dense["INFORMATION_DATE"].where(dense["HAD_CLAIM_THIS_MONTH"])
        last_claim_date = last_claim_date.groupby(policy_codes).ffill()
        months_since = ((dense["INFORMATION_DATE"] - last_claim_date).dt.days / 30.44).round().astype("float32")
        months_since = months_since.fillna(-1.0)

        out = pd.DataFrame({"POLICY_HASH": dense["POLICY_HASH"].astype(str), "INFORMATION_DATE": dense["INFORMATION_DATE"]})
        for k, v in roll_cols.items():
            out[k] = v
        out["MONTHS_SINCE_LAST_CLAIM"] = months_since.values

        table = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)

        print(f"  batch {i+1}/{n_batches} done ({len(out):,} rows)", flush=True)
        del dense, policy_codes, roll_cols, last_claim_date, months_since, out, table
        gc.collect()

    # policies that never claimed: zero-fill / sentinel, written as a final batch.
    # Must match the real batches' schema exactly (float32, same column order),
    # or pyarrow's ParquetWriter refuses the mismatched write.
    roll_col_names = [f"{c}_ROLL{w}M" for c in CLAIMS_COLS for w in ROLLING_WINDOWS]
    for col in roll_col_names:
        without_ids[col] = np.float32(0.0)
    without_ids["MONTHS_SINCE_LAST_CLAIM"] = np.float32(-1.0)
    without_ids = without_ids[["POLICY_HASH", "INFORMATION_DATE"] + roll_col_names + ["MONTHS_SINCE_LAST_CLAIM"]]
    table = pa.Table.from_pandas(without_ids, preserve_index=False, schema=writer.schema)
    writer.write_table(table)
    writer.close()

    print(f"Saved {out_path}", flush=True)

# ===========================================================================
# BLOCK: SURVEY recency features
# ===========================================================================
def build_survey_features():
    print("Loading spine IDs...", flush=True)
    spine_ids = load_spine_ids()

    print("Loading survey...", flush=True)
    survey_cols = ["POLICY_HASH", "INFORMATION_DATE", "OVERALL_EXPERIENCE_RATING_MEAN",
                   "NPS_RATING_MEAN", "SERVICE_RATING_MEAN"]
    survey = pd.read_parquet(SURVEY_PATH, columns=survey_cols)
    survey["INFORMATION_DATE"] = pd.to_datetime(survey["INFORMATION_DATE"])

    survey_policies = set(survey["POLICY_HASH"].unique())
    total_policies = spine_ids["POLICY_HASH"].nunique()
    print(f"  policies with >=1 survey ever: {len(survey_policies):,} / {total_policies:,}", flush=True)

    could_match = spine_ids["POLICY_HASH"].isin(survey_policies)
    with_ids = spine_ids.loc[could_match].sort_values("INFORMATION_DATE").reset_index(drop=True)
    without_ids = spine_ids.loc[~could_match].reset_index(drop=True)
    del spine_ids
    gc.collect()
    print(f"  rows needing asof-match: {len(with_ids):,} | rows with zero surveys ever: {len(without_ids):,}", flush=True)

    # rename survey's own date column before the asof-join so we can tell
    # apart "the spine row's date" from "which survey month actually matched" --
    # needed to compute months-since-last-survey correctly
    survey_sorted = survey.rename(columns={"INFORMATION_DATE": "SURVEY_DATE"}).sort_values("SURVEY_DATE").reset_index(drop=True)
    matched = pd.merge_asof(
        with_ids, survey_sorted,
        left_on="INFORMATION_DATE", right_on="SURVEY_DATE",
        by="POLICY_HASH", direction="backward", allow_exact_matches=True,
    )
    del with_ids, survey, survey_sorted
    gc.collect()

    has_match = matched["OVERALL_EXPERIENCE_RATING_MEAN"].notna()
    months_since_survey = ((matched["INFORMATION_DATE"] - matched["SURVEY_DATE"]).dt.days / 30.44).round().astype("float32")
    months_since_survey = months_since_survey.where(has_match, -1.0)  # sentinel: no survey matched (yet)

    out = pd.DataFrame({
        "POLICY_HASH": matched["POLICY_HASH"].astype(str),
        "INFORMATION_DATE": matched["INFORMATION_DATE"],
        "MONTHS_SINCE_LAST_SURVEY": months_since_survey,
        "LAST_OVERALL_EXPERIENCE_RATING": matched["OVERALL_EXPERIENCE_RATING_MEAN"].astype("float32"),
        "LAST_NPS_RATING": matched["NPS_RATING_MEAN"].astype("float32"),
        "LAST_SERVICE_RATING": matched["SERVICE_RATING_MEAN"].astype("float32"),
    })
    out["RECENT_LOW_RATING_FLAG"] = (out["LAST_OVERALL_EXPERIENCE_RATING"] <= 3).astype("float32")
    out.loc[~has_match, "RECENT_LOW_RATING_FLAG"] = 0.0
    out["HAS_EVER_SURVEYED"] = has_match.astype("int8")

    del matched
    gc.collect()

    # policies that never surveyed at all: sentinel fill
    # explicit float32 NaN (not plain np.nan, which pandas stores as float64) --
    # matching the matched side's dtype exactly avoids a dtype-mismatch on the
    # concat below (pandas currently handles it fine, but warns that a future
    # version's all-NA dtype inference will change; this sidesteps it rather
    # than relying on that behavior).
    nan32 = np.float32("nan")
    without_ids = without_ids.assign(
        MONTHS_SINCE_LAST_SURVEY=np.float32(-1.0),
        LAST_OVERALL_EXPERIENCE_RATING=nan32,
        LAST_NPS_RATING=nan32,
        LAST_SERVICE_RATING=nan32,
        RECENT_LOW_RATING_FLAG=np.float32(0.0),
        HAS_EVER_SURVEYED=np.int8(0),
    )

    result = pd.concat([out, without_ids], ignore_index=True)
    del out, without_ids
    gc.collect()

    out_path = f"{OUT_DIR}/features_survey.parquet"
    result.to_parquet(out_path, index=False)
    print(f"Saved {out_path} ({len(result):,} rows, {result.shape[1]} cols)", flush=True)


def build_spine_features():
    """
    Snapshot pass-through + trend features from the Monthly Active Policy Spine
    itself. Unlike Claims/Survey, every spine row already has this data --
    there's no join, just per-policy trend computation (lag, rolling count).

    Implemented in DuckDB rather than pandas: computing LAG()/windowed SUM()
    per policy over 12.95M rows in pandas needs either a full-table groupby
    (OOM'd before, see Claims) or manual policy-batching (works, but is more
    code and more ways to get it subtly wrong -- see the cummax/ffill bug).
    DuckDB's window functions do the same computation out-of-core, and we
    already proved DuckDB is memory-safe on this box for the balance check.

    Design (confirmed with real data before writing this):
      - HAD_COMPLAINT_FLAG: COMPLAINT_PER_MONTH is non-null for only 8,026 of
        12.95M rows -- 1 = a complaint was logged this month, 0 otherwise.
      - ARREARS_FLAG_LAG2M: ARREARS_FLAG is a clean N/Y flag (155,479 "Y" of
        12.95M rows, ~1.2%). Encoded to 0/1, then LAG(..., 2) per policy --
        this is the HBF-cautioned feature, lagged 2 months so it reflects an
        arrears state from BEFORE the prediction cutoff, not a same-month
        flag that could just be re-detecting an already-triggered reactive
        intervention. The first 2 months of any policy's history have no
        prior value -- LAG returns NULL there, which we leave as NULL
        (NaN) rather than guessing 0 or 1. LightGBM handles missing values
        natively (it learns the best split direction for them), so an
        honest "we don't know yet" is better than a fabricated default.
      - PREMIUM_CHANGE_3M: MONTHLY_PREMIUM minus its value 3 months ago, per
        policy. Same NULL-for-insufficient-history reasoning as above.
      - NEGATIVE_SENTIMENT_CALLS_LAST_6M: trailing 6-month count (current
        month + 5 preceding, matching the ROLL6M convention used for Claims)
        of CALL_NEGATIVE_SENTIMENT == 'Y' calls per policy. CALL_NEGATIVE_
        SENTIMENT is 'U'/'Y' (not 'N'/'Y' -- checked the raw values first),
        so we test for '=Y' explicitly rather than assuming a binary N/Y
        encoding like ARREARS_FLAG has.
      - All other SPINE_SNAPSHOT_COLS pass through unchanged, as this
        month's own snapshot value.
    """
    import duckdb

    out_path = f"{OUT_DIR}/features_spine.parquet"
    snapshot_select = ",\n        ".join(SPINE_SNAPSHOT_COLS)

    query = f"""
    COPY (
        SELECT
            POLICY_HASH,
            INFORMATION_DATE,
            {snapshot_select},
            CASE WHEN COMPLAINT_PER_MONTH IS NOT NULL THEN 1 ELSE 0 END
                AS HAD_COMPLAINT_FLAG,
            LAG(
                CASE WHEN ARREARS_FLAG = 'Y' THEN 1
                     WHEN ARREARS_FLAG = 'N' THEN 0
                     ELSE NULL END,
                2
            ) OVER (PARTITION BY POLICY_HASH ORDER BY INFORMATION_DATE)
                AS ARREARS_FLAG_LAG2M,
            MONTHLY_PREMIUM - LAG(MONTHLY_PREMIUM, 3) OVER (
                PARTITION BY POLICY_HASH ORDER BY INFORMATION_DATE
            ) AS PREMIUM_CHANGE_3M,
            SUM(CASE WHEN CALL_NEGATIVE_SENTIMENT = 'Y' THEN 1 ELSE 0 END) OVER (
                PARTITION BY POLICY_HASH ORDER BY INFORMATION_DATE
                ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
            ) AS NEGATIVE_SENTIMENT_CALLS_LAST_6M
        FROM read_parquet('{SPINE_PATH}')
    ) TO '{out_path}' (FORMAT PARQUET)
    """

    # IMPORTANT: without an explicit temp_directory, DuckDB spills to the
    # current working directory -- which here is the mounted external drive,
    # not local disk. That made a 3-way join that completes in ~5s take over
    # 180s just for the write step. Pointing spill files at a local scratch
    # dir (outside the mounted folder) is what actually fixed it.
    tmp_dir = os.path.expanduser("~/duckdb_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='3GB'")
    con.execute("PRAGMA threads=2")
    con.execute(f"PRAGMA temp_directory='{tmp_dir}'")
    con.execute(query)
    con.close()

    check = duckdb.connect().execute(
        f"SELECT COUNT(*), COUNT(DISTINCT POLICY_HASH) FROM read_parquet('{out_path}')"
    ).fetchone()
    print(f"Saved {out_path} ({check[0]:,} rows, {check[1]:,} distinct policies)", flush=True)



def build_joined_features():
    """
    Combines the three feature blocks -- Claims, Survey, Spine (snapshot +
    trend) -- into one final feature table keyed by (POLICY_HASH,
    INFORMATION_DATE). All three were built to cover every spine row exactly
    once, so this is a same-cardinality LEFT JOIN, not a fan-out -- verified
    below rather than assumed.

    Done in DuckDB for the same reason as build_spine_features(): reading
    three ~12.95M-row tables into pandas at once to merge them would very
    likely repeat the OOM problems we already hit and fixed elsewhere.
    """
    import duckdb

    claims_path = f"{OUT_DIR}/features_claims.parquet"
    survey_path = f"{OUT_DIR}/features_survey.parquet"
    spine_path = f"{OUT_DIR}/features_spine.parquet"
    out_path = f"{OUT_DIR}/features.parquet"

    query = f"""
    COPY (
        SELECT
            s.*,
            c.* EXCLUDE (POLICY_HASH, INFORMATION_DATE),
            v.* EXCLUDE (POLICY_HASH, INFORMATION_DATE)
        FROM read_parquet('{spine_path}') s
        LEFT JOIN read_parquet('{claims_path}') c
            ON s.POLICY_HASH = c.POLICY_HASH
           AND s.INFORMATION_DATE = c.INFORMATION_DATE
        LEFT JOIN read_parquet('{survey_path}') v
            ON s.POLICY_HASH = v.POLICY_HASH
           AND s.INFORMATION_DATE = v.INFORMATION_DATE
    ) TO '{out_path}' (FORMAT PARQUET)
    """

    # IMPORTANT: without an explicit temp_directory, DuckDB spills to the
    # current working directory -- which here is the mounted external drive,
    # not local disk. That made a 3-way join that completes in ~5s take over
    # 180s just for the write step. Pointing spill files at a local scratch
    # dir (outside the mounted folder) is what actually fixed it.
    tmp_dir = os.path.expanduser("~/duckdb_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='3GB'")
    con.execute("PRAGMA threads=2")
    con.execute(f"PRAGMA temp_directory='{tmp_dir}'")
    con.execute(query)
    con.close()

    con2 = duckdb.connect()
    con2.execute("PRAGMA memory_limit='2.5GB'")
    n_spine = con2.execute(f"SELECT COUNT(*) FROM read_parquet('{spine_path}')").fetchone()[0]
    n_out = con2.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
    n_cols = len(con2.execute(f"DESCRIBE SELECT * FROM read_parquet('{out_path}')").fetchdf())
    # any claims/survey column entirely null would indicate a join miss, not
    # expected since all three tables were built to cover every spine row
    null_claims = con2.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_path}') WHERE TOTAL_CLAIM_COUNT_ROLL3M IS NULL"
    ).fetchone()[0]
    null_survey = con2.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_path}') WHERE HAS_EVER_SURVEYED IS NULL"
    ).fetchone()[0]
    con2.close()

    print(f"Saved {out_path}: {n_out:,} rows, {n_cols} cols", flush=True)
    print(f"  row count matches spine ({n_spine:,}): {n_out == n_spine}", flush=True)
    print(f"  rows with no claims-join match: {null_claims:,} (expect 0)", flush=True)
    print(f"  rows with no survey-join match: {null_survey:,} (expect 0)", flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["claims", "survey", "spine", "join"])
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.step == "claims":
        build_claims_features()
    elif args.step == "survey":
        build_survey_features()
    elif args.step == "spine":
        build_spine_features()
    elif args.step == "join":
        build_joined_features()
    else:
        raise NotImplementedError(f"step '{args.step}' not written yet")
