"""
Step 3 - Model-ready data.

  1. Join the labels (step 1) to the features (step 2): WORK/model_table.parquet.
  2. Give every category a fixed numeric code (models/category_codes.json), so the same code means
     the same thing in every script, the saved models and the live demo.
  3. Write one array file per month (WORK/monthly_arrays/<split>/<month>_*), rows sorted by policy, so
     every later step reads the same rows in the same order and results are reproducible.
"""
import json
import os

import numpy as np

import config as C
from common import duckdb_conn, numeric_sql


def build_model_table(con):
    con.execute(f"""
    COPY (
        SELECT f.*, l.label, l.months_ahead, l.ctype, l.reason, l.split
        FROM read_parquet('{C.LABELS}') l
        JOIN read_parquet('{C.FEATURES}') f USING (POLICY_HASH, INFORMATION_DATE)
    ) TO '{C.MODEL_TABLE}' (FORMAT PARQUET)""")
    n_lab = con.execute(f"SELECT COUNT(*) FROM read_parquet('{C.LABELS}')").fetchone()[0]
    n_out = con.execute(f"SELECT COUNT(*) FROM read_parquet('{C.MODEL_TABLE}')").fetchone()[0]
    assert n_lab == n_out, f"join lost rows: {n_lab:,} labels vs {n_out:,} model rows"
    print(f"saved {C.MODEL_TABLE}: {n_out:,} rows")


def build_category_codes(con):
    # a blank HOSPITAL_EXCESS_AMT ('' on 47 rows) is treated like a missing value
    exprs = [f"array_agg(DISTINCT CASE WHEN {c} IS NULL OR {c} = '' THEN 'MISSING' ELSE {c} END)" if c == "HOSPITAL_EXCESS_AMT"
             else f"array_agg(DISTINCT COALESCE({c}, 'MISSING'))" for c in C.CATEGORICAL]
    row = con.execute(f"SELECT {', '.join(exprs)} FROM read_parquet('{C.MODEL_TABLE}')").fetchone()
    codes = {c: {v: float(i) for i, v in enumerate(sorted(set(vals)))} for c, vals in zip(C.CATEGORICAL, row)}
    os.makedirs(C.MODELS, exist_ok=True)
    json.dump(codes, open(C.CATEGORY_CODES, "w"), indent=1)
    print("saved", C.CATEGORY_CODES)
    return codes


def write_month(con, split, month, codes):
    sel = ", ".join([numeric_sql(c) for c in C.NUMERIC] + C.CATEGORICAL)
    df = con.execute(f"""
        SELECT CAST(POLICY_HASH AS VARCHAR) AS POLICY_HASH, months_ahead, ctype, reason, {sel}
        FROM read_parquet('{C.MODEL_TABLE}')
        WHERE CAST(INFORMATION_DATE AS DATE) = DATE '{month}'
        ORDER BY POLICY_HASH""").df()
    X = np.empty((len(df), len(C.ALL_FEATURES)), dtype=np.float32)
    for j, c in enumerate(C.NUMERIC):
        X[:, j] = df[c].to_numpy(dtype="float32")
    for j, c in enumerate(C.CATEGORICAL):
        col = df[c].replace("", np.nan) if c == "HOSPITAL_EXCESS_AMT" else df[c]
        X[:, len(C.NUMERIC) + j] = col.map(codes[c]).to_numpy(dtype="float32")   # NULL -> NaN (missing)
    base = os.path.join(C.CACHE, split, month)
    np.save(base + "_X.npy", X)
    np.save(base + "_ahead.npy", df["months_ahead"].to_numpy(dtype=np.int8))
    df[["POLICY_HASH", "ctype", "reason"]].assign(INFORMATION_DATE=month).to_parquet(base + "_keys.parquet", index=False)
    return len(df)


def main():
    os.makedirs(C.WORK, exist_ok=True)
    con = duckdb_conn()
    build_model_table(con)
    codes = build_category_codes(con)
    months = con.execute(f"""SELECT DISTINCT CAST(CAST(INFORMATION_DATE AS DATE) AS VARCHAR), split
                             FROM read_parquet('{C.MODEL_TABLE}') ORDER BY 1""").fetchall()
    for month, split in months:
        os.makedirs(os.path.join(C.CACHE, split), exist_ok=True)
        n = write_month(con, split, month, codes)
        print(f"  {split:5s} {month}: {n:,} policies", flush=True)
    con.close()


if __name__ == "__main__":
    main()
