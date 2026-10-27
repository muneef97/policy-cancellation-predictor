"""
Small helpers shared by the pipeline scripts: database connection, the monthly feature arrays,
calibration, ranking tables and the retention decision rule.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

import config as C


# ---- DuckDB (used for the large joins; spills to WORK rather than the data folder) -------
def duckdb_conn():
    import duckdb
    tmp = os.environ.get("HBF_DUCKDB_TMP", os.path.join(C.WORK, "duckdb_tmp"))
    os.makedirs(tmp, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{os.environ.get('HBF_DUCKDB_MEMORY', '3GB')}'")
    con.execute(f"PRAGMA threads={int(os.environ.get('HBF_DUCKDB_THREADS', 2))}")
    con.execute(f"PRAGMA temp_directory='{tmp}'")
    con.execute("SET preserve_insertion_order = false")   # lets large joins stream instead of buffering
    return con


def numeric_sql(col):
    # LHC_LOADING_RATE arrives as text with mixed formats ("12.00", "12.000")
    return f"TRY_CAST({col} AS DOUBLE) AS {col}" if col == "LHC_LOADING_RATE" else col


# ---- Monthly feature arrays (written by 03_prepare_model_data.py) ----------------------
# CACHE/<split>/<YYYY-MM-DD>_X.npy      float32, one row per policy, columns = config.ALL_FEATURES
# CACHE/<split>/<YYYY-MM-DD>_ahead.npy  int8: 0 = stays, 1/2/3 = cancels 1, 2 or 3 months ahead
# CACHE/<split>/<YYYY-MM-DD>_keys.parquet  POLICY_HASH, INFORMATION_DATE, ctype, reason
def month_files(split):
    files = sorted(glob.glob(os.path.join(C.CACHE, split, "*_X.npy")))
    assert files, f"no monthly arrays for '{split}' in {C.CACHE}: run 03_prepare_model_data.py first"
    return files


def month_of(path):
    return os.path.basename(path)[:10]


def load_month(path, cols=None):
    X = np.load(path, mmap_mode="r")
    if cols is not None:
        X = X[:, cols]
    ahead = np.load(path.replace("_X.npy", "_ahead.npy"))
    keys = pd.read_parquet(path.replace("_X.npy", "_keys.parquet"))
    return np.asarray(X, dtype=np.float32), ahead, keys


def load_split(split, cols=None, row_filter=None):
    """Stack every month of a split. row_filter(ahead, keys) -> boolean mask keeps only some rows."""
    Xs, aheads, keys = [], [], []
    for p in month_files(split):
        X, ahead, k = load_month(p, cols)
        if row_filter is not None:
            m = row_filter(ahead, k)
            X, ahead, k = X[m], ahead[m], k.loc[m].reset_index(drop=True)
        Xs.append(X); aheads.append(ahead); keys.append(k)
    return np.concatenate(Xs), np.concatenate(aheads), pd.concat(keys, ignore_index=True)


def model_columns(features=None):
    """Column positions (in the monthly arrays) and categorical positions (within the model's columns)."""
    features = features or C.MODEL_FEATURES
    cols = [C.ALL_FEATURES.index(f) for f in features]
    cat_idx = [j for j, f in enumerate(features) if f in C.CATEGORICAL]
    return cols, cat_idx


def cancel_type(keys):
    return keys["ctype"].fillna(C.UNCLASSIFIED).replace("", C.UNCLASSIFIED).to_numpy()


# ---- Category codes ---------------------------------------------------------------------
def category_codes():
    return json.load(open(C.CATEGORY_CODES))


def decode(col, values):
    inv = {v: k for k, v in category_codes()[col].items()}
    return np.array([inv.get(float(v), "(missing)") if np.isfinite(v) else "(missing)" for v in values], dtype=object)


# ---- Model outputs ----------------------------------------------------------------------
def load_model(name="final"):
    import lightgbm as lgb
    meta = json.load(open(os.path.join(C.MODELS, name, "meta.json")))
    return lgb.Booster(model_file=os.path.join(C.MODELS, name, "model.txt")), meta


def calibrate(score, scale_pos_weight):
    """The model is trained with leavers weighted up by w (scale_pos_weight), which inflates its
    odds by w. This standard prior correction turns the score back into a real-world probability."""
    return score / (score + scale_pos_weight * (1 - score))


def top_share_table(y, score, shares=(0.01, 0.02, 0.05, 0.10), ahead=None):
    """For each list size (riskiest 1%, 2%, ...): share of the list that really leaves (precision),
    share of all leavers on the list (recall), and, optionally, recall by months ahead."""
    order = np.argsort(-score, kind="stable")
    rank = np.empty(len(score), np.int64); rank[order] = np.arange(len(score))
    rows = []
    for k in shares:
        on = rank < k * len(score)
        r = {"top": k, "precision": float(y[on].mean()), "recall": float(on[y == 1].mean()),
             "lift": float(y[on].mean() / y.mean())}
        if ahead is not None:
            for m in (1, 2, 3):
                r[f"recall_ahead{m}"] = float(on[ahead == m].mean())
        rows.append(r)
    return rows


# ---- Retention decision rule (HBF's formula, challenge pack Q4 example) ------------------
ACTIONS = ["Digital, no discount", "Digital + discount", "Phone, no discount", "Phone + discount", "No action"]


def option_values(p, annual_premium, success=C.BASE_SUCCESS, phone_lift=C.PHONE_LIFT, discount_lift=C.DISCOUNT_LIFT):
    """Expected net value of each option: p x success x (value saved - discount) - contact cost.
    Returns an (n, 4) array in the order of ACTIONS[:4]."""
    p = np.asarray(p, float); disc = np.asarray(annual_premium, float) * C.MAX_DISCOUNT
    V, phone = C.VALUE_RETAINED, success * phone_lift
    return np.column_stack([
        p * success * V - C.DIGITAL_COST,
        p * success * discount_lift * (V - disc) - C.DIGITAL_COST,
        p * phone * V - C.PHONE_COST,
        p * phone * discount_lift * (V - disc) - C.PHONE_COST,
    ])


def best_action(p, annual_premium, **kw):
    """Index into ACTIONS of the best option (4 = no action when nothing beats $0), and its value."""
    v = option_values(p, annual_premium, **kw)
    best, val = v.argmax(1), v.max(1)
    return np.where(val > 0, best, 4), np.maximum(val, 0)


def annual_premium(monthly, fill):
    """Annual premium for the discount size; a blank premium is filled with the training median."""
    m = np.asarray(monthly, float)
    return np.where(np.isfinite(m), m, fill) * 12


# ---- Files ------------------------------------------------------------------------------
def save_json(obj, *path):
    p = os.path.join(*path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return p


def read_json(*path):
    return json.load(open(os.path.join(*path)))
