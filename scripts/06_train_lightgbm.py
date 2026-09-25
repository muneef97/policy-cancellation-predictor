"""
Step 5 (continued) - LightGBM + SHAP, 3-month window.

Why this loads data differently from the baseline (05_train_baseline.py):
LightGBM handles missing values and categorical columns natively, so none of
the baseline's machinery is needed here -- no one-hot encoding, no median
imputation, no feature scaling. That means far fewer output columns than the
baseline's 128-column one-hot design matrix, which in turn means we CAN
preallocate one big numpy array for the whole train split and fill it in
streamed chunks (rather than partial_fit-ing through chunks like the
baseline had to) -- single allocation, no repeated concat/copy overhead.

Categorical columns are passed to LightGBM as integer category codes in a
plain float32 array (not a pandas 'category' dtype DataFrame), with
categorical_feature=<column indices> telling LightGBM which columns to treat
that way. Codes come from a fixed, pre-enumerated category list (built once
via SQL DISTINCT, see get_categories() in 05_train_baseline.py) rather than
whatever happens to appear in a given chunk.

Run:
    python3 scripts/06_train_lightgbm.py --step train --window 3
    python3 scripts/06_train_lightgbm.py --step shap --window 3
"""
import argparse
import gc
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import importlib
baseline_mod = importlib.import_module("05_train_baseline")

OUT_DIR = "DATA_derived"
MODEL_DIR = "models"
BATCH_ROWS = 400_000

NUMERIC_COLS = baseline_mod.NUMERIC_COLS
CATEGORICAL_COLS = baseline_mod.CATEGORICAL_COLS
ALL_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
CAT_FEATURE_IDX = list(range(len(NUMERIC_COLS), len(ALL_FEATURE_COLS)))


def numeric_select_sql(c):
    return baseline_mod.numeric_select_sql(c)


def duckdb_conn():
    return baseline_mod.duckdb_conn()


def get_split_count(window_months, split_name):
    con = duckdb_conn()
    path = f"{OUT_DIR}/model_data_{window_months}m.parquet"
    n = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}') WHERE split='{split_name}'"
    ).fetchone()[0]
    con.close()
    return n


def load_split_array(window_months, split_name, cat_code_maps):
    """Streams a split into ONE preallocated float32 array (rows=split size,
    cols=len(ALL_FEATURE_COLS)) -- numeric cols keep their real value (NaN
    preserved), categorical cols become integer codes (NaN for anything not
    in the fixed category list, which LightGBM treats as missing)."""
    n_rows = get_split_count(window_months, split_name)
    X = np.empty((n_rows, len(ALL_FEATURE_COLS)), dtype=np.float32)
    y = np.empty(n_rows, dtype=np.int8)

    path = f"{OUT_DIR}/model_data_{window_months}m.parquet"
    select_cols = ", ".join([numeric_select_sql(c) for c in NUMERIC_COLS] + CATEGORICAL_COLS)
    con = duckdb_conn()
    query = f"""
        SELECT label, {select_cols}
        FROM read_parquet('{path}')
        WHERE split = '{split_name}'
    """
    reader = con.execute(query).to_arrow_reader(batch_size=BATCH_ROWS)

    pos = 0
    for batch in reader:
        df = batch.to_pandas()
        n = len(df)
        for j, c in enumerate(NUMERIC_COLS):
            X[pos:pos + n, j] = df[c].to_numpy(dtype="float32")
        for j, c in enumerate(CATEGORICAL_COLS):
            col = df[c]
            if c == "HOSPITAL_EXCESS_AMT":
                col = col.replace("", np.nan)
            codes = col.map(cat_code_maps[c])  # NaN for anything unmapped
            X[pos:pos + n, len(NUMERIC_COLS) + j] = codes.to_numpy(dtype="float32")
        y[pos:pos + n] = df["label"].to_numpy(dtype="int8")
        pos += n
    con.close()
    assert pos == n_rows, f"row count mismatch: filled {pos}, expected {n_rows}"
    return X, y


def build_code_maps(categories):
    return {c: {val: float(i) for i, val in enumerate(categories[c])} for c in CATEGORICAL_COLS}


def train_lightgbm(window_months):
    import time
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score

    print("Enumerating categorical values (fixed code mapping)...", flush=True)
    categories = baseline_mod.get_categories(window_months)
    cat_code_maps = build_code_maps(categories)
    for c in CATEGORICAL_COLS:
        print(f"  {c}: {len(categories[c])} categories", flush=True)

    print("\nLoading train split into preallocated array...", flush=True)
    t0 = time.time()
    X_train, y_train = load_split_array(window_months, "train", cat_code_maps)
    print(f"  X_train {X_train.shape}, took {time.time()-t0:.1f}s", flush=True)
    gc.collect()

    print("Loading val split...", flush=True)
    X_val, y_val = load_split_array(window_months, "val", cat_code_maps)
    print(f"  X_val {X_val.shape}", flush=True)
    gc.collect()

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos
    print(f"\nscale_pos_weight = {scale_pos_weight:.2f} (train base rate {100*n_pos/len(y_train):.2f}%)", flush=True)

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=ALL_FEATURE_COLS,
                             categorical_feature=CAT_FEATURE_IDX, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, free_raw_data=False)

    # deterministic + force_row_wise: without these, LightGBM's own docs say
    # multi-threaded histogram accumulation (num_threads=4 here) can sum
    # floating-point values in a different order from run to run, giving
    # slightly different split gains and, over 150 boosting rounds, a
    # genuinely different tree structure -- a fixed `seed` alone does not
    # guarantee this away. Added after independently confirming (and fixing)
    # an analogous non-determinism bug in the baseline's own training step;
    # not something to assume is fine without checking here too.
    params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "scale_pos_weight": scale_pos_weight,
        "num_leaves": 31,
        "max_bin": 127,
        "learning_rate": 0.08,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.6,
        "bagging_freq": 1,
        "min_data_in_leaf": 100,
        "num_threads": 4,
        "verbose": -1,
        "seed": 42,
        "deterministic": True,
        "force_col_wise": True,
    }

    # Fixed round count rather than early stopping: at ~1s/round on this box,
    # early stopping's "confirm no improvement for N more rounds" overhead
    # doesn't fit the tool environment's 180s per-command ceiling. Instead we
    # run a bounded number of rounds and pick the single best one afterward
    # from the recorded eval history -- same effect, bounded worst-case time.
    N_ROUNDS = 150
    print(f"\nTraining LightGBM ({N_ROUNDS} fixed rounds, best round picked after)...", flush=True)
    t0 = time.time()
    evals_result = {}
    booster = lgb.train(
        params, train_set,
        num_boost_round=N_ROUNDS,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.log_evaluation(period=20),
                   lgb.record_evaluation(evals_result)],
    )
    val_pr_curve = evals_result["val"]["average_precision"]
    best_round = int(np.argmax(val_pr_curve)) + 1
    print(f"Training took {time.time()-t0:.1f}s over {N_ROUNDS} rounds. "
          f"Best round by val PR-AUC: {best_round} (val PR-AUC={max(val_pr_curve):.4f})", flush=True)

    val_proba = booster.predict(X_val, num_iteration=best_round)
    roc_auc = roc_auc_score(y_val, val_proba)
    pr_auc = average_precision_score(y_val, val_proba)
    base_rate = y_val.mean()

    print(f"\n=== LightGBM ({window_months}m window) -- VAL set ===", flush=True)
    print(f"  n_val={len(y_val):,}  base rate: {base_rate*100:.2f}%", flush=True)
    print(f"  ROC-AUC: {roc_auc:.4f}", flush=True)
    print(f"  PR-AUC : {pr_auc:.4f}  (random model would score ~{base_rate:.4f})", flush=True)

    order = np.argsort(-val_proba)
    y_sorted = y_val[order]
    print("\n  Precision/recall if contacting the top-K% highest-risk policies:", flush=True)
    for pct in [1, 5, 10, 20]:
        k = int(len(y_val) * pct / 100)
        caught = y_sorted[:k].sum()
        precision = caught / k
        recall = caught / y_val.sum()
        print(f"    top {pct:>2}%  (n={k:,}): precision={precision*100:.2f}%  recall={recall*100:.2f}%", flush=True)

    # WA vs interstate breakdown -- decode STATE_CODE back from its category
    # code so we can slice the val predictions by it
    state_col_idx = ALL_FEATURE_COLS.index("STATE_CODE")
    state_codes = X_val[:, state_col_idx]
    wa_code = cat_code_maps["STATE_CODE"].get("WA")
    is_wa = state_codes == wa_code

    print("\n  === WA vs interstate breakdown (val set) ===", flush=True)
    for label, mask in [("WA", is_wa), ("Other (interstate)", ~is_wa)]:
        yv, pv = y_val[mask], val_proba[mask]
        if yv.sum() == 0 or len(yv) == 0:
            print(f"    {label}: no positive examples, skipping", flush=True)
            continue
        seg_roc = roc_auc_score(yv, pv)
        seg_pr = average_precision_score(yv, pv)
        print(f"    {label}: n={mask.sum():,}  base_rate={yv.mean()*100:.2f}%  "
              f"ROC-AUC={seg_roc:.4f}  PR-AUC={seg_pr:.4f}", flush=True)
        seg_order = np.argsort(-pv)
        seg_sorted = yv[seg_order]
        for pct in [5, 10]:
            k = max(1, int(len(yv) * pct / 100))
            caught = seg_sorted[:k].sum()
            print(f"      top {pct}% (n={k:,}): precision={caught/k*100:.2f}%  recall={caught/yv.sum()*100:.2f}%", flush=True)

    print("\nTop 20 features by gain:", flush=True)
    importance = booster.feature_importance(importance_type="gain")
    imp_order = np.argsort(-importance)
    for i in imp_order[:20]:
        print(f"    {ALL_FEATURE_COLS[i]:40s} {importance[i]:,.0f}", flush=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    booster.save_model(f"{MODEL_DIR}/lightgbm_{window_months}m.txt")
    import pickle
    with open(f"{MODEL_DIR}/lightgbm_{window_months}m_meta.pkl", "wb") as f:
        pickle.dump({"categories": categories, "cat_code_maps": cat_code_maps,
                     "feature_cols": ALL_FEATURE_COLS, "cat_idx": CAT_FEATURE_IDX,
                     "best_round": best_round}, f)
    print(f"\nSaved model to {MODEL_DIR}/lightgbm_{window_months}m.txt", flush=True)


def run_shap(window_months):
    """SHAP analysis on the saved LightGBM model, against the full val split
    (not a subsample -- LightGBM's TreeExplainer uses the exact fast
    TreeSHAP algorithm, cost scales with sample count with a small constant,
    so ~1M rows is not a real concern on this box, unlike the baseline's
    design-matrix build).

    Two specific questions this needs to answer, both flagged earlier as
    open items rather than assumed away:
      1. Does MARKETING_OPTIN's dominant *gain*-based importance (>3x the
         next feature) hold up as a real, well-behaved SHAP effect, or is it
         an artifact of how gain rewards a feature that happens to make a
         few very clean, very early splits?
      2. The linear baseline showed a counter-intuitive NEGATIVE coefficient
         on NEGATIVE_SENTIMENT_CALLS_LAST_6M (more negative-sentiment calls
         -> LOWER predicted risk) -- does the tree model show the same
         direction, or was that a baseline-specific quirk (e.g. correlation
         with a confound the linear model couldn't separate out)?
    """
    import time
    import pickle
    import shap
    import lightgbm as lgb

    print("Loading saved model + metadata...", flush=True)
    booster = lgb.Booster(model_file=f"{MODEL_DIR}/lightgbm_{window_months}m.txt")
    with open(f"{MODEL_DIR}/lightgbm_{window_months}m_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    categories = meta["categories"]
    cat_code_maps = meta["cat_code_maps"]
    feature_cols = meta["feature_cols"]
    best_round = meta["best_round"]

    print("Loading val split into array...", flush=True)
    t0 = time.time()
    X_val, y_val = load_split_array(window_months, "val", cat_code_maps)
    print(f"  X_val {X_val.shape}, took {time.time()-t0:.1f}s", flush=True)
    gc.collect()

    print(f"\nComputing SHAP values (TreeExplainer, best_round={best_round})...", flush=True)
    t0 = time.time()
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_val)
    # LightGBM Booster (non-sklearn API) with objective='binary' is a single
    # continuous-output model to shap -- TreeExplainer returns one 2D array
    # (log-odds/margin space), not a per-class list. Handle both shapes
    # defensively rather than assuming which one comes back.
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values
    print(f"  took {time.time()-t0:.1f}s, sv shape={sv.shape}", flush=True)

    mean_abs_shap = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs_shap)
    print("\nTop 20 features by mean |SHAP value| (val set, log-odds space):", flush=True)
    for i in order[:20]:
        print(f"    {feature_cols[i]:40s} {mean_abs_shap[i]:.4f}", flush=True)

    # Q1: MARKETING_OPTIN -- does it hold up, and what does it actually say?
    idx = feature_cols.index("MARKETING_OPTIN")
    codes = X_val[:, idx]
    inv_map = {v: k for k, v in cat_code_maps["MARKETING_OPTIN"].items()}
    print("\nMARKETING_OPTIN: mean SHAP contribution by category (val set):", flush=True)
    for code_val in sorted(set(codes[~np.isnan(codes)])):
        mask = codes == code_val
        cat_name = inv_map.get(code_val, f"code={code_val}")
        print(f"    {cat_name:20s} n={mask.sum():>10,}  mean_shap={sv[mask, idx].mean():+.4f}  "
              f"base_rate={y_val[mask].mean()*100:.2f}%", flush=True)
    nan_mask = np.isnan(codes)
    if nan_mask.sum() > 0:
        print(f"    {'MISSING':20s} n={nan_mask.sum():>10,}  mean_shap={sv[nan_mask, idx].mean():+.4f}  "
              f"base_rate={y_val[nan_mask].mean()*100:.2f}%", flush=True)

    # Q2: NEGATIVE_SENTIMENT_CALLS_LAST_6M -- direction check against the
    # baseline's counter-intuitive negative coefficient.
    idx2 = feature_cols.index("NEGATIVE_SENTIMENT_CALLS_LAST_6M")
    vals = X_val[:, idx2]
    shap_feat = sv[:, idx2]
    valid = ~np.isnan(vals)
    corr = np.corrcoef(vals[valid], shap_feat[valid])[0, 1]
    print(f"\nNEGATIVE_SENTIMENT_CALLS_LAST_6M: corr(feature value, SHAP contribution) = {corr:+.4f}", flush=True)
    uniq_vals = sorted(set(vals[valid].astype(int).tolist()))
    for v in uniq_vals[:6]:
        mask = vals == v
        if mask.sum() == 0:
            continue
        print(f"    calls={v}  n={mask.sum():>10,}  mean_shap={shap_feat[mask].mean():+.4f}  "
              f"base_rate={y_val[mask].mean()*100:.2f}%", flush=True)
    if len(uniq_vals) > 6:
        mask = vals >= uniq_vals[6]
        print(f"    calls>={uniq_vals[6]}  n={mask.sum():>10,}  mean_shap={shap_feat[mask].mean():+.4f}  "
              f"base_rate={y_val[mask].mean()*100:.2f}%", flush=True)

    # Persist the raw SHAP array + importance table FIRST -- these are the
    # substantive result and the ~148s TreeExplainer call that produced them
    # shouldn't be at risk of being thrown away by a slow plotting step
    # downstream (which is exactly what happened on the first attempt: the
    # summary_plot beeswarm over the full 1,077,079-row val set ran past the
    # tool's 175s ceiling and got killed before anything was saved).
    #
    # Saved to LOCAL scratch, not the OneDrive-mounted DATA_derived folder --
    # this array is ~430MB (float64, 1.08M x 50) and writing that much to a
    # network/FUSE-mounted folder is exactly the kind of catastrophically
    # slow spill-to-disk pattern already hit once before (Step 4/5's DuckDB
    # temp_directory bug). shap_plot loads it back from the same local path.
    shap_tmp_dir = os.path.expanduser("~/shap_tmp")
    os.makedirs(shap_tmp_dir, exist_ok=True)
    shap_npy_path = f"{shap_tmp_dir}/shap_values_{window_months}m.npy"
    t_save = time.time()
    np.save(shap_npy_path, sv)
    print(f"\nSaved raw SHAP array to {shap_npy_path} ({time.time()-t_save:.1f}s)", flush=True)

    import pandas as pd
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    imp_path = f"{OUT_DIR}/shap_importance_{window_months}m.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"Saved SHAP importance table to {imp_path}", flush=True)

    print("\nRun --step shap_plot separately to render the summary plot "
          "(kept as its own step so the ~150s SHAP computation above is never "
          "at risk of being lost to a slow downstream plotting call).", flush=True)


def run_shap_plot(window_months):
    """Renders the SHAP summary plot from the already-saved .npy array (see
    run_shap()) -- split into its own step because the SHAP computation
    itself takes ~150s, leaving too little of this tool's 175-180s-per-call
    budget for matplotlib import + a beeswarm render + savefig to reliably
    finish too. This step alone is fast."""
    import time
    import pickle
    import shap

    with open(f"{MODEL_DIR}/lightgbm_{window_months}m_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    cat_code_maps = meta["cat_code_maps"]
    feature_cols = meta["feature_cols"]

    shap_npy_path = os.path.expanduser(f"~/shap_tmp/shap_values_{window_months}m.npy")
    sv = np.load(shap_npy_path)
    print(f"Loaded SHAP array {sv.shape} from {shap_npy_path}", flush=True)

    print("Loading val split into array (needed for the plot's feature values)...", flush=True)
    X_val, y_val = load_split_array(window_months, "val", cat_code_maps)
    assert X_val.shape[0] == sv.shape[0], "val split row count doesn't match saved SHAP array"

    # Deterministic content-hash-based subsample for the plot, not the full
    # 1M+ rows -- a beeswarm plot's per-feature point layout is superlinear
    # in sample count and doesn't need every row to be visually meaningful.
    t0 = time.time()
    PLOT_N = 20_000
    if len(sv) > PLOT_N:
        rng = np.random.RandomState(42)
        plot_idx = rng.choice(len(sv), size=PLOT_N, replace=False)
    else:
        plot_idx = np.arange(len(sv))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("presentation", exist_ok=True)
    plt.figure()
    shap.summary_plot(sv[plot_idx], X_val[plot_idx], feature_names=feature_cols,
                       show=False, max_display=15)
    plt.tight_layout()
    out_png = f"presentation/shap_summary_{window_months}m.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved SHAP summary plot to {out_png} ({time.time()-t0:.1f}s, "
          f"{len(plot_idx):,}-row sample)", flush=True)


def load_val_with_type(window_months, cat_code_maps):
    """Same feature-array construction as load_split_array(), but ALSO
    streams CANCELLATION_REASON_CLASSIFICATION alongside it, row-aligned,
    from the SAME query -- fetching it via a separate query afterward would
    risk misalignment (no guaranteed stable row order across two independent
    DuckDB executions, the exact lesson learned from the baseline's
    reproducibility bug). Val only -- this is an evaluation step, not
    something that needs train-split scale."""
    n_rows = get_split_count(window_months, "val")
    X = np.empty((n_rows, len(ALL_FEATURE_COLS)), dtype=np.float32)
    y = np.empty(n_rows, dtype=np.int8)
    cancel_type = np.empty(n_rows, dtype=object)

    path = f"{OUT_DIR}/model_data_{window_months}m.parquet"
    select_cols = ", ".join([numeric_select_sql(c) for c in NUMERIC_COLS] + CATEGORICAL_COLS)
    con = duckdb_conn()
    query = f"""
        SELECT label, CANCELLATION_REASON_CLASSIFICATION, {select_cols}
        FROM read_parquet('{path}')
        WHERE split = 'val'
    """
    reader = con.execute(query).to_arrow_reader(batch_size=BATCH_ROWS)

    pos = 0
    for batch in reader:
        df = batch.to_pandas()
        n = len(df)
        for j, c in enumerate(NUMERIC_COLS):
            X[pos:pos + n, j] = df[c].to_numpy(dtype="float32")
        for j, c in enumerate(CATEGORICAL_COLS):
            col = df[c]
            if c == "HOSPITAL_EXCESS_AMT":
                col = col.replace("", np.nan)
            codes = col.map(cat_code_maps[c])
            X[pos:pos + n, len(NUMERIC_COLS) + j] = codes.to_numpy(dtype="float32")
        y[pos:pos + n] = df["label"].to_numpy(dtype="int8")
        cancel_type[pos:pos + n] = df["CANCELLATION_REASON_CLASSIFICATION"].fillna("(unclassified)").to_numpy(dtype=object)
        pos += n
    con.close()
    assert pos == n_rows, f"row count mismatch: filled {pos}, expected {n_rows}"
    return X, y, cancel_type


def run_eval_by_type(window_months):
    """Breaks the LightGBM model's val-set performance down by cancellation
    type (Switcher / System Lapsed / Non-Recoverable / Leave PHI /
    unclassified). One unified risk score drives contact decisions in
    practice, so the business-relevant question isn't a separate per-type
    PR-AUC (which would need an arbitrary choice of what counts as
    "negative" for a one-vs-rest comparison) -- it's: using the SAME global
    top-K%-contacted thresholds already reported for the overall model, what
    recall do we get on EACH type, and what's the composition of who actually
    gets contacted."""
    import pickle
    from sklearn.metrics import roc_auc_score, average_precision_score

    print("Loading saved model + metadata...", flush=True)
    booster = lgb.Booster(model_file=f"{MODEL_DIR}/lightgbm_{window_months}m.txt")
    with open(f"{MODEL_DIR}/lightgbm_{window_months}m_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    cat_code_maps = meta["cat_code_maps"]
    best_round = meta["best_round"]

    print("Loading val split (features + cancellation type, row-aligned)...", flush=True)
    X_val, y_val, cancel_type = load_val_with_type(window_months, cat_code_maps)
    print(f"  X_val {X_val.shape}", flush=True)

    val_proba = booster.predict(X_val, num_iteration=best_round)
    base_rate = y_val.mean()
    roc_auc = roc_auc_score(y_val, val_proba)
    pr_auc = average_precision_score(y_val, val_proba)
    print(f"\n=== LightGBM ({window_months}m window) -- VAL set, overall (sanity check) ===", flush=True)
    print(f"  n_val={len(y_val):,}  base rate: {base_rate*100:.2f}%  ROC-AUC: {roc_auc:.4f}  PR-AUC: {pr_auc:.4f}", flush=True)

    types_present = sorted(set(cancel_type[y_val == 1].tolist()))
    order = np.argsort(-val_proba)
    y_sorted = y_val[order]
    type_sorted = cancel_type[order]

    print("\n=== Recall by cancellation type, at each top-K% contacted threshold ===", flush=True)
    print("(Using ONE unified risk score/threshold, as HBF's retention team would in practice --", flush=True)
    print(" not a separate model or threshold per type.)", flush=True)
    for pct in [1, 5, 10, 20]:
        k = int(len(y_val) * pct / 100)
        contacted_types = type_sorted[:k][y_sorted[:k] == 1]
        print(f"\n  top {pct:>2}% contacted (n={k:,}):", flush=True)
        for t in types_present:
            total_t = int(((cancel_type == t) & (y_val == 1)).sum())
            caught_t = int((contacted_types == t).sum())
            recall_t = caught_t / total_t if total_t > 0 else float("nan")
            print(f"    {t:20s} n_total={total_t:>7,}  caught={caught_t:>7,}  recall={recall_t*100:5.1f}%", flush=True)

    print("\n=== Composition of who gets contacted (top 5%), among actual cancellers caught ===", flush=True)
    k5 = int(len(y_val) * 5 / 100)
    caught_mask = y_sorted[:k5] == 1
    caught_types_5pct = type_sorted[:k5][caught_mask]
    total_caught = len(caught_types_5pct)
    for t in types_present:
        n_t = int((caught_types_5pct == t).sum())
        print(f"    {t:20s} {n_t:>7,} of {total_caught:,} caught cancellers ({n_t/total_caught*100:5.1f}%)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True,
                         choices=["train", "shap", "shap_plot", "eval_by_type"])
    parser.add_argument("--window", type=int, required=True)
    args = parser.parse_args()

    if args.step == "train":
        train_lightgbm(args.window)
    elif args.step == "shap":
        run_shap(args.window)
    elif args.step == "shap_plot":
        run_shap_plot(args.window)
    else:
        import lightgbm as lgb
        run_eval_by_type(args.window)
