"""
Step 5 - Modeling, starting with the 3-month prediction window.

Order of work (per-window, reusable for the other 6 windows later):
  1. Join the window's label+split file to the shared features table -> one
     model-ready parquet (DuckDB join, same memory-safe pattern as Step 4).
  2. Baseline: Logistic Regression (all categoricals one-hot encoded --
     all are low cardinality, checked first: max 10 distinct values) with
     class_weight='balanced'. This is the "how far can a simple linear
     model get" floor that LightGBM needs to beat to justify its complexity.
  3. LightGBM with native categorical support + class weighting, then SHAP.
  4. Evaluate both on val (never touch test until final model choice).

Run:
    python3 scripts/05_train_baseline.py --step build --window 3
    python3 scripts/05_train_baseline.py --step baseline --window 3
"""
import argparse
import gc
import os
import numpy as np
import pandas as pd

FEATURES_PATH = "DATA_derived/features.parquet"
LABEL_PATH_TMPL = "DATA_derived/label_window_{n}m_split.parquet"
OUT_DIR = "DATA_derived"
MODEL_DIR = "models"

CATEGORICAL_COLS = [
    "POLICY_OWNER_AGE_BAND", "SCALE", "STATE_CODE", "CHANNEL_ACQUISITION",
    "TENURE_BAND", "YOUNGEST_DEPENDANT_AGE_BAND", "PRODUCT_HOLDINGS",
    "HOSPITAL_TIER", "EXTRAS_TIER", "HOSPITAL_EXCESS_AMT", "PAYMENT_METHOD_DESC",
    "PAYMENT_FREQUENCY_DESC", "ARREARS_FLAG", "MARKETING_OPTIN",
    "CALL_NEGATIVE_SENTIMENT", "RESOLUTION_TIME",
]

ID_COLS = ["POLICY_HASH", "INFORMATION_DATE"]
LABEL_META_COLS = ["label", "split", "CANCELLATION_REASON_CLASSIFICATION", "CANCELLATIONREASONTYPE"]


def duckdb_conn():
    import duckdb
    tmp_dir = os.path.expanduser("~/duckdb_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='3GB'")
    con.execute("PRAGMA threads=2")
    con.execute(f"PRAGMA temp_directory='{tmp_dir}'")
    return con


def _trim_memory():
    """Ask glibc to return freed heap pages to the OS. DuckDB connections can
    leave RSS elevated after .close() even though the memory is logically
    free -- without this, three short-lived 'lightweight' DuckDB connections
    in a row can leave enough resident memory behind to tip a later
    allocation over the edge, even though each one individually stays well
    under its own memory_limit."""
    import ctypes
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def build_model_dataset(window_months):
    label_path = LABEL_PATH_TMPL.format(n=window_months)
    out_path = f"{OUT_DIR}/model_data_{window_months}m.parquet"

    query = f"""
    COPY (
        SELECT f.*, l.label, l.split,
               l.CANCELLATION_REASON_CLASSIFICATION, l.CANCELLATIONREASONTYPE
        FROM read_parquet('{label_path}') l
        JOIN read_parquet('{FEATURES_PATH}') f
          ON l.POLICY_HASH = f.POLICY_HASH AND l.INFORMATION_DATE = f.INFORMATION_DATE
    ) TO '{out_path}' (FORMAT PARQUET)
    """
    con = duckdb_conn()
    con.execute(query)
    con.close()

    con2 = duckdb_conn()
    n_label = con2.execute(f"SELECT COUNT(*) FROM read_parquet('{label_path}')").fetchone()[0]
    n_out = con2.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
    by_split = con2.execute(f"""
        SELECT split, COUNT(*) n, ROUND(100.0*AVG(label),3) pos_rate
        FROM read_parquet('{out_path}') GROUP BY split ORDER BY split
    """).fetchdf()
    con2.close()

    print(f"Saved {out_path}: {n_out:,} rows (label file had {n_label:,} -- match: {n_out == n_label})", flush=True)
    print(by_split.to_string(), flush=True)


NUMERIC_COLS = [
    "ACTIVE_ADULT_COUNT", "ACTIVE_CHILD_COUNT", "INCOME_TIER", "MONTHLY_PREMIUM",
    "LHC_LOADING_RATE",
    "TOTAL_CLAIM_COUNT_ROLL3M", "TOTAL_BENEFIT_PAID_ROLL3M", "TOTAL_OUT_OF_POCKET_ROLL3M",
    "HOSPITAL_CLAIM_COUNT_ROLL3M", "REVERSAL_FLAG_ROLL3M", "NO_BENEFIT_PAID_FLAG_ROLL3M",
    "TOTAL_CLAIM_COUNT_ROLL6M", "TOTAL_BENEFIT_PAID_ROLL6M", "TOTAL_OUT_OF_POCKET_ROLL6M",
    "HOSPITAL_CLAIM_COUNT_ROLL6M", "REVERSAL_FLAG_ROLL6M", "NO_BENEFIT_PAID_FLAG_ROLL6M",
    "TOTAL_CLAIM_COUNT_ROLL12M", "TOTAL_BENEFIT_PAID_ROLL12M", "TOTAL_OUT_OF_POCKET_ROLL12M",
    "HOSPITAL_CLAIM_COUNT_ROLL12M", "REVERSAL_FLAG_ROLL12M", "NO_BENEFIT_PAID_FLAG_ROLL12M",
    "MONTHS_SINCE_LAST_CLAIM",
    "MONTHS_SINCE_LAST_SURVEY", "LAST_OVERALL_EXPERIENCE_RATING", "LAST_NPS_RATING",
    "LAST_SERVICE_RATING", "RECENT_LOW_RATING_FLAG", "HAS_EVER_SURVEYED",
    "HAD_COMPLAINT_FLAG", "ARREARS_FLAG_LAG2M", "PREMIUM_CHANGE_3M",
    "NEGATIVE_SENTIMENT_CALLS_LAST_6M",
]

# NULL here genuinely means "insufficient history / never happened" (see Step 4
# design notes), not "value unknown" -- these get a _MISSING indicator so a
# linear model can still see that signal, then median-imputed (median computed
# on TRAIN only) since sklearn can't take NaN directly. LightGBM later needs
# none of this -- it handles missing values natively.
NUMERIC_COLS_WITH_NULLS = [
    "INCOME_TIER", "MONTHLY_PREMIUM", "LHC_LOADING_RATE",
    "LAST_OVERALL_EXPERIENCE_RATING", "LAST_NPS_RATING", "LAST_SERVICE_RATING",
    "ARREARS_FLAG_LAG2M", "PREMIUM_CHANGE_3M",
]

BATCH_ROWS = 400_000


def numeric_select_sql(c):
    if c == "LHC_LOADING_RATE":
        # stored as VARCHAR with inconsistent formatting ("12.00" vs "12.000")
        # -- checked first: TRY_CAST collapses these correctly, no failures.
        return f"TRY_CAST({c} AS DOUBLE) AS {c}"
    return c


def compute_medians(window_months):
    """Single aggregate query covering every null-prone numeric column at
    once (train split only), instead of one full-column scan per column.
    Doing 8 separate full scans of a 756MB parquet file back-to-back turned
    out to matter: on this box (3.8GB RAM, no swap, page cache counted
    against the sandbox's memory ceiling), the accumulated page cache from
    dozens of near-back-to-back full scans (medians + scaling + categories
    combined were 57 separate scans) was enough to tip the later
    load_sample() allocation into an OOM that never reproduced when that
    same query was tested in isolation. One pass per function fixes that at
    the root instead of papering over it with memory_limit tweaks."""
    con = duckdb_conn()
    con.execute("PRAGMA memory_limit='768MB'")
    path = f"{OUT_DIR}/model_data_{window_months}m.parquet"
    select_exprs = []
    for c in NUMERIC_COLS_WITH_NULLS:
        expr = "TRY_CAST(LHC_LOADING_RATE AS DOUBLE)" if c == "LHC_LOADING_RATE" else c
        select_exprs.append(f"median({expr}) AS {c}")
    query = f"SELECT {', '.join(select_exprs)} FROM read_parquet('{path}') WHERE split='train'"
    row = con.execute(query).fetchone()
    medians = {c: float(v) for c, v in zip(NUMERIC_COLS_WITH_NULLS, row)}
    con.close()
    _trim_memory()
    return medians


def compute_scaling(window_months):
    """Mean/std per numeric column from TRAIN only, needed because SGD's
    gradient updates are dominated by whichever feature happens to have the
    largest raw scale (e.g. TOTAL_BENEFIT_PAID in the thousands vs a 0/1
    flag) if features aren't standardized first -- this was the actual bug
    behind the first run's near-random ROC-AUC of 0.55.

    Single query for all 33 columns (one full scan) instead of 33 separate
    ones -- see compute_medians() for why that matters here."""
    con = duckdb_conn()
    con.execute("PRAGMA memory_limit='768MB'")
    path = f"{OUT_DIR}/model_data_{window_months}m.parquet"
    select_exprs = []
    for c in NUMERIC_COLS:
        expr = "TRY_CAST(LHC_LOADING_RATE AS DOUBLE)" if c == "LHC_LOADING_RATE" else c
        select_exprs.append(f"AVG({expr}) AS {c}__mean")
        select_exprs.append(f"STDDEV_POP({expr}) AS {c}__std")
    query = f"SELECT {', '.join(select_exprs)} FROM read_parquet('{path}') WHERE split='train'"
    row = con.execute(query).fetchone()
    stats = {}
    for i, c in enumerate(NUMERIC_COLS):
        mean, std = row[2 * i], row[2 * i + 1]
        stats[c] = (float(mean or 0.0), float(std) if std and std > 1e-9 else 1.0)
    con.close()
    _trim_memory()
    return stats


def get_categories(window_months):
    """Enumerate each categorical column's distinct values up front (all are
    low cardinality, checked earlier: max 10) so the OneHotEncoder can be
    built with fixed categories rather than needing to fit on loaded data.

    Single query with one array_agg(DISTINCT ...) per column (one full scan)
    instead of 16 separate SELECT DISTINCT scans -- see compute_medians()
    for why that matters here."""
    con = duckdb_conn()
    con.execute("PRAGMA memory_limit='768MB'")
    path = f"{OUT_DIR}/model_data_{window_months}m.parquet"
    select_exprs = []
    for c in CATEGORICAL_COLS:
        if c == "HOSPITAL_EXCESS_AMT":
            # '' only appears on 47 of 11.29M rows -- almost certainly a
            # parsing artifact, not a deliberate distinct value, so it's
            # folded into the same MISSING bucket as NULL.
            expr = f"CASE WHEN {c} IS NULL OR {c} = '' THEN 'MISSING' ELSE {c} END"
        else:
            expr = f"COALESCE({c}, 'MISSING')"
        select_exprs.append(f"array_agg(DISTINCT {expr}) AS {c}")
    query = f"SELECT {', '.join(select_exprs)} FROM read_parquet('{path}')"
    row = con.execute(query).fetchone()
    cats = {c: sorted(set(v)) for c, v in zip(CATEGORICAL_COLS, row)}
    con.close()
    _trim_memory()
    return cats


def make_encoder(categories):
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np
    cat_list = [categories[c] for c in CATEGORICAL_COLS]
    return OneHotEncoder(categories=cat_list, handle_unknown="ignore",
                          sparse_output=True, dtype=np.float32)


def chunk_reader(window_months, split_name):
    """Streams one split in BATCH_ROWS-row arrow batches -- keeps peak memory
    bounded regardless of the split's total row count (this box has 3.8GB RAM
    and no swap; a plain fetchdf() on the full 7.5M-row train split OOM'd).

    ORDER BY hash(...) is not cosmetic: without a fixed row order, the row
    order model_data_*.parquet happens to come out in depends on incidental
    physical layout of its input files (verified by rebuilding it twice after
    an unrelated upstream fix and getting a different-but-internally-stable
    order both times). SGDClassifier.partial_fit's convergence path is
    genuinely sensitive to example order over just a few epochs, so an
    unguaranteed order made the baseline's reported metrics silently
    non-reproducible across reruns -- caught when a rebuild shifted PR-AUC
    from 0.352 to 0.255 with literally the same code. hash() gives a fixed,
    reproducible pseudo-random order regardless of how the upstream files
    were built or rebuilt."""
    path = f"{OUT_DIR}/model_data_{window_months}m.parquet"
    select_cols = ", ".join(
        [numeric_select_sql(c) for c in NUMERIC_COLS] + CATEGORICAL_COLS
    )
    con = duckdb_conn()
    # No ORDER BY here (deliberately) -- this is only used for val/test
    # scoring now, and ROC-AUC / PR-AUC are computed over the full set of
    # (label, score) pairs regardless of what order they arrived in, so
    # there's nothing to gain from paying for a sort here. (Training moved
    # to load_sample()'s single deterministic batch fit -- see its docstring
    # for why streaming+partial_fit was abandoned.)
    query = f"""
        SELECT label, {select_cols}
        FROM read_parquet('{path}')
        WHERE split = '{split_name}'
    """
    reader = con.execute(query).to_arrow_reader(batch_size=BATCH_ROWS)
    for batch in reader:
        yield batch.to_pandas()
    con.close()


def build_design_matrix(df, medians, scaling, encoder):
    """Mutates df in place rather than df.copy() -- this function always
    receives a df that is freshly built by its one caller and never reused
    afterward (load_sample() immediately returns whatever this returns), so
    a defensive full-frame copy here was pure waste. On a 1.5M-row x 48-col
    frame with 16 string columns, that copy (plus a second copy of just the
    string columns a few lines later) meant up to 3 overlapping copies of
    the same data alive at once -- confirmed as the actual OOM cause here:
    an isolated test of the DuckDB fetch alone (no design-matrix build)
    succeeded fine, but the full pipeline died every time at this step."""
    import numpy as np
    from scipy import sparse

    df["HOSPITAL_EXCESS_AMT"] = df["HOSPITAL_EXCESS_AMT"].replace("", np.nan)

    num_blocks = []
    num_names = []
    for c in NUMERIC_COLS:
        col = df[c].astype("float64")
        if c in NUMERIC_COLS_WITH_NULLS:
            missing = col.isna().astype("float32").to_numpy().reshape(-1, 1)
            num_blocks.append(missing)
            num_names.append(f"{c}_MISSING")
            col = col.fillna(medians[c])
        mean, std = scaling[c]
        col = (col - mean) / std
        num_blocks.append(col.to_numpy(dtype="float32").reshape(-1, 1))
        num_names.append(c)
    X_num = np.hstack(num_blocks).astype("float32")

    for c in CATEGORICAL_COLS:
        df[c] = df[c].where(df[c].notna(), "MISSING")
    X_cat = encoder.transform(df[CATEGORICAL_COLS])

    X = sparse.hstack([sparse.csr_matrix(X_num), X_cat], format="csr")
    if not hasattr(build_design_matrix, "_names_printed"):
        build_design_matrix.feature_names = num_names + list(
            encoder.get_feature_names_out(CATEGORICAL_COLS)
        )
        build_design_matrix._names_printed = True
    labels = df["label"].to_numpy()
    del df
    return X, labels


def load_sample(window_months, split_name, n_rows, medians, scaling, encoder):
    """Loads a fixed, deterministic sample (or the whole split, for val/test)
    as ONE design matrix -- used instead of chunk_reader()'s streaming for
    anything that goes through a proper batch .fit() rather than
    partial_fit.

    Sampling is done with a hash-of-content filter (abs(hash(POLICY_HASH))
    % divisor = 0), NOT DuckDB's native USING SAMPLE (reservoir, seed).
    Reservoir sampling is only deterministic relative to a FIXED physical
    row order, and that order is NOT stable across rebuilds of
    model_data_*.parquet -- confirmed directly: rebuilding the file from
    identical upstream logical content changed the reservoir sample enough
    to shift PR-AUC from 0.3576 to 0.3579 (small, but real -- proof the
    sample itself wasn't reproducible, only the query against one fixed
    file was). A hash of POLICY_HASH is a property of the DATA, not its
    position on disk, so it selects the exact same logical rows regardless
    of how any given rebuild happens to lay them out physically."""
    path = f"{OUT_DIR}/model_data_{window_months}m.parquet"
    select_cols = ", ".join(
        [numeric_select_sql(c) for c in NUMERIC_COLS] + CATEGORICAL_COLS
    )
    con = duckdb_conn()
    con.execute("PRAGMA memory_limit='1GB'")
    if n_rows:
        split_total = con.execute(
            f"SELECT count(*) FROM read_parquet('{path}') WHERE split = '{split_name}'"
        ).fetchone()[0]
        divisor = max(1, round(split_total / n_rows))
        query = f"""
            SELECT label, {select_cols}
            FROM read_parquet('{path}')
            WHERE split = '{split_name}' AND abs(hash(POLICY_HASH)) % {divisor} = 0
        """
    else:
        query = f"SELECT label, {select_cols} FROM read_parquet('{path}') WHERE split = '{split_name}'"
    df = con.execute(query).fetchdf()
    con.close()
    return build_design_matrix(df, medians, scaling, encoder)


def run_baseline(window_months):
    import time
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    print("Computing train medians (for null-prone numeric cols)...", flush=True)
    medians = compute_medians(window_months)
    print(f"  {medians}", flush=True)

    print("Computing train mean/std (for feature scaling)...", flush=True)
    scaling = compute_scaling(window_months)
    print(f"  {scaling}", flush=True)

    print("Enumerating categorical values...", flush=True)
    categories = get_categories(window_months)
    encoder = make_encoder(categories)
    # OneHotEncoder needs a .fit() call with matching column shape -- fit it on
    # one row built directly from its own declared categories, which is exact
    # and needs no real data.
    import pandas as pd
    one_row = pd.DataFrame({c: [categories[c][0]] for c in CATEGORICAL_COLS})
    encoder.fit(one_row)

    n_features = len(NUMERIC_COLS) + len(NUMERIC_COLS_WITH_NULLS) + sum(len(v) for v in categories.values())
    print(f"  design matrix will have {n_features} columns", flush=True)
    gc.collect()
    _trim_memory()

    # Switched from streaming SGDClassifier.partial_fit to a single batch
    # LogisticRegression.fit() on a fixed-size deterministic sample. The
    # streaming approach turned out to be genuinely non-reproducible: even
    # after forcing a fixed row order (ORDER BY hash(POLICY_HASH)), repeated
    # runs on identical data still produced different converged models
    # (observed ROC-AUC ranging 0.949-0.961 across runs) -- online SGD's
    # convergence path depends on more than just logical row order once
    # DuckDB's own multi-threaded sort/stream execution is in the mix, and
    # chasing that down further wasn't worth it for a disposable reference
    # model. A proper batch fit on a fixed sample has no such path-dependence:
    # given the same data, lbfgs converges to the same optimum every time.
    #
    # Sample size: measured (via a standalone instrumented run) that loading
    # 1,500,000 rows into a single pandas DataFrame + building the design
    # matrix peaked at ~3.7-4GB RSS on this box -- right at the 3.8GB/no-swap
    # ceiling, and it OOM'd (exit 137) every time despite the query itself
    # succeeding fine in isolation. Root cause: DuckDB leaves ~600MB of
    # non-reclaimable buffer-pool RSS resident even after the metadata
    # queries close their connections, and fetchdf()+build_design_matrix()
    # together cost roughly 3x the DataFrame's own logical memory (Arrow
    # intermediate buffers, float64->float32 conversions, the one-hot sparse
    # block). 750,000 rows measured at a safe ~2.3GB peak (vs 3.8GB total)
    # -- still ~12,500 positive examples at this window's 1.67% base rate,
    # far more than a 128-feature logistic regression needs.
    N_SAMPLE = 750_000
    print(f"\nLoading a fixed {N_SAMPLE:,}-row train sample (single batch, not streamed)...", flush=True)
    t0 = time.time()
    X_train, y_train = load_sample(window_months, "train", N_SAMPLE, medians, scaling, encoder)
    print(f"  X_train {X_train.shape}, took {time.time()-t0:.1f}s, "
          f"base rate {100*y_train.mean():.2f}%", flush=True)

    clf = LogisticRegression(class_weight="balanced", max_iter=300, solver="lbfgs")
    print("Fitting LogisticRegression (single batch call)...", flush=True)
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"  fit took {time.time()-t0:.1f}s", flush=True)
    del X_train, y_train
    gc.collect()

    print("Scoring val split (streamed)...", flush=True)
    y_true_all, y_score_all = [], []
    for chunk_df in chunk_reader(window_months, "val"):
        X_chunk, y_chunk = build_design_matrix(chunk_df, medians, scaling, encoder)
        proba = clf.predict_proba(X_chunk)[:, 1]
        y_true_all.append(y_chunk)
        y_score_all.append(proba)
    y_val = np.concatenate(y_true_all)
    val_proba = np.concatenate(y_score_all)

    roc_auc = roc_auc_score(y_val, val_proba)
    pr_auc = average_precision_score(y_val, val_proba)
    base_rate = y_val.mean()

    print(f"\n=== Baseline (Logistic Regression, {window_months}m window) -- VAL set ===", flush=True)
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

    import pickle
    with open(f"{MODEL_DIR}/baseline_logreg_{window_months}m.pkl", "wb") as f:
        pickle.dump({"model": clf, "medians": medians, "scaling": scaling,
                     "categories": categories, "encoder": encoder,
                     "feature_names": getattr(build_design_matrix, "feature_names", None)}, f)
    print(f"\nSaved model to {MODEL_DIR}/baseline_logreg_{window_months}m.pkl", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["build", "baseline", "state_check"])
    parser.add_argument("--window", type=int, required=True)
    args = parser.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)

    if args.step == "build":
        build_model_dataset(args.window)
    elif args.step == "baseline":
        run_baseline(args.window)
    else:
        raise NotImplementedError(f"step '{args.step}' not written yet")
