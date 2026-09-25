"""
Step 9 (pitch support) - Policy explorer data for the live-demo artifact.

Builds a small, curated JSON of REAL individual policies (3-month window),
each with: risk score, percentile bucket, display fields (age band, tenure,
state, premium, arrears flag, payment method, hospital tier, product), the
top-3 model reasons (via LightGBM's native pred_contrib, i.e. real per-row
SHAP-equivalent feature contributions -- not invented), a recommended action
(reusing the EXACT constants/logic from 07_cost_benefit.py, never re-derived),
and, for the "resolved" tab only, the real known outcome.

Two pools, both drawn from the SAME validated val-split load used everywhere
else in this project (no new loading logic, to avoid re-introducing a bug
already fixed once):
  - "resolved": rows from any month BEFORE the most recent one in val -- the
    3-month outcome window has long since closed, so label is a real known
    fact. Used for the "proof it works" tab.
  - "prospective": rows from the MOST RECENT month in val -- technically we
    do have a label for these too (the underlying data extends far enough to
    compute it), but the demo deliberately does NOT surface it for this pool,
    since the honest framing is "this is what the model says about the
    current book right now" (mirrors exactly how Section 1 of
    07_cost_benefit.py treats this same month). Used for the "not yet
    resolved" tab.

Run:
    python3 scripts/09_policy_explorer.py --window 3
"""
import argparse
import json
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import importlib
baseline_mod = importlib.import_module("05_train_baseline")
lgb_mod = importlib.import_module("06_train_lightgbm")
cb_mod = importlib.import_module("07_cost_benefit")

OUT_DIR = "DATA_derived"
MODEL_DIR = "models"
PRESENTATION_DIR = "presentation"

DISPLAY_COLS = [
    "POLICY_OWNER_AGE_BAND", "STATE_CODE", "TENURE_BAND", "HOSPITAL_TIER",
    "PAYMENT_METHOD_DESC", "ARREARS_FLAG", "PRODUCT_HOLDINGS",
]

FEATURE_LABELS = {
    "ACTIVE_ADULT_COUNT": "Number of adults on the policy",
    "ACTIVE_CHILD_COUNT": "Number of children on the policy",
    "INCOME_TIER": "Income tier",
    "MONTHLY_PREMIUM": "Monthly premium",
    "LHC_LOADING_RATE": "Loyalty loading (joined later in life)",
    "TOTAL_CLAIM_COUNT_ROLL3M": "How many claims they've made (3mo)",
    "TOTAL_BENEFIT_PAID_ROLL3M": "Benefits paid out recently (3mo)",
    "TOTAL_OUT_OF_POCKET_ROLL3M": "Out-of-pocket costs recently (3mo)",
    "HOSPITAL_CLAIM_COUNT_ROLL3M": "Hospital claims recently (3mo)",
    "REVERSAL_FLAG_ROLL3M": "Claim reversals (3mo)",
    "NO_BENEFIT_PAID_FLAG_ROLL3M": "Claims with no benefit paid (3mo)",
    "TOTAL_CLAIM_COUNT_ROLL6M": "How many claims they've made (6mo)",
    "TOTAL_BENEFIT_PAID_ROLL6M": "Benefits paid out recently (6mo)",
    "TOTAL_OUT_OF_POCKET_ROLL6M": "Out-of-pocket costs recently (6mo)",
    "HOSPITAL_CLAIM_COUNT_ROLL6M": "Hospital claims recently (6mo)",
    "REVERSAL_FLAG_ROLL6M": "Claim reversals (6mo)",
    "NO_BENEFIT_PAID_FLAG_ROLL6M": "Claims with no benefit paid (6mo)",
    "TOTAL_CLAIM_COUNT_ROLL12M": "How many claims they've made (12mo)",
    "TOTAL_BENEFIT_PAID_ROLL12M": "Benefits paid out recently (12mo)",
    "TOTAL_OUT_OF_POCKET_ROLL12M": "Out-of-pocket costs recently (12mo)",
    "HOSPITAL_CLAIM_COUNT_ROLL12M": "Hospital claims recently (12mo)",
    "REVERSAL_FLAG_ROLL12M": "Claim reversals (12mo)",
    "NO_BENEFIT_PAID_FLAG_ROLL12M": "Claims with no benefit paid (12mo)",
    "MONTHS_SINCE_LAST_CLAIM": "Time since their last claim",
    "MONTHS_SINCE_LAST_SURVEY": "Time since their last survey response",
    "LAST_OVERALL_EXPERIENCE_RATING": "Their last satisfaction rating",
    "LAST_NPS_RATING": "Their last NPS rating",
    "LAST_SERVICE_RATING": "Their last service rating",
    "RECENT_LOW_RATING_FLAG": "Recent low satisfaction rating",
    "HAS_EVER_SURVEYED": "Whether they've ever completed a survey",
    "HAD_COMPLAINT_FLAG": "Recent complaint on file",
    "ARREARS_FLAG_LAG2M": "Payment arrears history",
    "PREMIUM_CHANGE_3M": "Recent premium change",
    "NEGATIVE_SENTIMENT_CALLS_LAST_6M": "Negative-sentiment calls, last 6mo",
    "POLICY_OWNER_AGE_BAND": "Age",
    "SCALE": "Membership type (single/couple/family)",
    "STATE_CODE": "State",
    "CHANNEL_ACQUISITION": "How they joined",
    "TENURE_BAND": "How long they've been a member",
    "YOUNGEST_DEPENDANT_AGE_BAND": "Youngest dependant's age",
    "PRODUCT_HOLDINGS": "What cover they hold",
    "HOSPITAL_TIER": "Hospital cover tier",
    "EXTRAS_TIER": "Extras cover tier",
    "HOSPITAL_EXCESS_AMT": "Hospital excess amount",
    "PAYMENT_METHOD_DESC": "How they pay",
    "PAYMENT_FREQUENCY_DESC": "How often they pay",
    "ARREARS_FLAG": "Currently in arrears",
    "MARKETING_OPTIN": "Marketing sign-up status",
    "CALL_NEGATIVE_SENTIMENT": "Recent call sentiment",
    "RESOLUTION_TIME": "How long issues take to resolve",
}


def load_val_full(window_months, cat_code_maps):
    """Same validated pattern as load_val_with_premium_and_type() in
    07_cost_benefit.py, extended with POLICY_HASH + the raw display columns
    needed for the policy-explorer cards. One single query -- never two,
    per the row-order lesson from the baseline reproducibility bug."""
    n_rows = lgb_mod.get_split_count(window_months, "val")
    X = np.empty((n_rows, len(lgb_mod.ALL_FEATURE_COLS)), dtype=np.float32)
    y = np.empty(n_rows, dtype=np.int8)
    premium = np.empty(n_rows, dtype=np.float64)
    cancel_type = np.empty(n_rows, dtype=object)
    info_date = np.empty(n_rows, dtype=object)
    policy_hash = np.empty(n_rows, dtype=object)
    display = {c: np.empty(n_rows, dtype=object) for c in DISPLAY_COLS}

    path = f"{OUT_DIR}/model_data_{window_months}m.parquet"
    # DISPLAY_COLS are already a subset of CATEGORICAL_COLS -- selecting them a
    # second time under the same names would produce duplicate-named columns
    # in the query result (pandas then returns a DataFrame, not a Series, for
    # df[c], which breaks .map()). So select each categorical column ONCE and
    # capture its raw string value (for display) before mapping it to a code
    # (for the model).
    select_cols = ", ".join(
        [lgb_mod.numeric_select_sql(c) for c in lgb_mod.NUMERIC_COLS] + lgb_mod.CATEGORICAL_COLS
    )
    con = lgb_mod.duckdb_conn()
    query = f"""
        SELECT label, CANCELLATION_REASON_CLASSIFICATION,
               MONTHLY_PREMIUM AS RAW_PREMIUM,
               INFORMATION_DATE, POLICY_HASH, {select_cols}
        FROM read_parquet('{path}')
        WHERE split = 'val'
    """
    reader = con.execute(query).to_arrow_reader(batch_size=lgb_mod.BATCH_ROWS)

    pos = 0
    for batch in reader:
        df = batch.to_pandas()
        n = len(df)
        for j, c in enumerate(lgb_mod.NUMERIC_COLS):
            X[pos:pos + n, j] = df[c].to_numpy(dtype="float32")
        for j, c in enumerate(lgb_mod.CATEGORICAL_COLS):
            col = df[c]
            if c in DISPLAY_COLS:
                display[c][pos:pos + n] = col.astype(object).where(col.notna(), None).to_numpy(dtype=object)
            if c == "HOSPITAL_EXCESS_AMT":
                col = col.replace("", np.nan)
            codes = col.map(cat_code_maps[c])
            X[pos:pos + n, len(lgb_mod.NUMERIC_COLS) + j] = codes.to_numpy(dtype="float32")
        y[pos:pos + n] = df["label"].to_numpy(dtype="int8")
        premium[pos:pos + n] = df["RAW_PREMIUM"].to_numpy(dtype="float64")
        cancel_type[pos:pos + n] = df["CANCELLATION_REASON_CLASSIFICATION"].fillna("(unclassified)").to_numpy(dtype=object)
        info_date[pos:pos + n] = df["INFORMATION_DATE"].astype(str).to_numpy(dtype=object)
        policy_hash[pos:pos + n] = df["POLICY_HASH"].astype(str).to_numpy(dtype=object)
        pos += n
    con.close()
    assert pos == n_rows, f"row count mismatch: filled {pos}, expected {n_rows}"
    return X, y, premium, cancel_type, info_date, policy_hash, display


def bucket_for(score, thresholds):
    q99, q95, q90, q75 = thresholds
    if score >= q99:
        return "Top 1%"
    if score >= q95:
        return "Top 1-5%"
    if score >= q90:
        return "Top 5-10%"
    if score >= q75:
        return "Top 10-25%"
    return "Rest (bottom 75%)"


def best_action(churn_prob, annual_premium, success_rate_multiplier):
    """Exact same 4-option decision as 07_cost_benefit.py: digital/phone x
    with/without a 12%-of-annual-premium discount, picking whichever has the
    highest expected net value (or 'No action' if none clears zero)."""
    V = cb_mod.VALUE_RETAINED_FLAT
    max_discount = annual_premium * cb_mod.MAX_DISCOUNT_PCT_OF_ANNUAL

    digital_rate = cb_mod.BASELINE_SUCCESS_RATE * success_rate_multiplier
    phone_rate = digital_rate * cb_mod.BASELINE_PHONE_MULTIPLIER
    digital_disc_rate = digital_rate * cb_mod.BASELINE_DISCOUNT_MULTIPLIER
    phone_disc_rate = phone_rate * cb_mod.BASELINE_DISCOUNT_MULTIPLIER

    options = {
        "Digital nudge, no discount": cb_mod.expected_net_value(
            churn_prob, digital_rate, V, 0.0, cb_mod.DIGITAL_CONTACT_COST),
        "Digital nudge + discount": cb_mod.expected_net_value(
            churn_prob, digital_disc_rate, V, max_discount, cb_mod.DIGITAL_CONTACT_COST),
        "Phone call, no discount": cb_mod.expected_net_value(
            churn_prob, phone_rate, V, 0.0, cb_mod.PHONE_CONTACT_COST),
        "Phone call + discount": cb_mod.expected_net_value(
            churn_prob, phone_disc_rate, V, max_discount, cb_mod.PHONE_CONTACT_COST),
    }
    best_name = max(options, key=options.get)
    best_val = options[best_name]
    discount_pct = round(cb_mod.MAX_DISCOUNT_PCT_OF_ANNUAL * 100)
    if best_val <= 0:
        return "No action (not worth the cost)", best_val
    label = best_name
    if "discount" in best_name.lower() and "no discount" not in best_name.lower():
        label = (f"Digital nudge + {discount_pct}% discount" if "Digital" in best_name
                  else f"Phone call + {discount_pct}% discount")
    return label, best_val


def top3_reasons(booster, X_rows, best_round):
    contrib = booster.predict(X_rows, pred_contrib=True, num_iteration=best_round)
    reasons = []
    for row in contrib:
        feat_contrib = row[:-1]  # last col is the bias/expected-value term
        order = np.argsort(-np.abs(feat_contrib))[:3]
        names = [lgb_mod.ALL_FEATURE_COLS[i] for i in order]
        reasons.append([FEATURE_LABELS.get(n, n) for n in names])
    return reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=3)
    args = ap.parse_args()
    w = args.window

    print("Loading model + metadata...", flush=True)
    import lightgbm as lgb
    booster = lgb.Booster(model_file=f"{MODEL_DIR}/lightgbm_{w}m.txt")
    with open(f"{MODEL_DIR}/lightgbm_{w}m_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    cat_code_maps = meta["cat_code_maps"]
    best_round = meta["best_round"]

    print("Loading val split (features + display fields, one pass)...", flush=True)
    X, y, premium, cancel_type, info_date, policy_hash, display = load_val_full(w, cat_code_maps)
    print(f"  X {X.shape}", flush=True)

    print("Scoring full val set...", flush=True)
    churn_prob = booster.predict(X, num_iteration=best_round)

    medians = baseline_mod.compute_medians(w)
    if np.isnan(premium).any():
        premium = np.where(np.isnan(premium), medians["MONTHLY_PREMIUM"], premium)
    annual_premium = premium * 12.0

    q99, q95, q90, q75 = np.percentile(churn_prob, [99, 95, 90, 75])
    thresholds = (q99, q95, q90, q75)
    print(f"  thresholds: top1%={q99:.4f} top5%={q95:.4f} top10%={q90:.4f} top25%={q75:.4f}", flush=True)

    most_recent_month = max(info_date)
    is_recent = info_date == most_recent_month
    print(f"  most recent month: {most_recent_month} ({is_recent.sum():,} rows)", flush=True)

    rng = np.random.default_rng(7)

    def pick(mask, extra_mask=None, n=1):
        idx = np.where(mask & (extra_mask if extra_mask is not None else True))[0]
        if len(idx) == 0:
            return []
        return list(rng.choice(idx, size=min(n, len(idx)), replace=False))

    resolved_pool = ~is_recent
    prospective_pool = is_recent

    picked_resolved = []
    type_bucket_plan = [
        ("Switcher", "Top 1%", 1), ("Switcher", "Top 1-5%", 1),
        ("System Lapsed", "Top 1%", 1), ("System Lapsed", "Top 1-5%", 1),
        ("Leave PHI", "Top 1%", 1), ("Leave PHI", "Top 1-5%", 1),
        ("Non-Recoverable", "Top 1%", 1),
        ("(unclassified)", "Top 1-5%", 1),
    ]
    bucket_masks = {
        "Top 1%": churn_prob >= q99,
        "Top 1-5%": (churn_prob >= q95) & (churn_prob < q99),
        "Top 5-10%": (churn_prob >= q90) & (churn_prob < q95),
        "Top 10-25%": (churn_prob >= q75) & (churn_prob < q90),
        "Rest (bottom 75%)": churn_prob < q75,
    }
    for t, b, n in type_bucket_plan:
        mask = resolved_pool & (y == 1) & (cancel_type == t) & bucket_masks[b]
        picked_resolved += pick(mask, n=n)

    # honest miss: an actual canceller the model scored LOW (false negative)
    picked_resolved += pick(resolved_pool & (y == 1) & bucket_masks["Rest (bottom 75%)"], n=1)
    # true negatives: actual stayers scored low, for specificity
    picked_resolved += pick(resolved_pool & (y == 0) & bucket_masks["Rest (bottom 75%)"], n=2)

    picked_resolved = [int(i) for i in dict.fromkeys(picked_resolved)]

    picked_prospective = []
    for b in ["Top 1%", "Top 1-5%", "Top 5-10%", "Top 10-25%", "Rest (bottom 75%)"]:
        picked_prospective += pick(prospective_pool & bucket_masks[b], n=2)
    picked_prospective = [int(i) for i in dict.fromkeys(picked_prospective)]

    print(f"  picked {len(picked_resolved)} resolved, {len(picked_prospective)} prospective", flush=True)

    def build_records(indices, tab):
        recs = []
        if not indices:
            return recs
        X_rows = X[indices]
        reasons_all = top3_reasons(booster, X_rows, best_round)
        for k, i in enumerate(indices):
            t = cancel_type[i] if tab == "resolved" else None
            mult = cb_mod.TYPE_SUCCESS_MULTIPLIER.get(t, 1.0) if tab == "resolved" else 1.0
            action_label, action_val = best_action(float(churn_prob[i]), float(annual_premium[i]), mult)
            rec = {
                "policy_hash": str(policy_hash[i])[:10],
                "tab": tab,
                "score_pct": round(float(churn_prob[i]) * 100, 1),
                "bucket": bucket_for(churn_prob[i], thresholds),
                "age_band": display["POLICY_OWNER_AGE_BAND"][i],
                "state": display["STATE_CODE"][i],
                "tenure_band": display["TENURE_BAND"][i],
                "hospital_tier": display["HOSPITAL_TIER"][i],
                "payment_method": display["PAYMENT_METHOD_DESC"][i],
                "arrears_flag": display["ARREARS_FLAG"][i],
                "product": display["PRODUCT_HOLDINGS"][i],
                "monthly_premium": round(float(premium[i]), 2),
                "reasons": reasons_all[k],
                "recommended_action": action_label,
                "recommended_action_value": round(float(action_val), 2),
            }
            if tab == "resolved":
                rec["cancellation_type"] = t
                rec["actual_label"] = int(y[i])
                rec["outcome_text"] = (
                    f"Actually cancelled within {w} months (as {t})" if y[i] == 1
                    else "Actually stayed — no cancellation"
                )
            recs.append(rec)
        return recs

    print("Computing top-3 reasons for the curated sample...", flush=True)
    resolved_records = build_records(picked_resolved, "resolved")
    prospective_records = build_records(picked_prospective, "prospective")

    out = {
        "window_months": w,
        "most_recent_month": str(most_recent_month),
        "resolved": resolved_records,
        "prospective": prospective_records,
    }
    out_path = f"{PRESENTATION_DIR}/policy_explorer_{w}m.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved {out_path} ({len(resolved_records)} resolved + {len(prospective_records)} prospective records)", flush=True)


if __name__ == "__main__":
    main()
