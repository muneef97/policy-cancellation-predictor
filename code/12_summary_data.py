"""
Step 12 - Collect the summary numbers the live demo shows into results/summary_data.json.

Adds three things not produced elsewhere:
  trend          monthly cancellation rate from the raw tables (cancellations by effective cease date
                 / active policies that month); the last month of the extract is still being
                 processed, so it is reported separately and left off the chart
  excluded       evidence for leaving out MARKETING_OPTIN (results/excluded_fields.json)
  examples       real (de-identified) example policies: caught leavers of each type,
                 honest misses and a false alarm (Dec 2025), plus Jan 2026 policies by risk band
plus what drives the final model's scores overall and by type (SHAP contributions).
"""
import numpy as np
import pandas as pd

import config as C
from common import (ACTIONS, annual_premium, best_action, cancel_type, decode, duckdb_conn, load_model, load_split,
                    model_columns, read_json, save_json)

OPTION_KEYS = ["digital_no_discount", "digital_discount", "phone_no_discount", "phone_discount", "no_action"]
ACTION_TEXT = {"Digital, no discount": "Digital nudge, no discount", "Digital + discount": "Digital nudge + 12% discount",
               "Phone, no discount": "Phone call, no discount", "Phone + discount": "Phone call + 12% discount",
               "No action": "No action (not worth the cost)"}
DISPLAY = {"age": "POLICY_OWNER_AGE_BAND", "state": "STATE_CODE", "tenure": "TENURE_BAND", "tier": "HOSPITAL_TIER",
           "pay": "PAYMENT_METHOD_DESC", "arrears": "ARREARS_FLAG", "product": "PRODUCT_HOLDINGS"}


def trend():
    con = duckdb_conn()
    df = con.execute(f"""
        WITH a AS (SELECT date_trunc('month', INFORMATION_DATE) AS m, COUNT(*) AS active FROM read_parquet('{C.SPINE}') GROUP BY 1),
             c AS (SELECT date_trunc('month', POLICY_EFFECTIVE_CEASE_DATE) AS m, COUNT(*) AS cancels,
                          SUM(CASE WHEN CANCELLATION_REASON_CLASSIFICATION = 'Switcher' THEN 1 ELSE 0 END) AS switchers
                   FROM read_parquet('{C.CANCELLATIONS}') GROUP BY 1)
        SELECT a.m, a.active, COALESCE(c.cancels, 0) AS cancels, COALESCE(c.switchers, 0) AS switchers
        FROM a LEFT JOIN c USING (m) ORDER BY a.m""").df()
    con.close()
    df["rate"] = df.cancels / df.active
    full, last = df.iloc[:-1].copy(), df.iloc[-1]
    march = full.m.dt.month == 3
    normal = float(full.loc[~march, "rate"].mean())
    marches = [{"month": r.m.strftime("%Y-%m"), "rate": float(r.rate), "cancels": int(r.cancels), "active": int(r.active),
                "expected_normal": float(normal * r.active), "extra": float(r.cancels - normal * r.active)}
               for _, r in full[march].iterrows()]
    other = df.m.dt.month != 3
    return {"months": full.m.dt.strftime("%Y-%m").tolist(), "rate": full.rate.round(6).tolist(),
            "cancels": full.cancels.astype(int).tolist(), "active": full.active.astype(int).tolist(),
            "normal_rate": normal, "normal_cancels": float(full.loc[~march, "cancels"].mean()), "marches": marches,
            "march_ranks": full.rate.rank(ascending=False)[march].astype(int).tolist(),
            "switcher_share_march": float(full.loc[march, "switchers"].sum() / full.loc[march, "cancels"].sum()),
            "switcher_share_other": float(df.loc[other, "switchers"].sum() / df.loc[other, "cancels"].sum()),
            "excluded_last_month": {"month": last.m.strftime("%Y-%m"), "rate": float(last.rate)}}


def excluded_evidence():
    """Marketing opt-in never changes for a policy, yet 'Unknown' almost always ends in a cancellation."""
    con = duckdb_conn()
    df = con.execute(f"""
        WITH pol AS (SELECT POLICY_HASH, any_value(MARKETING_OPTIN) AS optin, COUNT(DISTINCT MARKETING_OPTIN) AS nvals,
                            MAX(INFORMATION_DATE) AS last_m FROM read_parquet('{C.SPINE}') GROUP BY 1),
             can AS (SELECT DISTINCT POLICY_HASH FROM read_parquet('{C.CANCELLATIONS}')),
             mx AS (SELECT MAX(INFORMATION_DATE) AS m FROM read_parquet('{C.SPINE}'))
        SELECT optin, COUNT(*) AS policies, AVG(CAST(c.POLICY_HASH IS NOT NULL AS INT)) AS ever_cancelled,
               AVG(CAST(last_m = (SELECT m FROM mx) AS INT)) AS active_at_end, AVG(CAST(nvals > 1 AS INT)) AS ever_changes
        FROM pol LEFT JOIN can c USING (POLICY_HASH) GROUP BY 1 ORDER BY 1""").df()
    con.close()
    return df.to_dict(orient="records")


def main():
    booster, meta = load_model("final")
    cols, _ = model_columns(meta["features"])
    metrics = read_json(C.MODELS, "final", "metrics.json")
    plan = read_json(C.RESULTS, "retention_plan.json")
    Xfull, ahead, keys = load_split("val")
    X = np.ascontiguousarray(Xfull[:, cols])
    s = booster.predict(X)
    vs = pd.read_parquet(f"{C.MODELS}/final/val_scores.parquet")
    assert np.allclose(vs["score"].to_numpy(), s), "scores differ from step 4: re-run from step 4"
    w = meta["scale_pos_weight"]
    p = s / (s + w * (1 - s))
    y = (ahead > 0).astype(int)
    ctype = cancel_type(keys)
    month = keys["INFORMATION_DATE"].to_numpy()
    recent = month == max(month)
    rank = np.empty(len(s), np.int64); rank[np.argsort(-s, kind="stable")] = np.arange(len(s))
    pr = rank / len(s)

    by_type = {t: {"n": int(((y == 1) & (ctype == t)).sum()),
                   "share": float(((y == 1) & (ctype == t)).sum() / y.sum()),
                   **{f"recall{k}": float((pr[(y == 1) & (ctype == t)] < k / 100).mean()) for k in (1, 5, 10)}}
               for t in C.ALL_TYPES}

    rng = np.random.default_rng(C.SEED)
    labels = [C.LABEL_TEXT[f] for f in meta["features"]]

    def drivers(mask, n):
        idx = rng.choice(np.where(mask)[0], size=min(n, int(mask.sum())), replace=False)
        c = np.abs(booster.predict(X[idx], pred_contrib=True)[:, :-1]).mean(0)
        imp = pd.Series(c, index=labels).sort_values(ascending=False)
        return [{"label": k, "share": float(v / imp.sum())} for k, v in imp.head(6).items()]
    drivers_overall = drivers(np.ones(len(y), bool), 20000)
    drivers_type = {t: drivers((y == 1) & (ctype == t), 4000) for t in C.ALL_TYPES}

    # real example policies (fixed seed): caught leavers of each type, two misses, a false alarm, a quiet stayer
    q = np.percentile(p, [99, 95, 90, 75])
    band = np.select([p >= q[0], p >= q[1], p >= q[2], p >= q[3]], ["Top 1%", "Top 1-5%", "Top 5-10%", "Top 10-25%"],
                     "Rest (bottom 75%)")
    pick_rng = np.random.default_rng(7)
    def pick(mask, n=1):
        idx = np.where(mask)[0]
        return [int(i) for i in pick_rng.choice(idx, size=min(n, len(idx)), replace=False)] if len(idx) else []
    known = ~recent
    resolved = []
    for t, b in [("Switcher", "Top 1%"), ("Switcher", "Top 1-5%"), ("System Lapsed", "Top 1%"), ("System Lapsed", "Top 1-5%"),
                 ("Leave PHI", "Top 1%"), ("Leave PHI", "Top 1-5%"), ("Non-Recoverable", "Top 1%"), (C.UNCLASSIFIED, "Top 1-5%")]:
        resolved += pick(known & (y == 1) & (ctype == t) & (band == b))
    resolved += pick(known & (y == 1) & (band == "Rest (bottom 75%)"), 2)
    resolved += pick(known & (y == 0) & (band == "Top 1%"))
    resolved += pick(known & (y == 0) & (band == "Rest (bottom 75%)"))
    prospective = [i for b in ["Top 1%", "Top 1-5%", "Top 5-10%", "Top 10-25%", "Rest (bottom 75%)"] for i in pick(recent & (band == b), 2)]
    prem = Xfull[:, C.ALL_FEATURES.index("MONTHLY_PREMIUM")].astype(float)
    annual = annual_premium(prem, plan["plan"]["train_median_premium"])
    shown = {k: decode(col, Xfull[:, C.ALL_FEATURES.index(col)]) for k, col in DISPLAY.items()}

    def records(idx, known_outcome):
        contrib = booster.predict(X[idx], pred_contrib=True)[:, :-1]
        out = []
        for row, i in zip(contrib, idx):
            lift = C.TYPE_SUCCESS_LIFT.get(ctype[i], 1.0) if known_outcome and y[i] else 1.0
            act, val = best_action([p[i]], [annual[i]], success=C.BASE_SUCCESS * lift)
            rec = {"id": str(keys["POLICY_HASH"].iloc[i])[:10], "score": round(float(p[i]) * 100, 1),
                   "pctRank": round(float(pr[i]) * 100, 2), "bucket": str(band[i]),
                   **{k: (None if v[i] == "(missing)" else v[i]) for k, v in shown.items()},
                   "premium": round(float(prem[i]) if np.isfinite(prem[i]) else plan["plan"]["train_median_premium"], 2),
                   "reasons": [labels[j] for j in np.argsort(-np.abs(row))[:3]],
                   "action": ACTION_TEXT[ACTIONS[int(act[0])]],
                   "actionVal": round(float(val[0]), 2)}
            if known_outcome:
                rec.update({"type": ctype[i] if y[i] else None, "label": int(y[i]),
                            "outcome": (f"Actually cancelled within 3 months (as {ctype[i]}, month {int(ahead[i])})" if y[i]
                                        else "Actually stayed - no cancellation within 3 months")})
            out.append(rec)
        return out

    rp, rj = plan["plan"], read_json(C.RESULTS, "risk_journeys.json")
    def opts(pl):
        return {k: {"n": pl["options"][a]["policies"], "share": pl["options"][a]["share"], "value": pl["options"][a]["value"]}
                for k, a in zip(OPTION_KEYS, ACTIONS)}
    top = [{**r, **{f"recall_lead{m}": r[f"recall_ahead{m}"] for m in (1, 2, 3)}} for r in metrics["top"]]
    how = read_json(C.RESULTS, "how_early.json")
    excluded = excluded_evidence()
    save_json({"marketing_optin_by_value": excluded}, C.RESULTS, "excluded_fields.json")
    summary = {
        "model": {"dropped": meta["excluded"], "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"],
                  "base_rate": metrics["base_rate"], "topk": top, "val_rows": metrics["val_rows"], "val_pos": metrics["val_leavers"]},
        "calibration": {"spw": w, "mean_pred": plan["calibration"]["mean_predicted"], "actual_rate": plan["calibration"]["actual_rate"],
                        "top1pct_mean_pred": plan["calibration"]["top1_mean_predicted"],
                        "top1pct_actual": plan["calibration"]["top1_actual"], "deciles": plan["calibration"]["deciles"]},
        "by_type": by_type, "drivers_overall": drivers_overall, "drivers_type": drivers_type,
        "cost_benefit": {"month": rp["month"], "baseline": {"total": rp["central"]["total_value"], "n": rp["policies"],
                                                            "options": opts(rp["central"])},
                         "sweep": [{"digital": x["success_rate"], "phone": x["success_rate"] * C.PHONE_LIFT,
                                    "total": x["total_value"], "options": opts(x)} for x in rp["sensitivity"]],
                         "expected_3m_cancellations": rp["expected_cancellations_3m"], "avg_max_discount": rp["average_max_discount"],
                         "discount_breakeven_mult": rp["discount_breakeven_lift"],
                         "discount_threshold_monthly_premium": rp["discount_pays_below_monthly_premium"],
                         "missing_premium_imputed": rp["blank_premiums_filled"], "train_median_premium": rp["train_median_premium"],
                         "by_type": {t: {"best_mean": v["best_mean_value"], "options": v["options_mean_value"],
                                         "share_worth_phone": v["share_worth_a_call"], "mean_prob": v["mean_probability"]}
                                     for t, v in plan["by_type"].items()}},
        "policies": {"resolved": records(resolved, True), "prospective": records(prospective, False)},
        "lead_time": {"exact": [{"month": e["months_ahead"], "roc_auc": e["roc_auc"], "pr_auc": e["pr_auc"],
                                 "base_rate": e["base_rate"], "top1_precision": e["top1_precision"]} for e in how["exact_month_models"]],
                      "within3_recall5_by_lead": {m: top[2][f"recall_lead{m}"] for m in (1, 2, 3)},
                      "within3_auc_by_lead": metrics["roc_auc_by_months_ahead"], "by_type": how["by_type"]},
        "clusters": rj["clusters"], "cluster_meta": {"k": rj["k"], "n_leavers": rj["n_leavers"], "silhouette": rj["silhouette"]},
        "leak_evidence": excluded, "trend": trend(),
        "type_drivers": read_json(C.RESULTS, "type_drivers.json"), "granularity": read_json(C.RESULTS, "granularity.json"),
        "final_check": read_json(C.RESULTS, "final_check.json"),
        "group_check": read_json(C.RESULTS, "group_check.json")["overall"],
    }
    save_json(summary, C.RESULTS, "summary_data.json")
    print(f"saved {C.RESULTS}/summary_data.json: ROC-AUC {metrics['roc_auc']:.4f}, plan ${rp['central']['total_value']:,.0f}, "
          f"{len(summary['policies']['resolved'])} + {len(summary['policies']['prospective'])} example policies")
    print("drivers:", [(d["label"], round(d["share"], 3)) for d in drivers_overall])


if __name__ == "__main__":
    main()
