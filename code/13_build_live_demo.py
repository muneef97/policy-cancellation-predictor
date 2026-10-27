"""
Step 13 - Build the live demo ("HBF Retention Hub"), one self-contained HTML file that works offline.

From the final model it exports:
  model      the trees in a compact form, so the page can re-score a policy in the browser (what-if)
  features   labels, categories and ranges for the what-if editor
  live book  every policy in the latest validation month (31 Jan 2026), scored and summarised as a
             segment cube (state x age x tenure x channel x hospital tier x risk band) so every chart
             can cross-filter the whole book; plus the riskiest 2,500 policies and 1,500 others with
             their top reasons and best action
  known month  31 Dec 2025, whose outcomes are known: precision and recall at every list size,
             overall, by type and by months ahead, and a calibration table
  plan cells a probability x premium histogram, so the planner can re-cost any assumptions instantly

Writes live_demo/HBF_Retention_Hub.html
"""
import json
import os

import numpy as np
import pandas as pd

import config as C
from common import (annual_premium, best_action, calibrate, cancel_type, load_model, load_month, model_columns,
                    month_files, read_json)

DIMS = ["STATE_CODE", "POLICY_OWNER_AGE_BAND", "TENURE_BAND", "CHANNEL_ACQUISITION", "HOSPITAL_TIER"]
BANDS = ["Top 1%", "Top 1-5%", "Top 5-10%", "Top 10-25%", "Bottom 75%"]


def compact_tree(node):
    """One LightGBM tree as flat arrays the browser can walk (see templates/model.js)."""
    nodes = []
    def walk(n):
        i = len(nodes); nodes.append(None)
        if "leaf_value" in n:
            nodes[i] = {"v": n["leaf_value"]}
            return i
        rec = {"f": n["split_feature"], "dl": 1 if n["default_left"] else 0, "m": {"None": 0, "Zero": 1, "NaN": 2}[n["missing_type"]]}
        if n["decision_type"] == "==":
            rec["c"] = [int(x) for x in str(n["threshold"]).split("||")]
        else:
            rec["t"] = n["threshold"]
        rec["l"] = walk(n["left_child"]); rec["r"] = walk(n["right_child"])
        nodes[i] = rec
        return i
    walk(node)
    return nodes


def feature_meta(X, features, cat_idx):
    codes = json.load(open(C.CATEGORY_CODES))
    out = []
    for j, f in enumerate(features):
        if j in cat_idx:
            out.append({"name": f, "label": C.LABEL_TEXT[f], "kind": "cat",
                        "cats": [c for c, _ in sorted(codes[f].items(), key=lambda kv: kv[1])]})
        else:
            col = X[:, j]; fin = col[np.isfinite(col)]
            q = np.percentile(fin, [1, 50, 99]) if len(fin) else [0, 0, 0]
            out.append({"name": f, "label": C.LABEL_TEXT[f], "kind": "num", "lo": float(q[0]), "med": float(q[1]), "hi": float(q[2]),
                        "int": bool(len(fin) and np.allclose(fin[:5000], np.round(fin[:5000]))), "missing": float(np.isnan(col).mean())})
    return out


def curves(y, s, groups=None):
    """Precision and recall for contact lists of 0.5%, 1%, ... 30% of the book."""
    order = np.argsort(-s, kind="stable"); ys = y[order]
    ks = np.round(np.arange(0.005, 0.3001, 0.005), 3)
    idx = (ks * len(s)).astype(int) - 1
    cum = np.cumsum(ys)
    out = {"k": ks.tolist(), "precision": (cum[idx] / (idx + 1)).round(4).tolist(), "recall": (cum[idx] / ys.sum()).round(4).tolist()}
    if groups is not None:
        g = groups[order]
        out["by"] = {str(k): (np.cumsum(ys & (g == k))[idx] / max(1, (ys & (g == k)).sum())).round(4).tolist() for k in np.unique(g)}
    return out


def main():
    from sklearn.metrics import average_precision_score, roc_auc_score
    booster, meta = load_model("final")
    features, w = meta["features"], meta["scale_pos_weight"]
    cols, cat_idx = model_columns(features)
    summary = read_json(C.RESULTS, "summary_data.json")
    median_prem = summary["cost_benefit"]["train_median_premium"]
    dump = booster.dump_model()
    assert dump["feature_names"] == features
    known_p, live_p = month_files("val")[0], month_files("val")[-1]

    # ---- live book ----
    X, _, keys = load_month(live_p, cols)
    s = booster.predict(X); p = calibrate(s, w)
    rank = pd.Series(p).rank(ascending=False, pct=True).to_numpy()
    prem = X[:, features.index("MONTHLY_PREMIUM")].astype(float)
    annual = annual_premium(prem, median_prem)
    act, actv = best_action(p, annual)
    inprog = (np.isnan(prem) | (prem <= 1)).astype(int)      # blank or $0 premium: cancellation probably already in progress
    band = np.select([rank <= .01, rank <= .05, rank <= .10, rank <= .25], [0, 1, 2, 3], 4)
    df = pd.DataFrame({**{d: np.nan_to_num(X[:, features.index(d)], nan=-1).astype(int) for d in DIMS},
                       "band": band, "ip": inprog, "p": p, "prem": annual, "rp": p * annual, "fl": (rank <= .05).astype(int),
                       **{f"a{a}": (act == a).astype(int) for a in range(5)}})
    cube = df.groupby(DIMS + ["band", "ip"]).agg(n=("p", "size"), ep=("p", "sum"), prem=("prem", "sum"), rp=("rp", "sum"),
                                                 fl=("fl", "sum"), **{f"a{a}": (f"a{a}", "sum") for a in range(5)}).reset_index()
    cube_rows = [[int(v) for v in r[:len(DIMS) + 3]] + [round(float(r.ep), 3), round(float(r.prem), 0), round(float(r.rp), 1)]
                 + [int(getattr(r, c)) for c in ["fl", "a0", "a1", "a2", "a3", "a4"]] for r in cube.itertuples(index=False)]
    order = np.argsort(-p)
    pick = np.concatenate([order[:2500], np.random.default_rng(29).choice(order[2500:], 1500, replace=False)])
    contrib = booster.predict(X[pick], pred_contrib=True)[:, :-1]
    policies = [{"id": str(keys["POLICY_HASH"].iloc[i])[:10], "p": round(float(p[i]), 6), "s": float(s[i]),
                 "rank": round(float(rank[i]), 6), "act": int(act[i]), "ev": round(float(actv[i]), 2),
                 "x": [None if np.isnan(v) else round(float(v), 4) for v in X[i]],
                 "why": [[int(j), round(float(c[j]), 4)] for j in np.argsort(-np.abs(c))[:4]], "ip": int(inprog[i])}
                for c, i in zip(contrib, pick)]
    live = {"n": int(len(p)), "expected_cancellations": float(p.sum()), "premium": float(annual.sum()),
            "premium_at_risk": float((p * annual).sum()), "flagged": int((rank <= .05).sum()), "in_progress": int(inprog.sum()),
            "in_progress_top5": int(((rank <= .05) & (inprog == 1)).sum()),
            "flag_threshold_p": float(np.sort(p)[::-1][int(.05 * len(p)) - 1]),
            "p_hist": np.histogram(np.log10(np.clip(p, 1e-5, 1)), bins=50, range=(-5, 0))[0].tolist()}
    # probability x annual premium histogram for the planner
    pi = np.clip(np.digitize(p, np.logspace(-5, 0, 161)) - 1, 0, 159)
    am_edges = np.concatenate([np.linspace(0, np.percentile(annual, 99.5), 80), [annual.max() + 1]])
    ai = np.clip(np.digitize(annual, am_edges) - 1, 0, len(am_edges) - 2)
    g = pd.DataFrame({"pi": pi, "ai": ai, "p": p, "a": annual}).groupby(["pi", "ai"]).agg(n=("p", "size"), p=("p", "mean"), a=("a", "mean"))
    book_cells = [[int(r.n), round(float(r.p), 6), round(float(r.a), 2)] for r in g.itertuples()]

    # ---- known month: how well it did ----
    Xk, ak, kk = load_month(known_p, cols)
    pk = calibrate(booster.predict(Xk), w); yk = ak > 0
    ct = np.where(yk, cancel_type(kk), "none")
    dec = pd.qcut(pk, 10, labels=False)
    perf = {"month": str(kk["INFORMATION_DATE"].iloc[0]), "n": int(len(yk)), "pos": int(yk.sum()), "base_rate": float(yk.mean()),
            "overall": curves(yk, pk), "by_type": curves(yk, pk, ct), "by_lead": curves(yk, pk, ak.astype(int)),
            "type_counts": {t: int((ct == t).sum()) for t in C.ALL_TYPES},
            "calibration": [{"pred": float(pk[dec == d].mean()), "actual": float(yk[dec == d].mean())} for d in range(10)],
            "roc_auc": float(roc_auc_score(yk, pk)), "pr_auc": float(average_precision_score(yk, pk))}

    data = {"features": feature_meta(X, features, cat_idx), "trees": [compact_tree(t["tree_structure"]) for t in dump["tree_info"]],
            "spw": w, "book_cells": book_cells, "deck": summary,   # the page reads it under its original key
            "months": {"live": os.path.basename(live_p)[:10], "known": perf["month"]},
            "dims": DIMS, "bands": BANDS, "cube": cube_rows, "live": live, "policies": policies, "perf": perf}
    payload = json.dumps(data, separators=(",", ":"), default=float).replace("</", "<\\/")   # keep </script> out of the JSON
    html = open(os.path.join(C.TEMPLATES, "live_demo.html"), encoding="utf-8").read()
    html = html.replace("__MODEL_JS__", open(os.path.join(C.TEMPLATES, "model.js"), encoding="utf-8").read()).replace("__DATA__", payload)
    assert "__DATA__" not in html and "__MODEL_JS__" not in html
    os.makedirs(C.LIVE_DEMO, exist_ok=True)
    out = os.path.join(C.LIVE_DEMO, "HBF_Retention_Hub.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"wrote {out} ({os.path.getsize(out) // 1024:,} KB): {len(p):,} live policies, {len(cube_rows):,} cube cells, "
          f"known-month ROC-AUC {perf['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
