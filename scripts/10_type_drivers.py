"""
Step 10 (pitch support) - Per-cancellation-type driving features.

For the "four questions" slide: instead of one overall driver list, compute
the top-3 real LightGBM feature contributions (pred_contrib, same mechanism
as 09_policy_explorer.py -- SHAP-equivalent, not invented) separately for
each cancellation type's actual cancellers in the held-out val split.

Samples up to SAMPLE_N actual cancellers per type (fixed seed) rather than
the full val set, purely for speed -- pred_contrib on ~1M rows is slow and
unnecessary for a stable mean-abs-contribution ranking.

Run:
    python3 scripts/10_type_drivers.py --window 3
"""
import argparse
import json
import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import importlib
lgb_mod = importlib.import_module("06_train_lightgbm")
pe_mod = importlib.import_module("09_policy_explorer")

OUT_DIR = "DATA_derived"
MODEL_DIR = "models"
PRESENTATION_DIR = "presentation"
SAMPLE_N = 4000

TYPES = ["Switcher", "System Lapsed", "Leave PHI", "Non-Recoverable", "(unclassified)"]


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

    print("Loading val split (reusing the validated 09_policy_explorer loader)...", flush=True)
    X, y, premium, cancel_type, info_date, policy_hash, display = pe_mod.load_val_full(w, cat_code_maps)
    print(f"  X {X.shape}", flush=True)

    rng = np.random.default_rng(11)
    out = {"window_months": w, "types": {}}

    # overall (all actual cancellers, any type) for comparison
    all_cancel_idx = np.where(y == 1)[0]
    print(f"  total actual cancellers in val: {len(all_cancel_idx):,}", flush=True)

    def top_features_for(idx):
        if len(idx) > SAMPLE_N:
            idx = rng.choice(idx, size=SAMPLE_N, replace=False)
        contrib = booster.predict(X[idx], pred_contrib=True, num_iteration=best_round)
        feat_contrib = contrib[:, :-1]  # drop bias/expected-value column
        mean_abs = np.mean(np.abs(feat_contrib), axis=0)
        order = np.argsort(-mean_abs)[:5]
        return [
            {"feature": lgb_mod.ALL_FEATURE_COLS[i],
             "label": pe_mod.FEATURE_LABELS.get(lgb_mod.ALL_FEATURE_COLS[i], lgb_mod.ALL_FEATURE_COLS[i]),
             "mean_abs_contrib": round(float(mean_abs[i]), 5)}
            for i in order
        ], int(len(idx))

    overall_top, overall_n = top_features_for(all_cancel_idx)
    out["overall"] = {"top_features": overall_top, "sample_n": overall_n, "population_n": int(len(all_cancel_idx))}
    print("  overall top:", [f["label"] for f in overall_top], flush=True)

    for t in TYPES:
        idx = np.where((y == 1) & (cancel_type == t))[0]
        if len(idx) == 0:
            print(f"  {t}: no rows, skipping", flush=True)
            continue
        top, n = top_features_for(idx)
        out["types"][t] = {"top_features": top, "sample_n": n, "population_n": int(len(idx))}
        print(f"  {t} (n={len(idx):,}, sampled {n}):", [f["label"] for f in top], flush=True)

    out_path = f"{PRESENTATION_DIR}/type_drivers_{w}m.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
