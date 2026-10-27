"""
Step 4 - Train the models (LightGBM), learn on Sep 2024 - Oct 2025, judge on Dec 2025 - Jan 2026.

  final           "will this policy cancel within 3 months?"  - the main model, used for every result and the live demo
  exact_month_1/2/3  "will it cancel in exactly month 1 / 2 / 3?" - how early the signal appears (Question 3)

Leavers are rare (about 1.85%), so they are weighted up by the stayer:leaver ratio during training
(scale_pos_weight); common.calibrate() turns scores back into real-world probabilities.
The fixed 150 trees are evaluated on validation after every tree and the model is saved at the
round with the best validation PR-AUC.

Each model folder gets model.txt, meta.json, metrics.json and val_scores.parquet.
"""
import gc
import os
import time

import numpy as np

import config as C
from common import calibrate, load_split, model_columns, save_json, top_share_table

TARGETS = {"final": "within3", "exact_month_1": "exact1", "exact_month_2": "exact2", "exact_month_3": "exact3"}


def target(ahead, kind):
    """Rows used and their 0/1 label. For 'exact k', people leaving sooner than k are left out
    (they are no longer customers by month k)."""
    if kind == "within3":
        return np.ones(len(ahead), bool), (ahead > 0).astype(np.int8)
    k = int(kind[-1])
    return (ahead == 0) | (ahead >= k), (ahead == k).astype(np.int8)


def train(name, kind):
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score
    t0 = time.time()
    cols, cat_idx = model_columns()
    Xtr, a_tr, _ = load_split("train", cols)
    rows, ytr = target(a_tr, kind)
    if not rows.all():
        Xtr, ytr = Xtr[rows], ytr[rows]
    spw = float((len(ytr) - ytr.sum()) / ytr.sum())
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=C.MODEL_FEATURES, categorical_feature=cat_idx,
                      params={"max_bin": 127, "verbose": -1}, free_raw_data=True).construct()
    del Xtr; gc.collect()

    Xva, a_va, kva = load_split("val", cols)
    rows_v, yva_all = target(a_va, kind)
    dva = lgb.Dataset(Xva[rows_v], label=yva_all[rows_v], reference=dtr)
    hist = {}
    booster = lgb.train(C.lgb_params(spw), dtr, num_boost_round=C.ROUNDS, valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.record_evaluation(hist)])
    best = int(np.argmax(hist["val"]["average_precision"])) + 1

    s = booster.predict(Xva, num_iteration=best)          # every validation row, for reporting
    y, sk = yva_all[rows_v], s[rows_v]
    m = {"model": name, "target": kind, "best_round": best, "scale_pos_weight": spw,
         "train_rows": int(len(ytr)), "train_leavers": int(ytr.sum()),
         "val_rows": int(len(y)), "val_leavers": int(y.sum()), "base_rate": float(y.mean()),
         "roc_auc": float(roc_auc_score(y, sk)), "pr_auc": float(average_precision_score(y, sk)),
         "top": top_share_table(y, sk, ahead=a_va if kind == "within3" else None)}
    if kind == "within3":
        stay = a_va == 0
        m["roc_auc_by_months_ahead"] = {k: float(roc_auc_score(a_va[stay | (a_va == k)] > 0, s[stay | (a_va == k)]))
                                        for k in (1, 2, 3)}
    out = os.path.join(C.MODELS, name)
    os.makedirs(out, exist_ok=True)
    booster.save_model(os.path.join(out, "model.txt"), num_iteration=best)
    save_json({"features": C.MODEL_FEATURES, "categorical_index": cat_idx, "excluded": C.EXCLUDED,
               "target": kind, "best_round": best, "scale_pos_weight": spw}, out, "meta.json")
    save_json(m, out, "metrics.json")
    kva.assign(months_ahead=a_va, score=s, calibrated=calibrate(s, spw)).to_parquet(
        os.path.join(out, "val_scores.parquet"), index=False)
    t1 = m["top"][0]
    print(f"{name}: best round {best}, ROC-AUC {m['roc_auc']:.4f}, PR-AUC {m['pr_auc']:.4f}, "
          f"riskiest 1% precision {t1['precision']:.3f} ({t1['lift']:.1f}x), {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for name, kind in TARGETS.items():
        train(name, kind)
