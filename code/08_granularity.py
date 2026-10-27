"""
Step 8 - Question 2: how granular can we predict cancellations?

If we already know a policy will cancel within 3 months, can the model tell WHY?
  Level 1  which of the four types (Switcher, Leave PHI, System Lapsed, Non-Recoverable)
  Level 2  the detailed reason within a type (e.g. Deceased vs Moving overseas).
           System Lapsed has a single reason, so it has no level 2.
One multi-class model per question, trained on Sep 2024 - Oct 2025 leavers and judged on
Dec 2025 - Jan 2026 leavers. Reasons with fewer than 1,500 training rows are grouped as "Other".

Measures: accuracy compared with always guessing the most common answer, and, for each class,
ranking skill against the rest (50% = no better than guessing) and how often it is named correctly.

Writes results/granularity.json
"""
import numpy as np

import config as C
from common import load_split, model_columns, save_json

MIN_ROWS = 1500


def fit(Xtr, ytr, Xva, yva, classes, cat_idx):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    params = C.lgb_params()
    params.update({"objective": "multiclass", "num_class": len(classes), "metric": "multi_logloss"})
    params.pop("scale_pos_weight")
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=C.MODEL_FEATURES, categorical_feature=cat_idx,
                      params={"max_bin": 127, "verbose": -1})
    hist = {}
    b = lgb.train(params, dtr, num_boost_round=C.ROUNDS, valid_sets=[lgb.Dataset(Xva, label=yva, reference=dtr)],
                  valid_names=["val"], callbacks=[lgb.record_evaluation(hist)])
    best = int(np.argmin(hist["val"]["multi_logloss"])) + 1
    P = b.predict(Xva, num_iteration=best)
    pred = P.argmax(1)
    per_class = {}
    for i, c in enumerate(classes):
        yi = (yva == i).astype(int)
        wrong = np.bincount(pred[(yva == i) & (pred != i)], minlength=len(classes))
        per_class[c] = {"share": float(yi.mean()), "ranking": float(roc_auc_score(yi, P[:, i])),
                        "named_correctly": float((pred[yva == i] == i).mean()),
                        "most_often_mistaken_for": classes[int(wrong.argmax())] if wrong.sum() else None,
                        "mistaken_share": float(wrong.max() / (yva == i).sum())}
    return {"leavers": int(len(yva)), "best_round": best, "accuracy": float((pred == yva).mean()),
            "accuracy_guessing_most_common": float((yva == np.bincount(ytr).argmax()).mean()), "classes": per_class}


def main():
    cols, cat_idx = model_columns()
    leavers = lambda a, k: a > 0
    Xtr, _, ktr = load_split("train", cols, leavers)
    Xva, _, kva = load_split("val", cols, leavers)
    out = {}

    code = {t: i for i, t in enumerate(C.TYPES)}
    mt, mv = ktr["ctype"].isin(C.TYPES).to_numpy(), kva["ctype"].isin(C.TYPES).to_numpy()
    out["types"] = fit(Xtr[mt], ktr.loc[mt, "ctype"].map(code).to_numpy(),
                       Xva[mv], kva.loc[mv, "ctype"].map(code).to_numpy(), C.TYPES, cat_idx)

    out["reasons"] = {}
    for t in ["Switcher", "Leave PHI", "Non-Recoverable"]:
        a, b = (ktr["ctype"] == t).to_numpy(), (kva["ctype"] == t).to_numpy()
        counts = ktr.loc[a, "reason"].value_counts()
        keep = [r for r in counts.index if counts[r] >= MIN_ROWS]
        classes = keep + (["Other"] if len(keep) < len(counts) else [])
        to_code = lambda s: s.where(s.isin(keep), "Other").map({c: i for i, c in enumerate(classes)}).to_numpy()
        out["reasons"][t] = fit(Xtr[a], to_code(ktr.loc[a, "reason"]), Xva[b], to_code(kva.loc[b, "reason"]), classes, cat_idx)

    save_json(out, C.RESULTS, "granularity.json")
    for name, r in [("Four types", out["types"])] + [(f"Reasons within {t}", r) for t, r in out["reasons"].items()]:
        print(f"\n{name}: {r['leavers']:,} leavers, accuracy {r['accuracy']:.1%} "
              f"(always guessing the most common: {r['accuracy_guessing_most_common']:.1%})")
        for c, v in r["classes"].items():
            print(f"   {c:36s} share {v['share']:6.1%}  ranking {v['ranking']:6.1%}  named correctly {v['named_correctly']:6.1%}")


if __name__ == "__main__":
    main()
