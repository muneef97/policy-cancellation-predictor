"""
Step 7 - Question 1: which variables predict EACH cancellation type?

One model per type (Switcher, Leave PHI, System Lapsed, Non-Recoverable):
  leavers of that type vs policies that stay; leavers of other types are left out, so each model
  learns what separates its own leavers from stayers. Same features and settings as the final model.
  Training uses every training leaver and a fixed 10% sample of stayers (leavers are then over-
  represented about 10 times, which is enough; heavier weighting made the models overfit).

For each type, on the validation months (Dec 2025 - Jan 2026):
  ranking skill and share caught in its riskiest 5%;
  how much the model relies on each field, measured two independent ways:
    SHAP     average size of each field's push on the score
    shuffle  how much ranking skill (ROC-AUC) is lost when that field is shuffled
  the top 3 "signals": fields ranked by the average of their two positions;
  for each signal, the real 3-month rate of that type in each group (e.g. each tenure band), so the
  group most at risk can be named in plain words;
  clue groups (related fields added together first, so one signal isn't split across several).

Writes models/type_<name>/model.txt and results/type_drivers.json
"""
import os

import numpy as np
import pandas as pd

import config as C
from common import cancel_type, decode, load_model, load_split, model_columns, save_json

STAYER_SAMPLE = 0.10
GROUPS = {
    "Price and cover level": ["MONTHLY_PREMIUM", "PRODUCT_HOLDINGS", "HOSPITAL_TIER", "EXTRAS_TIER",
                              "HOSPITAL_EXCESS_AMT", "LHC_LOADING_RATE"],
    "Recent price rise": ["PREMIUM_CHANGE_3M"],
    "Age and household": ["POLICY_OWNER_AGE_BAND", "SCALE", "ACTIVE_ADULT_COUNT", "ACTIVE_CHILD_COUNT",
                          "YOUNGEST_DEPENDANT_AGE_BAND"],
    "Membership history": ["TENURE_BAND", "CHANNEL_ACQUISITION"],
    "State": ["STATE_CODE"],
    "Payment habits": ["PAYMENT_METHOD_DESC", "PAYMENT_FREQUENCY_DESC", "ARREARS_FLAG_LAG2M"],
    "Claims": [f for f in C.MODEL_FEATURES if any(w in f for w in ("CLAIM", "BENEFIT", "OUT_OF_POCKET", "REVERSAL"))],
    "Service experience": ["MONTHS_SINCE_LAST_SURVEY", "LAST_OVERALL_EXPERIENCE_RATING", "LAST_NPS_RATING",
                           "LAST_SERVICE_RATING", "RECENT_LOW_RATING_FLAG", "HAS_EVER_SURVEYED", "HAD_COMPLAINT_FLAG",
                           "NEGATIVE_SENTIMENT_CALLS_LAST_6M", "CALL_NEGATIVE_SENTIMENT", "RESOLUTION_TIME"],
}
RATE_FIELDS = ["STATE_CODE", "TENURE_BAND", "POLICY_OWNER_AGE_BAND", "PAYMENT_METHOD_DESC", "SCALE",
               "HOSPITAL_TIER", "MONTHLY_PREMIUM"]


def slug(t):
    return "type_" + t.lower().replace(" ", "_").replace("-", "_")


BLANK = "(blank)"


def readable(col, values):
    """Group a field's values for a rate table: categories as they are, numbers in five equal-sized bands."""
    if col in C.CATEGORICAL:
        return decode(col, values)
    v = pd.Series(values, dtype=float)
    ok = v.notna() & ((v > 1) if col == "MONTHLY_PREMIUM" else True)   # a $0 premium means the policy is mid-change
    out = pd.Series(BLANK, index=v.index, dtype=object)
    if v[ok].nunique() <= 6:
        out[ok] = v[ok].map(lambda x: f"{x:g}")
    else:
        q = pd.qcut(v[ok], 5, duplicates="drop")
        money = "$" if col == "MONTHLY_PREMIUM" else ""
        out[ok] = [f"{money}{iv.left:,.0f} to {money}{iv.right:,.0f}" for iv in q]
    return out.to_numpy()


def rate_table(col, values, y):
    r = pd.DataFrame({"value": readable(col, values), "y": y}).groupby("value").y.agg(["mean", "size"])
    r = r[r["size"] >= 2000].sort_values("mean", ascending=False)        # ignore tiny groups
    return [{"value": v, "rate": float(x["mean"]), "policies": int(x["size"])} for v, x in r.iterrows()]


def shuffle_importance(b, X, y, best, rng):
    """ROC-AUC lost when each field is shuffled (all leavers of the type + 150,000 stayers)."""
    from sklearn.metrics import roc_auc_score
    idx = np.concatenate([np.where(y == 1)[0], rng.choice(np.where(y == 0)[0], min(150000, int((y == 0).sum())), replace=False)])
    Xs, ys = X[idx].copy(), y[idx]
    base = roc_auc_score(ys, b.predict(Xs, num_iteration=best))
    drop = {}
    for j, f in enumerate(C.MODEL_FEATURES):
        keep = Xs[:, j].copy(); Xs[:, j] = rng.permutation(keep)
        drop[f] = base - roc_auc_score(ys, b.predict(Xs, num_iteration=best))
        Xs[:, j] = keep
    return pd.Series(drop)


def main():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    cols, cat_idx = model_columns()
    rng = np.random.default_rng(C.SEED)
    Xtr, a_tr, k_tr = load_split("train", cols, lambda a, k: (a > 0) | (rng.random(len(a)) < STAYER_SAMPLE))
    t_tr = np.where(a_tr > 0, cancel_type(k_tr), "STAY")
    Xva, a_va, k_va = load_split("val", cols)
    t_va = np.where(a_va > 0, cancel_type(k_va), "STAY")
    general, _ = load_model("final")
    group_idx = {g: [C.MODEL_FEATURES.index(f) for f in fs if f in C.MODEL_FEATURES] for g, fs in GROUPS.items()}
    labels = [C.LABEL_TEXT[f] for f in C.MODEL_FEATURES]
    shap_rng = np.random.default_rng(C.SEED)
    report = {}
    for t in C.TYPES:
        m = (t_tr == "STAY") | (t_tr == t)
        y = (t_tr[m] == t).astype(np.int8)
        dtr = lgb.Dataset(Xtr[m], label=y, feature_name=C.MODEL_FEATURES, categorical_feature=cat_idx,
                          params={"max_bin": 127, "verbose": -1})
        mv = np.where((t_va == "STAY") | (t_va == t))[0]
        yv = (t_va[mv] == t).astype(np.int8)
        Xv = Xva[mv]
        hist = {}
        b = lgb.train(C.lgb_params(1.0), dtr, num_boost_round=C.ROUNDS, valid_sets=[lgb.Dataset(Xv, label=yv, reference=dtr)],
                      valid_names=["val"], callbacks=[lgb.record_evaluation(hist)])
        best = int(np.argmax(hist["val"]["average_precision"])) + 1
        out = os.path.join(C.MODELS, slug(t)); os.makedirs(out, exist_ok=True)
        b.save_model(os.path.join(out, "model.txt"), num_iteration=best)

        s = b.predict(Xv, num_iteration=best)
        top5 = np.argsort(-s)[: int(0.05 * len(s))]
        sample = shap_rng.choice(len(mv), size=min(20000, len(mv)), replace=False)
        con = b.predict(Xv[sample], pred_contrib=True, num_iteration=best)[:, :-1]
        shap = pd.Series(np.abs(con).mean(0), index=C.MODEL_FEATURES)
        clue = pd.Series(shap.to_numpy(), index=labels).sort_values(ascending=False)
        shuffle = shuffle_importance(b, Xv, yv, best, shap_rng)
        both = (shap.rank(ascending=False) + shuffle.rank(ascending=False)) / 2
        signals = []
        for f in both.sort_values(kind="stable").index[:3]:
            table = rate_table(f, Xv[:, C.MODEL_FEATURES.index(f)], yv)
            priced = [x for x in table if not (f == "MONTHLY_PREMIUM" and x["value"] == BLANK)]
            signals.append({"field": f, "label": C.LABEL_TEXT[f], "shap_share": float(shap[f] / shap.sum()),
                            "shuffle_drop": float(shuffle[f]), "highest": priced[0], "lowest": priced[-1],
                            "times": priced[0]["rate"] / max(priced[-1]["rate"], 1e-9), "rates": table})
        grp = pd.Series({g: np.abs(con[:, ii].sum(1)).mean() for g, ii in group_idx.items()}).sort_values(ascending=False)
        rates = {}
        for f in RATE_FIELDS:
            rates[C.LABEL_TEXT[f]] = rate_table(f, Xv[:, C.MODEL_FEATURES.index(f)], yv)
        report[t] = {"val_leavers": int(yv.sum()), "rate": float(yv.mean()), "best_round": best,
                     "ranking": float(roc_auc_score(yv, s)),
                     "ranking_of_final_model": float(roc_auc_score(yv, general.predict(Xv))),
                     "caught_in_top5": float(yv[top5].sum() / yv.sum()),
                     "clues": [{"clue": k, "share": float(v / clue.sum())} for k, v in clue.head(8).items()],
                     "clue_groups": [{"group": k, "share": float(v / grp.sum())} for k, v in grp.items()],
                     "shuffle_top": [{"clue": C.LABEL_TEXT[k], "ranking_lost": float(v)}
                                     for k, v in shuffle.sort_values(ascending=False).head(8).items()],
                     "signals": signals, "rates": rates}
        r = report[t]
        print(f"\n{t}: {r['val_leavers']:,} leavers in Dec-Jan, ranking {r['ranking']:.1%} "
              f"(final model {r['ranking_of_final_model']:.1%}), caught in its riskiest 5% {r['caught_in_top5']:.1%}")
        print("  clues:  " + ", ".join(f"{c['clue']} {c['share']:.0%}" for c in r["clues"][:6]))
        print("  groups: " + ", ".join(f"{g['group']} {g['share']:.0%}" for g in r["clue_groups"]))
        for i, sg in enumerate(signals, 1):
            print(f"  signal {i}: {sg['label']} (SHAP {sg['shap_share']:.0%}, shuffle {sg['shuffle_drop'] * 100:.1f}) - "
                  f"highest {sg['highest']['value']} {sg['highest']['rate']:.2%}, lowest {sg['lowest']['value']} "
                  f"{sg['lowest']['rate']:.2%} ({sg['times']:.0f}x)")
    save_json(report, C.RESULTS, "type_drivers.json")


if __name__ == "__main__":
    main()
