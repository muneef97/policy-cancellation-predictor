"""
Step 5 - Final check on the locked test months (Mar - May 2026).

These months were never used to train, tune or choose anything. The final model is scored on
them once, and the same measures as validation are reported, pooled and month by month.
Note: a 31 March snapshot predicts April - June, so the March cancellation spike itself sits in the
validation months (31 Dec and 31 Jan snapshots), not here. May's window reaches into August 2026,
the last month of the extract, which HBF may still be processing.

Writes results/final_check.json
"""
import numpy as np

import config as C
from common import calibrate, load_model, load_split, model_columns, save_json, top_share_table


def summary(y, s, p, ahead):
    from sklearn.metrics import average_precision_score, roc_auc_score
    top = np.argsort(-p)[: int(0.01 * len(p))]
    return {"rows": int(len(y)), "leavers": int(y.sum()), "base_rate": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
            "top": top_share_table(y, s, ahead=ahead), "mean_predicted": float(p.mean()),
            "top1_mean_predicted": float(p[top].mean()), "top1_actual": float(y[top].mean())}


def main():
    booster, meta = load_model("final")
    cols, _ = model_columns(meta["features"])
    X, ahead, keys = load_split("test", cols)
    s = booster.predict(X)
    p = calibrate(s, meta["scale_pos_weight"])
    y = (ahead > 0).astype(np.int8)
    month = keys["INFORMATION_DATE"].to_numpy()
    res = {"months": sorted(set(month)), "all": summary(y, s, p, ahead),
           "by_month": {m: summary(y[month == m], s[month == m], p[month == m], ahead[month == m])
                        for m in sorted(set(month))}}
    save_json(res, C.RESULTS, "final_check.json")
    for name, r in [("Mar-May", res["all"])] + list(res["by_month"].items()):
        t1, t5 = r["top"][0], r["top"][2]
        print(f"{name:10s} rows {r['rows']:>9,}  cancel rate {r['base_rate']:.2%}  ROC-AUC {r['roc_auc']:.3f}  "
              f"riskiest 1% {t1['precision']:.1%} ({t1['lift']:.1f}x)  leavers in riskiest 5% {t5['recall']:.1%}  "
              f"predicted {r['mean_predicted']:.2%}")


if __name__ == "__main__":
    main()
