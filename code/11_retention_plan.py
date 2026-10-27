"""
Step 11 - Question 4: which intervention suits each policy, and what is it worth?

1. Calibration: the final model's scores are converted to real-world probabilities
   (common.calibrate) and checked against what actually happened on the validation months.
2. The plan: every policy in the latest validation month (31 Jan 2026) gets the option with the
   highest expected value under HBF's formula
       expected value = probability x success rate x ($1,000 - discount) - contact cost
   choosing between digital or phone, each with or without a 12%-of-premium discount, or no action.
   HBF did not give a success rate, so it is tested at 10%, 20% and 30% (20% is the central case).
3. By type: average best expected value per actual leaver of each cancellation type, using the
   challenge pack's description of how saveable each type is (config.TYPE_SUCCESS_LIFT).

Writes results/retention_plan.json
"""
import numpy as np
import pandas as pd

import config as C
from common import (ACTIONS, annual_premium, best_action, cancel_type, load_model, load_split, model_columns,
                    month_files, option_values, save_json)


def main():
    booster, meta = load_model("final")
    cols, _ = model_columns(meta["features"])
    prem_col = C.ALL_FEATURES.index("MONTHLY_PREMIUM")
    train_median = float(np.nanmedian(np.concatenate([np.load(p, mmap_mode="r")[:, prem_col] for p in month_files("train")])))

    X, ahead, keys = load_split("val")
    s = booster.predict(np.ascontiguousarray(X[:, cols]))
    w = meta["scale_pos_weight"]
    p = s / (s + w * (1 - s))
    y = (ahead > 0).astype(int)
    month = keys["INFORMATION_DATE"].to_numpy()
    annual = annual_premium(X[:, prem_col], train_median)

    # 1. calibration
    dec = pd.qcut(p, 10, labels=False, duplicates="drop")
    top1 = np.argsort(-p)[: int(0.01 * len(p))]
    calibration = {"scale_pos_weight": w, "mean_predicted": float(p.mean()), "actual_rate": float(y.mean()),
                   "top1_mean_predicted": float(p[top1].mean()), "top1_actual": float(y[top1].mean()),
                   "deciles": [{"decile": int(d), "predicted": float(p[dec == d].mean()), "actual": float(y[dec == d].mean()),
                                "policies": int((dec == d).sum())} for d in range(10)]}

    # 2. the plan for the latest month
    latest = month == max(month)
    pl, al = p[latest], annual[latest]

    def plan(success):
        act, val = best_action(pl, al, success=success)
        return {"success_rate": success, "total_value": float(val.sum()), "policies": int(len(pl)),
                "options": {a: {"policies": int((act == i).sum()), "share": float((act == i).mean()),
                                "value": float(val[act == i].sum())} for i, a in enumerate(ACTIONS)}}
    avg_discount = float((al * C.MAX_DISCOUNT).mean())
    plan_json = {"month": max(month), "policies": int(latest.sum()), "expected_cancellations_3m": float(pl.sum()),
                 "central": plan(C.BASE_SUCCESS), "sensitivity": [plan(r) for r in C.SUCCESS_RATES],
                 "average_max_discount": avg_discount,
                 "discount_breakeven_lift": C.VALUE_RETAINED / (C.VALUE_RETAINED - avg_discount),
                 # a discount pays (at the assumed 1.3x lift) while 12% of annual premium < $1,000 x (1 - 1/1.3)
                 "discount_pays_below_monthly_premium": (C.VALUE_RETAINED - C.VALUE_RETAINED / C.DISCOUNT_LIFT) / C.MAX_DISCOUNT / 12,
                 "phone_breakeven_probability": C.PHONE_COST / (C.BASE_SUCCESS * C.PHONE_LIFT * C.VALUE_RETAINED),
                 "blank_premiums_filled": int(np.isnan(X[latest, prem_col]).sum()), "train_median_premium": train_median}

    # 3. by type, actual leavers in both validation months
    ctype = np.where(y == 1, cancel_type(keys), "STAY")
    by_type = {}
    for t in C.ALL_TYPES:
        m = ctype == t
        v = option_values(p[m], annual[m], success=C.BASE_SUCCESS * C.TYPE_SUCCESS_LIFT[t])
        best = np.maximum(v.max(1), 0)
        by_type[t] = {"leavers": int(m.sum()), "mean_probability": float(p[m].mean()), "best_mean_value": float(best.mean()),
                      "options_mean_value": {a: float(v[:, i].mean()) for i, a in enumerate(ACTIONS[:4])},
                      "share_worth_a_call": float((v[:, 2] > np.maximum(v[:, 0], 0)).mean())}

    save_json({"calibration": calibration, "plan": plan_json, "by_type": by_type}, C.RESULTS, "retention_plan.json")
    c = plan_json["central"]
    print(f"calibration: predicted {calibration['mean_predicted']:.2%} vs actual {calibration['actual_rate']:.2%}")
    print(f"plan {plan_json['month']}: {plan_json['policies']:,} policies, expected value ${c['total_value']:,.0f}/month "
          f"(sensitivity: {[round(x['total_value']) for x in plan_json['sensitivity']]})")
    for a, o in c["options"].items():
        print(f"  {a:22s} {o['policies']:>8,} {o['share']:6.1%}  ${o['value']:>12,.0f}")
    print("by type, mean best value per leaver:", {t: round(v["best_mean_value"], 2) for t, v in by_type.items()})


if __name__ == "__main__":
    main()
