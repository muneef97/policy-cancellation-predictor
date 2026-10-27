"""
Step 6 - Does the model work equally well for every group? (test months, Mar - May 2026)

For each state, age band, tenure band, joining channel and income tier:
  ranking skill (ROC-AUC) within the group, share of the group's leavers caught by the one
  book-wide riskiest-5% list, predicted vs actual cancel rate, and share of the group on the list.
Groups with fewer than 100 leavers are flagged as too small to judge.

Writes results/group_check.csv and results/group_check.json
"""
import numpy as np
import pandas as pd

import config as C
from common import calibrate, decode, load_model, load_split, model_columns, save_json

GROUPS = {"State": "STATE_CODE", "Age band": "POLICY_OWNER_AGE_BAND", "Tenure": "TENURE_BAND",
          "Joining channel": "CHANNEL_ACQUISITION", "Income tier": "INCOME_TIER"}
SMALL = 100


def main():
    from sklearn.metrics import roc_auc_score
    booster, meta = load_model("final")
    cols, _ = model_columns(meta["features"])
    X, ahead, _ = load_split("test")                      # all columns: groups include income tier
    s = booster.predict(np.ascontiguousarray(X[:, cols]))
    p = calibrate(s, meta["scale_pos_weight"])
    y = (ahead > 0).astype(np.int8)
    listed = np.zeros(len(s), bool); listed[np.argsort(-s, kind="stable")[: int(0.05 * len(s))]] = True

    values = {}
    for name, col in GROUPS.items():
        v = X[:, C.ALL_FEATURES.index(col)]
        values[name] = (np.where(np.isnan(v), "(blank)", np.char.add("Tier ", np.nan_to_num(v).astype(int).astype(str)))
                        if col == "INCOME_TIER" else decode(col, v))
    values["WA vs rest"] = np.where(values["State"] == "WA", "WA", "Other states")

    rows = []
    for g, v in values.items():
        for val in sorted(set(v)):
            m = v == val; yg = y[m]
            rows.append({"group": g, "value": val, "share_of_book": float(m.mean()), "leavers": int(yg.sum()),
                         "actual_rate": float(yg.mean()), "predicted_rate": float(p[m].mean()),
                         "ranking": float(roc_auc_score(yg, s[m])) if 0 < yg.sum() < len(yg) else None,
                         "caught_in_top5": float(listed[m & (y == 1)].mean()) if yg.sum() else None,
                         "on_top5_list": float(listed[m].mean()), "small_group": bool(yg.sum() < SMALL)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{C.RESULTS}/group_check.csv", index=False)
    save_json({"overall": {"ranking": float(roc_auc_score(y, s)), "caught_in_top5": float(listed[y == 1].mean()),
                           "predicted_rate": float(p.mean()), "actual_rate": float(y.mean())}, "groups": rows},
              C.RESULTS, "group_check.json")
    pd.set_option("display.width", 200)
    show = df.assign(**{c: (df[c] * 100).round(1) for c in ["share_of_book", "ranking", "caught_in_top5", "on_top5_list"]},
                     actual_rate=(df.actual_rate * 100).round(2), predicted_rate=(df.predicted_rate * 100).round(2))
    for g, d in show.groupby("group", sort=False):
        print(f"\n{g}\n" + d.drop(columns="group").to_string(index=False))


if __name__ == "__main__":
    main()
