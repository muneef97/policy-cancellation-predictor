"""
Step 9 - Question 3: how early can each type be predicted?

Uses the final model's validation scores (Dec 2025 - Jan 2026), ranked across the whole book.
For each cancellation type and each lead time (leaving 1, 2 or 3 months after the snapshot):
  share caught in the riskiest 5% and 10% of the book, and ranking skill against stayers.
Adds the exact-month models from step 4 (one model per lead month) as an overall view.

Writes results/how_early.json
"""
import os

import numpy as np
import pandas as pd

import config as C
from common import cancel_type, read_json, save_json


def main():
    from sklearn.metrics import roc_auc_score
    v = pd.read_parquet(os.path.join(C.MODELS, "final", "val_scores.parquet"))
    rank = v["score"].rank(ascending=False, method="first").to_numpy() / len(v)
    ahead, s = v["months_ahead"].to_numpy(), v["score"].to_numpy()
    ctype = np.where(ahead > 0, cancel_type(v), "STAY")
    stay = ahead == 0
    rows = []
    for t in C.ALL_TYPES + ["All leavers"]:
        for m in (1, 2, 3):
            f = (ahead == m) & ((ctype == t) | (t == "All leavers"))
            rows.append({"type": t, "months_ahead": m, "leavers": int(f.sum()),
                         "caught_in_top5": float((rank[f] <= 0.05).mean()), "caught_in_top10": float((rank[f] <= 0.10).mean()),
                         "ranking": float(roc_auc_score(np.r_[np.ones(f.sum()), np.zeros(stay.sum())], np.r_[s[f], s[stay]]))})
    exact = []
    for k in (1, 2, 3):
        m = read_json(C.MODELS, f"exact_month_{k}", "metrics.json")
        exact.append({"months_ahead": k, "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "base_rate": m["base_rate"],
                      "top1_precision": m["top"][0]["precision"], "top1_lift": m["top"][0]["lift"]})
    save_json({"by_type": rows, "exact_month_models": exact}, C.RESULTS, "how_early.json")
    df = pd.DataFrame(rows)
    for c in ["caught_in_top5", "caught_in_top10", "ranking"]:
        df[c] = (df[c] * 100).round(1)
    print(df.to_string(index=False))
    print("\nexact-month models:", [(e["months_ahead"], round(e["roc_auc"], 3), round(e["top1_lift"], 1)) for e in exact])


if __name__ == "__main__":
    main()
