"""
Step 10 - Question 3, continued: how did each leaver's risk build up before they left?

Leavers are grouped by their risk over the 6 months before they cancelled.
To avoid a policy being scored by a model that learned from its own history, policies are split in
two halves by a hash of their ID; a model trained on one half (training months only) scores the other
half in every month (out-of-fold scores). Scores become a within-month percentile, so "top 5%"
means the same in every month. A leaver's anchor is the first month they cancel within the next
month; only leavers with all 6 months observed are grouped (k-means, 4 groups).
The locked test months are not used.

Writes results/risk_journeys.json and results/risk_journeys.parquet
"""
import os

import numpy as np
import pandas as pd

import config as C
from common import load_month, model_columns, month_files, save_json

HISTORY = 6
GROUPS = 4


def half_of(hashes):
    return (pd.util.hash_pandas_object(pd.Series(hashes), index=False).to_numpy() % 2).astype(np.int8)


def train_halves(cols, cat_idx):
    import lightgbm as lgb
    models = []
    for h in (0, 1):
        Xs, ys = [], []
        for p in month_files("train"):
            X, ahead, k = load_month(p, cols)
            m = half_of(k["POLICY_HASH"]) == h
            Xs.append(X[m]); ys.append((ahead[m] > 0).astype(np.int8))
        X, y = np.concatenate(Xs), np.concatenate(ys)
        params = C.lgb_params(float((len(y) - y.sum()) / y.sum())); params["metric"] = "none"
        models.append(lgb.train(params, lgb.Dataset(X, label=y, feature_name=C.MODEL_FEATURES, categorical_feature=cat_idx),
                                num_boost_round=C.ROUNDS))
        print(f"  half {h}: trained on {len(y):,} rows", flush=True)
    return models


def score_months(models, cols):
    out = []
    for p in month_files("train") + month_files("gap") + month_files("val"):
        X, ahead, k = load_month(p, cols)
        h = half_of(k["POLICY_HASH"])
        s = np.empty(len(k))
        for i in (0, 1):                   # the model trained on half i scores the other half
            s[h != i] = models[i].predict(X[h != i])
        k = k.assign(months_ahead=ahead, pct=pd.Series(s).rank(ascending=False, pct=True).to_numpy())  # ~0 = riskiest
        out.append(k[["POLICY_HASH", "INFORMATION_DATE", "ctype", "months_ahead", "pct"]])
    s = pd.concat(out, ignore_index=True)
    s["INFORMATION_DATE"] = pd.to_datetime(s["INFORMATION_DATE"])
    return s


def journeys(s):
    anchors = (s[s.months_ahead == 1].sort_values("INFORMATION_DATE").groupby("POLICY_HASH", as_index=False).first()
               [["POLICY_HASH", "INFORMATION_DATE", "ctype"]].rename(columns={"INFORMATION_DATE": "t0"}))
    h = s[s.POLICY_HASH.isin(anchors.POLICY_HASH)][["POLICY_HASH", "INFORMATION_DATE", "pct"]].merge(anchors, on="POLICY_HASH")
    h["before"] = (h.t0.dt.year - h.INFORMATION_DATE.dt.year) * 12 + (h.t0.dt.month - h.INFORMATION_DATE.dt.month)
    h = h[(h.before >= 0) & (h.before < HISTORY)]
    wide = h.pivot_table(index="POLICY_HASH", columns="before", values="pct").dropna()
    wide.columns = [f"m{-int(c)}" for c in wide.columns]
    cols = [f"m{-i}" for i in range(HISTORY - 1, -1, -1)]
    return wide[cols].reset_index().merge(anchors[["POLICY_HASH", "ctype"]], on="POLICY_HASH"), cols


def describe(traj, labels, cols):
    rows = []
    for c in range(GROUPS):
        t = traj[labels == c]
        arr = t[cols].to_numpy()
        ever = (arr <= 0.10).any(1)
        types = t["ctype"].fillna(C.UNCLASSIFIED).replace("", C.UNCLASSIFIED).value_counts(normalize=True)
        rows.append({"cluster": c, "leavers": len(t), "share": len(t) / len(traj),
                     **{f"median_top_pct_{col}": float(t[col].median()) for col in cols},
                     "share_in_top10_at_5_months_before": float((t["m-5"] <= 0.10).mean()),
                     "share_in_top10_at_last_month": float((t["m0"] <= 0.10).mean()),
                     "share_ever_top10": float(ever.mean()),
                     "top_types": types.head(5).round(3).to_dict()})
    return rows


def main():
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    cols, cat_idx = model_columns()
    s = score_months(train_halves(cols, cat_idx), cols)
    traj, mcols = journeys(s)
    Z = -np.log10(np.clip(traj[mcols].to_numpy(), 1e-4, 1))       # top 1% vs 5% matters more than 40% vs 60%
    sample = np.random.RandomState(C.SEED).choice(len(Z), size=min(15000, len(Z)), replace=False)
    silhouette = {}
    for k in range(2, 7):
        lab = KMeans(n_clusters=k, n_init=10, random_state=C.SEED).fit(Z).labels_
        silhouette[k] = float(silhouette_score(Z[sample], lab[sample]))
    km = KMeans(n_clusters=GROUPS, n_init=20, random_state=C.SEED).fit(Z)
    order = np.argsort(km.cluster_centers_[:, 0])[::-1]        # most warning 5 months out first
    labels = np.array([{old: new for new, old in enumerate(order)}[c] for c in km.labels_])
    rows = describe(traj, labels, mcols)
    traj.assign(group=labels).to_parquet(os.path.join(C.RESULTS, "risk_journeys.parquet"), index=False)
    save_json({"n_leavers": int(len(traj)), "k": GROUPS, "clusters": rows, "silhouette": silhouette},
              C.RESULTS, "risk_journeys.json")
    print(f"{len(traj):,} leavers with 6 months of history; silhouette by k:", {k: round(v, 3) for k, v in silhouette.items()})
    for r in rows:
        print(f"  group {r['cluster']}: {r['share']:.0%} of leavers, in top 10% five months before "
              f"{r['share_in_top10_at_5_months_before']:.0%}, in the final month {r['share_in_top10_at_last_month']:.0%}")


if __name__ == "__main__":
    main()
