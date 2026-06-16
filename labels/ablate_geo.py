"""
ablate_geo.py (labels) — retrain the fraud_positive model WITHOUT practice_state and
primary_taxonomy, to test whether the model's signal is real billing-anomaly detection
or just "memorize that AZ behavioral-health is where the AHCCCS positives are."

Keeps entity_type and ALL the peer-normalized rate features (those reference
taxonomy/state only as a normalization baseline, not as a "this sector is bad" lookup).
Same temporal split / HP / 3 seeds as train_fraud, so PR-AUC is comparable to the
full-feature fraud_positive (val 0.574 / test 0.550).

If PR-AUC holds → the billing features carry the signal and we de-biased the geo/sector
concentration for free. If it collapses → the model was leaning on the categoricals.

Run:
    python -m labels.ablate_geo
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.model import config as c
from src.model.train import LGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS
from src.model.train_fraud import build_frame, neg_split, metrics, TRAIN_MAX_YEAR, VAL_YEAR, TEST_YEARS

GEO = ["primary_taxonomy", "practice_state"]


def log(m=""): print(m, flush=True)


def temporal_eval(X, base, tag):
    cats = [col for col in c.CATEGORICAL_FEATURES if col in X.columns]
    yr = base["excl_year_all"]
    split = pd.Series(index=base.index, dtype=object)
    split[base["neg"]] = base.loc[base["neg"], "npi"].map(neg_split).values
    neg_tr, neg_va, neg_te = (split == "train").values, (split == "val").values, (split == "test").values
    pos = base["pos"].values
    tr = (pos & (yr <= TRAIN_MAX_YEAR).values) | neg_tr
    va = (pos & (yr == VAL_YEAR).values) | neg_va
    te = (pos & yr.isin(TEST_YEARS).values) | neg_te
    yv, yt = pos[va].astype(int), pos[te].astype(int)
    rows = []
    for seed in (0, 1, 2):
        p = dict(LGB_PARAMS); p["seed"] = seed
        dtr = lgb.Dataset(X[tr], label=pos[tr].astype(int), categorical_feature=cats)
        dval = lgb.Dataset(X[va], label=yv, reference=dtr)
        b = lgb.train(p, dtr, NUM_BOOST_ROUND, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)])
        mv, mt = metrics(yv, b.predict(X[va])), metrics(yt, b.predict(X[te]))
        rows.append((mv["pr_auc"], mv["p50"], mt["pr_auc"], mt["p100"]))
    r = np.array(rows)
    log(f"  {tag:16s} val PR {r[:,0].mean():.3f} [{r[:,0].min():.3f}-{r[:,0].max():.3f}] "
        f"P@50 {r[:,1].mean():.2f} | test PR {r[:,2].mean():.3f} [{r[:,2].min():.3f}-{r[:,2].max():.3f}] "
        f"P@100 {r[:,3].mean():.2f}  ({X.shape[1]} feats)")


def main():
    X, base = build_frame()
    log("\nTEMPORAL EVAL — fraud_positive, geo/sector ablation (3 seeds):")
    temporal_eval(X, base, "full features")
    temporal_eval(X.drop(columns=GEO), base, "drop state+tax")


if __name__ == "__main__":
    main()
