"""
train_fraud.py (model) — production retrain on the expanded fraud_positive label.

ONE-VARIABLE change from the current production model (src/model/train.py): the
training label moves from provider_on_leie (578 all-LEIE) to fraud_positive (889 =
fraud-relevant LEIE ∪ AHCCCS ∪ NV — see labels/VERDICT.md). Everything else is
reused verbatim: build_feature_matrix (same 42 leakage-free features + blocklist)
and LGB_PARAMS. The existing train.py / score.py are left byte-identical and remain
the runnable fallback until the lead diff is approved (GATE 4).

Because 371 of the 889 positives (the AHCCCS/NV net-new) sit OUTSIDE the 308k PU
frame, the training frame is rebuilt from the full scored universe:
  positives = fraud_positive
  negatives = confident-clean (anomaly_score==0 & ~not_scored), NOT on any exclusion
              list, and NOT sharing a company_id with a positive (company-level
              disjointness, per GATE 0.3).

This module does two things:
  (1) GATE 2 — reproduce the temporal result through the production feature/HP code:
      train ≤2023 / val 2024 / test 2025-26, 3 seeds → eval_temporal.csv.
  (2) Fit the production scorer on ALL labeled rows (random holdout for early
      stopping, same as train.py) → lgbm_fraud.txt + feature_list.json.

Outputs go to Model/fraud_positive/ (current production artifacts/ untouched).

Run:
    python -m src.model.train_fraud
"""

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from . import config
from .data import build_feature_matrix, require
from .train import LGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS

LABELS = config.MODEL_DATA_DIR / "labels" / "expanded_labels.parquet"
OUT_DIR = config.MODEL_DATA_DIR / "fraud_positive"
SEEDS = [0, 1, 2]
TRAIN_MAX_YEAR, VAL_YEAR, TEST_YEARS = 2023, 2024, (2025, 2026)


def log(m=""):
    print(m, flush=True)


def neg_split(npi):
    """deterministic 70/15/15 train/val/test for clean negatives, by NPI."""
    h = int(npi) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def metrics(y, s):
    order = np.argsort(-s)
    yr = y[order]
    out = {"n": int(len(y)), "n_pos": int(y.sum()),
           "pr_auc": float(average_precision_score(y, s)),
           "roc_auc": float(roc_auc_score(y, s))}
    for k in (50, 100):
        out[f"p{k}"] = float(yr[:k].sum() / k)
    return out


def build_frame():
    """Full-universe frame with the fraud_positive label, company-disjoint negatives,
    and the temporal split year. Returns (X, base) aligned by row."""
    df = pd.read_parquet(config.SCORED_UNIVERSE_PARQUET)
    X = build_feature_matrix(df).reset_index(drop=True)             # asserts blocklist intact
    base = df[["npi", "anomaly_score", "not_scored"]].reset_index(drop=True)
    lab = pd.read_parquet(LABELS).set_index("npi")
    base = base.join(lab, on="npi")
    cmap = pd.read_parquet(config.NPI_TO_COMPANY_MAP).set_index("npi")["company_id"]
    base["company_id"] = base["npi"].map(cmap)

    base["pos"] = base["fraud_positive"].fillna(False)
    excluded_any = base["on_leie_any"].fillna(False) | base["on_ahcccs"].fillna(False) | base["on_nv"].fillna(False)
    pos_company = set(base.loc[base["pos"], "company_id"].dropna())
    base["neg"] = ((base["anomaly_score"] == 0) & (~base["not_scored"].fillna(True))
                   & (~excluded_any) & (~base["company_id"].isin(pos_company)))
    require("positive and negative sets are disjoint", not bool((base["pos"] & base["neg"]).any()))
    log(f"    frame: {len(base):,} rows | {int(base['pos'].sum())} positives | "
        f"{int(base['neg'].sum()):,} clean negatives | {X.shape[1]} features")
    return X, base


def temporal_eval(X, base):
    """GATE 2: 3-seed temporal val/test through the production feature/HP code."""
    yr = base["excl_year_all"]
    split = pd.Series(index=base.index, dtype=object)
    split[base["neg"]] = base.loc[base["neg"], "npi"].map(neg_split).values
    neg_tr, neg_va, neg_te = (split == "train").values, (split == "val").values, (split == "test").values
    pos = base["pos"].values
    tr_pos = pos & (yr <= TRAIN_MAX_YEAR).values
    va = (pos & (yr == VAL_YEAR).values) | neg_va
    te = (pos & yr.isin(TEST_YEARS).values) | neg_te
    yv, yt = pos[va].astype(int), pos[te].astype(int)
    log(f"    temporal: train+ {tr_pos.sum()} | val+ {yv.sum()} test+ {yt.sum()} | "
        f"neg tr {neg_tr.sum():,}/va {neg_va.sum():,}/te {neg_te.sum():,}")
    rows = []
    for seed in SEEDS:
        params = dict(LGB_PARAMS); params["seed"] = seed
        dtr = lgb.Dataset(X[tr_pos | neg_tr], label=pos[tr_pos | neg_tr].astype(int),
                          categorical_feature=config.CATEGORICAL_FEATURES)
        dval = lgb.Dataset(X[va], label=yv, reference=dtr)
        b = lgb.train(params, dtr, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)])
        mv, mt = metrics(yv, b.predict(X[va])), metrics(yt, b.predict(X[te]))
        rows.append({"seed": seed, "best_iter": b.best_iteration,
                     "val_pr": mv["pr_auc"], "val_p50": mv["p50"], "val_p100": mv["p100"],
                     "test_pr": mt["pr_auc"], "test_p100": mt["p100"]})
        log(f"      seed{seed}: val PR {mv['pr_auc']:.3f} P@50 {mv['p50']:.2f} | test PR {mt['pr_auc']:.3f} P@100 {mt['p100']:.2f}")
    res = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DIR / "eval_temporal.csv", index=False)
    log(f"    val PR-AUC {res['val_pr'].mean():.3f} [{res['val_pr'].min():.3f}-{res['val_pr'].max():.3f}] | "
        f"test PR-AUC {res['test_pr'].mean():.3f} [{res['test_pr'].min():.3f}-{res['test_pr'].max():.3f}]")
    return res


def fit_production(X, base):
    """Fit the scorer on ALL labeled rows (random holdout for early stopping)."""
    lab_mask = (base["pos"] | base["neg"]).values
    Xl, yl = X[lab_mask], base.loc[lab_mask, "pos"].astype("int8").values
    Xt, Xv, yt, yv = train_test_split(Xl, yl, test_size=config.VAL_FRACTION,
                                      stratify=yl, random_state=config.SEED)
    params = dict(LGB_PARAMS); params["seed"] = config.SEED
    dtr = lgb.Dataset(Xt, label=yt, categorical_feature=config.CATEGORICAL_FEATURES)
    dval = lgb.Dataset(Xv, label=yv, reference=dtr)
    b = lgb.train(params, dtr, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dval], valid_names=["val"],
                  callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(200)])
    log(f"    production fit: {len(Xl):,} labeled rows, best_iter {b.best_iteration}")
    b.save_model(str(OUT_DIR / "lgbm_fraud.txt"), num_iteration=b.best_iteration)
    feats = list(Xl.columns)
    (OUT_DIR / "feature_list.json").write_text(json.dumps(
        {"features": feats, "categorical": config.CATEGORICAL_FEATURES, "label": "fraud_positive",
         "categories": {c: Xl[c].cat.categories.tolist()
                        for c in config.CATEGORICAL_FEATURES if c in Xl.columns}}, indent=2))
    return b


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    log("[1/3] Building fraud_positive training frame (full universe)")
    X, base = build_frame()
    log("[2/3] GATE 2 — temporal reproduction (3 seeds)")
    res = temporal_eval(X, base)
    require("temporal test PR-AUC in ~0.55 neighborhood", res["test_pr"].mean() > 0.40,
            f"mean {res['test_pr'].mean():.3f}")
    require("no seed collapse (min val PR-AUC well above 0.04)", res["val_pr"].min() > 0.30,
            f"min {res['val_pr'].min():.3f}")
    log("[3/3] Fitting production scorer on all labeled rows")
    fit_production(X, base)
    log(f"    → {OUT_DIR}")


if __name__ == "__main__":
    main()
