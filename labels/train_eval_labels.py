"""
train_eval_labels.py (labels) — does a richer/cleaner positive label improve the
model? Head-to-head on an identical temporal split, with a COMMON evaluation
target so the PR-AUC numbers are comparable across variants.

Design (the fairness comes from holding eval fixed and varying only training):
  * Common eval target = fraud_positive (889): fraud-relevant LEIE ∪ AHCCCS ∪ NV.
    val = fraud positives excluded in 2024; test = excluded in 2025-26.
  * Common negatives = confident-clean (anomaly_score==0 & ~not_scored) and NOT
    excluded by ANY source; split train/val/test by a deterministic NPI hash.
  * Variants differ ONLY in the TRAINING positive set (each filtered to ≤2023):
      A_leie578     train on all LEIE            (current production label)
      B_leie_fraud  train on fraud-relevant LEIE (cleaned)
      C_expanded    train on fraud_positive      (cleaned + state enforcement)
  * Same LightGBM HP as src/model/train.py, early-stop on val PR-AUC, 3 seeds.

So every variant is scored on the SAME val/test rows (future fraud + same clean
negatives); the only thing that moves the metric is the training label. Read-only
on src/model; nothing in the production pipeline changes.

Run:
    python -m labels.train_eval_labels
"""

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.model import config as mcfg
from src.model.data import build_feature_matrix
from src.model.train import LGB_PARAMS

LABELS = mcfg.MODEL_DATA_DIR / "labels" / "expanded_labels.parquet"
OUT = mcfg.MODEL_DATA_DIR / "labels" / "label_compare.csv"
SEEDS = [0, 1, 2]
TRAIN_MAX_YEAR, VAL_YEAR, TEST_YEARS = 2023, 2024, (2025, 2026)
VARIANTS = {                       # variant -> column in expanded_labels giving its train positives
    "A_leie578": "on_leie_any",
    "B_leie_fraud": "on_leie_fraud",
    "C_expanded": "fraud_positive",
}


def log(m=""):
    print(m, flush=True)


def neg_split(npi):
    """deterministic 70/15/15 train/val/test for clean negatives, by NPI."""
    h = int(npi) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def metrics(y, s):
    order = np.argsort(-s)
    yr = y[order]
    out = {
        "n": len(y), "n_pos": int(y.sum()), "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
    }
    for k in (50, 100):
        out[f"p@{k}"] = float(yr[:k].sum() / k)
    return out


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()

    log("[1/3] Loading features + labels")
    df = pd.read_parquet(mcfg.SCORED_UNIVERSE_PARQUET)
    X = build_feature_matrix(df).reset_index(drop=True)
    base = df[["npi", "anomaly_score", "not_scored"]].reset_index(drop=True)
    lab = pd.read_parquet(LABELS).set_index("npi")
    base = base.join(lab, on="npi")
    log(f"  {len(X):,} rows × {X.shape[1]} features")

    excluded_any = base["on_leie_any"].fillna(False) | base["on_ahcccs"].fillna(False) | base["on_nv"].fillna(False)
    clean = (base["anomaly_score"] == 0) & (~base["not_scored"].fillna(True)) & (~excluded_any)
    split = pd.Series(index=base.index, dtype=object)
    split[clean] = base.loc[clean, "npi"].map(neg_split).values
    neg_train = (split == "train").values
    neg_val = (split == "val").values
    neg_test = (split == "test").values

    yr = base["excl_year_all"]
    # Two eval targets. fraud_pos is what we want to find; leie_fraud is the SUBSET that
    # excludes the state lists C trains on — the bias check (C must win here too, not just
    # on its own kind). Both scored on the SAME clean negatives.
    targets = {
        "fraud_pos": base["fraud_positive"].fillna(False).values,
        "leie_fraud": base["on_leie_fraud"].fillna(False).values,
    }
    evals = {}      # name -> (val_mask, y_val, test_mask, y_test)
    for tname, tarr in targets.items():
        vmask = (tarr & (yr == VAL_YEAR).values) | neg_val
        tmask = (tarr & yr.isin(TEST_YEARS).values) | neg_test
        evals[tname] = (vmask, tarr[vmask], tmask, tarr[tmask])
        log(f"  target {tname:11s}: val pos {int(tarr[vmask].sum())} test pos {int(tarr[tmask].sum())}")
    log(f"  clean negatives: train {neg_train.sum():,} val {neg_val.sum():,} test {neg_test.sum():,}")

    log("[2/3] Training variants (3 seeds each)")
    params = dict(LGB_PARAMS)
    # early-stop on the fraud_pos val target (the real objective)
    es_vmask, es_yval, _, _ = evals["fraud_pos"]
    rows = []
    for vname, col in VARIANTS.items():
        pos = base[col].fillna(False).values
        train_pos = pos & (yr <= TRAIN_MAX_YEAR).values
        tr_mask = train_pos | neg_train
        Xtr, ytr = X[tr_mask], train_pos[tr_mask].astype(int)
        for seed in SEEDS:
            params["seed"] = seed
            dtr = lgb.Dataset(Xtr, label=ytr)
            dval = lgb.Dataset(X[es_vmask], label=es_yval.astype(int), reference=dtr)
            booster = lgb.train(
                params, dtr, num_boost_round=10000, valid_sets=[dval],
                callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)],
            )
            row = {"variant": vname, "n_train_pos": int(train_pos.sum()), "seed": seed,
                   "best_iter": booster.best_iteration}
            for tname, (vmask, yv, tmask, yt) in evals.items():
                mv, mt = metrics(yv, booster.predict(X[vmask])), metrics(yt, booster.predict(X[tmask]))
                row[f"{tname}_val_pr"] = mv["pr_auc"]
                row[f"{tname}_val_p50"] = mv["p@50"]
                row[f"{tname}_val_p100"] = mv["p@100"]
                row[f"{tname}_test_pr"] = mt["pr_auc"]
            rows.append(row)
            log(f"  {vname:13s} seed{seed}  "
                f"fraud_pos val PR {row['fraud_pos_val_pr']:.3f} P@50 {row['fraud_pos_val_p50']:.2f} | "
                f"leie_fraud val PR {row['leie_fraud_val_pr']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)

    log("\n[3/3] " + "=" * 62 + "\nLABEL HEAD-TO-HEAD (3-seed mean ± range)\n" + "=" * 70)
    for tname in targets:
        vp = evals[tname][1]
        log(f"\n  EVAL TARGET = {tname}  (val pos {int(vp.sum())}, base rate {vp.mean():.3%})")
        log(f"  {'variant':13s} {'train+':>7s} {'val PR-AUC':>20s} {'val P@50':>9s} {'val P@100':>10s} {'test PR-AUC':>20s}")
        for v in VARIANTS:
            g = res[res["variant"] == v]
            log(f"  {v:13s} {g['n_train_pos'].iloc[0]:>7d} "
                f"{g[f'{tname}_val_pr'].mean():>9.3f} [{g[f'{tname}_val_pr'].min():.3f}-{g[f'{tname}_val_pr'].max():.3f}] "
                f"{g[f'{tname}_val_p50'].mean():>9.2f} {g[f'{tname}_val_p100'].mean():>10.2f} "
                f"{g[f'{tname}_test_pr'].mean():>9.3f} [{g[f'{tname}_test_pr'].min():.3f}-{g[f'{tname}_test_pr'].max():.3f}]")
    log(f"\n  → {OUT}")


if __name__ == "__main__":
    main()
