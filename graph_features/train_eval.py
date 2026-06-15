"""
train_eval.py (graph_features) — Phases 0,4,5: baseline sanity, parallel models,
verdict.

Augments the existing NPI LightGBM with graph-feature columns and asks, honestly,
whether structure adds signal — on the SAME temporal split the GNN used, with the
SAME model hyperparameters (only the feature columns change).

Variants (each ≥3 seeds):
  billing            — the existing model's 42 features (baseline)
  billing+structure  — + degree / component (label-free structure thesis)
  billing+proximity  — + excluded-proximity (temporally clean; "near known-bad")
  billing+all        — everything

Reports val/test PR-AUC + precision@K, runs the structure-only leakage ablation,
and a SHAP read on which graph features (if any) the model actually uses. Writes
eval_headtohead.csv, leakage_ablation.csv, shap_summary.txt, VERDICT.md.

GATE 0 (--gate0): reproduce the existing model (~0.465 PR-AUC) on its RANDOM split
to prove the training harness is faithful before the temporal experiment.

Run:
    python -m graph_features.train_eval --gate0     # harness fidelity check
    python -m graph_features.train_eval             # the experiment
"""

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.model import config as mcfg
from src.model.data import build_feature_matrix
from graph_features import config

# the existing model's hyperparameters (CLAUDE.md) — heavy regularization, unchanged
LGB = dict(objective="binary", metric="average_precision", learning_rate=0.03,
           num_leaves=15, min_data_in_leaf=100, lambda_l2=10.0, feature_fraction=0.6,
           bagging_fraction=0.8, bagging_freq=1, is_unbalance=True, verbosity=-1)
ROUNDS, STOP, SEEDS = 10000, 300, [0, 1, 2]
PKS = [50, 100]


def log(m=""):
    print(m, flush=True)


def patk(y, s, k):
    return y[np.argsort(-s)[:k]].mean()


def fit_eval(Xtr, ytr, Xva, yva, Xte, yte, seed):
    dtr = lgb.Dataset(Xtr, ytr); dva = lgb.Dataset(Xva, yva, reference=dtr)
    b = lgb.train({**LGB, "seed": seed}, dtr, num_boost_round=ROUNDS, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(STOP, verbose=False)])
    sv = b.predict(Xva, num_iteration=b.best_iteration)
    st = b.predict(Xte, num_iteration=b.best_iteration)
    return b, {"val_pr": average_precision_score(yva, sv), "val_roc": roc_auc_score(yva, sv),
               "test_pr": average_precision_score(yte, st),
               **{f"val_p{k}": patk(yva, sv, k) for k in PKS},
               **{f"test_p{k}": patk(yte, st, k) for k in PKS}}


def gate0():
    """Reproduce the existing model on its random PU split (harness fidelity)."""
    log("GATE 0 — reproduce existing model on RANDOM PU split (expect ~0.465)")
    df = pd.read_parquet(mcfg.PU_TRAINING_PARQUET)
    X = build_feature_matrix(df); y = df[mcfg.LABEL].astype(int).to_numpy()
    tri, vai = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=42)
    _, m = fit_eval(X.iloc[tri], y[tri], X.iloc[vai], y[vai], X.iloc[vai], y[vai], 42)
    log(f"  reproduced val PR-AUC {m['val_pr']:.4f} (original 0.465) → "
        f"{'OK' if abs(m['val_pr']-0.465) < 0.06 else 'CHECK'}")


def load_experiment():
    """NPI features (42) + graph features, aligned, with temporal split masks."""
    df = pd.read_parquet(config.SCORED_UNIVERSE)
    X = build_feature_matrix(df).reset_index(drop=True)
    base = list(X.columns)
    X["npi"] = df["npi"].values
    g = pd.read_parquet(config.NPI_GRAPH_FEATURES)
    X = X.merge(g, on="npi", how="left", validate="1:1")
    gcols = config.STRUCTURE_FEATS + config.PROXIMITY_FEATS
    sp = pd.read_parquet(config.SPLIT_MASKS, columns=["npi", "split"])
    ls = pd.read_parquet(config.NODES_PARQUET, columns=["npi", "label_status"])
    meta = X[["npi"]].merge(sp, on="npi", how="left").merge(ls, on="npi", how="left")
    y = (meta["label_status"].to_numpy() == "positive").astype(int)
    lab = np.isin(meta["label_status"].to_numpy(), ["positive", "reliable_neg"])
    masks = {s: ((meta["split"].to_numpy() == s) & lab) for s in ["train", "val", "test"]}
    return X.drop(columns=["npi"]), y, masks, base, gcols


def run_variant(name, cols, X, y, masks):
    Xc = X[cols]
    rows = [fit_eval(Xc[masks["train"]], y[masks["train"]], Xc[masks["val"]], y[masks["val"]],
                     Xc[masks["test"]], y[masks["test"]], s)[1] for s in SEEDS]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg["val_pr_range"] = [min(r["val_pr"] for r in rows), max(r["val_pr"] for r in rows)]
    log(f"  {name:20s} val PR {agg['val_pr']:.4f} {agg['val_pr_range']} | "
        f"val P@50 {agg['val_p50']:.2f} P@100 {agg['val_p100']:.2f} | test PR {agg['test_pr']:.4f}")
    return {"variant": name, **agg}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate0", action="store_true")
    args = ap.parse_args()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.gate0:
        gate0(); return

    log("Loading experiment table (42 billing + 10 graph features, temporal split)")
    X, y, masks, base, gcols = load_experiment()
    for s in ["train", "val", "test"]:
        log(f"  {s}: {int(masks[s].sum()):,} labeled ({int(y[masks[s]].sum())} pos)")

    log("\nGATE 4-5 — parallel models (≥3 seeds, same HP, same temporal split)")
    variants = {
        "billing": base,
        "billing+structure": base + config.STRUCTURE_FEATS,
        "billing+proximity": base + config.PROXIMITY_FEATS,
        "billing+all": base + gcols,
    }
    res = [run_variant(n, c, X, y, masks) for n, c in variants.items()]
    pd.DataFrame(res).to_csv(config.OUT_DIR / "eval_headtohead.csv", index=False)

    # leakage ablation: structure-only is the honest structure thesis test
    base_pr = next(r for r in res if r["variant"] == "billing")["val_pr"]
    struct_pr = next(r for r in res if r["variant"] == "billing+structure")["val_pr"]
    prox_pr = next(r for r in res if r["variant"] == "billing+proximity")["val_pr"]
    base_rng = next(r for r in res if r["variant"] == "billing")["val_pr_range"]
    pd.DataFrame([
        {"comparison": "structure lift", "delta_val_pr": struct_pr - base_pr},
        {"comparison": "proximity lift", "delta_val_pr": prox_pr - base_pr},
    ]).to_csv(config.OUT_DIR / "leakage_ablation.csv", index=False)

    # SHAP: how much does the model actually use the graph features?
    log("\nSHAP on billing+all (graph-feature usage)")
    import shap
    Xall = X[base + gcols]
    b = fit_eval(Xall[masks["train"]], y[masks["train"]], Xall[masks["val"]], y[masks["val"]],
                 Xall[masks["val"]], y[masks["val"]], 0)[0]
    sv = shap.TreeExplainer(b).shap_values(Xall[masks["val"]])
    sv = sv[1] if isinstance(sv, list) else sv
    mabs = np.abs(sv).mean(0)
    imp = pd.Series(mabs, index=Xall.columns).sort_values(ascending=False)
    gf_rank = {f: int(imp.index.get_loc(f)) + 1 for f in gcols}
    (config.OUT_DIR / "shap_summary.txt").write_text(
        "mean|SHAP| ranking (1=most used). Graph-feature ranks among "
        f"{len(imp)} features:\n" +
        "\n".join(f"  {f}: rank {gf_rank[f]}  (mean|shap| {imp[f]:.4f})" for f in gcols) +
        "\n\nTop 15 overall:\n" + imp.head(15).to_string())
    log("  graph-feature SHAP ranks (of %d): %s" % (len(imp),
        {f: gf_rank[f] for f in sorted(gcols, key=lambda x: gf_rank[x])[:5]}))

    # verdict
    lift = struct_pr - base_pr
    beyond_noise = struct_pr > base_rng[1]
    verdict = ("ADDS SIGNAL" if (lift > 0 and beyond_noise) else "NEUTRAL")
    log("\n" + "=" * 60 + "\nGATE 5 — VERDICT\n" + "=" * 60)
    log(f"  baseline (billing) val PR-AUC {base_pr:.4f} {base_rng}")
    log(f"  +structure {struct_pr:.4f} (Δ {lift:+.4f}) | +proximity {prox_pr:.4f} "
        f"(Δ {prox_pr-base_pr:+.4f}) | +all {next(r for r in res if r['variant']=='billing+all')['val_pr']:.4f}")
    log(f"  structure lift beyond seed-range? {beyond_noise} → {verdict}")
    (config.OUT_DIR / "verdict.json").write_text(json.dumps(
        {"baseline_val_pr": base_pr, "baseline_range": base_rng,
         "structure_val_pr": struct_pr, "structure_lift": lift,
         "proximity_val_pr": prox_pr, "proximity_lift": prox_pr - base_pr,
         "structure_beyond_noise": bool(beyond_noise), "verdict": verdict,
         "results": res}, indent=2))


if __name__ == "__main__":
    main()
