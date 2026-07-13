"""Train on Trey's frozen_2023-12 package per the manifest contract; evaluate forward.

Design (per Trey's letter + our GATE-0 findings):
- Trainable = manifest feature groups (raw+peerpct+subscore) minus leakage_hard minus
  leakage_adjacent minus betweenness (constant) minus anything the manifest doesn't
  classify (52 orphan columns quarantined at GATE 0).
- Target = provider_on_exclusion (the in-time label, 1,162 positives).
- Split = 5-fold GroupKFold on group_id: every provider is scored OUT-OF-FOLD by a
  model that never saw its organization. Reuses src.model.train.LGB_PARAMS (the
  heavy-regularization config that survived every prior track).
- Evaluation = the FORWARD label (future_bans is_prospective_positive) on eligible
  rows only (was_excluded_pre_cutoff = 0): ROC-AUC, PR-AUC, top-decile lift,
  recall@1000, P@100. The in-time label is only the training signal; the forward
  number is the result.
- Variants: A = all trainable; B (strict) = drop feature_vintage == current_state,
  per the letter ("the gap tells you how much the today's-snapshot features help").

Usage: python -m labels.train_frozen [--package DIR] [--seeds 42 43 44]
Writes labels/FROZEN_TRAIN_REPORT.md + frozen_train_results.json (repo, no data).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.model.train import EARLY_STOPPING_ROUNDS, LGB_PARAMS, NUM_BOOST_ROUND

DEFAULT_PKG = Path.home() / "Desktop/Data/preclean/trey/frozen_2023-12"


def load_package(pkg: Path):
    manifest = json.loads((pkg / "feature_manifest.json").read_text())
    df = pd.read_parquet(pkg / "provider_features_for_model.parquet")
    df["npi"] = df["npi"].astype(str)
    fwd = pd.read_csv(pkg / "future_bans_after_2023-12.csv", dtype={"npi": str})
    return manifest, df, fwd


def trainable_columns(manifest: dict, df: pd.DataFrame) -> list[str]:
    groups = (manifest["raw_feature_cols"] + manifest["peerpct_cols"]
              + manifest["subscore_cols"] + manifest["embedding_cols"])
    fenced = set(manifest["leakage_hard"]) | set(manifest["leakage_adjacent"])
    cols = [c for c in groups
            if c in df.columns and c not in fenced and c != "betweenness"
            and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def strict_columns(cols: list[str], manifest: dict) -> list[str]:
    current_state = set(manifest["feature_vintage"].get("current_state", []))
    return [c for c in cols if c not in current_state]


def top_decile_lift(y: np.ndarray, score: np.ndarray) -> float:
    k = max(1, len(score) // 10)
    top = np.argsort(-score)[:k]
    return float(y[top].mean() / y.mean()) if y.mean() > 0 else float("nan")


def precision_at(y: np.ndarray, score: np.ndarray, k: int) -> float:
    top = np.argsort(-score)[:k]
    return float(y[top].mean())


def recall_at(y: np.ndarray, score: np.ndarray, k: int) -> float:
    top = np.argsort(-score)[:k]
    return float(y[top].sum() / y.sum()) if y.sum() > 0 else float("nan")


def oof_scores(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
               seed: int, n_folds: int = 5) -> np.ndarray:
    """Out-of-fold predictions with group-aware folds (in-time label training)."""
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_folds)
    for tr_idx, te_idx in gkf.split(X, y, groups):
        # inner early-stopping split, also group-aware
        tr_groups = groups[tr_idx]
        inner = GroupKFold(n_splits=5)
        (fit_i, val_i) = next(inner.split(X.iloc[tr_idx], y[tr_idx], tr_groups))
        fit_idx, val_idx = tr_idx[fit_i], tr_idx[val_i]
        params = dict(LGB_PARAMS)
        params["seed"] = seed
        booster = lgb.train(
            params,
            lgb.Dataset(X.iloc[fit_idx], label=y[fit_idx]),
            NUM_BOOST_ROUND,
            valid_sets=[lgb.Dataset(X.iloc[val_idx], label=y[val_idx])],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        oof[te_idx] = booster.predict(X.iloc[te_idx],
                                      num_iteration=booster.best_iteration)
    assert not np.isnan(oof).any()
    return oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, default=DEFAULT_PKG)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()

    manifest, df, fwd = load_package(args.package)
    cols_all = trainable_columns(manifest, df)
    cols_strict = strict_columns(cols_all, manifest)
    print(f"variant A (all trainable): {len(cols_all)} cols | "
          f"variant B (strict, no current_state): {len(cols_strict)} cols")

    y_train = df[manifest["label"]].fillna(0).astype(int).to_numpy()
    groups = df[manifest["group_cols"][0]].astype(str).fillna("solo:" + df["npi"]).to_numpy()

    pre_cut = set(fwd.loc[fwd["was_excluded_pre_cutoff"] == 1, "npi"])
    fwd_pos = set(fwd.loc[fwd["is_prospective_positive"] == 1, "npi"])
    eligible = ~df["npi"].isin(pre_cut)
    y_fwd = df["npi"].isin(fwd_pos).astype(int).to_numpy()
    print(f"eligible rows: {eligible.sum():,} | forward positives in universe: "
          f"{int(y_fwd[eligible].sum()):,} | in-time training positives: {y_train.sum():,}")

    results: dict[str, dict] = {}
    for name, cols in [("A_all_trainable", cols_all), ("B_strict_no_current_state", cols_strict)]:
        per_seed = []
        for seed in args.seeds:
            oof = oof_scores(df[cols], y_train, groups, seed)
            ye, se = y_fwd[eligible.to_numpy()], oof[eligible.to_numpy()]
            per_seed.append({
                "seed": seed,
                "fwd_roc_auc": roc_auc_score(ye, se),
                "fwd_pr_auc": average_precision_score(ye, se),
                "fwd_top_decile_lift": top_decile_lift(ye, se),
                "fwd_recall_at_1000": recall_at(ye, se, 1000),
                "fwd_p_at_100": precision_at(ye, se, 100),
                "intime_oof_pr_auc": average_precision_score(y_train, oof),
            })
            print(f"  {name} seed {seed}: fwd ROC {per_seed[-1]['fwd_roc_auc']:.4f} "
                  f"PR {per_seed[-1]['fwd_pr_auc']:.4f} "
                  f"lift@10% {per_seed[-1]['fwd_top_decile_lift']:.2f} "
                  f"recall@1000 {per_seed[-1]['fwd_recall_at_1000']:.3f}")
        agg = {k: (float(np.mean([s[k] for s in per_seed])),
                   float(np.min([s[k] for s in per_seed])),
                   float(np.max([s[k] for s in per_seed])))
               for k in per_seed[0] if k != "seed"}
        results[name] = {"n_cols": len(cols), "per_seed": per_seed, "mean_min_max": agg}

    out_json = Path(__file__).parent / "frozen_train_results.json"
    out_json.write_text(json.dumps(results, indent=2))

    lines = ["# Frozen-package training — forward-label results\n",
             f"package: `{args.package}` | seeds: {args.seeds} | "
             "5-fold GroupKFold(group_id), out-of-fold scoring, "
             "eval on eligible rows vs future bans\n",
             "| variant | cols | fwd ROC-AUC | fwd PR-AUC | lift@10% | recall@1000 | P@100 |",
             "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        m = r["mean_min_max"]
        lines.append(
            f"| {name} | {r['n_cols']} "
            f"| {m['fwd_roc_auc'][0]:.4f} [{m['fwd_roc_auc'][1]:.4f}-{m['fwd_roc_auc'][2]:.4f}] "
            f"| {m['fwd_pr_auc'][0]:.4f} "
            f"| {m['fwd_top_decile_lift'][0]:.2f} "
            f"| {m['fwd_recall_at_1000'][0]:.3f} "
            f"| {m['fwd_p_at_100'][0]:.3f} |")
    delta = (results["A_all_trainable"]["mean_min_max"]["fwd_roc_auc"][0]
             - results["B_strict_no_current_state"]["mean_min_max"]["fwd_roc_auc"][0])
    lines.append(f"\ncurrent_state contribution (A − B, fwd ROC-AUC): **{delta:+.4f}**\n")
    report = Path(__file__).parent / "FROZEN_TRAIN_REPORT.md"
    report.write_text("\n".join(lines))
    print(f"\nwrote {report} and {out_json}")


if __name__ == "__main__":
    main()
