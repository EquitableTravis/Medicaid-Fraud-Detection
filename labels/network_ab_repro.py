"""Independent reproduction of Trey's network A/B on the frozen package.

NOT a port of his network_ab.py — reimplemented from the written protocol so a
reproduction actually means something. Mirrors his three designs and adds our
audit extensions as extra arms:

Designs
  FULL      : all eligible rows, forward label.
  MATCHED   : every forward-positive case + 3 controls matched on
              (taxonomy, state, net_paid quartile), ladder relaxing to
              (taxonomy, quartile) then taxonomy-only. Controls are ordinary
              never-banned peers (realistic), matched on the FORWARD label.

Arms (feature sets; base = manifest trainable minus betweenness)
  none        : base only (no network columns anywhere)
  full_net    : base + structural + label-adjacent network flags
                (within_2_hops_of_exclusion, subscore_ownership_integrity,
                 has_excluded_owner) — Trey's MATCHED design
  structural  : base + shell_score + related_party_density — Trey's
                MATCHED_STRUCTURAL design (the +0.035 claim)
  rpd_only    : base + related_party_density alone            [audit ext. 1a]
  shell_deprox: base + related_party_density + shell_score residualized on
                the exclusion-proximity channel (OLS residual vs
                within_2_hops_of_exclusion + excluded-distance proxy)
                                                              [audit ext. 1b]

Audit extension 2 (recency): every design also runs with forward positives
banned within --recency-months of the cutoff removed, to blunt
investigations-already-in-flight. Cutoff = 2023-12-31.

Stats: 5-fold grouped CV on the matched set; delta CIs by cluster bootstrap
over match groups (1,000 resamples), mirroring the written protocol.

Usage: python -m labels.network_ab_repro [--package DIR] [--seeds 42 43 44]
Writes labels/NETWORK_AB_REPRO.md + network_ab_repro.json.
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
from labels.train_frozen import load_package, trainable_columns, top_decile_lift

DEFAULT_PKG = Path.home() / "Desktop/Data/preclean/trey/frozen_2023-12"
CUTOFF = pd.Timestamp("2023-12-31")

STRUCTURAL = ["shell_score", "related_party_density"]
LABEL_ADJ_NET = ["within_2_hops_of_exclusion", "subscore_ownership_integrity",
                 "has_excluded_owner"]


def build_matched(df: pd.DataFrame, y_fwd: np.ndarray, eligible: np.ndarray,
                  rng: np.random.Generator, n_controls: int = 3) -> pd.DataFrame:
    """Case-control set: each forward positive + n matched never-banned controls."""
    pool = df[eligible].copy()
    pool["_y"] = y_fwd[eligible]
    pool["_quart"] = pd.qcut(pool["net_paid"].fillna(0), 4, labels=False, duplicates="drop")
    pool["_tax"] = pool["primary_taxonomy"].fillna("UNK")
    pool["_state"] = pool["practice_state"].fillna("UNK")

    cases = pool[pool["_y"] == 1]
    controls_pool = pool[pool["_y"] == 0]
    by_full = controls_pool.groupby(["_tax", "_state", "_quart"], observed=True).indices
    by_taxq = controls_pool.groupby(["_tax", "_quart"], observed=True).indices
    by_tax = controls_pool.groupby(["_tax"], observed=True).indices

    rows, group_ids = [], []
    used: set[int] = set()
    for gid, (_, case) in enumerate(cases.iterrows()):
        keys = [(by_full, (case["_tax"], case["_state"], case["_quart"])),
                (by_taxq, (case["_tax"], case["_quart"])),
                (by_tax, case["_tax"])]
        picked: list[int] = []
        for table, key in keys:
            cand = [i for i in table.get(key, []) if i not in used]
            rng.shuffle(cand)
            picked += cand[: n_controls - len(picked)]
            if len(picked) >= n_controls:
                break
        if not picked:
            continue
        used.update(picked)
        rows.append(case)
        group_ids.append(gid)
        for i in picked:
            rows.append(controls_pool.iloc[i])
            group_ids.append(gid)
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["_match_group"] = group_ids
    return out


def deprox_shell(df: pd.DataFrame) -> pd.Series:
    """shell_score with the exclusion-proximity channel partialled out (OLS residual)."""
    y = df["shell_score"].fillna(0).to_numpy()
    X = df[["within_2_hops_of_exclusion"]].fillna(0).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return pd.Series(y - X @ beta, index=df.index, name="shell_score_deprox")


def cv_scores(frame: pd.DataFrame, cols: list[str], y: np.ndarray,
              groups: np.ndarray, seed: int) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(frame[cols], y, groups):
        params = dict(LGB_PARAMS)
        params["seed"] = seed
        fit = lgb.train(params, lgb.Dataset(frame.iloc[tr][cols], label=y[tr]),
                        NUM_BOOST_ROUND,
                        valid_sets=[lgb.Dataset(frame.iloc[te][cols], label=y[te])],
                        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        oof[te] = fit.predict(frame.iloc[te][cols], num_iteration=fit.best_iteration)
    return oof


def bootstrap_delta(y, s_with, s_without, clusters, metric, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx_by_c = {c: np.where(clusters == c)[0] for c in uniq}
    deltas = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_c[c] for c in pick])
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        deltas.append(metric(y[idx], s_with[idx]) - metric(y[idx], s_without[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, default=DEFAULT_PKG)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--recency-months", type=int, default=6)
    args = ap.parse_args()

    manifest, df, fwd = load_package(args.package)
    base = trainable_columns(manifest, df)
    # base must exclude the network columns entirely
    net_all = set(STRUCTURAL + LABEL_ADJ_NET)
    base = [c for c in base if c not in net_all]

    fwd["first_excl_date"] = pd.to_datetime(fwd["first_excl_date"], errors="coerce")
    pre = set(fwd.loc[fwd["was_excluded_pre_cutoff"] == 1, "npi"])
    pos_all = set(fwd.loc[fwd["is_prospective_positive"] == 1, "npi"])
    recent = fwd["first_excl_date"] <= CUTOFF + pd.DateOffset(months=args.recency_months)
    pos_late = set(fwd.loc[(fwd["is_prospective_positive"] == 1) & ~recent, "npi"])

    df["shell_score_deprox"] = deprox_shell(df)
    eligible = (~df["npi"].isin(pre)).to_numpy()

    arms = {
        "none": base,
        "full_net": base + STRUCTURAL + LABEL_ADJ_NET,
        "structural": base + STRUCTURAL,
        "rpd_only": base + ["related_party_density"],
        "shell_deprox": base + ["related_party_density", "shell_score_deprox"],
    }
    results = {}
    for label_name, pos_set in [("all_forward", pos_all),
                                (f"late_only_gt{args.recency_months}mo", pos_late)]:
        y_fwd = df["npi"].isin(pos_set).astype(int).to_numpy()
        matched_frames = []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            matched = build_matched(df, y_fwd, eligible, rng)
            matched_frames.append(matched)
        print(f"[{label_name}] positives in universe: {int(y_fwd[eligible].sum()):,} | "
              f"matched rows (seed {args.seeds[0]}): {len(matched_frames[0]):,}")

        for arm_name, cols in arms.items():
            if arm_name == "none":
                continue
            per_seed = []
            for seed, matched in zip(args.seeds, matched_frames):
                ym = matched["_y"].to_numpy()
                gm = matched["_match_group"].to_numpy()
                s_with = cv_scores(matched, cols, ym, gm, seed)
                s_without = cv_scores(matched, arms["none"], ym, gm, seed)
                droc, rlo, rhi = bootstrap_delta(ym, s_with, s_without, gm, roc_auc_score)
                dpr, plo, phi = bootstrap_delta(ym, s_with, s_without, gm,
                                                average_precision_score)
                per_seed.append({"seed": seed,
                                 "roc_with": roc_auc_score(ym, s_with),
                                 "roc_without": roc_auc_score(ym, s_without),
                                 "d_roc": droc, "d_roc_ci": [rlo, rhi],
                                 "d_pr": dpr, "d_pr_ci": [plo, phi]})
                print(f"  [{label_name}] {arm_name} seed {seed}: "
                      f"ROC {per_seed[-1]['roc_with']:.4f} vs {per_seed[-1]['roc_without']:.4f} "
                      f"dROC {droc:+.4f} [{rlo:+.4f},{rhi:+.4f}] dPR {dpr:+.4f}")
            results[f"{label_name}::{arm_name}"] = per_seed

    out = Path(__file__).parent / "network_ab_repro.json"
    out.write_text(json.dumps(results, indent=2))

    lines = ["# Network A/B — independent reproduction + audit extensions\n",
             f"package: `{args.package}` | seeds {args.seeds} | matched 3:1 on "
             "(taxonomy, state, net_paid quartile) ladder | grouped 5-fold CV | "
             "cluster bootstrap CIs (1,000)\n",
             "| label | arm | ROC with | ROC without | ΔROC [95% CI] | ΔPR [95% CI] |",
             "|---|---|---|---|---|---|"]
    for key, per_seed in results.items():
        label_name, arm = key.split("::")
        m = {k: np.mean([s[k] for s in per_seed]) for k in
             ["roc_with", "roc_without", "d_roc", "d_pr"]}
        lo = np.mean([s["d_roc_ci"][0] for s in per_seed])
        hi = np.mean([s["d_roc_ci"][1] for s in per_seed])
        plo = np.mean([s["d_pr_ci"][0] for s in per_seed])
        phi = np.mean([s["d_pr_ci"][1] for s in per_seed])
        lines.append(f"| {label_name} | {arm} | {m['roc_with']:.4f} | {m['roc_without']:.4f} "
                     f"| {m['d_roc']:+.4f} [{lo:+.4f},{hi:+.4f}] "
                     f"| {m['d_pr']:+.4f} [{plo:+.4f},{phi:+.4f}] |")
    report = Path(__file__).parent / "NETWORK_AB_REPRO.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {report} and {out}")


if __name__ == "__main__":
    main()
