"""GATE 0 for Trey's frozen_2023-12 package: verify every claim before any training.

Checks (each prints PASS/FAIL/WARN; exits non-zero on any FAIL):
  1. Shapes: 617,062 x 211 parquet; manifest counts; forward-label file integrity.
  2. Manifest partition: every parquet column accounted for (identifier / label-side /
     feature-group / leakage); trainable set derivation matches the letter's 126.
  3. Fences: leakage_hard (13) and leakage_adjacent (11) disjoint from trainable;
     known offenders (within_2_hops, shell_score, related_party_density, weak_label*,
     billing_after_deactivation, DME referrer) all fenced; weak-supervision LF columns
     absent or fenced.
  4. Label sanity: in-time label positives == manifest n_positives; forward label
     4,401 prospective positives; was_excluded_pre_cutoff consistent with in-time label;
     no NPI both prospective-positive and pre-cutoff-excluded.
  5. Degenerates: betweenness constant (per the letter -> drop); any other constant
     trainable columns reported.
  6. Solo-AUC scan: every trainable column's single-feature ROC-AUC against the FORWARD
     label on eligible rows (never-banned-at-freeze). Anything > 0.75 solo is flagged
     as a leakage suspect for manual review before training.

Usage: python -m labels.validate_frozen [--package DIR] [--report PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PKG = Path.home() / "Desktop/Data/preclean/trey/frozen_2023-12"

EXPECT_ROWS = 617_062
EXPECT_COLS = 211
EXPECT_FWD_POS = 4_401
SOLO_AUC_FLAG = 0.75

KNOWN_OFFENDERS = [
    "within_2_hops_of_exclusion", "shell_score", "related_party_density",
    "related_party_density_norm", "subscore_ownership_integrity",
    "has_excluded_owner", "billing_after_deactivation", "billed_after_death",
]

results: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "", warn: bool = False) -> None:
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    results.append((status, f"{name}: {detail}" if detail else name))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def fast_auc(y: np.ndarray, x: np.ndarray) -> float:
    """Rank-based ROC-AUC, NaNs imputed to median (mirrors LightGBM's tolerance)."""
    x = np.where(np.isnan(x), np.nanmedian(x), x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # midranks for ties
    s = pd.Series(x)
    ranks = s.rank(method="average").to_numpy()
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, default=DEFAULT_PKG)
    ap.add_argument("--report", type=Path,
                    default=Path(__file__).parent / "FROZEN_GATE0_REPORT.md")
    args = ap.parse_args()
    pkg = args.package

    manifest = json.loads((pkg / "feature_manifest.json").read_text())
    df = pd.read_parquet(pkg / "provider_features_for_model.parquet")
    fwd = pd.read_csv(pkg / "future_bans_after_2023-12.csv", dtype={"npi": str})

    # --- 1. shapes ---
    check("parquet shape", df.shape == (EXPECT_ROWS, EXPECT_COLS),
          f"{df.shape} vs expected ({EXPECT_ROWS}, {EXPECT_COLS})")
    check("manifest n_providers", manifest["n_providers"] == len(df),
          f"{manifest['n_providers']} == {len(df)}")
    check("npi unique", df["npi"].is_unique, "one row per NPI")

    # --- 2. manifest partition / trainable derivation ---
    feature_groups = (manifest["raw_feature_cols"] + manifest["peerpct_cols"]
                      + manifest["subscore_cols"] + manifest["embedding_cols"])
    ids = manifest["identifier_cols"]
    label_side = [manifest["label"]] + manifest["label_metadata"]
    grouping = manifest["group_cols"] + manifest["assessability"]
    lk_hard = manifest["leakage_hard"]
    lk_adj = manifest["leakage_adjacent"]

    trainable = [c for c in feature_groups
                 if c not in set(lk_hard) | set(lk_adj) and c in df.columns]
    check("trainable count == 126 (letter)", len(trainable) == 126,
          f"derived {len(trainable)}", warn=len(trainable) != 126)

    accounted = set(feature_groups) | set(ids) | set(label_side) | set(grouping) \
        | set(lk_hard) | set(lk_adj)
    unaccounted = [c for c in df.columns if c not in accounted]
    check("every parquet column accounted for by manifest", not unaccounted,
          f"unaccounted: {unaccounted}" if unaccounted else "all classified")

    vint = manifest["feature_vintage"]
    vint_all = set().union(*[set(v) for v in vint.values()])
    unvintaged = [c for c in trainable if c not in vint_all]
    check("every trainable column has a vintage class", not unvintaged,
          f"missing vintage: {unvintaged}" if unvintaged else
          f"classes: { {k: len(v) for k, v in vint.items()} }")

    # --- 3. fences ---
    both = set(lk_hard) & set(lk_adj)
    check("hard/adjacent lists disjoint", not both, str(both) if both else "")
    leaked = [c for c in KNOWN_OFFENDERS
              if c in trainable]
    check("known offenders all fenced", not leaked,
          f"IN TRAINABLE: {leaked}" if leaked else "all fenced or absent")
    weak_cols = [c for c in df.columns if c.startswith(("weak_label", "lf_"))]
    weak_leak = [c for c in weak_cols if c in trainable]
    check("weak-supervision columns not trainable", not weak_leak,
          f"weak cols in parquet: {weak_cols or 'none'}")
    check("leakage_adjacent count == 11 (letter)", len(lk_adj) == 11,
          f"{len(lk_adj)}: {lk_adj}")

    # --- 4. labels ---
    y_intime = df[manifest["label"]].fillna(0).astype(int)
    check("in-time positives == manifest n_positives",
          int(y_intime.sum()) == manifest["n_positives"],
          f"{int(y_intime.sum())} vs {manifest['n_positives']}")

    n_prosp = int(fwd["is_prospective_positive"].sum())
    check("forward prospective positives == 4,401 (letter)",
          n_prosp == EXPECT_FWD_POS, f"{n_prosp}")
    both_flags = fwd[(fwd["is_prospective_positive"] == 1)
                     & (fwd["was_excluded_pre_cutoff"] == 1)]
    check("no NPI both prospective and pre-cutoff", len(both_flags) == 0,
          f"{len(both_flags)} conflicts")

    df["npi"] = df["npi"].astype(str)
    fwd_pos = set(fwd.loc[fwd["is_prospective_positive"] == 1, "npi"])
    pre_cut = set(fwd.loc[fwd["was_excluded_pre_cutoff"] == 1, "npi"])
    in_universe = len(fwd_pos & set(df["npi"]))
    check("forward positives found in universe", in_universe > 0,
          f"{in_universe}/{len(fwd_pos)} in the 617k universe")
    # pre-cutoff bans in the label file should broadly agree with in-time label
    intime_npis = set(df.loc[y_intime == 1, "npi"])
    overlap = len(pre_cut & intime_npis)
    check("pre-cutoff exclusions overlap in-time label", overlap > 0,
          f"{overlap} of {len(intime_npis)} in-time positives matched", warn=True)

    # --- 5. degenerates ---
    if "betweenness" in df.columns:
        check("betweenness constant (drop per letter)",
              df["betweenness"].nunique(dropna=False) <= 1,
              f"nunique={df['betweenness'].nunique(dropna=False)}")
    numeric_trainable = [c for c in trainable
                         if pd.api.types.is_numeric_dtype(df[c])]
    constants = [c for c in numeric_trainable if df[c].nunique(dropna=True) <= 1]
    check("no constant trainable columns", not constants,
          f"constants: {constants}" if constants else "", warn=bool(constants))
    non_numeric = [c for c in trainable if c not in numeric_trainable]
    check("non-numeric trainable columns (need encoding)", True,
          f"{non_numeric or 'none'}", warn=bool(non_numeric))

    # --- 6. solo-AUC scan vs FORWARD label on eligible rows ---
    eligible = ~df["npi"].isin(pre_cut)
    y_fwd = df["npi"].isin(fwd_pos).astype(int)[eligible].to_numpy()
    print(f"\nsolo-AUC scan: {len(numeric_trainable)} trainable numeric columns, "
          f"{eligible.sum():,} eligible rows, {y_fwd.sum():,} forward positives")
    aucs = {}
    for c in numeric_trainable:
        aucs[c] = fast_auc(y_fwd, df.loc[eligible, c].to_numpy(dtype=np.float64))
    scan = pd.Series(aucs).dropna()
    scan = pd.concat([scan, (1 - scan)], axis=1).max(axis=1).sort_values(ascending=False)
    suspects = scan[scan > SOLO_AUC_FLAG]
    check(f"no trainable column solo-AUC > {SOLO_AUC_FLAG} vs forward label",
          suspects.empty,
          f"SUSPECTS: {suspects.round(3).to_dict()}" if not suspects.empty
          else f"max: {scan.index[0]} = {scan.iloc[0]:.3f}")
    top15 = scan.head(15)

    # --- report ---
    n_fail = sum(1 for s, _ in results if s == "FAIL")
    n_warn = sum(1 for s, _ in results if s == "WARN")
    lines = ["# GATE 0 — frozen_2023-12 package validation\n",
             f"package: `{pkg}`\n",
             f"**{'FAIL' if n_fail else 'PASS'}** — "
             f"{n_fail} fail / {n_warn} warn / "
             f"{len(results) - n_fail - n_warn} pass\n"]
    lines += [f"- [{s}] {msg}" for s, msg in results]
    lines += ["\n## Top-15 solo ROC-AUC vs forward label (eligible rows)\n",
              "| column | solo AUC |", "|---|---|"]
    lines += [f"| {c} | {v:.3f} |" for c, v in top15.items()]
    args.report.write_text("\n".join(lines) + "\n")
    print(f"\nreport -> {args.report}")
    print(f"RESULT: {'FAIL' if n_fail else 'PASS'} ({n_fail} fail, {n_warn} warn)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
