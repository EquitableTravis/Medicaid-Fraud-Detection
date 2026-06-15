"""
build_nodes.py (gnn) — Phase 1: the canonical node table.

Every scored NPI (~617k) becomes a node. Builds ONE nodes.parquet with a stable
row order (reset_index) — every later array (edge_index, X, masks) aligns to it.

Label taxonomy (PU setup), derived DIRECTLY from the scored universe (not the PU
parquet — the PU file is only used to RECONCILE the join):
  positive     : provider_on_leie
  reliable_neg : ~provider_on_leie & ~not_scored & anomaly_score == 0
  unlabeled    : everything else (unknown, not negative)

Features are the 42 leakage-free columns from the tabular model's
build_feature_matrix() — which already drops the label, the excluded-owner
leakage family, identifiers, and the detector-score columns (incl. anomaly_score,
per the locked decision). The label-source columns (provider_on_leie / not_scored
/ anomaly_score) are read separately to assign label_status and are NEVER features.

Output: ~/Desktop/Data/Model/gnn/nodes.parquet with columns
  row_index, npi, <42 features...>, label_status, not_scored
plus a printed GATE-1 inventory.

Run:
    python -m src.gnn.build_nodes
"""

import argparse

import numpy as np
import pandas as pd

from ..model import config as mcfg
from ..model.data import build_feature_matrix, excluded_columns, log, require
from . import config


def assign_label_status(df: pd.DataFrame) -> pd.Series:
    """positive / reliable_neg / unlabeled from the three label-source columns."""
    leie = df["provider_on_leie"].fillna(False).astype(bool)
    not_scored = df["not_scored"].fillna(True).astype(bool)
    anomaly = pd.to_numeric(df["anomaly_score"], errors="coerce")
    reliable_neg = (~leie) & (~not_scored) & (anomaly == 0)
    return pd.Series(
        np.select([leie, reliable_neg], ["positive", "reliable_neg"],
                  default="unlabeled"),
        index=df.index, name="label_status")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default=str(config.NODES_PARQUET))
    args = p.parse_args()
    out = config.Path(args.out) if isinstance(args.out, str) else args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    log("[1/4] Loading scored universe (the node universe)")
    df = pd.read_parquet(config.SCORED_UNIVERSE_PARQUET)
    require("universe rows == expected", len(df) == mcfg.EXPECTED_UNIVERSE_ROWS,
            f"{len(df):,} vs {mcfg.EXPECTED_UNIVERSE_ROWS:,}")
    require("npi unique", df["npi"].is_unique)
    for c in config.LABEL_SOURCE_COLS:
        require(f"label-source column present: {c}", c in df.columns)

    log("[2/4] Assigning label_status + building the 42-feature matrix")
    label_status = assign_label_status(df)
    X = build_feature_matrix(df)                       # 42 leakage-free features
    feats = list(X.columns)
    # Hard leakage assertion: no blocklist / detector / identifier / anomaly col leaked in.
    bad = set(feats) & excluded_columns(df.columns)
    require("no leakage/identifier/detector column in features", not bad, str(bad))
    require("anomaly_score excluded from features", "anomaly_score" not in feats)
    require("feature count == 42", len(feats) == 42, f"{len(feats)} features")

    log("[3/4] Reconciling universe label set against the PU parquet")
    pu_npis = set(pd.read_parquet(config.PU_TRAINING_PARQUET, columns=["npi"])["npi"])
    labeled_mask = label_status.isin(["positive", "reliable_neg"])
    labeled_npis = set(df.loc[labeled_mask, "npi"])
    n_pos = int((label_status == "positive").sum())
    n_neg = int((label_status == "reliable_neg").sum())
    n_unl = int((label_status == "unlabeled").sum())
    require("positives ~= 578", abs(n_pos - mcfg.EXPECTED_PU_POSITIVES) <= 1,
            f"{n_pos} vs {mcfg.EXPECTED_PU_POSITIVES}")
    require("universe {positive ∪ reliable_neg} == PU npi set",
            labeled_npis == pu_npis,
            f"universe-labeled {len(labeled_npis):,} vs PU {len(pu_npis):,}; "
            f"sym-diff {len(labeled_npis ^ pu_npis):,}")

    log("[4/4] Writing nodes.parquet (stable row order)")
    nodes = X.copy()
    nodes.insert(0, "npi", df["npi"].values)
    nodes["label_status"] = label_status.values
    nodes["not_scored"] = df["not_scored"].fillna(True).astype(bool).values
    nodes = nodes.reset_index(drop=True)
    nodes.insert(0, "row_index", np.arange(len(nodes), dtype=np.int64))
    nodes.to_parquet(out, index=False)

    # ---- GATE 1 inventory ----
    log("\n" + "=" * 60)
    log("GATE 1 — node table inventory")
    log("=" * 60)
    log(f"  nodes: {len(nodes):,}  (one row per scored NPI)")
    log(f"  features: {len(feats)} leakage-free columns")
    log(f"    categorical: {[c for c in mcfg.CATEGORICAL_FEATURES if c in feats]}")
    log(f"  label_status:  positive={n_pos:,}  reliable_neg={n_neg:,}  unlabeled={n_unl:,}")
    log(f"    positive rate among labeled: {n_pos / (n_pos + n_neg):.4%}")
    log(f"  universe↔PU join: {{positive ∪ reliable_neg}} == PU parquet npi set "
        f"({len(labeled_npis):,} == {len(pu_npis):,}) ✓")
    log(f"  PU total {len(pu_npis):,} = positives {n_pos:,} + reliable_neg {n_neg:,} "
        f"= {n_pos + n_neg:,} ✓")
    log(f"  not_scored carried for the Phase-10 reliability gate: "
        f"{int(nodes['not_scored'].sum()):,} unreliable")
    log(f"  written → {out}")


if __name__ == "__main__":
    main()
