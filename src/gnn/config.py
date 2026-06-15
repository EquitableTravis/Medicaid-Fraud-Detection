"""
config.py (gnn) — paths, device, and label definitions for the GNN training track.

Reuses the tabular model's config (feature/leakage constants, input parquet
paths) so the two tracks can never disagree on what a feature is. GNN artifacts
live under ~/Desktop/Data/Model/gnn/ (outside the repo, HIPAA). The graph is
built in-memory (no Neo4j on the training path) — these are just file locations
and the label taxonomy.
"""

from pathlib import Path

from ..model import config as mcfg  # reuse the tabular model's constants

# ---- paths ----
GNN_DATA_DIR = mcfg.MODEL_DATA_DIR / "gnn"          # ~/Desktop/Data/Model/gnn
PRECLEAN_DIR = mcfg.DATA_DIR / "preclean"           # NPPES/PECOS/Caught/owners
LEIE_CSV = PRECLEAN_DIR / "Caught.csv"              # exclusion dates (EXCLDATE) for temporal split
NODES_PARQUET = GNN_DATA_DIR / "nodes.parquet"

# Re-export the tabular model's input parquets (resolved via mcfg.find_input)
SCORED_UNIVERSE_PARQUET = mcfg.SCORED_UNIVERSE_PARQUET   # 617,062 x 57 (the node universe)
PU_TRAINING_PARQUET = mcfg.PU_TRAINING_PARQUET           # 308,038 x 57 (reconciliation only)
NPI_TO_COMPANY_MAP = mcfg.NPI_TO_COMPANY_MAP             # company_id = structural-split group key

# ---- label taxonomy (PU setup; derived directly from the scored universe) ----
# positive      : provider_on_leie
# reliable_neg  : ~provider_on_leie & ~not_scored & anomaly_score == 0  (confident-clean)
# unlabeled     : everything else (unknown, NOT negative) — message passing only, no gradient
LABEL = mcfg.LABEL                                  # "provider_on_leie"
# Columns needed to derive label_status but NEVER fed as features (detector-score cols).
LABEL_SOURCE_COLS = ["provider_on_leie", "not_scored", "anomaly_score"]

SEED = 42


def get_device():
    """Apple Silicon MPS if available, else CPU (per the plan)."""
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"
