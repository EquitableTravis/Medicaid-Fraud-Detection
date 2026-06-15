"""
config.py (graph_features) — paths for the graph-features probe.

This initiative ADDS graph-structure feature columns to the existing NPI-grain
LightGBM (the production model is NPI-grain → rollup; there is no separate company
model). It clones nothing destructively: the original `src/model/` is read-only
here. We reuse the already-built identifier graph (the GNN's owner/PAC/address/AO/
fax edges) and the temporal split — no need to rebuild them.

Artifacts go to ~/Desktop/Data/Model/graph_features/ (never overwriting the
existing model's artifacts).
"""

from pathlib import Path

from src.model import config as mcfg   # read-only reuse of label/feature/path constants (absolute; graph_features is top-level)

# reuse the GNN artifacts (identifier graph + temporal split, already verified)
GNN_DIR = mcfg.MODEL_DATA_DIR / "gnn"
NODES_PARQUET = GNN_DIR / "nodes.parquet"            # row_index, npi, label_status, 42 feats
SPLIT_MASKS = GNN_DIR / "split_masks.parquet"        # npi, split, excl_year (temporal)
EDGES_ALL = GNN_DIR / "edges_all.npy"                # [2,E] unioned identifier edges
EDGE_TYPES = ["owner", "pac", "addr", "ao", "fax"]   # per-type degree
SCORED_UNIVERSE = mcfg.SCORED_UNIVERSE_PARQUET       # the model's NPI feature table

# outputs (all new)
OUT_DIR = mcfg.MODEL_DATA_DIR / "graph_features"
NPI_GRAPH_FEATURES = OUT_DIR / "npi_graph_features.parquet"

TRAIN_CUTOFF_YEAR = 2023        # excluded-proximity features use ONLY LEIE known ≤ this (temporal)
MAX_BFS_HOP = 4                 # cap dist_to_excluded; unreachable → MAX_BFS_HOP + 1 sentinel

# feature groups for the leakage ablation (the honest test of the structure thesis)
STRUCTURE_FEATS = ["comp_size", "deg_total", "deg_owner", "deg_pac", "deg_addr",
                   "deg_ao", "deg_fax"]
PROXIMITY_FEATS = ["comp_excluded_ratio", "dist_to_excluded", "n_excluded_neighbors"]
