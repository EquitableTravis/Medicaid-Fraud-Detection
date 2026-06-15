"""
features.py (gnn) — Phase 4: node feature matrix & encoders.

Produces the model-ready tensors aligned to the canonical row order:
  X_num  — standardized numeric features, float32 [N, F_num]
  X_cat  — integer category codes,        int64   [N, 3]  (entity/taxonomy/state)

Numeric = the 39 numeric billing features (the 42 minus the 3 categoricals) PLUS
light, label-free structural/identity features computed from the graph:
  provider_age (NPPES enumeration date), per-edge-type degree (5), total degree,
  component size. All log1p'd where skewed.

Leakage discipline: the **scaler (median impute + mean/std) is fit on the TRAIN
mask only**, then applied to all nodes — so val/test statistics never leak. Edges
carry no labels, so graph-derived features are split-safe. Categoricals are encoded
to integer codes (0 = unknown/NaN) with vocab built over all nodes; vocab sizes are
recorded to size the in-model embeddings.

Output (~/Desktop/Data/Model/gnn/): X_num.npy, X_cat.npy, features_meta.json.

Run (after build_nodes, build_edges, splits):
    python -m src.gnn.features
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..model import config as mcfg
from . import config

EDGE_TYPES = ["owner", "ao", "pac", "fax", "addr"]
CLIP = 5.0


def log(m=""):
    print(m, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default=str(config.GNN_DATA_DIR))
    args = p.parse_args()
    out = config.Path(args.out) if isinstance(args.out, str) else args.out

    log("[1/5] Loading nodes, splits, node-attrs")
    nodes = pd.read_parquet(config.NODES_PARQUET)
    N = len(nodes)
    split = pd.read_parquet(config.GNN_DATA_DIR / "split_masks.parquet",
                            columns=["row_index", "split"])
    train_mask = (split.sort_values("row_index")["split"].to_numpy() == "train")
    attrs = pd.read_parquet(config.GNN_DATA_DIR / "nppes_node_attrs.parquet",
                            columns=["provider_age_years"])
    require_align = len(split) == N == len(attrs)
    assert require_align, "artifact row-count mismatch"
    log(f"  {N:,} nodes | train mask {int(train_mask.sum()):,}")

    # numeric vs categorical feature columns
    non_feat = {"row_index", "npi", "label_status", "not_scored"}
    feat_cols = [c for c in nodes.columns if c not in non_feat]
    cat_cols = [c for c in mcfg.CATEGORICAL_FEATURES if c in feat_cols]   # 3
    num_cols = [c for c in feat_cols if c not in cat_cols]                # 39
    log(f"  features: {len(num_cols)} numeric + {len(cat_cols)} categorical")

    log("[2/5] Graph-derived structural features (label-free)")
    extra = {}
    age = attrs["provider_age_years"].to_numpy(dtype=np.float64)
    extra["provider_age"] = np.where(age < 0, 0.0, age)                   # 4 tiny negatives → 0
    deg_total = np.zeros(N, dtype=np.float64)
    for t in EDGE_TYPES:
        ei = np.load(config.GNN_DATA_DIR / f"edges_{t}.npy")
        d = np.bincount(ei[0], minlength=N).astype(np.float64) if ei.shape[1] else np.zeros(N)
        extra[f"deg_{t}"] = np.log1p(d)
        deg_total += d
    extra["deg_total"] = np.log1p(deg_total)
    allei = np.load(config.GNN_DATA_DIR / "edges_all.npy")
    g = coo_matrix((np.ones(allei.shape[1], np.int8), (allei[0], allei[1])), shape=(N, N))
    _, lbl = connected_components(g, directed=False)
    extra["component_size"] = np.log1p(np.bincount(lbl)[lbl].astype(np.float64))
    log(f"  added {len(extra)} structural features: {list(extra)}")

    log("[3/5] Numeric matrix → median-impute + standardize (fit on TRAIN only)")
    Xn = nodes[num_cols].to_numpy(dtype=np.float64)
    Xn = np.column_stack([Xn] + [extra[k] for k in extra])
    all_num_names = num_cols + list(extra)
    med = np.nanmedian(np.where(train_mask[:, None], Xn, np.nan), axis=0)   # train medians
    med = np.where(np.isnan(med), 0.0, med)
    inds = np.where(np.isnan(Xn))
    Xn[inds] = np.take(med, inds[1])                                         # impute
    mean = Xn[train_mask].mean(axis=0)
    std = Xn[train_mask].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    Xn = np.clip((Xn - mean) / std, -CLIP, CLIP).astype(np.float32)
    assert not np.isnan(Xn).any() and not np.isinf(Xn).any(), "NaN/inf in X_num"
    log(f"  X_num {Xn.shape} | train mean≈0 ({np.abs(Xn[train_mask].mean()):.3f}) "
        f"std≈1 ({Xn[train_mask].std():.2f}) | no NaN/inf ✓")

    log("[4/5] Categorical → integer codes (0=unknown) + vocab")
    Xc = np.zeros((N, len(cat_cols)), dtype=np.int64)
    vocab_sizes = {}
    for j, c in enumerate(cat_cols):
        vals = nodes[c].astype(str).fillna("nan")
        uniq = [u for u in sorted(vals.unique()) if u not in ("nan", "")]
        code = {u: i + 1 for i, u in enumerate(uniq)}                       # 0 reserved
        Xc[:, j] = vals.map(code).fillna(0).astype(np.int64).to_numpy()
        vocab_sizes[c] = len(uniq) + 1
    log(f"  X_cat {Xc.shape} | vocab sizes {vocab_sizes}")

    log("[5/5] Writing arrays + meta")
    np.save(out / "X_num.npy", Xn)
    np.save(out / "X_cat.npy", Xc)
    (out / "features_meta.json").write_text(json.dumps({
        "n_nodes": N, "numeric_features": all_num_names, "n_numeric": Xn.shape[1],
        "categorical_features": cat_cols, "vocab_sizes": vocab_sizes,
        "clip": CLIP, "scaler_fit": "train_mask_only",
        "train_median": med.tolist(), "train_mean": mean.tolist(), "train_std": std.tolist(),
    }, indent=2))

    log("\n" + "=" * 60 + "\nGATE 4 — feature matrix\n" + "=" * 60)
    log(f"  X_num: {Xn.shape} float32 (no NaN/inf, scaler train-only, clipped ±{CLIP})")
    log(f"  X_cat: {Xc.shape} int64 | vocab {vocab_sizes} → embedding dims to size in Phase 6")
    log(f"  numeric = {len(num_cols)} billing + {len(extra)} structural = {Xn.shape[1]}")
    log(f"  written → {out}")


if __name__ == "__main__":
    main()
