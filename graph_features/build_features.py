"""
build_features.py (graph_features) — Phase 1-2: per-NPI graph features.

Reuses the already-built identifier graph (GNN owner/PAC/address/AO/fax edges) and
the temporal split. Computes NPI-level structural features that the existing
LightGBM can split on:

  STRUCTURE (label-free): comp_size, deg_total, deg_<type>×5
  PROXIMITY (excluded-relative, TEMPORALLY DISCIPLINED): comp_excluded_ratio,
            dist_to_excluded, n_excluded_neighbors — computed using ONLY LEIE
            exclusions known as of the training cutoff (≤2023), matching how the
            label is defined. Using future exclusions here would be a label echo.

The structure/proximity split is the Phase-5 leakage ablation: structure-only is
the honest test of "does graph structure add signal"; proximity is the (also
legitimate, if temporally clean) "near a known-bad provider" signal.

Isolated NPIs get explicit 0 / sentinel values (no NaNs). Output:
~/Desktop/Data/Model/graph_features/npi_graph_features.parquet keyed by npi.

Run:
    python -m graph_features.build_features
"""

import argparse

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from graph_features import config


def log(m=""):
    print(m, flush=True)


def multi_source_bfs(adj_csr, sources, n, max_hop):
    """Distance (in hops) from every node to the nearest source; sources at 0.
    Unreached → max_hop+1 sentinel. Iterative frontier expansion (fast for a
    small source set over a sparse graph)."""
    dist = np.full(n, max_hop + 1, dtype=np.int16)
    frontier = np.array(sorted(sources), dtype=np.int64)
    dist[frontier] = 0
    indptr, indices = adj_csr.indptr, adj_csr.indices
    for h in range(1, max_hop + 1):
        if frontier.size == 0:
            break
        nbrs = np.concatenate([indices[indptr[u]:indptr[u + 1]] for u in frontier]) \
            if frontier.size else np.array([], dtype=np.int64)
        nbrs = np.unique(nbrs)
        new = nbrs[dist[nbrs] == max_hop + 1]
        dist[new] = h
        frontier = new
    return dist


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default=str(config.NPI_GRAPH_FEATURES))
    args = p.parse_args()
    out = config.Path(args.out) if isinstance(args.out, str) else args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    log("[1/5] Loading node order + temporal split (excl years)")
    nodes = pd.read_parquet(config.NODES_PARQUET, columns=["row_index", "npi", "label_status"]
                            ).sort_values("row_index").reset_index(drop=True)
    N = len(nodes)
    sp = pd.read_parquet(config.SPLIT_MASKS, columns=["npi", "excl_year"])
    yr = nodes.merge(sp, on="npi", how="left")["excl_year"].to_numpy()
    pos = (nodes["label_status"].to_numpy() == "positive")
    # excluded-as-of-cutoff: positives excluded on/before the train cutoff year ONLY
    excl_cut = pos & (np.nan_to_num(yr, nan=9999) <= config.TRAIN_CUTOFF_YEAR)
    log(f"  {N:,} nodes | positives {int(pos.sum())} | "
        f"excluded≤{config.TRAIN_CUTOFF_YEAR} (proximity source set) {int(excl_cut.sum())}")

    log("[2/5] Building adjacency (all identifier edge types) + components")
    allei = np.load(config.EDGES_ALL)
    A = csr_matrix((np.ones(allei.shape[1], np.int8), (allei[0], allei[1])), shape=(N, N))
    A.data[:] = 1
    _, comp = connected_components(A, directed=False)
    comp_size_arr = np.bincount(comp)
    comp_size = comp_size_arr[comp].astype(np.int32)

    log("[3/5] Degree (total + per edge type)")
    feats = {"comp_size": comp_size}
    deg_total = np.asarray(A.sum(axis=1)).ravel().astype(np.int32)
    feats["deg_total"] = deg_total
    for t in config.EDGE_TYPES:
        ei = np.load(config.GNN_DIR / f"edges_{t}.npy")
        feats[f"deg_{t}"] = (np.bincount(ei[0], minlength=N).astype(np.int32)
                             if ei.shape[1] else np.zeros(N, np.int32))

    log("[4/5] Proximity features (temporal: excluded≤cutoff only)")
    excl_vec = excl_cut.astype(np.float32)
    feats["n_excluded_neighbors"] = np.asarray(A.dot(excl_vec)).ravel().astype(np.int32)
    feats["dist_to_excluded"] = multi_source_bfs(A, np.where(excl_cut)[0], N, config.MAX_BFS_HOP)
    # fraction of each node's component that is excluded-as-of-cutoff
    comp_excl = np.bincount(comp, weights=excl_vec, minlength=len(comp_size_arr))
    feats["comp_excluded_ratio"] = (comp_excl[comp] / np.maximum(comp_size, 1)).astype(np.float32)

    log("[5/5] Writing")
    df = pd.DataFrame({"npi": nodes["npi"].values, **feats})
    assert not df.drop(columns=["npi"]).isna().any().any(), "NaN in graph features"
    df.to_parquet(out, index=False)

    log("\n" + "=" * 60 + "\nGATE 1-2 — graph features\n" + "=" * 60)
    log(f"  {N:,} NPIs × {len(feats)} features → {out.name}")
    log(f"  structure: {config.STRUCTURE_FEATS}")
    log(f"  proximity: {config.PROXIMITY_FEATS} (excluded source = LEIE≤{config.TRAIN_CUTOFF_YEAR})")
    log(f"  dist_to_excluded: reached≤{config.MAX_BFS_HOP}hops "
        f"{int((df['dist_to_excluded']<=config.MAX_BFS_HOP).sum()):,} | "
        f"sentinel(unreached) {int((df['dist_to_excluded']==config.MAX_BFS_HOP+1).sum()):,}")
    log(f"  nodes with an excluded neighbor: {int((df['n_excluded_neighbors']>0).sum()):,}")
    log(f"  no NaN ✓ | written → {out}")


if __name__ == "__main__":
    main()
