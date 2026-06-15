"""
check_edges.py (gnn) — deep GATE-3 verification beyond the build-time report.

Read-only audit of the edge artifacts: array integrity (shape/dtype/index range,
self-loops, duplicates, undirected symmetry, unioned == concat), per-edge-type
connectivity contribution, the largest surviving groups (spot-check they're real),
the provider-age feature distribution, and — most important — whether the 578 LEIE
positives are actually CONNECTED (a GNN can only help nodes that have neighbors).

Run:
    python -m src.gnn.check_edges
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..model import config as mcfg
from . import config

TYPES = ["owner", "ao", "pac", "fax", "addr"]


def log(m=""):
    print(m, flush=True)


def largest_component(ei, N):
    if ei.shape[1] == 0:
        return 0, N
    g = coo_matrix((np.ones(ei.shape[1], np.int8), (ei[0], ei[1])), shape=(N, N))
    ncomp, lbl = connected_components(g, directed=False)
    sizes = np.bincount(lbl)
    return int(sizes.max()), ncomp


def main():
    d = config.GNN_DATA_DIR
    nodes = pd.read_parquet(config.NODES_PARQUET, columns=["row_index", "npi", "label_status"])
    N = len(nodes)
    log(f"nodes: {N:,}")

    # ---- 1. per-type array integrity ----
    log("\n[1] array integrity per edge type")
    per = {t: np.load(d / f"edges_{t}.npy") for t in TYPES}
    total_dir = 0
    for t, ei in per.items():
        assert ei.dtype == np.int64 and ei.ndim == 2 and ei.shape[0] == 2, f"{t} bad shape/dtype"
        in_range = (ei.min() >= 0 and ei.max() < N) if ei.shape[1] else True
        self_loops = int((ei[0] == ei[1]).sum())
        # duplicate undirected pairs within type
        pairs = np.sort(ei, axis=0)
        uniq = np.unique(pairs, axis=1).shape[1]
        dup_frac = 1 - uniq / ei.shape[1] if ei.shape[1] else 0
        # symmetry: every (s,d) has (d,s)
        fwd = set(map(tuple, ei.T[:min(200000, ei.shape[1])]))
        sym = all((b, a) in fwd for a, b in list(fwd)[:5000])
        total_dir += ei.shape[1]
        log(f"  {t:6s}: {ei.shape[1]:>10,} dir-edges ({ei.shape[1]//2:,} undirected) | "
            f"in_range={in_range} self_loops={self_loops} dup_frac={dup_frac:.3f} sym={sym}")

    # ---- 2. unioned == concat ----
    log("\n[2] unioned consistency")
    allei = np.load(d / "edges_all.npy")
    tag = np.load(d / "edge_type_all.npy")
    log(f"  edges_all {allei.shape[1]:,} dir-edges | sum per-type {total_dir:,} | "
        f"match={allei.shape[1] == total_dir}")
    log(f"  edge_type_all len {len(tag):,} == edges_all cols {allei.shape[1]:,}: "
        f"{len(tag) == allei.shape[1]} | type codes present {sorted(np.unique(tag).tolist())}")

    # ---- 3. per-type connectivity contribution ----
    log("\n[3] connectivity contribution per type (does one type make the blob?)")
    for t, ei in per.items():
        big, ncomp = largest_component(ei, N)
        log(f"  {t:6s}: largest component {big:,} ({big/N:.1%}) | components {ncomp:,}")
    big_all, ncomp_all = largest_component(allei, N)
    log(f"  ALL   : largest component {big_all:,} ({big_all/N:.1%}) | components {ncomp_all:,}")

    # ---- 4. label connectivity (the critical check) ----
    log("\n[4] are the labels connected? (GNN can only help nodes with neighbors)")
    deg = np.bincount(allei[0], minlength=N)
    pos = nodes["label_status"].to_numpy() == "positive"
    neg = nodes["label_status"].to_numpy() == "reliable_neg"
    unl = nodes["label_status"].to_numpy() == "unlabeled"
    for name, m in [("positive", pos), ("reliable_neg", neg), ("unlabeled", unl)]:
        dgm = deg[m]
        iso = int((dgm == 0).sum())
        log(f"  {name:13s}: n={int(m.sum()):>7,} | connected {int((dgm>0).sum()):>7,} "
            f"({(dgm>0).mean():.1%}) | isolated {iso:,} | median deg {int(np.median(dgm))} "
            f"| max deg {int(dgm.max())}")
    # positive↔positive adjacency (label clustering → Phase-5 split must separate)
    pos_idx = set(np.where(pos)[0].tolist())
    pp = sum(1 for a, b in allei.T[::2] if a in pos_idx and b in pos_idx)  # count over one direction
    log(f"  positive↔positive direct edges: {pp:,} (label clustering → structural split in Phase 5)")

    # ---- 5. largest surviving groups (spot-check they're real) ----
    log("\n[5] largest surviving groups per type — real shared entities?")
    names = pd.read_parquet(mcfg.SCORED_UNIVERSE_PARQUET, columns=["npi", "org_legal_name"])
    name_of = dict(zip(names["npi"], names["org_legal_name"].fillna("")))
    for t in TYPES:
        ei = per[t]
        if ei.shape[1] == 0:
            continue
        top_node = int(np.argmax(np.bincount(ei[0], minlength=N)))
        nbrs = ei[1][ei[0] == top_node]
        sample_npis = nodes.iloc[np.r_[top_node, nbrs[:3]]]["npi"].tolist()
        orgs = [name_of.get(x, "")[:32] for x in sample_npis]
        log(f"  {t:6s}: top-degree node deg={len(nbrs)} | sample orgs: {orgs}")

    # ---- 6. provider-age feature ----
    log("\n[6] provider-age feature")
    attrs = pd.read_parquet(d / "nppes_node_attrs.parquet")
    age = attrs["provider_age_years"]
    log(f"  non-null {age.notna().sum():,}/{len(age):,} | min {age.min():.1f} | "
        f"median {age.median():.1f} | max {age.max():.1f} | negative-age rows "
        f"{int((age < 0).sum()):,}")
    require_ok = (age.dropna() >= 0).all() or int((age < 0).sum()) < 50
    log(f"  row order aligned to nodes: {bool((attrs['row_index'].to_numpy()==np.arange(N)).all())}")

    log("\nVERDICT: review the four critical lines — (a) no self-loops / in-range, "
        "(b) no single type = the whole blob, (c) positives mostly CONNECTED, "
        "(d) age feature sane.")


if __name__ == "__main__":
    main()
