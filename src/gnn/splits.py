"""
splits.py (gnn) — Phase 5: temporal + structural splits, leakage defense.

Two requirements, intersected at the GROUP level:
  Temporal  — train on LEIE exclusions up to year T, validate/test on later years.
              Answers the deployment question: do we surface providers BEFORE
              they're caught? Exclusion years come from Caught.csv (EXCLDATE).
  Structural— an entire owner-group falls on ONE side, never straddling, so a
              train positive and a val positive can't sit 1–2 hops apart and leak
              the label through an edge. Group key = `company_id` from the rollup
              (pac>owner>name) — the owner-group that built the company rollup.

Group assignment:
  - group with ≥1 positive → bucket by its LATEST positive exclusion year
    (conservative: a group with any recent exclusion goes to val/test, never train).
  - group with no positive (reliable_neg / unlabeled only) → deterministic hash of
    company_id into train/val/test at the target proportions (structural random).
Cutoff years are chosen data-drivenly from the positive-year distribution (printed,
so GATE 5's "cutoff justified" is self-documenting).

Loss/metrics use LABELED nodes only (positives ∪ reliable_neg) within each split;
unlabeled nodes carry no gradient but stay in the graph for message passing.

Output: ~/Desktop/Data/Model/gnn/split_masks.parquet
  (row_index, npi, company_id, label_status, excl_year, split ∈ {train,val,test})

Run (after build_nodes; needs npi_to_company_map + Caught.csv):
    python -m src.gnn.splits
"""

import argparse
import hashlib

import numpy as np
import pandas as pd

from ..model import config as mcfg
from . import config

TRAIN_FRAC, VAL_FRAC = 0.70, 0.85          # negative-group hash cut points
POS_TRAIN_CUM, POS_VAL_CUM = 0.45, 0.68    # cumulative-positive targets → year cutoffs
                                           # (percentiles collapse on the 2022-26-skewed years)


def log(m=""):
    print(m, flush=True)


def ghash(s: str) -> float:
    """Deterministic [0,1) hash of a group id (negative-group structural split)."""
    h = hashlib.md5(str(s).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def load_excl_years(npis_pos):
    """npi -> earliest exclusion YEAR for our positive NPIs, from Caught.csv."""
    leie = pd.read_csv(config.LEIE_CSV, dtype=str, keep_default_na=False)
    leie = leie[leie["NPI"].isin(npis_pos)].copy()
    yr = pd.to_datetime(leie["EXCLDATE"], format="%Y%m%d", errors="coerce").dt.year
    leie = leie.assign(yr=yr).dropna(subset=["yr"])
    return leie.groupby("NPI")["yr"].min().astype(int).to_dict()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default=str(config.GNN_DATA_DIR / "split_masks.parquet"))
    args = p.parse_args()

    log("[1/5] Loading nodes + company-group key")
    nodes = pd.read_parquet(config.NODES_PARQUET, columns=["row_index", "npi", "label_status"])
    N = len(nodes)
    nmap = pd.read_parquet(mcfg.NPI_TO_COMPANY_MAP, columns=["npi", "company_id"])
    nodes = nodes.merge(nmap, on="npi", how="left", validate="1:1")
    nodes["company_id"] = nodes["company_id"].fillna("npi:" + nodes["npi"])  # singleton fallback
    log(f"  {N:,} nodes | {nodes['company_id'].nunique():,} owner-groups")

    log("[2/5] Joining LEIE exclusion years (temporal axis)")
    pos = nodes["label_status"] == "positive"
    excl = load_excl_years(set(nodes.loc[pos, "npi"]))
    nodes["excl_year"] = nodes["npi"].map(excl)
    pos_years = nodes.loc[pos, "excl_year"].dropna()
    log(f"  positives with a parsed exclusion year: {len(pos_years)}/{int(pos.sum())}")
    hist = pos_years.astype(int).value_counts().sort_index()
    log("  positive exclusion-year histogram: " +
        " ".join(f"{y}:{c}" for y, c in hist.items()))
    cum = hist.cumsum() / hist.sum()
    years = cum.index.tolist()
    t_val = next(y for y in years if cum[y] >= POS_TRAIN_CUM)             # train ≤ t_val
    t_test = next((y for y in years if cum[y] >= POS_VAL_CUM and y > t_val), t_val + 1)
    log(f"  chosen cutoffs: train ≤ {t_val}, val ({t_val}, {t_test}], test > {t_test} "
        f"(cum-positive {POS_TRAIN_CUM:.0%}/{POS_VAL_CUM:.0%}; ensures positives in every split)")

    log("[3/5] Assigning each owner-group to one split")
    # group temporal year = latest positive exclusion year in the group (NaN if none)
    grp_year = (nodes.dropna(subset=["excl_year"]).groupby("company_id")["excl_year"]
                .max().astype(int).to_dict())

    def assign(cid):
        y = grp_year.get(cid)
        if y is not None:                      # group has ≥1 positive → temporal
            return "train" if y <= t_val else ("val" if y <= t_test else "test")
        u = ghash(cid)                         # negative/unlabeled group → structural hash
        return "train" if u < TRAIN_FRAC else ("val" if u < VAL_FRAC else "test")

    grp_split = {cid: assign(cid) for cid in nodes["company_id"].unique()}
    nodes["split"] = nodes["company_id"].map(grp_split)

    log("[4/5] Leakage diagnostics")
    # (a) assert no group straddles
    straddle = nodes.groupby("company_id")["split"].nunique()
    require_no_straddle = int((straddle > 1).sum())
    log(f"  owner-groups straddling >1 split: {require_no_straddle} (must be 0)")
    assert require_no_straddle == 0, "a group straddles splits — bug"
    # (b) cross-split edges. Total crossings are mostly harmless (negative/unlabeled
    #     endpoints carry no label). What can actually leak a label is a LABELED↔LABELED
    #     cross-split edge — above all a positive↔positive one. Report those.
    split_of = nodes["split"].to_numpy()
    labeled = nodes["label_status"].isin(["positive", "reliable_neg"]).to_numpy()
    ispos = (nodes["label_status"] == "positive").to_numpy()
    ei = np.load(config.GNN_DATA_DIR / "edges_all.npy")
    cross = split_of[ei[0]] != split_of[ei[1]]
    lab_lab = labeled[ei[0]] & labeled[ei[1]]
    pp = ispos[ei[0]] & ispos[ei[1]]
    log(f"  cross-split edges (all): {int(cross.sum()//2):,}/{ei.shape[1]//2:,} "
        f"({cross.mean():.1%}) — mostly negative/unlabeled, harmless")
    log(f"  cross-split LABELED↔LABELED: {int((cross & lab_lab).sum()//2):,} "
        f"| positive↔positive: {int((cross & pp).sum()//2):,}  "
        f"(of {int(pp.sum()//2)} total pos↔pos) — the real leakage exposure; Phase-8 ablation tests it")

    log("[5/5] Writing split_masks.parquet")
    nodes[["row_index", "npi", "company_id", "label_status", "excl_year", "split"]].to_parquet(
        args.out, index=False)

    # ---- GATE 5 report ----
    log("\n" + "=" * 60 + "\nGATE 5 — split inventory\n" + "=" * 60)
    lab = nodes[nodes["label_status"].isin(["positive", "reliable_neg"])]
    for s in ["train", "val", "test"]:
        sl = lab[lab["split"] == s]
        npos = int((sl["label_status"] == "positive").sum())
        nneg = int((sl["label_status"] == "reliable_neg").sum())
        log(f"  {s:5s}: labeled {len(sl):>7,}  (positives {npos:>4}  reliable_neg {nneg:>7,})")
    log(f"  unlabeled (graph only, no gradient): "
        f"{int((nodes['label_status']=='unlabeled').sum()):,}")
    log(f"  temporal cutoffs justified above; no owner-group straddles ✓; "
        f"cross-split edges {cross.mean():.2%}")
    log(f"  written → {args.out}")


if __name__ == "__main__":
    main()
