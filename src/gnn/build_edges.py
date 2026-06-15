"""
build_edges.py (gnn) — Phase 3: the edge tables (GATE 3, the hard part).

Builds NPI↔NPI undirected edges for 5 edge types, all keyed on the canonical
`row_index` from nodes.parquet:
  shared_owner  — owner files (ASSOCIATE ID - OWNER) bridged owner→NPI via PECOS
  shared_ao     — NPPES authorized-official name
  shared_pac    — PECOS PECOS_ASCT_CNTL_ID
  shared_fax    — NPPES practice fax
  shared_addr   — NPPES practice address (street|city|state|zip5)
(+ optional shared_phone / shared_mail with --with-phone / --with-mailing)

A single streamed NPPES pass yields the AO / fax / address keys AND the
enumeration date (→ provider age feature, written to nppes_node_attrs.parquet).
Normalizers (`norm_address`, `norm_phone`, `norm_name_base`, `org_key`, `_clean`)
are reused from src/graph_work so keying matches the validated Neo4j graph.

Mega-clique control (per type): a group of n NPIs sharing a key →
  n ≤ FULL_CAP        : full clique (all C(n,2) pairs)
  FULL_CAP < n ≤ HUB  : star-connect to one representative (n-1 edges); logged
  n > HUB             : dropped entirely (infrastructure hub); logged
Tighter caps on infrastructure-prone keys (fax/addr/phone/mail) than on operator
keys (owner/pac/ao). Every star/drop goes to edge_drop_log.csv.

Outputs (~/Desktop/Data/Model/gnn/): edges_<type>.npy [2,E] per type,
edges_all.npy + edge_type_all.npy (unioned + integer type tag),
nppes_node_attrs.parquet, edge_drop_log.csv. Prints the GATE-3 report.

Run (after build_nodes):
    python -m src.gnn.build_edges            # 5 core types
    python -m src.gnn.build_edges --with-phone --with-mailing
"""

import argparse
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..graph_work.etl.build_graph import (_clean, norm_address, norm_name_base,
                                          norm_phone, org_key)
from ..model import config as mcfg
from . import config

# Per-type clique caps: (FULL_CAP, HUB_CAP). Operator keys looser; infra keys tighter.
CAPS = {
    "owner": (30, 300), "ao": (30, 300), "pac": (30, 300),
    "fax": (20, 150), "addr": (20, 150), "phone": (15, 100), "mail": (15, 100),
}
TYPE_CODE = {"owner": 0, "ao": 1, "pac": 2, "fax": 3, "addr": 4, "phone": 5, "mail": 6}
REFERENCE_DATE = pd.Timestamp("2025-05-11")   # NPPES extract vintage; for provider-age feature
CHUNK = 200_000
READ = dict(dtype=str, on_bad_lines="skip", encoding="utf-8", encoding_errors="replace")

NPPES_COLS = {
    "NPI": "npi",
    "Provider Enumeration Date": "enum_date",
    "Authorized Official Last Name": "ao_last",
    "Authorized Official First Name": "ao_first",
    "Provider First Line Business Practice Location Address": "prac1",
    "Provider Second Line Business Practice Location Address": "prac2",
    "Provider Business Practice Location Address City Name": "prac_city",
    "Provider Business Practice Location Address State Name": "prac_state",
    "Provider Business Practice Location Address Postal Code": "prac_zip",
    "Provider Business Practice Location Address Fax Number": "prac_fax",
    "Provider Business Practice Location Address Telephone Number": "prac_phone",
    "Provider First Line Business Mailing Address": "mail1",
    "Provider Second Line Business Mailing Address": "mail2",
    "Provider Business Mailing Address City Name": "mail_city",
    "Provider Business Mailing Address State Name": "mail_state",
    "Provider Business Mailing Address Postal Code": "mail_zip",
}
OWNER_GLOB = config.PRECLEAN_DIR / "owners"


def log(m=""):
    print(m, flush=True)


def edges_from_groups(key_to_rows, etype, drop_log):
    """Emit undirected [2,E] edges with per-type clique caps; record stars/drops.
    Pairs are DEDUPED within the type (an NPI pair sharing several keys of the same
    type — e.g. multiple co-owners — must be ONE edge, not N, or message passing
    double-counts that neighbor)."""
    full_cap, hub_cap = CAPS[etype]
    pairs = set()                       # unique undirected (min,max) pairs
    n_full = n_star = n_drop = 0
    for key, rows in key_to_rows.items():
        rows = sorted(set(rows))
        n = len(rows)
        if n < 2:
            continue
        if n <= full_cap:
            for a, b in itertools.combinations(rows, 2):
                pairs.add((a, b))       # rows sorted → a < b
            n_full += 1
        elif n <= hub_cap:
            rep = rows[0]
            for r in rows[1:]:
                pairs.add((rep, r))     # rep is min → rep < r
            n_star += 1
            drop_log.append({"edge_type": etype, "key": str(key)[:80],
                             "group_size": n, "action": "star"})
        else:
            n_drop += 1
            drop_log.append({"edge_type": etype, "key": str(key)[:80],
                             "group_size": n, "action": "dropped_hub"})
    if not pairs:
        ei = np.empty((2, 0), dtype=np.int64)
    else:
        arr = np.array(sorted(pairs), dtype=np.int64).T            # [2, P] unique
        ei = np.hstack([arr, arr[[1, 0]]])                         # both directions
    log(f"  {etype:6s}: {ei.shape[1]:>10,} edges ({ei.shape[1]//2:,} unique pairs) | "
        f"groups full={n_full:,} star={n_star:,} dropped_hub={n_drop:,}")
    return ei


def groups_from_keycol(row_index, keys):
    """key -> [row_index...] for a per-node key column (blank keys dropped)."""
    df = pd.DataFrame({"r": row_index, "k": keys})
    df = df[df["k"].astype(bool) & (df["k"] != "")]
    return {k: list(v) for k, v in df.groupby("k")["r"]}


def stream_nppes(nppes_path, universe, with_phone, with_mail):
    """One pass: per-npi AO/fax/addr (+ optional phone/mail) keys + enum_date."""
    cols = list(NPPES_COLS)
    acc = {}
    seen = 0
    for ch in pd.read_csv(nppes_path, usecols=cols, chunksize=CHUNK, **READ):
        seen += len(ch)
        ch = ch.rename(columns=NPPES_COLS)
        sub = ch[ch["npi"].isin(universe)]
        for r in sub.itertuples(index=False):
            ao = norm_name_base(r.ao_last, r.ao_first)
            addr, _ = norm_address(r.prac1, r.prac2, r.prac_city, r.prac_state, r.prac_zip)
            fax = norm_phone(r.prac_fax)
            rec = {"ao": ao, "addr": addr, "fax": fax, "enum": _clean(r.enum_date)}
            if with_phone:
                rec["phone"] = norm_phone(r.prac_phone)
            if with_mail:
                m, _ = norm_address(r.mail1, r.mail2, r.mail_city, r.mail_state, r.mail_zip)
                rec["mail"] = m
            acc[r.npi] = rec
        if seen % (CHUNK * 10) == 0:
            log(f"    nppes … {seen:,} rows scanned, {len(acc):,} universe NPIs matched")
    log(f"  nppes pass done: {len(acc):,}/{len(universe):,} universe NPIs found")
    return acc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--with-phone", action="store_true")
    p.add_argument("--with-mailing", dest="with_mail", action="store_true")
    args = p.parse_args()
    out = config.GNN_DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    log("[1/6] Loading node table (row-index map)")
    nodes = pd.read_parquet(config.NODES_PARQUET, columns=["row_index", "npi"])
    N = len(nodes)
    idx = dict(zip(nodes["npi"], nodes["row_index"]))
    universe = set(idx)
    log(f"  {N:,} nodes")

    log("[2/6] Streaming NPPES (AO / fax / address / enum-date in one pass)")
    nppes = stream_nppes(config.LEIE_CSV.parent / "NPPES.csv", universe,
                         args.with_phone, args.with_mail)
    # per-node key columns aligned to row order
    npis = nodes["npi"].tolist()
    ao = [nppes.get(n, {}).get("ao", "") for n in npis]
    fax = [nppes.get(n, {}).get("fax", "") for n in npis]
    addr = [nppes.get(n, {}).get("addr", "") for n in npis]
    enum = [nppes.get(n, {}).get("enum", "") for n in npis]
    rows = nodes["row_index"].to_numpy()

    log("[3/6] Reading PECOS (PAC key + enrollment→NPI bridge)")
    enroll_to_npi, pac_groups = {}, defaultdict(list)
    for ch in pd.read_csv(config.LEIE_CSV.parent / "PECOS.csv", chunksize=CHUNK, **READ):
        ch = ch.fillna("")
        sub = ch[ch["NPI"].isin(universe)]
        for r in sub.itertuples(index=False):
            if r.ENRLMT_ID:
                enroll_to_npi[r.ENRLMT_ID] = r.NPI
            if r.PECOS_ASCT_CNTL_ID:
                pac_groups[r.PECOS_ASCT_CNTL_ID].append(idx[r.NPI])
    log(f"  {len(enroll_to_npi):,} enrollment→NPI bridges, {len(pac_groups):,} PAC keys")

    log("[4/6] Reading owner files (owner→NPI via enrollment bridge)")
    owner_groups = defaultdict(list)
    n_owner_files = 0
    for f in sorted(OWNER_GLOB.glob("*.csv")):
        n_owner_files += 1
        df = pd.read_csv(f, **READ).fillna("")
        df["_npi"] = df["ENROLLMENT ID"].map(enroll_to_npi)
        hit = df[df["_npi"].notna() & df["_npi"].isin(universe)]
        for _, r in hit.iterrows():
            assoc = r["ASSOCIATE ID - OWNER"]
            if not str(assoc).strip():
                is_org = bool(str(r["ORGANIZATION NAME - OWNER"]).strip())
                assoc = ("ORG:" + org_key(r["ORGANIZATION NAME - OWNER"]) if is_org
                         else "PER:" + norm_name_base(r["LAST NAME - OWNER"], r["FIRST NAME - OWNER"]))
            if str(assoc).strip() not in ("", "ORG:", "PER:"):
                owner_groups[assoc].append(idx[r["_npi"]])
    log(f"  {len(owner_groups):,} owner keys from {n_owner_files} owner file(s)")

    log("[5/6] Building edges (per-type clique caps)")
    drop_log = []
    edge_sets = {
        "owner": edges_from_groups(owner_groups, "owner", drop_log),
        "ao":    edges_from_groups(groups_from_keycol(rows, ao), "ao", drop_log),
        "pac":   edges_from_groups(pac_groups, "pac", drop_log),
        "fax":   edges_from_groups(groups_from_keycol(rows, fax), "fax", drop_log),
        "addr":  edges_from_groups(groups_from_keycol(rows, addr), "addr", drop_log),
    }
    if args.with_phone:
        phone = [nppes.get(n, {}).get("phone", "") for n in npis]
        edge_sets["phone"] = edges_from_groups(groups_from_keycol(rows, phone), "phone", drop_log)
    if args.with_mail:
        mail = [nppes.get(n, {}).get("mail", "") for n in npis]
        edge_sets["mail"] = edges_from_groups(groups_from_keycol(rows, mail), "mail", drop_log)

    log("[6/6] Writing arrays + node attrs + drop log")
    for t, ei in edge_sets.items():
        np.save(out / f"edges_{t}.npy", ei)
    all_ei = np.hstack([ei for ei in edge_sets.values() if ei.shape[1]])
    all_tag = np.concatenate([np.full(ei.shape[1], TYPE_CODE[t], dtype=np.int8)
                              for t, ei in edge_sets.items() if ei.shape[1]])
    np.save(out / "edges_all.npy", all_ei)
    np.save(out / "edge_type_all.npy", all_tag)
    enum_dt = pd.to_datetime(pd.Series(enum), format="%m/%d/%Y", errors="coerce")
    age_years = (REFERENCE_DATE - enum_dt).dt.days / 365.25
    pd.DataFrame({"row_index": rows, "npi": npis, "enum_date": enum,
                  "provider_age_years": age_years.to_numpy()}).to_parquet(
        out / "nppes_node_attrs.parquet", index=False)
    pd.DataFrame(drop_log, columns=["edge_type", "key", "group_size", "action"]).to_csv(
        out / "edge_drop_log.csv", index=False)

    gate3_report(N, edge_sets, all_ei, nodes, age_years, drop_log)


# ---------------- GATE 3 report ----------------

def gate3_report(N, edge_sets, all_ei, nodes, age_years, drop_log):
    log("\n" + "=" * 64 + "\nGATE 3 — edge / graph sanity\n" + "=" * 64)
    total = all_ei.shape[1]
    log(f"  total edges (undirected, both dirs): {total:,} across {len(edge_sets)} types")
    log(f"  provider-age feature: {int(age_years.notna().sum()):,}/{N:,} non-null "
        f"(median {np.nanmedian(age_years):.1f}y)")
    dl = pd.DataFrame(drop_log)
    if len(dl):
        log(f"  drop log: {int((dl.action=='star').sum()):,} star, "
            f"{int((dl.action=='dropped_hub').sum()):,} dropped hubs "
            f"(biggest dropped group {int(dl[dl.action=='dropped_hub'].group_size.max()) if (dl.action=='dropped_hub').any() else 0:,})")

    # degree distribution over the unioned graph
    deg = np.bincount(all_ei[0], minlength=N)
    iso = int((deg == 0).sum())
    log(f"  degree: max {deg.max():,} | p99 {np.percentile(deg,99):.0f} | "
        f"p50 {np.percentile(deg,50):.0f} | mean {deg.mean():.2f}")
    log(f"  isolated nodes (degree 0): {iso:,} ({iso/N:.1%})")
    log(f"  connected nodes: {N-iso:,} ({(N-iso)/N:.1%})")

    # connectivity
    g = coo_matrix((np.ones(total, dtype=np.int8), (all_ei[0], all_ei[1])), shape=(N, N))
    ncomp, lbl = connected_components(g, directed=False)
    sizes = np.bincount(lbl)
    big = sizes.max()
    log(f"  connected components: {ncomp:,} | largest {big:,} ({big/N:.1%} of nodes)")
    multi = int((sizes >= 2).sum())
    log(f"  non-trivial components (≥2 nodes): {multi:,} | singletons {int((sizes==1).sum()):,}")
    log("  GATE check: graph is neither one blob (largest < ~60% of N) nor all dust "
        f"({(N-iso)/N:.0%} connected) → "
        f"{'OK' if big < 0.6*N and (N-iso) > 0.3*N else 'REVIEW'}")

    # eyeball known multi-NPI operators (org names from the scored parquet)
    log("\n  eyeball — known multi-NPI operators (are their NPIs connected?):")
    names = pd.read_parquet(mcfg.SCORED_UNIVERSE_PARQUET, columns=["npi", "org_legal_name"])
    name_of = dict(zip(names["npi"], names["org_legal_name"].fillna("")))
    nodes = nodes.assign(org=nodes["npi"].map(name_of).fillna(""))
    for pat in ["TOTAL RENAL CARE", "BAYADA", "PEDIATRIC SERVICES OF AMERICA",
                "PINNACLE TREATMENT", "AVEANNA"]:
        sub = nodes[nodes["org"].str.upper().str.contains(pat, na=False)]
        if sub.empty:
            log(f"    {pat:28s}: no NPI matched by name")
            continue
        rs = sub["row_index"].to_numpy()
        comps = lbl[rs]
        # how many fall in the single most-common shared component
        vals, cnts = np.unique(comps, return_counts=True)
        top = cnts.max()
        log(f"    {pat:28s}: {len(rs):>4} NPIs | {top}/{len(rs)} share one component "
            f"({'connected ✓' if top >= 2 else 'isolated ✗'})")
    log(f"\n  written → {config.GNN_DATA_DIR}")


if __name__ == "__main__":
    main()
