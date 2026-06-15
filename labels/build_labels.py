"""
build_labels.py (labels) — expanded + cleaned positive-label inventory.

The model is bottlenecked by 578 LEIE positives, many of which are NOT billing
fraud (license revocation, loan default, etc.). This builds a richer, cleaner
positive set, keyed to the 617k scored universe, from three sources:

  fraud_leie  — LEIE rows whose EXCLTYPE is fraud-relevant (1128 a1/a2/a3/b1/b2/
                b3/b7 — convictions/fraud/kickbacks), dropping license/loan noise.
  ahcccs      — Arizona Medicaid (AHCCCS) provider suspensions/terminations
                (credible-allegation-of-fraud actions — the AZ behavioral-health
                wave that federal LEIE largely misses).
  nv          — Nevada Medicaid sanctions/terminations.

Each positive carries an exclusion YEAR (for the temporal split). Output is an
inventory table + a printed report of counts, overlaps, net-new positives beyond
the current LEIE label, and universe coverage. Read-only on src/model. No
existing file is modified.

Run:
    python -m labels.build_labels
"""

import argparse
import re

import numpy as np
import pandas as pd

from src.model import config as mcfg
from src.model.build_pursuit_pipeline import parse_ahcccs, pdf_text

FRAUD_RELEVANT = {"1128a1", "1128a2", "1128a3", "1128b1", "1128b2", "1128b3", "1128b7"}
PRECLEAN = mcfg.DATA_DIR / "preclean"
ENFORCE = mcfg.MODEL_DATA_DIR / "enforcement_lists"
OUT = mcfg.MODEL_DATA_DIR / "labels" / "expanded_labels.parquet"
NPI_RE = re.compile(r"\b[12]\d{9}\b")
DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")


def log(m=""):
    print(m, flush=True)


def year_from(s):
    m = DATE_RE.search(str(s))
    if not m:
        return None
    y = int(m.group(3))
    return y if 2006 <= y <= 2026 else None     # NPIs only exist since 2006; clip parse junk


def leie_positives(universe):
    """fraud-relevant + all LEIE NPIs in the universe, each with an exclusion year."""
    df = pd.read_csv(PRECLEAN / "Caught.csv", dtype=str, keep_default_na=False)
    df = df[df["NPI"].str.fullmatch(r"[12]\d{9}") & df["NPI"].isin(universe)]
    df["yr"] = pd.to_datetime(df["EXCLDATE"], format="%Y%m%d", errors="coerce").dt.year
    df = df[df["yr"].between(2006, 2026)]
    any_leie = set(df["NPI"])
    any_year = df.groupby("NPI")["yr"].min().astype(int).to_dict()
    fr = df[df["EXCLTYPE"].isin(FRAUD_RELEVANT)]
    fr_year = fr.groupby("NPI")["yr"].min().astype(int).to_dict()
    return any_leie, set(fr["NPI"]), fr_year, any_year


def ahcccs_positives(universe):
    npis, _ = parse_ahcccs(ENFORCE / "ahcccs_suspensions.pdf")   # {npi: "AHCCCS Suspend mm/dd/yyyy"}
    return {n: year_from(s) for n, s in npis.items() if n in universe}


def nv_positives(universe):
    """NV master exclusions: NPI + best-effort termination year from the same line."""
    out = {}
    for line in pdf_text(ENFORCE / "nv_exclusions.pdf").splitlines():
        m = NPI_RE.search(line)
        if m and m.group(0) in universe:
            out[m.group(0)] = year_from(line)
    return out


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    log("[1/4] Loading universe + current LEIE label")
    uni = pd.read_parquet(mcfg.SCORED_UNIVERSE_PARQUET, columns=["npi", "provider_on_leie"])
    universe = set(uni["npi"])
    cur_leie = set(uni.loc[uni["provider_on_leie"].fillna(False), "npi"])
    log(f"  universe {len(universe):,} NPIs | current provider_on_leie positives {len(cur_leie):,}")

    log("[2/4] Fraud-relevant LEIE (drop license/loan exclusions)")
    any_leie, fraud_leie, fr_year, any_year = leie_positives(universe)
    log(f"  LEIE in universe: any-type {len(any_leie):,} | fraud-relevant {len(fraud_leie):,} "
        f"(dropped {len(any_leie)-len(fraud_leie):,} non-fraud exclusions)")

    log("[3/4] State enforcement (AHCCCS + NV)")
    az = ahcccs_positives(universe)
    nv = nv_positives(universe)
    log(f"  AHCCCS suspensions in universe: {len(az):,} | NV sanctions in universe: {len(nv):,}")

    log("[4/4] Assembling label table")
    fraud_pos = set(fraud_leie) | set(az) | set(nv)

    def min_year(*dicts):
        """earliest non-null year for an NPI across the given {npi:year} maps."""
        out = {}
        for d in dicts:
            for n, y in d.items():
                if y and (n not in out or y < out[n]):
                    out[n] = y
        return out

    fraud_year = min_year(fr_year, az, nv)              # year for the fraud_positive target
    excl_year_all = min_year(any_year, az, nv)          # year for ANY excluded node (covers variant A)
    rows = []
    for n in universe:
        rows.append({
            "npi": n,
            "on_leie_any": n in any_leie,
            "on_leie_fraud": n in fraud_leie,
            "on_ahcccs": n in az,
            "on_nv": n in nv,
            "fraud_positive": n in fraud_pos,
            "excl_year": fraud_year.get(n),             # fraud-target year
            "excl_year_all": excl_year_all.get(n),      # any-source year (for temporal split of every variant)
        })
    lab = pd.DataFrame(rows)
    lab.to_parquet(OUT, index=False)

    # ---- inventory report ----
    net_new = fraud_pos - cur_leie
    cleaned_out = cur_leie - fraud_pos          # current positives dropped as non-fraud / not re-confirmed
    az_only = set(az) - any_leie
    nv_only = set(nv) - any_leie
    log("\n" + "=" * 62 + "\nLABEL INVENTORY\n" + "=" * 62)
    log(f"  current label (all LEIE):            {len(cur_leie):>5,}")
    log(f"  NEW fraud_positive set:              {len(fraud_pos):>5,}")
    log(f"    = fraud-relevant LEIE  {len(fraud_leie):>5,}")
    log(f"    ∪ AHCCCS               {len(az):>5,}  ({len(az_only):,} not on LEIE)")
    log(f"    ∪ NV                   {len(nv):>5,}  ({len(nv_only):,} not on LEIE)")
    log(f"  net-NEW positives vs current label:  {len(net_new):>5,}")
    log(f"  current positives NOT in new set:    {len(cleaned_out):>5,} "
        f"(non-fraud LEIE types dropped)")
    yr = lab.loc[lab["fraud_positive"], "excl_year"].dropna().astype(int)
    log(f"  fraud_positive with a year: {len(yr):,}/{len(fraud_pos):,}")
    log("  year histogram: " + " ".join(f"{y}:{c}" for y, c in yr.value_counts().sort_index().items()))
    log(f"  → {OUT}")


if __name__ == "__main__":
    main()
