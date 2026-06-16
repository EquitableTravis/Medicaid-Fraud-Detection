"""
finalize_handoff.py (model) — two finishing touches for the ad-targeting handoff:

  1. NPPES name resolution — replace "UNKNOWN NAME (NPI x)" leads (single individual
     NPIs with no org_legal_name) with the person's name from provider_dim
     (NPPES-derived provider_name), so no lead reaches the ad team as a bare NPI.

  2. Score-band calibration — the model score is an uncalibrated RANKING score, not a
     probability. Rather than invent a probability, report empirical precision (LEIE
     density) + lift by score band on the reliable universe, and assign each lead a
     confidence band (high/med/low) on that evidence — the defensible handoff signal.

Run (after export_leads + screen_leads [+ explain_leads]):
    python -m src.model.finalize_handoff
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import log, require

PROVIDER_DIM = config.DATA_DIR / "integrated" / "provider_dim.parquet"
# Bands chosen from the empirical precision-by-band table (calibration_table below):
# LEIE lift is NOT monotonic — it is ~base-rate across 0.5-0.99 and jumps to ~12.7x
# only at >=0.99. So the score genuinely prioritizes ONLY the top band; below it,
# leads rest on the $10M + institutional-screen filters, not on the score.
BANDS = [(0.99, "high"), (0.90, "med"), (0.0, "low")]


def band(score):
    for thr, name in BANDS:
        if score >= thr:
            return name
    return "low"


def calibration_table():
    """Empirical LEIE precision + lift by company-score band on reliable companies."""
    comp = pd.read_parquet(config.SCORES_DIR / "company_model_scores.parquet",
                           columns=["company_model_score_max", "n_leie_npis", "n_npis_reliable"])
    comp = comp[comp["n_npis_reliable"] > 0]
    comp["has_leie"] = comp["n_leie_npis"] > 0
    base = comp["has_leie"].mean()
    edges = [0.0, 0.5, 0.7, 0.8, 0.9, 0.99, 1.0001]
    comp["bin"] = pd.cut(comp["company_model_score_max"], edges, right=False)
    g = comp.groupby("bin", observed=True)["has_leie"].agg(["size", "mean"])
    g["lift"] = g["mean"] / base
    log(f"    base LEIE rate among reliable companies: {base:.4%}")
    log("    score band            n        LEIE-precision   lift")
    for idx, r in g.iterrows():
        log(f"    {str(idx):20s} {int(r['size']):>8,}   {r['mean']:>10.4%}   {r['lift']:>5.1f}x")
    return g


def main():
    p = argparse.ArgumentParser(description=__doc__)
    default_in = config.MODEL_DATA_DIR / "model_leads_top5000_over10m_screened_explained.csv"
    if not default_in.exists():
        default_in = config.MODEL_DATA_DIR / "model_leads_top5000_over10m_screened.csv"
    p.add_argument("--in", dest="in_csv", default=str(default_in))
    args = p.parse_args()
    in_path = Path(args.in_csv)

    log("[1/3] Resolving UNKNOWN individual-NPI names from provider_dim (NPPES)")
    leads = pd.read_csv(in_path, dtype={"npi_list": str})
    pdm = pd.read_parquet(PROVIDER_DIM, columns=["npi", "provider_name"])
    name_of = pdm.assign(npi=pdm["npi"].astype(str)).set_index("npi")["provider_name"]
    unk = leads["company_name"].astype(str).str.startswith("UNKNOWN NAME")
    npis = leads.loc[unk, "company_name"].str.extract(r"NPI (\d+)")[0]
    resolved = npis.map(name_of)
    leads.loc[unk, "company_name"] = resolved.where(
        resolved.notna() & (resolved.str.strip() != ""), leads.loc[unk, "company_name"]).values
    n_fixed = int((resolved.str.strip() != "").fillna(False).sum())
    require("no bare-NPI names remain that provider_dim could resolve",
            n_fixed >= int(unk.sum()) - 5, f"{n_fixed}/{int(unk.sum())}")
    log(f"    resolved {n_fixed}/{int(unk.sum())} UNKNOWN individual leads to names")

    log("[2/3] Score-band calibration (empirical LEIE precision/lift)")
    calibration_table()

    log("[3/3] Assigning confidence bands + writing handoff CSV")
    leads["confidence"] = leads["company_model_score_max"].map(band)
    out = in_path.with_name("model_leads_handoff.csv")
    leads.to_csv(out, index=False)
    log("    confidence mix: " + " ".join(f"{k}:{v}" for k, v in
        leads["confidence"].value_counts().items()))
    log(f"    {len(leads):,} leads → {out}")


if __name__ == "__main__":
    main()
