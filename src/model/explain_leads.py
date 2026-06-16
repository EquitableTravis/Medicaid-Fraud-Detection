"""
explain_leads.py (model) — attach per-lead SHAP feature drivers to the company lead
list, so every lead carries WHY it scored high (like the rest of the pipeline's
anomaly_contributing_features).

For each company lead, takes its highest-scoring reliable constituent NPI and pulls
that NPI's top positive LightGBM SHAP contributions (pred_contrib). Lean: contributions
are computed only for the lead constituents, not the full 617k universe.

Run (after export_leads + screen_leads):
    python -m src.model.explain_leads
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from .data import log, require
from .score import build_universe_matrix

TOP_N = 4   # feature drivers per lead


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_csv", default=str(
        config.MODEL_DATA_DIR / "model_leads_top5000_over10m_screened.csv"))
    p.add_argument("--artifacts-dir", default=str(config.ARTIFACTS_DIR))
    args = p.parse_args()
    in_path = Path(args.in_csv)

    log("[1/4] Loading model, leads, provider scores")
    adir = Path(args.artifacts_dir)
    booster = lgb.Booster(model_file=str(adir / "lgbm_leie.txt"))
    spec = json.loads((adir / "feature_list.json").read_text())
    leads = pd.read_csv(in_path, dtype={"npi_list": str})
    prov = pd.read_parquet(config.SCORES_DIR / "provider_model_scores.parquet",
                           columns=["npi", "model_score", "score_reliable"])
    score_of = prov.set_index("npi")["model_score"]
    reliable = set(prov.loc[prov["score_reliable"], "npi"])

    log("[2/4] Picking each lead's top reliable constituent NPI")
    def top_npi(npis):
        cand = [n for n in str(npis).split("|") if n in reliable] or str(npis).split("|")
        return max(cand, key=lambda n: score_of.get(n, -1))
    leads["_driver_npi"] = leads["npi_list"].map(top_npi)

    log("[3/4] Computing SHAP contributions for the lead constituents")
    df = pd.read_parquet(config.SCORED_UNIVERSE_PARQUET)
    sub = df[df["npi"].isin(set(leads["_driver_npi"]))].reset_index(drop=True)
    X = build_universe_matrix(sub, spec)
    contrib = booster.predict(X, pred_contrib=True)        # (n, n_feat+1); last col = base
    feats = spec["features"]
    require("contrib width matches features+1", contrib.shape[1] == len(feats) + 1)
    rows = {}
    for i, npi in enumerate(sub["npi"]):
        c = contrib[i, :-1]
        top = np.argsort(-c)[:TOP_N]
        rows[npi] = "; ".join(f"{feats[j]} ({c[j]:+.2f})" for j in top if c[j] > 0)
    leads["top_drivers"] = leads["_driver_npi"].map(rows)

    log("[4/4] Writing explained lead list")
    leads = leads.drop(columns=["_driver_npi"])
    out = in_path.with_name(in_path.stem + "_explained.csv")
    leads.to_csv(out, index=False)
    log(f"    {len(leads):,} leads with SHAP drivers → {out}")
    log("    example: " + str(leads.iloc[0]["company_name"]) + " ← " + str(leads.iloc[0]["top_drivers"])[:120])


if __name__ == "__main__":
    main()
