"""
explain_flags.py (labels) — for the audited CLEANED sample, show WHY each lead was
flagged (top SHAP driver + the underlying peer-normalized feature value), so a
"looks legit on the web" verdict can be tested against the actual billing anomaly.

Framework:
  * The model's features are RATES normalized WITHIN taxonomy peers (per-patient /
    per-line intensity as percentile + robust-z vs same-taxonomy providers). Net_paid
    is NOT a feature. So "they're a big legit X" is already controlled for — the peer
    group IS other X's.
  * If the top driver is a peer-normalized rate at an extreme percentile → the flag is
    REAL (they bill unlike their own peers); a clean website does not explain it away.
  * If the top driver is `primary_taxonomy` → the flag is "this specialty is fraud-prone"
    (guilt-by-specialty) → explainable/weaker for a legitimate provider in that specialty.

Run:
    python -m labels.explain_flags
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.model import config as c
from src.model.score import build_universe_matrix

CLEANED = c.MODEL_DATA_DIR / "output" / "final" / "model_leads_CLEANED.csv"
RATE_HINT = ("_pct_tax", "_rz_tax", "_pct_taxstate", "_rz_taxstate")


def log(m=""): print(m, flush=True)


def main():
    d = pd.read_csv(CLEANED, dtype={"npi_list": str})
    samp = d.sample(n=25, random_state=7).sort_values("new_rank")     # same sample as the audit

    prov = pd.read_parquet(c.SCORES_DIR / "provider_model_scores.parquet",
                           columns=["npi", "model_score"]).set_index("npi")["model_score"]
    samp["driver_npi"] = samp["npi_list"].map(
        lambda x: max(str(x).split("|"), key=lambda n: prov.get(n, -1)))

    adir = c.ARTIFACTS_DIR
    booster = lgb.Booster(model_file=str(adir / "lgbm_leie.txt"))
    spec = json.loads((adir / "feature_list.json").read_text())
    feats = spec["features"]

    df = pd.read_parquet(c.SCORED_UNIVERSE_PARQUET)
    sub = df[df["npi"].isin(set(samp["driver_npi"]))].reset_index(drop=True)
    X = build_universe_matrix(sub, spec)
    contrib = booster.predict(X, pred_contrib=True)
    rowidx = {npi: i for i, npi in enumerate(sub["npi"])}

    log(f"\n{'='*78}\nWHY EACH SAMPLE LEAD WAS FLAGGED (top driver + peer-normalized value)\n{'='*78}")
    peer_real = guilt_tax = 0
    for r in samp.itertuples():
        i = rowidx[r.driver_npi]
        c_ = contrib[i, :-1]
        top = np.argsort(-c_)[:3]
        f0 = feats[top[0]]
        # raw value of the top driver in the universe row
        val = sub.iloc[i].get(f0, None)
        is_rate = f0.endswith(RATE_HINT)
        # peer percentile of the headline intensity feature if present
        pct = sub.iloc[i].get("lines_per_patient_instance_pct_tax", np.nan)
        psz = sub.iloc[i].get("peer_group_size_tax", np.nan)
        kind = "REAL peer-rate anomaly" if is_rate else ("guilt-by-specialty" if f0 == "primary_taxonomy" else "other")
        if is_rate: peer_real += 1
        elif f0 == "primary_taxonomy": guilt_tax += 1
        drivers = ", ".join(f"{feats[j]}({c_[j]:+.1f})" for j in top)
        vtxt = f"{val:.3f}" if isinstance(val, (int, float, np.floating)) and pd.notna(val) else str(val)
        log(f"\n#{int(r.new_rank)} {str(r.company_name)[:44]} [{r.specialty}]  peer_n={'' if pd.isna(psz) else int(psz)}")
        log(f"   top driver: {f0} = {vtxt}  → {kind}")
        log(f"   lines/patient percentile vs taxonomy: {'' if pd.isna(pct) else f'{pct:.1%}'}")
        log(f"   SHAP: {drivers}")
    log(f"\n{'-'*78}")
    log(f"top driver = peer-normalized RATE anomaly: {peer_real}/25  | = primary_taxonomy (guilt-by-specialty): {guilt_tax}/25")


if __name__ == "__main__":
    main()
