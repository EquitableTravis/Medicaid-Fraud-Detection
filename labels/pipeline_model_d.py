"""
pipeline_model_d.py (labels) — final lead list from Model D (trained on ALL
exclusions: fraud-relevant LEIE ∪ all 38 state Medicaid exclusion lists).

Filter stack (as specified):
  score all NPIs (Model D) -> reliability gate (score_reliable = ~not_scored)
  -> company rollup (max constituent score over reliable constituents)
  -> consolidated billing >= $10M
  -> company_model_score_max >= 0.90
  -> hardened institutional screens (hospital/FQHC/gov/tribal/academic/school/nonprofit)
  -> NPPES name resolution + SHAP drivers + confidence bands

Outputs to ~/Desktop/Model_D_Leads/:
  leads_final.csv               all leads passing the filters
  leads_final_NOVEL.csv         training-positive (already-excluded) companies removed
  removed_institutional_audit.csv
  README.md

Run: python -m labels.pipeline_model_d
"""

import json, re
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.model import config as c
from src.model.data import build_feature_matrix, log
from src.model.score import rollup_to_company, PROVIDER_CONTEXT_COLS
from src.model.export_leads import resolve_names, OUT_COLS
from src.model.train import LGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS
from labels.build_leads_fraud import screen

OUTDIR = Path.home() / "Desktop" / "Model_D_Leads"
PROVIDER_DIM = c.DATA_DIR / "integrated" / "provider_dim.parquet"
SCORE_MIN, MIN_NET = 0.90, 10_000_000
FR = {"1128a1","1128a2","1128a3","1128b1","1128b2","1128b3","1128b7"}


def leie_sets():
    df = pd.read_csv(c.DATA_DIR/"preclean"/"Caught.csv", dtype=str, keep_default_na=False)
    df = df[df["NPI"].str.fullmatch(r"[12]\d{9}")]
    return set(df["NPI"]), set(df.loc[df["EXCLTYPE"].isin(FR),"NPI"])


def band(s): return "high" if s>=0.99 else ("med" if s>=0.90 else "low")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log("[1/6] Building Model D label + clean negatives")
    df = pd.read_parquet(c.SCORED_UNIVERSE_PARQUET).reset_index(drop=True)
    X = build_feature_matrix(df).reset_index(drop=True)           # full features (incl. geo)
    npis = df["npi"].values; uni = set(npis)
    cmap = pd.read_parquet(c.NPI_TO_COMPANY_MAP).set_index("npi")["company_id"]
    company_id = df["npi"].map(cmap)
    leie_any, leie_fraud = leie_sets()
    st = pd.read_csv(c.MODEL_DATA_DIR/"labels"/"all_state_exclusions_npis.csv", dtype={"npi":str})
    state_npis = set(st.loc[st["npi"].isin(uni),"npi"])
    fp = set(pd.read_parquet(c.MODEL_DATA_DIR/"labels"/"expanded_labels.parquet").query("fraud_positive")["npi"])

    npi_s = pd.Series(npis)
    pos = npi_s.isin(leie_fraud | state_npis).values                # Model D positives
    excluded_any = npi_s.isin(leie_any | state_npis | fp).values    # never a negative
    pos_company = set(company_id[pos].dropna())
    clean = (df["anomaly_score"].values==0) & (~df["not_scored"].fillna(True).values) \
            & (~excluded_any) & (~company_id.isin(pos_company).values)
    log(f"    {pos.sum():,} positives | {clean.sum():,} clean negatives")

    log("[2/6] Fitting Model D production scorer (all labeled rows)")
    lab = pos | clean
    Xl, yl = X[lab], pos[lab].astype("int8")
    cats = [col for col in c.CATEGORICAL_FEATURES if col in X.columns]
    Xt, Xv, yt, yv = train_test_split(Xl, yl, test_size=c.VAL_FRACTION, stratify=yl, random_state=c.SEED)
    p = dict(LGB_PARAMS); p["seed"]=c.SEED
    booster = lgb.train(p, lgb.Dataset(Xt,label=yt,categorical_feature=cats), NUM_BOOST_ROUND,
                        valid_sets=[lgb.Dataset(Xv,label=yv)],
                        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS,verbose=False),lgb.log_evaluation(0)])
    log(f"    best_iter {booster.best_iteration} on {len(Xl):,} labeled rows")

    log("[3/6] Score universe + reliability gate + company rollup")
    prov = df[PROVIDER_CONTEXT_COLS].copy()
    prov.insert(1, "model_score", booster.predict(X))
    prov["score_reliable"] = ~prov["not_scored"].fillna(True)
    prov = prov.sort_values("model_score", ascending=False).reset_index(drop=True)
    comp = rollup_to_company(prov)

    log(f"[4/6] Filter: $10M+ AND company_model_score_max >= {SCORE_MIN}")
    leads = comp[(comp["company_net_paid"]>=MIN_NET) & (comp["company_model_score_max"]>=SCORE_MIN)].copy()
    leads = leads.sort_values(["company_model_score_max","company_net_paid"], ascending=False)
    leads = resolve_names(leads, prov[["npi","model_score","org_legal_name"]])
    leads["rank"] = range(1, len(leads)+1)
    excl_all = leie_any | state_npis | fp
    leads["n_excluded_npis"] = leads["npi_list"].map(lambda s: sum(n in excl_all for n in str(s).split("|")))
    leads["n_fraud_pos_npis"] = leads["n_excluded_npis"]
    log(f"    {len(leads):,} companies pass $10M + score>={SCORE_MIN}")

    log("[5/6] Institutional screens + SHAP drivers + names + bands")
    kept, audit = screen(leads[OUT_COLS+["n_fraud_pos_npis","n_excluded_npis"]], prov)
    kept = kept.reset_index(drop=True)
    # SHAP drivers for each lead's top reliable constituent NPI
    npi_idx = {n:i for i,n in enumerate(df["npi"])}
    score_of = prov.set_index("npi")["model_score"]; reliable=set(prov.loc[prov["score_reliable"],"npi"])
    def top_npi(s):
        cand=[n for n in str(s).split("|") if n in reliable] or str(s).split("|")
        return max(cand, key=lambda n: score_of.get(n,-1))
    kept["_dnpi"]=kept["npi_list"].map(top_npi)
    contrib = booster.predict(X.iloc[[npi_idx[n] for n in kept["_dnpi"]]], pred_contrib=True)
    feats=list(X.columns)
    kept["top_drivers"]=[ "; ".join(f"{feats[j]} ({contrib[i,j]:+.2f})"
                          for j in np.argsort(-contrib[i,:-1])[:4] if contrib[i,j]>0) for i in range(len(kept))]
    # NPPES name resolution for UNKNOWN
    pdm = pd.read_parquet(PROVIDER_DIM, columns=["npi","provider_name"])
    name_of = pdm.assign(npi=pdm["npi"].astype(str)).set_index("npi")["provider_name"]
    unk = kept["company_name"].astype(str).str.startswith("UNKNOWN NAME")
    res = kept.loc[unk,"company_name"].str.extract(r"NPI (\d+)")[0].map(name_of)
    kept.loc[unk,"company_name"] = res.where(res.notna()&(res.str.strip()!=""), kept.loc[unk,"company_name"]).values
    pstate = df.sort_values("net_paid",ascending=False).drop_duplicates("npi").set_index("npi")["practice_state"]
    kept["state"]=kept["_dnpi"].map(pstate)
    kept["confidence"]=kept["company_model_score_max"].map(band)
    kept["rank"]=range(1,len(kept)+1)

    log("[6/6] Writing -> Desktop/Model_D_Leads/")
    cols=["rank","company_name","confidence","company_model_score_max","company_net_paid","n_npis",
          "n_npis_reliable","n_excluded_npis","specialty","state","top_drivers","merge_confidence","npi_list"]
    final=kept[cols].rename(columns={"company_model_score_max":"model_score"})
    final.to_csv(OUTDIR/"leads_final.csv", index=False)
    novel=final[final["n_excluded_npis"]==0].copy(); novel["rank"]=range(1,len(novel)+1)
    novel.to_csv(OUTDIR/"leads_final_NOVEL.csv", index=False)
    audit.to_csv(OUTDIR/"removed_institutional_audit.csv", index=False)
    mix=kept["confidence"].value_counts().to_dict()
    (OUTDIR/"README.md").write_text(
f"""# Model D — Final Lead List

Model D = LightGBM trained on ALL exclusions (fraud-relevant federal LEIE + all 38 state
Medicaid exclusion lists; {pos.sum():,} positives). Best model on the temporal benchmark
(neutral future-LEIE PR-AUC 0.56 vs 0.46 fraud_positive vs 0.27 all-LEIE).

## Filters applied
Model D score -> reliability gate (drop not_scored) -> company rollup (max reliable
constituent score) -> consolidated billing >= ${MIN_NET:,} -> company score >= {SCORE_MIN}
-> institutional screens ({len(audit):,} removed) -> NPPES names + SHAP drivers + bands.

## Files
- `leads_final.csv` — {len(final):,} companies (${final['company_net_paid'].sum()/1e9:.0f}B billing). Confidence: {mix}.
- `leads_final_NOVEL.csv` — {len(novel):,} with already-excluded constituents removed (genuinely new leads).
- `removed_institutional_audit.csv` — screened-out institutions (quarantined, not deleted).

Columns: rank, company_name, confidence (high>=0.99 / med 0.90-0.99), model_score,
company_net_paid, n_npis, n_excluded_npis (already on an exclusion list), specialty, state,
top_drivers (SHAP), npi_list. **Leads for human review, never determinations of fraud.**
""")
    log(f"    leads_final {len(final):,} | NOVEL {len(novel):,} | removed {len(audit):,}")
    log(f"    confidence: {mix}")


if __name__ == "__main__":
    main()
