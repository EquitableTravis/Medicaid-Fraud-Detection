"""
pipeline_nogeo.py (labels) — END-TO-END final product for the no-geo model.

Runs the fraud_positive model trained WITHOUT practice_state / primary_taxonomy
(flags on billing behavior only, no guilt-by-specialty) through the entire pipeline
and writes the finished lead list to a new Desktop folder:

  fit no-geo model → score 617k → reliability gate → company rollup → $10M + top-K
  → hardened institutional screens → per-lead SHAP drivers → NPPES name resolution
  → confidence bands → final CSV + README.

Run:
    python -m labels.pipeline_nogeo
"""

import json
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
from src.model.train_fraud import build_frame
from labels.build_leads_fraud import screen

GEO = ["primary_taxonomy", "practice_state"]
OUTDIR = Path.home() / "Desktop" / "NoGeo_Model_Leads"
PROVIDER_DIM = c.DATA_DIR / "integrated" / "provider_dim.parquet"
TOP_N_DRIVERS = 4


def band(score):
    return "high" if score >= 0.99 else ("med" if score >= 0.90 else "low")


def calibration(comp):
    cc = comp[comp["n_npis_reliable"] > 0].copy()
    cc["has_leie"] = cc["n_leie_npis"] > 0
    base = cc["has_leie"].mean()
    edges = [0.0, 0.5, 0.7, 0.8, 0.9, 0.99, 1.0001]
    cc["bin"] = pd.cut(cc["company_model_score_max"], edges, right=False)
    g = cc.groupby("bin", observed=True)["has_leie"].agg(["size", "mean"])
    g["lift"] = g["mean"] / base
    return base, g


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    log("[1/6] Fit no-geo model (no state / no taxonomy)")
    X, base = build_frame()
    Xg = X.drop(columns=GEO)
    cats = [col for col in c.CATEGORICAL_FEATURES if col in Xg.columns]
    labm = (base["pos"] | base["neg"]).values
    Xl, yl = Xg[labm], base.loc[labm, "pos"].astype("int8").values
    Xt, Xv, yt, yv = train_test_split(Xl, yl, test_size=c.VAL_FRACTION, stratify=yl, random_state=c.SEED)
    p = dict(LGB_PARAMS); p["seed"] = c.SEED
    booster = lgb.train(p, lgb.Dataset(Xt, label=yt, categorical_feature=cats), NUM_BOOST_ROUND,
                        valid_sets=[lgb.Dataset(Xv, label=yv)],
                        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)])
    log(f"    {Xg.shape[1]} features, {len(Xl):,} labeled rows, best_iter {booster.best_iteration}")

    log("[2/6] Score universe + reliability gate + company rollup")
    df = pd.read_parquet(c.SCORED_UNIVERSE_PARQUET).reset_index(drop=True)
    assert (df["npi"].values == base["npi"].values).all(), "row misalignment"
    prov = df[PROVIDER_CONTEXT_COLS].copy()
    prov.insert(1, "model_score", booster.predict(Xg))
    prov["score_reliable"] = ~prov["not_scored"].fillna(True)
    prov = prov.sort_values("model_score", ascending=False).reset_index(drop=True)
    comp = rollup_to_company(prov)
    base_rate, caltab = calibration(comp)

    log("[3/6] $10M + top-5000 + name resolution + hardened screens")
    elig = comp[(comp["company_net_paid"] >= 10_000_000) & (comp["company_model_score_max"] >= 0)]
    leads = elig.sort_values(["company_model_score_max", "company_net_paid"], ascending=False).head(5000).copy()
    leads = resolve_names(leads, prov[["npi", "model_score", "org_legal_name"]])
    leads["rank"] = range(1, len(leads) + 1)
    leads["n_fraud_pos_npis"] = 0
    kept, audit = screen(leads[OUT_COLS + ["n_fraud_pos_npis"]], prov)
    kept = kept.reset_index(drop=True)
    log(f"    {len(leads):,} → screened {len(kept):,} (removed {len(audit):,})")

    log("[4/6] Per-lead SHAP drivers")
    npi_idx = {n: i for i, n in enumerate(df["npi"])}
    score_of = prov.set_index("npi")["model_score"]
    reliable = set(prov.loc[prov["score_reliable"], "npi"])
    def top_npi(npis):
        cand = [n for n in str(npis).split("|") if n in reliable] or str(npis).split("|")
        return max(cand, key=lambda n: score_of.get(n, -1))
    kept["_dnpi"] = kept["npi_list"].map(top_npi)
    rows = [npi_idx[n] for n in kept["_dnpi"]]
    contrib = booster.predict(Xg.iloc[rows], pred_contrib=True)
    feats = list(Xg.columns)
    drivers = []
    for i in range(len(kept)):
        cc = contrib[i, :-1]
        top = np.argsort(-cc)[:TOP_N_DRIVERS]
        drivers.append("; ".join(f"{feats[j]} ({cc[j]:+.2f})" for j in top if cc[j] > 0))
    kept["top_drivers"] = drivers

    log("[5/6] NPPES name resolution + state + confidence bands")
    pdm = pd.read_parquet(PROVIDER_DIM, columns=["npi", "provider_name"])
    name_of = pdm.assign(npi=pdm["npi"].astype(str)).set_index("npi")["provider_name"]
    unk = kept["company_name"].astype(str).str.startswith("UNKNOWN NAME")
    res = kept.loc[unk, "company_name"].str.extract(r"NPI (\d+)")[0].map(name_of)
    kept.loc[unk, "company_name"] = res.where(res.notna() & (res.str.strip() != ""),
                                              kept.loc[unk, "company_name"]).values
    pstate = df.sort_values("net_paid", ascending=False).drop_duplicates("npi").set_index("npi")["practice_state"]
    kept["state"] = kept["_dnpi"].map(pstate)
    kept["confidence"] = kept["company_model_score_max"].map(band)
    kept["rank"] = range(1, len(kept) + 1)

    log("[6/6] Writing final product → Desktop/NoGeo_Model_Leads/")
    cols = ["rank", "company_name", "confidence", "company_model_score_max", "company_net_paid",
            "n_npis", "n_npis_reliable", "n_leie_npis", "specialty", "state", "top_drivers",
            "merge_confidence", "npi_list"]
    final = kept[cols].rename(columns={"company_model_score_max": "model_score"})
    final_csv = OUTDIR / "leads_final.csv"
    final.to_csv(final_csv, index=False)
    audit.to_csv(OUTDIR / "removed_institutional_audit.csv", index=False)

    cal_md = "\n".join(f"| {str(i)} | {int(r['size']):,} | {r['mean']:.3%} | {r['lift']:.1f}x |"
                       for i, r in caltab.iterrows())
    mix = kept["confidence"].value_counts()
    (OUTDIR / "README.md").write_text(f"""# No-Geo Model — Final Lead List

Generated by `labels/pipeline_nogeo.py` from the **fraud_positive** model trained
WITHOUT `practice_state` and `primary_taxonomy` — so every lead is flagged on billing
behavior (rate anomalies vs taxonomy peers), never on "what specialty/state it is."

## What this is
- **{len(final):,} company leads**, each ≥$10M consolidated Medicaid billing, ranked by max
  constituent model score, after institutional screens (hospital/FQHC/government/tribal/
  academic/school/nonprofit removed — {len(audit):,} removed, audit included).
- Model: LightGBM, label = `fraud_positive` (889 = fraud-relevant LEIE ∪ AHCCCS ∪ NV),
  {Xg.shape[1]} billing features (state + taxonomy dropped). Temporal validation: val PR-AUC
  ~0.56, test ~0.52, P@50 1.00.
- Columns: `top_drivers` = the SHAP features that drove each flag; `confidence` band; `state`
  + `specialty` = dominant constituent NPI; `npi_list` = constituents.

## Confidence bands (empirical LEIE precision by score band; base rate {base_rate:.3%})
| score band | companies | LEIE precision | lift |
|---|---|---|---|
{cal_md}

Lift concentrates at the top: **high = score ≥0.99**, med = 0.90–0.99, low = <0.90.
Confidence mix: {", ".join(f"{k} {v}" for k, v in mix.items())}.

## Important
These are **leads for human review, not determinations of fraud**. The flags are real
peer-relative billing anomalies, but a high score is not proof of wrongdoing — many large
legitimate providers in intensive care models also bill unlike their peers. Prioritize the
**high band + single-NPI high-$ individual outliers**. Out-of-sample validation (the frozen
forward test against the next LEIE/AHCCCS pull) is the pending check on real predictive value.

Generated {len(final):,} leads · ${final['company_net_paid'].sum()/1e9:.0f}B total billing.
""")
    log(f"    {len(final):,} leads, ${final['company_net_paid'].sum()/1e9:.1f}B → {final_csv}")
    log(f"    confidence: " + ", ".join(f"{k}={v}" for k, v in mix.items()))


if __name__ == "__main__":
    main()
