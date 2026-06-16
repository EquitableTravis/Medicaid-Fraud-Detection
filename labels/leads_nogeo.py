"""
leads_nogeo.py (labels) — fit the fraud_positive model WITHOUT practice_state /
primary_taxonomy, regenerate the company lead list, and measure whether dropping the
geo/sector shortcut DE-CONCENTRATES the leads (less AZ / behavioral-health pile-up)
while keeping the billing-anomaly signal.

Reuses the production chain (rollup_to_company, resolve_names, the hardened screens).
X comes from train_fraud.build_frame (full-universe features, consistent encoding), so
fit and score share identical columns. Compares AZ share + taxonomy concentration vs
the full fraud_positive leads and vs CLEANED.

Run:
    python -m labels.leads_nogeo
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.model import config as c
from src.model.data import log
from src.model.score import rollup_to_company, PROVIDER_CONTEXT_COLS
from src.model.export_leads import resolve_names, OUT_COLS
from src.model.train import LGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS
from src.model.train_fraud import build_frame
from labels.build_leads_fraud import screen

GEO = ["primary_taxonomy", "practice_state"]
OUT = c.MODEL_DATA_DIR / "fraud_positive"


def conc(leads, prov_state, tag):
    az = leads["npi_list"].map(lambda x: prov_state.get(str(x).split("|")[0], "") == "AZ").mean()
    top_tax = leads["specialty"].value_counts(normalize=True).head(3)
    log(f"  {tag:22s} n={len(leads):>4}  AZ={az:.1%}  top-tax: " +
        ", ".join(f"{k} {v:.0%}" for k, v in top_tax.items()))


def main():
    log("[1/4] Fit fraud_positive WITHOUT state+taxonomy")
    X, base = build_frame()
    Xg = X.drop(columns=GEO)
    cats = [col for col in c.CATEGORICAL_FEATURES if col in Xg.columns]
    lab = (base["pos"] | base["neg"]).values
    Xl, yl = Xg[lab], base.loc[lab, "pos"].astype("int8").values
    Xt, Xv, yt, yv = train_test_split(Xl, yl, test_size=c.VAL_FRACTION, stratify=yl, random_state=c.SEED)
    p = dict(LGB_PARAMS); p["seed"] = c.SEED
    b = lgb.train(p, lgb.Dataset(Xt, label=yt, categorical_feature=cats), NUM_BOOST_ROUND,
                  valid_sets=[lgb.Dataset(Xv, label=yv)],
                  callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)])
    log(f"    best_iter {b.best_iteration} on {len(Xl):,} labeled rows ({Xg.shape[1]} feats)")

    log("[2/4] Score universe + rollup")
    df = pd.read_parquet(c.SCORED_UNIVERSE_PARQUET).reset_index(drop=True)
    assert (df["npi"].values == base["npi"].values).all(), "row misalignment"
    prov = df[PROVIDER_CONTEXT_COLS].copy()
    prov.insert(1, "model_score", b.predict(Xg))
    prov["score_reliable"] = ~prov["not_scored"].fillna(True)
    prov = prov.sort_values("model_score", ascending=False).reset_index(drop=True)
    comp = rollup_to_company(prov)

    log("[3/4] $10M + top-5000 + name resolution + screens")
    elig = comp[(comp["company_net_paid"] >= 10_000_000) & (comp["company_model_score_max"] >= 0)]
    leads = elig.sort_values(["company_model_score_max", "company_net_paid"], ascending=False).head(5000).copy()
    leads = resolve_names(leads, prov[["npi", "model_score", "org_legal_name"]])
    leads["rank"] = range(1, len(leads) + 1)
    leads["n_fraud_pos_npis"] = 0
    kept, _ = screen(leads[OUT_COLS + ["n_fraud_pos_npis"]], prov)
    kept.to_csv(OUT / "leads_nogeo_screened.csv", index=False)

    log("[4/4] Concentration comparison (dominant-NPI state + taxonomy)")
    pstate = df.sort_values("net_paid", ascending=False).drop_duplicates("npi").set_index("npi")["practice_state"]
    conc(kept, pstate, "NO-GEO fraud_positive")
    full = pd.read_csv(OUT / "leads_fraud_screened.csv", dtype={"npi_list": str})
    conc(full, pstate, "FULL fraud_positive")
    cl = pd.read_csv(c.MODEL_DATA_DIR / "output" / "final" / "model_leads_CLEANED.csv", dtype={"npi_list": str})
    conc(cl, pstate, "CLEANED (production)")

    # how much did the lead set change vs full fraud_positive?
    k = lambda s: frozenset(str(s).split("|"))
    nk, fk = set(kept["npi_list"].map(k)), set(full["npi_list"].map(k))
    log(f"\n  no-geo vs full fraud_positive: shared {len(nk & fk)}, "
        f"no-geo-only {len(nk - fk)}, dropped-from-full {len(fk - nk)}")
    log(f"  → {OUT}/leads_nogeo_screened.csv")


if __name__ == "__main__":
    main()
