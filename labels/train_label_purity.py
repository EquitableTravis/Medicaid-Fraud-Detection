"""
Model D label-purity A/B/C on the NEWEST model config (no-geo + care-model
cluster features, temporal protocol from labels/train_model_d_nogeo.py).

Arms (only the positive set varies; features/negatives/splits/HP identical):
  D_original   = fraud-relevant LEIE (a1,a2,a3,b1,b2,b3,b7) | ALL state-list NPIs
                 (what the newest model actually trains on today)
  D_conviction = strict fraud convictions only:
                 LEIE {a1,a3,b1,b7} | state records whose basis text is
                 explicitly fraud (OpenSanctions description evidence)
  D_convic_az  = D_conviction | AZ AHCCCS suspensions (statutory basis =
                 credible allegation of fraud; allegation-grade, not conviction)

All arms evaluated on the SAME neutral target: future STRICT-fraud LEIE
exclusions (val=2024, test=2025-26) — "does it find actual fraudsters it has
never seen." Positives with unknown/non-fraud basis are dropped from training
in the corrected arms but stay OUT of negatives in every arm (same `excluded`
set), so negatives are identical across arms.

Durable outputs -> ~/Desktop/Data/Model/labels/:
  state_exclusion_basis.csv  (per-NPI basis category + evidence, all 30 states)
  fraud_conviction_labels.csv (the corrected positive sets with tier + year)
"""
import json
import sys
from pathlib import Path

REPO = Path.home() / "Desktop" / "medicaid-fraud-detection"
sys.path.insert(0, str(REPO))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from src.model import config as c
from src.model.data import build_feature_matrix
from src.model.train import LGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS
from labels.repeer_and_train_d import care_clusters, repeer, neg_split

TRAIN_MAX, VAL_Y, TEST_Y = 2023, 2024, (2025, 2026)
SEEDS = [0, 1, 2]
GEO = ["primary_taxonomy", "practice_state"]
FR = {"1128a1", "1128a2", "1128a3", "1128b1", "1128b2", "1128b3", "1128b7"}
STRICT = {"1128a1", "1128a3", "1128b1", "1128b7"}
BASIS_CSV = c.MODEL_DATA_DIR / "labels" / "state_exclusion_basis.csv"
LABELS_DIR = c.MODEL_DATA_DIR / "labels"


def log(m=""):
    print(m, flush=True)


def main():
    log("[1] universe + features (no-geo + cluster cols, the newest-model config)")
    df = pd.read_parquet(c.SCORED_UNIVERSE_PARQUET).reset_index(drop=True)
    X = build_feature_matrix(df).reset_index(drop=True)
    cl = care_clusters(df)
    Xc = pd.concat([X.drop(columns=[g for g in GEO if g in X.columns]),
                    repeer(df, cl).reset_index(drop=True)], axis=1)
    cats = [col for col in c.CATEGORICAL_FEATURES if col in Xc.columns]
    log(f"    {Xc.shape[1]} features ({X.shape[1]} base - geo + cluster cols)")

    npis = df["npi"].values
    idx = pd.Series(np.arange(len(npis)), index=npis)
    company_id = df["npi"].map(pd.read_parquet(c.NPI_TO_COMPANY_MAP).set_index("npi")["company_id"])

    def vec(s):
        m = np.zeros(len(npis), bool)
        ii = idx.reindex([x for x in s if x in idx.index]).dropna().astype(int).values
        m[ii] = True
        return m

    def yarr(dic):
        a = np.full(len(npis), np.nan)
        for k, v in dic.items():
            if k in idx.index:
                a[idx[k]] = v
        return a

    log("[2] labels")
    cg = pd.read_csv(c.DATA_DIR / "preclean" / "Caught.csv", dtype=str, keep_default_na=False)
    cg = cg[cg["NPI"].str.fullmatch(r"[12]\d{9}")]
    cg["yr"] = pd.to_datetime(cg["EXCLDATE"], format="%Y%m%d", errors="coerce").dt.year
    cg = cg[cg["yr"].between(2006, 2026)]
    any_y = cg.groupby("NPI")["yr"].min().astype(int).to_dict()
    fr_y = cg[cg["EXCLTYPE"].isin(FR)].groupby("NPI")["yr"].min().astype(int).to_dict()
    strict_y = cg[cg["EXCLTYPE"].isin(STRICT)].groupby("NPI")["yr"].min().astype(int).to_dict()

    st = pd.read_csv(LABELS_DIR / "all_state_exclusions_npis.csv", dtype={"npi": str})
    st = st[st["npi"].isin(set(npis))]
    st["year"] = pd.to_numeric(st["year"], errors="coerce")
    state_y = st.dropna(subset=["year"]).groupby("npi")["year"].min().astype(int).to_dict()
    state_all = set(st["npi"])

    basis = pd.read_csv(BASIS_CSV, dtype={"npi": str})
    state_fraud = set(basis.loc[basis["category"] == "fraud", "npi"]) & set(npis)
    az_alleg = set(st.loc[st["state"] == "AZ", "npi"])
    fp = set(pd.read_parquet(LABELS_DIR / "expanded_labels.parquet").query("fraud_positive")["npi"])

    leie_fraud = vec(set(fr_y)); leie_strict = vec(set(strict_y)); leie_any = vec(set(any_y))
    state_m = vec(state_all); state_fraud_m = vec(state_fraud); az_m = vec(az_alleg)

    y_fr = yarr(fr_y); y_strict = yarr(strict_y); y_state = yarr(state_y)

    def pos_year(mask_leie, y_leie, mask_state):
        y = np.where(mask_leie, y_leie, np.nan)
        ys = np.where(mask_state, y_state, np.nan)
        return np.fmin(np.where(np.isnan(y), ys, y), np.where(np.isnan(ys), y, ys))

    ARMS = {
        "D_original":   (leie_fraud | state_m,                 pos_year(leie_fraud, y_fr, state_m)),
        "D_conviction": (leie_strict | state_fraud_m,          pos_year(leie_strict, y_strict, state_fraud_m)),
        "D_convic_az":  (leie_strict | state_fraud_m | az_m,   pos_year(leie_strict, y_strict, state_fraud_m | az_m)),
    }
    for name, (m, _) in ARMS.items():
        log(f"    {name}: {int(m.sum())} positives")

    # negatives identical across arms: clean, on NO list, company-disjoint from anything list-adjacent
    excluded = leie_any | state_m | vec(fp)
    pos_company = set(company_id[excluded].dropna())
    clean = ((df["anomaly_score"].values == 0) & (~df["not_scored"].fillna(True).values)
             & (~excluded) & (~company_id.isin(pos_company).values))
    spl = np.array([neg_split(n) if clean[i] else "" for i, n in enumerate(npis)], dtype=object)
    neg_tr, neg_va, neg_te = spl == "train", spl == "val", spl == "test"
    log(f"    clean negatives: {int(clean.sum()):,} (identical across arms)")

    # neutral target: future STRICT-fraud LEIE
    t_mask, t_year = leie_strict, y_strict

    log("[3] training 3 arms x 3 seeds (temporal, neutral future strict-fraud target)")
    results = {}
    for name, (pos, y_pos) in ARMS.items():
        trpos = pos & (y_pos <= TRAIN_MAX)
        tr = trpos | neg_tr
        arm = []
        for seed in SEEDS:
            Xa, Xb, ya, yb = train_test_split(Xc[tr], trpos[tr].astype(int), test_size=0.2,
                                              stratify=trpos[tr].astype(int), random_state=seed)
            p = dict(LGB_PARAMS); p["seed"] = seed
            b = lgb.train(p, lgb.Dataset(Xa, label=ya, categorical_feature=cats), NUM_BOOST_ROUND,
                          valid_sets=[lgb.Dataset(Xb, label=yb)],
                          callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                                     lgb.log_evaluation(0)])
            va = (t_mask & (t_year == VAL_Y)) | neg_va
            te = (t_mask & np.isin(t_year, TEST_Y)) | neg_te
            sv, stt = b.predict(Xc[va]), b.predict(Xc[te])
            yv, yt = t_mask[va].astype(int), t_mask[te].astype(int)
            ov = np.argsort(-sv)
            arm.append((average_precision_score(yv, sv), average_precision_score(yt, stt),
                        float(yv[ov][:50].sum() / 50)))
            log(f"    {name} seed {seed}: val {arm[-1][0]:.3f} test {arm[-1][1]:.3f} P@50 {arm[-1][2]:.2f}")
        results[name] = arm

    log("\n" + "=" * 72)
    log(f"{'arm':14s} {'n_pos':>6s} {'val PR-AUC (mean [min-max])':>30s} {'test':>8s} {'valP@50':>8s}")
    for name, arm in results.items():
        a = np.array(arm)
        log(f"{name:14s} {int(ARMS[name][0].sum()):>6d} "
            f"{a[:,0].mean():>10.3f} [{a[:,0].min():.3f}-{a[:,0].max():.3f}] "
            f"{a[:,1].mean():>8.3f} {a[:,2].mean():>8.2f}")

    log("\n[4] durable label artifacts")
    basis.to_csv(LABELS_DIR / "state_exclusion_basis.csv", index=False)
    rows = []
    for npi in set(np.array(npis)[ARMS["D_convic_az"][0]]):
        i = idx[npi]
        tier = ("leie_strict_fraud" if leie_strict[i]
                else "state_fraud_text" if state_fraud_m[i] else "az_credible_allegation")
        yr = ARMS["D_convic_az"][1][i]
        rows.append((npi, tier, int(yr) if not np.isnan(yr) else ""))
    pd.DataFrame(rows, columns=["npi", "tier", "year"]).to_csv(
        LABELS_DIR / "fraud_conviction_labels.csv", index=False)
    log(f"    -> {LABELS_DIR/'state_exclusion_basis.csv'}")
    log(f"    -> {LABELS_DIR/'fraud_conviction_labels.csv'}")
    (LABELS_DIR / "label_purity_results.json").write_text(
        json.dumps({k: v for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
