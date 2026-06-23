"""
train_model_d.py (labels) — Model D = trained on ALL exclusions (fraud-relevant LEIE
∪ all 38 state Medicaid exclusion lists). Head-to-head vs A (all-LEIE) and C
(fraud_positive = LEIE+AZ+NV), on the identical temporal harness.

Fair design: each TRAINING label (A/C/D) is trained on its own positives ≤2023 +
company-disjoint clean negatives, with early stopping on an internal random holdout
(NOT a temporal target — avoids target leakage). Every trained model is then scored on
THREE common held-out future targets (val 2024 / test 2025-26):
  T_leie  = future fraud-relevant LEIE  (neutral federal ground truth — the key test)
  T_state = future state exclusions     (the new ground truth D adds)
  T_all   = future any-exclusion (D's label)
So the comparison isn't biased toward any one model's training label.

Run: python -m labels.train_model_d
"""

import re
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from src.model import config as c
from src.model.data import build_feature_matrix
from src.model.train import LGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS

TRAIN_MAX, VAL_Y, TEST_Y = 2023, 2024, (2025, 2026)
SEEDS = [0, 1, 2]


def log(m=""): print(m, flush=True)
def neg_split(npi): h = int(npi) % 100; return "train" if h < 70 else ("val" if h < 85 else "test")


def leie_years():
    df = pd.read_csv(c.DATA_DIR/"preclean"/"Caught.csv", dtype=str, keep_default_na=False)
    df = df[df["NPI"].str.fullmatch(r"[12]\d{9}")]
    df["yr"] = pd.to_datetime(df["EXCLDATE"], format="%Y%m%d", errors="coerce").dt.year
    df = df[df["yr"].between(2006, 2026)]
    FR = {"1128a1","1128a2","1128a3","1128b1","1128b2","1128b3","1128b7"}
    any_y = df.groupby("NPI")["yr"].min().astype(int).to_dict()
    fr = df[df["EXCLTYPE"].isin(FR)]
    fr_y = fr.groupby("NPI")["yr"].min().astype(int).to_dict()
    return any_y, fr_y


def main():
    log("[1/4] Loading features, labels, all-state exclusions")
    df = pd.read_parquet(c.SCORED_UNIVERSE_PARQUET).reset_index(drop=True)
    X = build_feature_matrix(df).reset_index(drop=True)
    npis = df["npi"].values
    base = df[["npi","anomaly_score","not_scored"]].copy()
    base["company_id"] = base["npi"].map(
        pd.read_parquet(c.NPI_TO_COMPANY_MAP).set_index("npi")["company_id"])
    universe = set(npis)
    exp = pd.read_parquet(c.MODEL_DATA_DIR/"labels"/"expanded_labels.parquet").set_index("npi")
    any_y, fr_y = leie_years()
    st = pd.read_csv("/tmp/stateexcl/state_npis.csv", dtype={"npi":str})
    st = st[st["npi"].isin(universe)]; st["year"]=pd.to_numeric(st["year"],errors="coerce")
    state_npis = set(st["npi"]); state_y = st.dropna(subset=["year"]).groupby("npi")["year"].min().astype(int).to_dict()

    idx = pd.Series(np.arange(len(npis)), index=npis)
    def vec(s): m=np.zeros(len(npis),bool); ii=idx.reindex([x for x in s if x in idx.index]).dropna().astype(int).values; m[ii]=True; return m
    def yarr(dic):
        a=np.full(len(npis),np.nan)
        for k,v in dic.items():
            if k in idx.index: a[idx[k]]=v
        return a
    leie_any=vec(set(any_y)); leie_fraud=vec(set(fr_y)); state=vec(state_npis)
    fp = exp["fraud_positive"].fillna(False).reindex(npis).fillna(False).values
    y_any=yarr(any_y); y_fr=yarr(fr_y); y_state=yarr(state_y)
    y_fp = exp["excl_year_all"].reindex(npis).values.astype(float)
    # D positives + year = min(fraud-LEIE, state)
    Dpos = leie_fraud | state
    y_d = np.fmin(np.where(leie_fraud,y_fr,np.nan), np.where(state,y_state,np.nan))
    y_d = np.where(np.isnan(y_d), np.where(leie_fraud,y_fr,y_state), y_d)

    # negatives: confident-clean, not on ANY exclusion list, company-disjoint from any positive
    excluded = leie_any | state | fp
    posset = set(npis[Dpos | fp | leie_any])
    pos_company = set(base.loc[base["npi"].isin(posset),"company_id"].dropna())
    clean = (base["anomaly_score"].values==0) & (~base["not_scored"].fillna(True).values) & (~excluded) & (~base["company_id"].isin(pos_company).values)
    spl = np.array([neg_split(n) if clean[i] else "" for i,n in enumerate(npis)],dtype=object)
    neg_tr, neg_va, neg_te = spl=="train", spl=="val", spl=="test"
    log(f"    universe {len(npis):,} | clean negs tr/va/te {neg_tr.sum():,}/{neg_va.sum():,}/{neg_te.sum():,}")

    # training labels: (name, pos_mask, year_array)
    TRAIN = {"A_all_LEIE":(leie_any,y_any), "C_fraud_pos":(fp,y_fp), "D_all_excl":(Dpos,y_d)}
    # eval targets: (name, pos_mask, year_array)
    TARGETS = {"T_leie_fraud":(leie_fraud,y_fr), "T_state":(state,y_state), "T_all_excl":(Dpos,y_d)}
    for t,(m,_) in TARGETS.items(): log(f"    target {t}: {int(m.sum())} positives")

    log("[2/4] Training A/C/D (3 seeds; early-stop on internal holdout)")
    def metrics(y,s):
        order=np.argsort(-s); yr=y[order]
        return average_precision_score(y,s), float(yr[:50].sum()/50)
    results={}  # (label,target)->list of (val_pr,test_pr,val_p50)
    for lname,(pmask,yarr_) in TRAIN.items():
        trpos = pmask & (yarr_<=TRAIN_MAX)
        tr = trpos | neg_tr
        Xtr_all = X[tr]; ytr_all = trpos[tr].astype(int)
        cats=[col for col in c.CATEGORICAL_FEATURES if col in X.columns]
        for seed in SEEDS:
            Xa,Xb,ya,yb = train_test_split(Xtr_all,ytr_all,test_size=0.2,stratify=ytr_all,random_state=seed)
            p=dict(LGB_PARAMS); p["seed"]=seed
            b=lgb.train(p,lgb.Dataset(Xa,label=ya,categorical_feature=cats),NUM_BOOST_ROUND,
                        valid_sets=[lgb.Dataset(Xb,label=yb)],
                        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS,verbose=False),lgb.log_evaluation(0)])
            for tname,(tmask,tyr) in TARGETS.items():
                va=(tmask&(tyr==VAL_Y))|neg_va; te=(tmask&np.isin(tyr,TEST_Y))|neg_te
                vp,vp50=metrics(tmask[va].astype(int),b.predict(X[va]))
                tp,_=metrics(tmask[te].astype(int),b.predict(X[te]))
                results.setdefault((lname,tname),[]).append((vp,tp,vp50))
        log(f"    {lname}: trained (train+ {int(trpos.sum())})")

    log("\n[3/4] " + "="*70 + "\nRESULTS — val/test PR-AUC (3-seed mean), rows=training label, cols=eval target\n"+"="*70)
    for tname in TARGETS:
        log(f"\n  EVAL TARGET = {tname}  (val pos {int(((TARGETS[tname][0])&(TARGETS[tname][1]==VAL_Y)).sum())})")
        log(f"  {'train label':14s} {'val PR-AUC':>20s} {'test PR-AUC':>20s} {'val P@50':>9s}")
        for lname in TRAIN:
            r=np.array(results[(lname,tname)])
            log(f"  {lname:14s} {r[:,0].mean():>9.3f}[{r[:,0].min():.3f}-{r[:,0].max():.3f}] "
                f"{r[:,1].mean():>9.3f}[{r[:,1].min():.3f}-{r[:,1].max():.3f}] {r[:,2].mean():>9.2f}")
    log("\n[4/4] done")


if __name__ == "__main__":
    main()
