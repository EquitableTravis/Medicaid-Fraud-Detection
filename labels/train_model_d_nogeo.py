"""
train_model_d_nogeo.py (labels) — Model D (all exclusions: fraud-relevant LEIE ∪ all 38
state lists) WITH vs WITHOUT the geo/sector features (practice_state, primary_taxonomy).
Same temporal harness as train_model_d: 3 seeds, internal early-stop (no target leak),
evaluated on the neutral future-LEIE target + state + all-exclusion targets.

Run: python -m labels.train_model_d_nogeo
"""
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from src.model import config as c
from src.model.data import build_feature_matrix
from src.model.train import LGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS

TRAIN_MAX, VAL_Y, TEST_Y = 2023, 2024, (2025, 2026)
SEEDS = [0,1,2]; GEO = ["primary_taxonomy","practice_state"]
FR = {"1128a1","1128a2","1128a3","1128b1","1128b2","1128b3","1128b7"}
def log(m=""): print(m, flush=True)
def neg_split(npi): h=int(npi)%100; return "train" if h<70 else ("val" if h<85 else "test")

def main():
    df = pd.read_parquet(c.SCORED_UNIVERSE_PARQUET).reset_index(drop=True)
    X = build_feature_matrix(df).reset_index(drop=True)
    npis = df["npi"].values; uni=set(npis)
    company_id = df["npi"].map(pd.read_parquet(c.NPI_TO_COMPANY_MAP).set_index("npi")["company_id"])
    # leie years
    cg = pd.read_csv(c.DATA_DIR/"preclean"/"Caught.csv", dtype=str, keep_default_na=False)
    cg = cg[cg["NPI"].str.fullmatch(r"[12]\d{9}")]; cg["yr"]=pd.to_datetime(cg["EXCLDATE"],format="%Y%m%d",errors="coerce").dt.year
    cg=cg[cg["yr"].between(2006,2026)]
    any_y=cg.groupby("NPI")["yr"].min().astype(int).to_dict()
    fr_y=cg[cg["EXCLTYPE"].isin(FR)].groupby("NPI")["yr"].min().astype(int).to_dict()
    st=pd.read_csv(c.MODEL_DATA_DIR/"labels"/"all_state_exclusions_npis.csv",dtype={"npi":str}); st=st[st["npi"].isin(uni)]
    st["year"]=pd.to_numeric(st["year"],errors="coerce"); state_npis=set(st["npi"])
    state_y=st.dropna(subset=["year"]).groupby("npi")["year"].min().astype(int).to_dict()
    fp=set(pd.read_parquet(c.MODEL_DATA_DIR/"labels"/"expanded_labels.parquet").query("fraud_positive")["npi"])
    idx=pd.Series(np.arange(len(npis)),index=npis)
    def vec(s): m=np.zeros(len(npis),bool); ii=idx.reindex([x for x in s if x in idx.index]).dropna().astype(int).values; m[ii]=True; return m
    def yarr(dic):
        a=np.full(len(npis),np.nan)
        for k,v in dic.items():
            if k in idx.index: a[idx[k]]=v
        return a
    leie_any=vec(set(any_y)); leie_fraud=vec(set(fr_y)); state=vec(state_npis)
    y_fr=yarr(fr_y); y_state=yarr(state_y)
    Dpos = leie_fraud|state
    y_d=np.fmin(np.where(leie_fraud,y_fr,np.nan),np.where(state,y_state,np.nan))
    y_d=np.where(np.isnan(y_d),np.where(leie_fraud,y_fr,y_state),y_d)
    # negatives: clean, not on any list, company-disjoint
    excluded=leie_any|state|vec(fp)
    pos_company=set(company_id[Dpos|vec(fp)|leie_any].dropna())
    clean=(df["anomaly_score"].values==0)&(~df["not_scored"].fillna(True).values)&(~excluded)&(~company_id.isin(pos_company).values)
    spl=np.array([neg_split(n) if clean[i] else "" for i,n in enumerate(npis)],dtype=object)
    neg_tr,neg_va,neg_te=spl=="train",spl=="val",spl=="test"
    TARGETS={"T_leie_fraud (NEUTRAL)":(leie_fraud,y_fr),"T_state":(state,y_state),"T_all_excl":(Dpos,y_d)}
    log(f"    D positives {int(Dpos.sum())} | clean negs {int(clean.sum()):,}")

    def metrics(y,s):
        order=np.argsort(-s); yr=y[order]; return average_precision_score(y,s), float(yr[:50].sum()/50)
    def run(feat_drop, tag):
        Xf = X.drop(columns=feat_drop) if feat_drop else X
        cats=[col for col in c.CATEGORICAL_FEATURES if col in Xf.columns]
        trpos=Dpos&(y_d<=TRAIN_MAX); tr=trpos|neg_tr
        out={}
        for seed in SEEDS:
            Xa,Xb,ya,yb=train_test_split(Xf[tr],trpos[tr].astype(int),test_size=0.2,stratify=trpos[tr].astype(int),random_state=seed)
            p=dict(LGB_PARAMS); p["seed"]=seed
            b=lgb.train(p,lgb.Dataset(Xa,label=ya,categorical_feature=cats),NUM_BOOST_ROUND,valid_sets=[lgb.Dataset(Xb,label=yb)],
                        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS,verbose=False),lgb.log_evaluation(0)])
            for tn,(tm,ty) in TARGETS.items():
                va=(tm&(ty==VAL_Y))|neg_va; te=(tm&np.isin(ty,TEST_Y))|neg_te
                vp,vp50=metrics(tm[va].astype(int),b.predict(Xf[va])); tp,_=metrics(tm[te].astype(int),b.predict(Xf[te]))
                out.setdefault(tn,[]).append((vp,tp,vp50))
        log(f"    {tag} ({Xf.shape[1]} feats) trained")
        return out

    log("[train] Model D — full vs no-geo (3 seeds)")
    full=run(None,"D full"); nogeo=run(GEO,"D no-geo")
    log("\n"+"="*68+"\nMODEL D: full features vs no-geo (state+taxonomy dropped)\n"+"="*68)
    for tn in TARGETS:
        f=np.array(full[tn]); g=np.array(nogeo[tn])
        log(f"\n  {tn}")
        log(f"    {'':10s} {'val PR-AUC':>20s} {'test PR-AUC':>20s} {'val P@50':>9s}")
        log(f"    D full     {f[:,0].mean():>9.3f}[{f[:,0].min():.3f}-{f[:,0].max():.3f}] {f[:,1].mean():>9.3f}[{f[:,1].min():.3f}-{f[:,1].max():.3f}] {f[:,2].mean():>9.2f}")
        log(f"    D no-geo   {g[:,0].mean():>9.3f}[{g[:,0].min():.3f}-{g[:,0].max():.3f}] {g[:,1].mean():>9.3f}[{g[:,1].min():.3f}-{g[:,1].max():.3f}] {g[:,2].mean():>9.2f}")

if __name__=="__main__":
    main()
