"""
repeer_and_train_d.py (labels, branch feat/care-model-peering) — re-peer the rate
anomaly features by CARE MODEL instead of self-reported NPPES taxonomy, then retrain
Model D and compare.

Problem: the existing peer features (*_rz_tax / *_pct_tax / *_taxstate) z-score each
provider's billing rates against same-`primary_taxonomy` providers. But taxonomy is a
self-reported specialty code that mismatches reimbursement/care model — e.g. FQHCs coded
"Family Medicine" get compared to fee-for-service GPs, so their bundled per-encounter
rate looks anomalous purely from the mismatch.

Method (care-model peering): cluster providers by their OPERATIONAL signature — service
breadth (n_distinct_hcpcs), concentration (top_hcpcs_share, hcpcs_hhi), visit intensity
(lines/patient), scale (service_volume), tenure, entity type — NOT the payment-dollar
magnitudes we're testing. Then recompute robust-z + percentile of the 4 rate metrics
WITHIN each care-model cluster. FQHCs cluster with FQHCs, labs with labs, etc., so the
z measures "anomalous vs your true peers."

Compares Model D (all-exclusions label) trained on taxonomy-peered vs cluster-peered
features, on the neutral/state/all temporal targets.

Run: python -m labels.repeer_and_train_d
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from src.model import config as c
from src.model.data import build_feature_matrix
from src.model.train import LGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS

TRAIN_MAX, VAL_Y, TEST_Y = 2023, 2024, (2025, 2026); SEEDS=[0,1,2]; K=30
FR={"1128a1","1128a2","1128a3","1128b1","1128b2","1128b3","1128b7"}
METRICS=["net_paid","paid_per_claim_line","lines_per_patient_instance","paid_per_patient_instance"]
TAXCOLS=[f"{m}_{s}" for m in METRICS for s in ["rz_tax","pct_tax","rz_taxstate","pct_taxstate"]]
def log(m=""): print(m,flush=True)
def neg_split(npi): h=int(npi)%100; return "train" if h<70 else ("val" if h<85 else "test")

def care_clusters(df):
    f=pd.DataFrame({
      "ldistinct":np.log1p(pd.to_numeric(df["n_distinct_hcpcs"],errors="coerce")),
      "topshare":pd.to_numeric(df["top_hcpcs_paid_share"],errors="coerce"),
      "hhi":pd.to_numeric(df["hcpcs_hhi"],errors="coerce"),
      "lvol":np.log1p(pd.to_numeric(df["service_volume"],errors="coerce")),
      "lpp":pd.to_numeric(df["lines_per_patient_instance"],errors="coerce"),
      "lmonths":np.log1p(pd.to_numeric(df["n_active_months"],errors="coerce")),
      "indiv":df["entity_type"].astype(str).str.contains("Indiv",case=False).astype(float),
    }).replace([np.inf,-np.inf],np.nan)
    f=f.fillna(f.median())
    Xs=StandardScaler().fit_transform(f)
    rel=~df["not_scored"].fillna(True).values
    km=MiniBatchKMeans(n_clusters=K,random_state=0,n_init=10,batch_size=10000)
    km.fit(Xs[rel])
    return km.predict(Xs)

def repeer(df, cl):
    out={}
    cl=pd.Series(cl)
    for m in METRICS:
        x=pd.to_numeric(df[m],errors="coerce").values
        s=pd.Series(x)
        med=cl.map(s.groupby(cl).median()).values
        mad=cl.map(pd.Series(np.abs(x-med)).groupby(cl).median()).values
        rz=np.clip(np.nan_to_num((x-med)/(1.4826*np.where(mad==0,np.nan,mad)),nan=0.0),-50,50)
        pct=s.groupby(cl).rank(pct=True).values
        out[f"{m}_rz_cluster"]=rz; out[f"{m}_pct_cluster"]=np.nan_to_num(pct,nan=0.5)
    return pd.DataFrame(out)

def main():
    log("[1/5] Load + build care-model clusters")
    df=pd.read_parquet(c.SCORED_UNIVERSE_PARQUET).reset_index(drop=True)
    cl=care_clusters(df)
    df["_cluster"]=cl
    rep=repeer(df,cl)
    Xtax=build_feature_matrix(df).reset_index(drop=True)
    present=[col for col in TAXCOLS if col in Xtax.columns]
    Xcl=pd.concat([Xtax.drop(columns=present).reset_index(drop=True), rep.reset_index(drop=True)],axis=1)
    log(f"    {K} clusters | taxonomy-peer feats {Xtax.shape[1]} | cluster-peer feats {Xcl.shape[1]} (swapped {len(present)} tax cols for {rep.shape[1]} cluster cols)")

    # sanity: FQHC examples' paid_per_claim anomaly, taxonomy-z vs cluster-z
    fqhc=["COMMUNITY HEALTH CENTERS OF AMERICA","STALLANT MEDICAL GROUP INC","VALLEY HEALTH PARTNERS COMMUNITY HEALTH CENTER"]
    nm=df["org_legal_name"].astype(str).str.upper()
    for name in fqhc:
        i=nm[nm.str.startswith(name[:25])].index
        if len(i):
            i=i[0]
            log(f"    {name[:34]:34s} paid/claim z: tax {df.at[i,'paid_per_claim_line_rz_tax']:+.1f} -> cluster {rep.at[i,'paid_per_claim_line_rz_cluster']:+.1f}")

    log("[2/5] Build Model D label + targets + negatives")
    npis=df["npi"].values; uni=set(npis)
    company_id=df["npi"].map(pd.read_parquet(c.NPI_TO_COMPANY_MAP).set_index("npi")["company_id"])
    cg=pd.read_csv(c.DATA_DIR/"preclean"/"Caught.csv",dtype=str,keep_default_na=False); cg=cg[cg["NPI"].str.fullmatch(r"[12]\d{9}")]
    cg["yr"]=pd.to_datetime(cg["EXCLDATE"],format="%Y%m%d",errors="coerce").dt.year; cg=cg[cg["yr"].between(2006,2026)]
    any_y=cg.groupby("NPI")["yr"].min().astype(int).to_dict(); fr_y=cg[cg["EXCLTYPE"].isin(FR)].groupby("NPI")["yr"].min().astype(int).to_dict()
    st=pd.read_csv(c.MODEL_DATA_DIR/"labels"/"all_state_exclusions_npis.csv",dtype={"npi":str}); st=st[st["npi"].isin(uni)]
    st["year"]=pd.to_numeric(st["year"],errors="coerce"); state_npis=set(st["npi"]); state_y=st.dropna(subset=["year"]).groupby("npi")["year"].min().astype(int).to_dict()
    fp=set(pd.read_parquet(c.MODEL_DATA_DIR/"labels"/"expanded_labels.parquet").query("fraud_positive")["npi"])
    idx=pd.Series(np.arange(len(npis)),index=npis)
    def vec(s): m=np.zeros(len(npis),bool); ii=idx.reindex([x for x in s if x in idx.index]).dropna().astype(int).values; m[ii]=True; return m
    def yarr(d):
        a=np.full(len(npis),np.nan)
        for k,v in d.items():
            if k in idx.index: a[idx[k]]=v
        return a
    leie_any=vec(set(any_y)); leie_fraud=vec(set(fr_y)); state=vec(state_npis)
    y_fr=yarr(fr_y); y_state=yarr(state_y); Dpos=leie_fraud|state
    y_d=np.fmin(np.where(leie_fraud,y_fr,np.nan),np.where(state,y_state,np.nan)); y_d=np.where(np.isnan(y_d),np.where(leie_fraud,y_fr,y_state),y_d)
    excluded=leie_any|state|vec(fp); pos_company=set(company_id[Dpos|vec(fp)|leie_any].dropna())
    clean=(df["anomaly_score"].values==0)&(~df["not_scored"].fillna(True).values)&(~excluded)&(~company_id.isin(pos_company).values)
    spl=np.array([neg_split(n) if clean[i] else "" for i,n in enumerate(npis)],dtype=object)
    neg_tr,neg_va,neg_te=spl=="train",spl=="val",spl=="test"
    TARGETS={"T_leie_fraud (NEUTRAL)":(leie_fraud,y_fr),"T_state":(state,y_state),"T_all_excl":(Dpos,y_d)}

    def metrics(y,s): order=np.argsort(-s); yr=y[order]; return average_precision_score(y,s),float(yr[:50].sum()/50)
    def run(Xf,tag):
        cats=[col for col in c.CATEGORICAL_FEATURES if col in Xf.columns]
        trpos=Dpos&(y_d<=TRAIN_MAX); tr=trpos|neg_tr; out={}
        for seed in SEEDS:
            Xa,Xb,ya,yb=train_test_split(Xf[tr],trpos[tr].astype(int),test_size=0.2,stratify=trpos[tr].astype(int),random_state=seed)
            p=dict(LGB_PARAMS); p["seed"]=seed
            b=lgb.train(p,lgb.Dataset(Xa,label=ya,categorical_feature=cats),NUM_BOOST_ROUND,valid_sets=[lgb.Dataset(Xb,label=yb)],
                        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS,verbose=False),lgb.log_evaluation(0)])
            for tn,(tm,ty) in TARGETS.items():
                va=(tm&(ty==VAL_Y))|neg_va; te=(tm&np.isin(ty,TEST_Y))|neg_te
                vp,vp50=metrics(tm[va].astype(int),b.predict(Xf[va])); tp,_=metrics(tm[te].astype(int),b.predict(Xf[te]))
                out.setdefault(tn,[]).append((vp,tp,vp50))
        log(f"    {tag} trained"); return out

    Xboth=pd.concat([Xtax.reset_index(drop=True), rep.reset_index(drop=True)],axis=1)
    log("[3/5] Train Model D — taxonomy-peer (baseline) / care-model-only (replace) / BOTH (add)")
    base=run(Xtax,"taxonomy-peer"); rep_res=run(Xcl,"care-model-only"); both=run(Xboth,"both")
    log("\n[5/5] "+"="*66+"\nMODEL D peering: taxonomy / care-model-only / BOTH\n"+"="*66)
    for tn in TARGETS:
        b=np.array(base[tn]); g=np.array(rep_res[tn]); h=np.array(both[tn])
        log(f"\n  {tn}")
        log(f"    {'':18s} {'val PR-AUC':>18s} {'test PR-AUC':>18s} {'val P@50':>9s}")
        log(f"    taxonomy (base) {b[:,0].mean():>8.3f}[{b[:,0].min():.3f}-{b[:,0].max():.3f}] {b[:,1].mean():>8.3f}[{b[:,1].min():.3f}-{b[:,1].max():.3f}] {b[:,2].mean():>9.2f}")
        log(f"    care-model only {g[:,0].mean():>8.3f}[{g[:,0].min():.3f}-{g[:,0].max():.3f}] {g[:,1].mean():>8.3f}[{g[:,1].min():.3f}-{g[:,1].max():.3f}] {g[:,2].mean():>9.2f}")
        log(f"    BOTH (add)      {h[:,0].mean():>8.3f}[{h[:,0].min():.3f}-{h[:,0].max():.3f}] {h[:,1].mean():>8.3f}[{h[:,1].min():.3f}-{h[:,1].max():.3f}] {h[:,2].mean():>9.2f}")

if __name__=="__main__":
    main()
