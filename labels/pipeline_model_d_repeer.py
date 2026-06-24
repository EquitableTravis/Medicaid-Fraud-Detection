"""
pipeline_model_d_repeer.py (labels, branch feat/care-model-peering) — regenerate Model D
leads using the BOTH feature set (taxonomy-peer + care-model-cluster-peer features), and
check whether the FQHC/RHC taxonomy-mismatch artifacts actually drop out of the top.

Same filter stack as pipeline_model_d: reliability gate -> company rollup -> $10M ->
score>=0.90 -> hardened screens -> NPPES names. Writes to ~/Desktop/Model_D_repeer_Leads/
and prints a before/after comparison vs the original (taxonomy-only) Model D leads.

Run: python -m labels.pipeline_model_d_repeer
"""
import re
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from src.model import config as c
from src.model.data import build_feature_matrix, log
from src.model.score import rollup_to_company, PROVIDER_CONTEXT_COLS
from src.model.export_leads import resolve_names, OUT_COLS
from src.model.train import LGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS
from labels.build_leads_fraud import screen
from labels.repeer_and_train_d import care_clusters, repeer

OUTDIR = Path.home()/"Desktop"/"Model_D_repeer_Leads"
OLD = Path.home()/"Desktop"/"Model_D_Leads"/"leads_final.csv"
PROVIDER_DIM = c.DATA_DIR/"integrated"/"provider_dim.parquet"
SCORE_MIN, MIN_NET = 0.90, 10_000_000
FR={"1128a1","1128a2","1128a3","1128b1","1128b2","1128b3","1128b7"}
FQHC_RE = re.compile(r"COMMUNITY HEALTH|HEALTH CENTER|RURAL HEALTH|FQHC|HEALTH CARE CENTER", re.I)
def band(s): return "high" if s>=0.99 else ("med" if s>=0.90 else "low")

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log("[1/6] Build BOTH feature set (taxonomy-peer + care-model-cluster-peer)")
    df=pd.read_parquet(c.SCORED_UNIVERSE_PARQUET).reset_index(drop=True)
    Xbase=build_feature_matrix(df).reset_index(drop=True)
    rep=repeer(df, care_clusters(df)).reset_index(drop=True)
    X=pd.concat([Xbase, rep], axis=1)
    log(f"    features: {Xbase.shape[1]} taxonomy + {rep.shape[1]} care-model = {X.shape[1]}")

    npis=df["npi"].values; uni=set(npis)
    cmap=pd.read_parquet(c.NPI_TO_COMPANY_MAP).set_index("npi")["company_id"]; company_id=df["npi"].map(cmap)
    cg=pd.read_csv(c.DATA_DIR/"preclean"/"Caught.csv",dtype=str,keep_default_na=False); cg=cg[cg["NPI"].str.fullmatch(r"[12]\d{9}")]
    leie_any=set(cg["NPI"]); leie_fraud=set(cg.loc[cg["EXCLTYPE"].isin(FR),"NPI"])
    st=pd.read_csv(c.MODEL_DATA_DIR/"labels"/"all_state_exclusions_npis.csv",dtype={"npi":str}); state_npis=set(st.loc[st["npi"].isin(uni),"npi"])
    fp=set(pd.read_parquet(c.MODEL_DATA_DIR/"labels"/"expanded_labels.parquet").query("fraud_positive")["npi"])
    ns=pd.Series(npis)
    pos=ns.isin(leie_fraud|state_npis).values
    excl=ns.isin(leie_any|state_npis|fp).values
    pos_company=set(company_id[pos].dropna())
    clean=(df["anomaly_score"].values==0)&(~df["not_scored"].fillna(True).values)&(~excl)&(~company_id.isin(pos_company).values)
    log(f"    {pos.sum():,} positives | {clean.sum():,} clean negatives")

    log("[2/6] Fit Model D (BOTH features) on all labeled rows")
    lab=pos|clean; Xl,yl=X[lab],pos[lab].astype("int8"); cats=[col for col in c.CATEGORICAL_FEATURES if col in X.columns]
    Xt,Xv,yt,yv=train_test_split(Xl,yl,test_size=c.VAL_FRACTION,stratify=yl,random_state=c.SEED)
    p=dict(LGB_PARAMS); p["seed"]=c.SEED
    b=lgb.train(p,lgb.Dataset(Xt,label=yt,categorical_feature=cats),NUM_BOOST_ROUND,valid_sets=[lgb.Dataset(Xv,label=yv)],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS,verbose=False),lgb.log_evaluation(0)])
    log(f"    best_iter {b.best_iteration}")

    log("[3/6] Score + reliability gate + rollup")
    prov=df[PROVIDER_CONTEXT_COLS].copy(); prov.insert(1,"model_score",b.predict(X))
    prov["score_reliable"]=~prov["not_scored"].fillna(True)
    prov=prov.sort_values("model_score",ascending=False).reset_index(drop=True)
    comp=rollup_to_company(prov)

    log(f"[4/6] $10M + score>={SCORE_MIN} + screens")
    leads=comp[(comp["company_net_paid"]>=MIN_NET)&(comp["company_model_score_max"]>=SCORE_MIN)].copy()
    leads=leads.sort_values(["company_model_score_max","company_net_paid"],ascending=False)
    leads=resolve_names(leads, prov[["npi","model_score","org_legal_name"]]); leads["rank"]=range(1,len(leads)+1)
    excl_all=leie_any|state_npis|fp
    leads["n_excluded_npis"]=leads["npi_list"].map(lambda s: sum(n in excl_all for n in str(s).split("|")))
    leads["n_fraud_pos_npis"]=leads["n_excluded_npis"]
    kept,audit=screen(leads[OUT_COLS+["n_fraud_pos_npis","n_excluded_npis"]],prov); kept=kept.reset_index(drop=True)

    log("[5/6] names + bands; write")
    pdm=pd.read_parquet(PROVIDER_DIM,columns=["npi","provider_name"]); name_of=pdm.assign(npi=pdm["npi"].astype(str)).set_index("npi")["provider_name"]
    unk=kept["company_name"].astype(str).str.startswith("UNKNOWN NAME")
    res=kept.loc[unk,"company_name"].str.extract(r"NPI (\d+)")[0].map(name_of)
    kept.loc[unk,"company_name"]=res.where(res.notna()&(res.str.strip()!=""),kept.loc[unk,"company_name"]).values
    pstate=df.sort_values("net_paid",ascending=False).drop_duplicates("npi").set_index("npi")["practice_state"]
    dnpi=kept["npi_list"].map(lambda s:str(s).split("|")[0])
    kept["state"]=dnpi.map(pstate); kept["confidence"]=kept["company_model_score_max"].map(band); kept["rank"]=range(1,len(kept)+1)
    cols=["rank","company_name","confidence","company_model_score_max","company_net_paid","n_npis","n_excluded_npis","specialty","state","merge_confidence","npi_list"]
    final=kept[cols].rename(columns={"company_model_score_max":"model_score"})
    final.to_csv(OUTDIR/"leads_final.csv",index=False)
    final[final["n_excluded_npis"]==0].assign(rank=lambda d:range(1,len(d)+1)).to_csv(OUTDIR/"leads_final_NOVEL.csv",index=False)

    log("[6/6] "+"="*60+"\nFQHC/RHC artifact check: taxonomy-only (old) vs BOTH (new)\n"+"="*60)
    old=pd.read_csv(OLD,dtype=str); old["rank"]=pd.to_numeric(old["rank"],errors="coerce")
    def fqhc_in_top(d,col,n):
        t=d.head(n); return int(t[col].astype(str).str.contains(FQHC_RE).sum())
    log(f"  total leads: old {len(old)} -> new {len(final)}")
    for n in (50,100,250):
        log(f"  FQHC/RHC-named in top {n}: old {fqhc_in_top(old,'company_name',n)} -> new {fqhc_in_top(final,'company_name',n)}")
    log("\n  the 3 example FQHCs (rank old -> new; 'gone' = no longer in list):")
    for name in ["COMMUNITY HEALTH CENTERS OF AMERICA","STALLANT MEDICAL GROUP","VALLEY HEALTH PARTNERS"]:
        o=old[old["company_name"].astype(str).str.upper().str.startswith(name[:22])]
        nw=final[final["company_name"].astype(str).str.upper().str.startswith(name[:22])]
        orank=int(o["rank"].iloc[0]) if len(o) else None
        nrank=int(nw["rank"].iloc[0]) if len(nw) else "gone"
        log(f"    {name[:34]:34s} rank {orank} -> {nrank}")
    log(f"\n  -> {OUTDIR}/leads_final.csv ({len(final)} leads, {int((final['confidence']=='high').sum())} high)")

if __name__=="__main__":
    main()
