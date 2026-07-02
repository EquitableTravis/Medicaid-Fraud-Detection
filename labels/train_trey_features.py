"""
train_trey_features.py (labels) — do Trey's extra features add signal?

Baseline = the newest model config: exclusion-risk label (fraud-relevant LEIE
∪ 30 state lists), no-geo feature set + care-model cluster cols, temporal
protocol, neutral future-strict-fraud target (labels/train_label_purity.py).

Treatment = same everything + the usable feature columns from
~/Desktop/data/preclean/trey/provider_scored.parquet (617,062 rows, NPI grain).

Column policy for the trey parquet (design mirrors src/model/config.py):
  DROPPED — label/leakage: provider_on_exclusion, provider_on_leie,
    billed_after_exclusion, excluded_after_billing, has_excluded_owner,
    exclusion_label_sources, excluded_owner_role,
    facility_excluded_owner_n_probable, layer3_probable_owner,
    within_2_hops_of_exclusion, weak_label*, confirmed_clean, clean_basis,
    subscore_ownership_integrity (ownership-exclusion derived), assessable
  DROPPED — detector-derived (negatives were selected on anomaly==0):
    anomaly_*, signals_tripped, priority_*, n_concept_signals,
    iforest_score_secondary, concentration, payment_intensity,
    service_intensity, specialty_mismatch, temporal, peer_basis, not_scored*,
    rule_reasons, anomaly_contributing_concepts
  DROPPED — identity/geo/taxonomy (no-geo config): state, city, zip,
    provider_name, org_legal_name, org_node_id, group_id, peer_group_key,
    nucc_grouping, nucc_classification (+ name-duplicates of existing cols)
  KEPT — behavior features: E&M coding shape, Part B/D utilization + drug mix,
    opioid shares, Open Payments concentration/correlation, address clustering,
    market saturation, PBJ understaffing, hospice live-discharge, deficiencies,
    peer-percentile variants, behavior subscores, shell/graph structure,
    billing residuals, consistency flags.

Run from repo root: python -m labels.train_trey_features
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from src.model import config as c
from src.model.data import build_feature_matrix
from src.model.train import LGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS
from labels.repeer_and_train_d import care_clusters, repeer, neg_split

TREY_PARQUET = Path.home() / "Desktop" / "data" / "preclean" / "trey" / "provider_scored.parquet"
LABELS_DIR = c.MODEL_DATA_DIR / "labels"
TRAIN_MAX, VAL_Y, TEST_Y = 2023, 2024, (2025, 2026)
SEEDS = [0, 1, 2]
GEO = ["primary_taxonomy", "practice_state"]
FR = {"1128a1", "1128a2", "1128a3", "1128b1", "1128b2", "1128b3", "1128b7"}
STRICT = {"1128a1", "1128a3", "1128b1", "1128b7"}

TREY_DROP = {
    # label / leakage
    "provider_on_exclusion", "provider_on_leie", "billed_after_exclusion",
    "excluded_after_billing", "has_excluded_owner", "exclusion_label_sources",
    "excluded_owner_role", "facility_excluded_owner_n_probable",
    "layer3_probable_owner", "within_2_hops_of_exclusion", "weak_label",
    "weak_label_score", "weak_label_votes", "confirmed_clean", "clean_basis",
    "assessable",
    # engineered columns NEVER sanctioned by Trey's handoff (HANDOFF_TRAVIS_V1):
    # ownership_integrity is the handoff's own "leakage flag"; shell_score /
    # expected_net_paid / billing_residual / weak_label* appear nowhere in its
    # column contract — unverified, stay out in every variant.
    "subscore_ownership_integrity",
    "shell_score", "expected_net_paid", "billing_residual",
    # detector-derived
    "anomaly_score", "anomaly_pct", "signals_tripped", "priority_tier",
    "priority_rank", "anomaly_lead_v3", "anomaly_score_v3", "n_concept_signals",
    "anomaly_contributing_concepts", "iforest_score_secondary", "concentration",
    "payment_intensity", "service_intensity", "specialty_mismatch", "temporal",
    "peer_basis", "not_scored", "not_scored_reason", "rule_reasons",
    # identity / geo / taxonomy (no-geo config)
    "state", "city", "zip", "provider_name", "org_legal_name", "org_node_id",
    "group_id", "peer_group_key", "nucc_grouping", "nucc_classification",
}

# The 12 handoff-sanctioned rules-engine subscores ("Trees can use raw +
# peerpct + subscores together"). Excluded under the v2 raw-only policy;
# included under --include-subscores (v3, per HANDOFF_TRAVIS_V1 contract).
SUBSCORES_12 = {
    "subscore_single_service_mill", "subscore_payment_outlier",
    "subscore_overutilization", "subscore_specialty_mismatch",
    "subscore_rapid_ramp", "subscore_upcoding", "subscore_pharma_kickback",
    "subscore_drug_outlier", "subscore_worthless_services",
    "subscore_hospice_ineligibility", "subscore_saturation_fraud",
    "subscore_pill_mill",
}


def get_trey_drop(include_subscores=False):
    return TREY_DROP if include_subscores else (TREY_DROP | SUBSCORES_12)


def log(m=""):
    print(m, flush=True)


def load_trey(existing_cols, include_subscores=False):
    t = pd.read_parquet(TREY_PARQUET)
    assert t["npi"].is_unique, "trey parquet npi not unique"
    drop = get_trey_drop(include_subscores)
    dup = {col for col in t.columns if col in existing_cols and col != "npi"}
    keep = [col for col in t.columns
            if col == "npi" or (col not in drop and col not in dup)]
    t = t[keep]
    bad = [col for col in t.columns if col != "npi" and not (
        pd.api.types.is_numeric_dtype(t[col]) or pd.api.types.is_bool_dtype(t[col]))]
    if bad:
        log(f"    dropping non-numeric trey cols: {bad}")
        t = t.drop(columns=bad)
    for col in t.columns:
        if col != "npi" and pd.api.types.is_bool_dtype(t[col]):
            t[col] = t[col].astype("int8")
    feats = [col for col in t.columns if col != "npi"]
    log(f"    trey features kept: {len(feats)} (dropped {len(drop)} listed + "
        f"{len(dup)} duplicates of existing cols)")
    return t, feats


def main():
    import argparse
    import hashlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-subscores", action="store_true",
                    help="v3: include the 12 handoff-sanctioned rules-engine subscores")
    ap.add_argument("--group-neg-split", action="store_true",
                    help="assign negatives to train/val/test by trey group_id (handoff grouped-CV)")
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()
    variant = "plus_trey_v3" if args.include_subscores else "plus_trey"
    log("[1] universe + baseline features (no-geo + cluster)")
    df = pd.read_parquet(c.SCORED_UNIVERSE_PARQUET).reset_index(drop=True)
    X = build_feature_matrix(df).reset_index(drop=True)
    cl = care_clusters(df)
    X_base = pd.concat([X.drop(columns=[g for g in GEO if g in X.columns]),
                        repeer(df, cl).reset_index(drop=True)], axis=1)
    cats = [col for col in c.CATEGORICAL_FEATURES if col in X_base.columns]

    log("[2] trey features")
    trey, trey_feats = load_trey(set(df.columns), include_subscores=args.include_subscores)
    order = df[["npi"]].merge(trey, on="npi", how="left", validate="1:1")
    assert len(order) == len(df)
    n_match = int(order[trey_feats[0]].notna().sum()) if trey_feats else 0
    log(f"    joined 1:1; rows with trey data: {n_match:,} / {len(df):,}")
    X_trey = pd.concat([X_base, order[trey_feats].reset_index(drop=True)], axis=1)

    log("[3] label (exclusion-risk: fraud-relevant LEIE ∪ state lists) + negatives")
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

    leie_fraud = vec(set(fr_y)); leie_strict = vec(set(strict_y)); leie_any = vec(set(any_y))
    state_m = vec(set(st["npi"]))
    y_fr = yarr(fr_y); y_strict = yarr(strict_y); y_state = yarr(state_y)

    pos = leie_fraud | state_m
    y_pos = np.fmin(np.where(leie_fraud, y_fr, np.nan), np.where(state_m, y_state, np.nan))
    y_pos = np.where(np.isnan(y_pos), np.where(leie_fraud, y_fr, y_state), y_pos)

    fp = set(pd.read_parquet(LABELS_DIR / "expanded_labels.parquet").query("fraud_positive")["npi"])
    excluded = leie_any | state_m | vec(fp)
    pos_company = set(company_id[excluded].dropna())
    clean = ((df["anomaly_score"].values == 0) & (~df["not_scored"].fillna(True).values)
             & (~excluded) & (~company_id.isin(pos_company).values))
    if args.group_neg_split:
        gid = pd.read_parquet(TREY_PARQUET, columns=["npi", "group_id"]).set_index("npi")["group_id"]
        gmap = pd.Series(npis).map(gid).fillna(pd.Series(npis).values).astype(str).values

        def gsplit(g):
            h = int(hashlib.md5(g.encode()).hexdigest(), 16) % 100
            return "train" if h < 70 else ("val" if h < 85 else "test")
        spl = np.array([gsplit(gmap[i]) if clean[i] else "" for i in range(len(npis))], dtype=object)
        log("    negatives split GROUP-AWARE by trey group_id (handoff CV rule)")
    else:
        spl = np.array([neg_split(n) if clean[i] else "" for i, n in enumerate(npis)], dtype=object)
    neg_tr, neg_va, neg_te = spl == "train", spl == "val", spl == "test"
    log(f"    positives {int(pos.sum())} | clean negatives {int(clean.sum()):,}")

    t_mask, t_year = leie_strict, y_strict  # neutral target: future strict-fraud LEIE

    log("[4] training baseline vs +trey (3 seeds, temporal)")
    results = {}
    arms = [("baseline", X_base), (variant, X_trey)]
    if args.skip_baseline:
        arms = arms[1:]
    for name, Xf in arms:
        trpos = pos & (y_pos <= TRAIN_MAX)
        tr = trpos | neg_tr
        arm = []
        for seed in SEEDS:
            Xa, Xb, ya, yb = train_test_split(Xf[tr], trpos[tr].astype(int), test_size=0.2,
                                              stratify=trpos[tr].astype(int), random_state=seed)
            p = dict(LGB_PARAMS); p["seed"] = seed
            b = lgb.train(p, lgb.Dataset(Xa, label=ya, categorical_feature=cats), NUM_BOOST_ROUND,
                          valid_sets=[lgb.Dataset(Xb, label=yb)],
                          callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                                     lgb.log_evaluation(0)])
            va = (t_mask & (t_year == VAL_Y)) | neg_va
            te = (t_mask & np.isin(t_year, TEST_Y)) | neg_te
            sv, ste = b.predict(Xf[va]), b.predict(Xf[te])
            yv, yt = t_mask[va].astype(int), t_mask[te].astype(int)
            ov = np.argsort(-sv)
            arm.append({"val_pr_auc": float(average_precision_score(yv, sv)),
                        "test_pr_auc": float(average_precision_score(yt, ste)),
                        "val_p50": float(yv[ov][:50].sum() / 50),
                        "best_iter": int(b.best_iteration)})
            log(f"    {name} ({Xf.shape[1]} feats) seed {seed}: "
                f"val {arm[-1]['val_pr_auc']:.3f} test {arm[-1]['test_pr_auc']:.3f} "
                f"P@50 {arm[-1]['val_p50']:.2f}")
            if name == variant and seed == SEEDS[-1]:
                imp = pd.DataFrame({"feature": Xf.columns,
                                    "gain": b.feature_importance("gain")}).sort_values(
                    "gain", ascending=False)
                trey_gain = imp[imp["feature"].isin(trey_feats)]
                log("\n    top 15 TREY features by gain (last seed):")
                for r in trey_gain.head(15).itertuples():
                    log(f"      {r.feature}: {r.gain:,.0f}")
                total = imp["gain"].sum()
                log(f"    trey share of total gain: {trey_gain['gain'].sum()/total:.1%}")
                imp.to_csv(LABELS_DIR / "trey_feature_importance.csv", index=False)
        results[name] = arm

    log("\n" + "=" * 70)
    for name, arm in results.items():
        v = np.array([a["val_pr_auc"] for a in arm]); t = np.array([a["test_pr_auc"] for a in arm])
        p50 = np.mean([a["val_p50"] for a in arm])
        log(f"{name:10s} val PR-AUC {v.mean():.3f} [{v.min():.3f}-{v.max():.3f}] | "
            f"test {t.mean():.3f} [{t.min():.3f}-{t.max():.3f}] | val P@50 {p50:.2f}")
    (LABELS_DIR / f"trey_ab_results_{variant}{'_groupsplit' if args.group_neg_split else ''}.json").write_text(json.dumps(results, indent=2))



if __name__ == "__main__":
    main()
