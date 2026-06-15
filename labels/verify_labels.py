"""
verify_labels.py (labels) — Phase 0 pre-merge leak verification (GATE 0).

Three checks that rule out the ways a label expansion can fake a temporal win.
If any fails, STOP — do not retrain production.

  0.1 Top-50 hand-trace      top test-ranked NPIs must include genuine 2024+
                             LEIE-fraud positives, not only AHCCCS/NV echoes.
  0.2 Temporal dating        every TRAIN positive first-seen ≤2023; train/test
                             positive NPI sets disjoint; test positives genuinely
                             first-excluded 2024+ (no earlier date).
  0.3 Negative disjointness  no expanded-positive COMPANY (company_id) also sits
                             in the clean-negative set under another NPI.

Writes labels/VERIFY.md. Read-only on src/model and on production artifacts.

Run:
    python -m labels.verify_labels
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.model import config as mcfg
from src.model.data import build_feature_matrix
from src.model.train import LGB_PARAMS
from labels.build_labels import leie_positives, ahcccs_positives, nv_positives
from labels.train_eval_labels import neg_split, TRAIN_MAX_YEAR, VAL_YEAR, TEST_YEARS

LABELS = mcfg.MODEL_DATA_DIR / "labels" / "expanded_labels.parquet"
VERIFY_MD = Path(__file__).resolve().parent / "VERIFY.md"


def log(m=""):
    print(m, flush=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    out = []                                   # markdown lines

    log("[load] features + labels + identity")
    df = pd.read_parquet(mcfg.SCORED_UNIVERSE_PARQUET)
    X = build_feature_matrix(df).reset_index(drop=True)
    base = df[["npi", "org_legal_name", "entity_type", "primary_taxonomy",
               "practice_state", "net_paid", "anomaly_score", "not_scored"]].reset_index(drop=True)
    lab = pd.read_parquet(LABELS).set_index("npi")
    base = base.join(lab, on="npi")
    cmap = pd.read_parquet(mcfg.NPI_TO_COMPANY_MAP).set_index("npi")["company_id"]
    base["company_id"] = base["npi"].map(cmap)

    # per-source years (re-derived, the same way build_labels does) for the dating check
    universe = set(base["npi"])
    _, _, fr_year, any_year = leie_positives(universe)
    az = ahcccs_positives(universe)
    nv = nv_positives(universe)
    src_years = {}
    for nm, d in (("leie", any_year), ("ahcccs", az), ("nv", nv)):
        for n, y in d.items():
            src_years.setdefault(n, {})[nm] = y

    # LEIE names for the hand-trace
    cg = pd.read_csv(mcfg.DATA_DIR / "preclean" / "Caught.csv", dtype=str, keep_default_na=False)
    cg = cg[cg["NPI"].isin(universe)]
    leie_name = {r.NPI: (r.BUSNAME or f"{r.FIRSTNAME} {r.LASTNAME}").strip() for r in cg.itertuples()}

    # masks. Negatives use COMPANY-level disjointness: exclude any NPI on a state/LEIE list
    # AND any NPI sharing a company_id with a fraud_positive (no sibling-NPI of a flagged org
    # counts as a clean negative). This is the policy carried into the production retrain.
    excluded_any = base["on_leie_any"].fillna(False) | base["on_ahcccs"].fillna(False) | base["on_nv"].fillna(False)
    pos_company = set(base.loc[base["fraud_positive"].fillna(False), "company_id"].dropna())
    on_pos_company = base["company_id"].isin(pos_company)
    clean = (base["anomaly_score"] == 0) & (~base["not_scored"].fillna(True)) & (~excluded_any) & (~on_pos_company)
    split = pd.Series(index=base.index, dtype=object)
    split[clean] = base.loc[clean, "npi"].map(neg_split).values
    neg_train, neg_val, neg_test = (split == "train").values, (split == "val").values, (split == "test").values
    yr = base["excl_year_all"]
    fp = base["fraud_positive"].fillna(False).values
    fp_test = fp & yr.isin(TEST_YEARS).values
    fp_val = fp & (yr == VAL_YEAR).values

    out.append("# Labels Track — Phase 0 Verification (GATE 0)\n")
    out.append("Date: 2026-06-15 · Branch: `feat/labels` · `python -m labels.verify_labels`\n")
    out.append("Three leak checks on the expanded `fraud_positive` label before any production "
               "retrain. Construction note: each NPI carries a single `excl_year_all` = earliest "
               "exclusion across LEIE/AHCCCS/NV, and split = that year (train ≤2023, val 2024, "
               "test 2025-26), so an NPI lands in exactly one split.\n")

    # ---------- 0.2 temporal dating ----------
    log("\n[0.2] Temporal dating of train positives")
    train_pos_c = base["fraud_positive"].fillna(False).values & (yr <= TRAIN_MAX_YEAR).values
    test_pos_c = fp_test
    tp_npis = set(base.loc[train_pos_c, "npi"])
    te_npis = set(base.loc[test_pos_c, "npi"])
    overlap = tp_npis & te_npis
    # test positives whose earliest source date is actually ≤2023 (would be a dating error)
    bad_test = [n for n in te_npis if min((y for y in src_years.get(n, {}).values() if y), default=9999) <= TRAIN_MAX_YEAR]
    # train positives that ALSO carry a >2023 date (legit: known-bad earlier, not in test)
    train_also_later = [n for n in tp_npis if max((y for y in src_years.get(n, {}).values() if y), default=0) > TRAIN_MAX_YEAR]
    p02 = not overlap and not bad_test
    out.append(f"## 0.2 Temporal dating — {'PASS' if p02 else 'FAIL'}\n")
    out.append(f"- train positives (≤{TRAIN_MAX_YEAR}): **{len(tp_npis)}** · test positives (2024+ via excl_year_all): **{len(te_npis)}**")
    out.append(f"- train ∩ test positive NPIs: **{len(overlap)}** (must be 0)")
    out.append(f"- test positives with an earlier (≤2023) source date — a dating error: **{len(bad_test)}** (must be 0)")
    out.append(f"- train positives that also carry a later (>2023) action: {len(train_also_later)} "
               f"(allowed — known-bad by 2023, and NOT in the test set, so no future-label leak)\n")
    log(f"  train {len(tp_npis)} test {len(te_npis)} | overlap {len(overlap)} bad_test {len(bad_test)} -> {'PASS' if p02 else 'FAIL'}")

    # ---------- 0.3 negative disjointness (company grain) ----------
    log("[0.3] Negative-set disjointness (company_id)")
    pos_companies = set(base.loc[fp_val | fp_test, "company_id"].dropna())
    neg_companies = set(base.loc[neg_val | neg_test, "company_id"].dropna())
    shared = pos_companies & neg_companies
    n_neg_in_shared = int((base.loc[neg_val | neg_test, "company_id"].isin(shared)).sum())
    p03 = len(shared) == 0
    out.append(f"## 0.3 Negative disjointness (company grain) — {'PASS' if p03 else 'REVIEW'}\n")
    out.append(f"- eval-positive companies: {len(pos_companies)} · eval-negative companies: {len(neg_companies)}")
    out.append(f"- companies on BOTH sides of the eval: **{len(shared)}** "
               f"({n_neg_in_shared} negative NPIs share a company_id with an eval positive)")
    if shared:
        out.append(f"- Interpretation: NPI-grain eval, so a fraud company can have some clean-billing NPIs; "
                   f"{len(shared)} shared companies out of {len(pos_companies)} positive companies "
                   f"({len(shared)/max(1,len(pos_companies)):.1%}). Material only if large.\n")
    else:
        out.append("- No company appears on both sides of the evaluation.\n")
    log(f"  shared companies {len(shared)} ({n_neg_in_shared} neg NPIs) -> {'PASS' if p03 else 'REVIEW'}")

    # ---------- 0.1 top-50 hand-trace ----------
    log("[0.1] Top-50 hand-trace (training C, scoring test period)")
    params = dict(LGB_PARAMS); params["seed"] = 0
    train_mask = train_pos_c | neg_train
    booster = lgb.train(params, lgb.Dataset(X[train_mask], label=train_pos_c[train_mask].astype(int)),
                        num_boost_round=600, callbacks=[lgb.log_evaluation(0)])
    test_mask = fp_test | neg_test
    idx = np.where(test_mask)[0]
    scores = booster.predict(X.iloc[idx])
    top = idx[np.argsort(-scores)[:50]]
    tt = base.iloc[top].copy()
    tt["is_pos"] = fp_test[top]

    def source(row):
        s = []
        if row["on_leie_fraud"]: s.append("LEIE-fraud")
        if row["on_ahcccs"]: s.append("AHCCCS")
        if row["on_nv"]: s.append("NV")
        return "+".join(s) if s else "neg"

    tt["source"] = tt.apply(source, axis=1)
    pos_hits = tt[tt["is_pos"]]
    n_pos = len(pos_hits)
    by_src = pos_hits["source"].value_counts().to_dict()
    leie_genuine = pos_hits[pos_hits["on_leie_fraud"].fillna(False)]
    p01 = len(leie_genuine) >= 5
    out.append(f"## 0.1 Top-50 hand-trace — {'PASS' if p01 else 'FAIL'}\n")
    out.append(f"- Of the top 50 test-period NPIs by score, **{n_pos} are true fraud positives** "
               f"(P@50 = {n_pos/50:.2f}). Source breakdown: {by_src}.")
    out.append(f"- Genuine 2024+ **LEIE-fraud** positives in the top 50 (not state-list echoes): "
               f"**{len(leie_genuine)}** (need ≥5).\n")
    out.append("Hand-traced genuine LEIE-fraud leads (NPI · name · taxonomy · state · year · $net):\n")
    out.append("| NPI | name | taxonomy | state | excl_yr | net_paid |")
    out.append("|---|---|---|---|---|---|")
    for r in leie_genuine.head(8).itertuples():
        nm = (r.org_legal_name or leie_name.get(r.npi, "") or "").strip()[:40]
        out.append(f"| {r.npi} | {nm} | {str(r.primary_taxonomy)[:24]} | {r.practice_state} "
                   f"| {int(r.excl_year_all) if pd.notna(r.excl_year_all) else '?'} | {r.net_paid:,.0f} |")
    out.append("")
    log(f"  P@50 {n_pos/50:.2f} | genuine LEIE-fraud in top50: {len(leie_genuine)} -> {'PASS' if p01 else 'FAIL'}")

    gate = p01 and p02 and p03
    out.insert(2, f"\n**GATE 0: {'PASS — proceed to Phase 1' if gate else 'NOT CLEAR — resolve before retrain'}** "
                  f"(0.1 {'✓' if p01 else '✗'} · 0.2 {'✓' if p02 else '✗'} · 0.3 {'✓' if p03 else '⚠'})\n")
    VERIFY_MD.write_text("\n".join(out))
    log(f"\nGATE 0: {'PASS' if gate else 'REVIEW'} -> {VERIFY_MD}")


if __name__ == "__main__":
    main()
