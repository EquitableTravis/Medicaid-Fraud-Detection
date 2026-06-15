"""
build_leads_fraud.py (labels) — Phase 3: score the universe on a given model and
run the UNCHANGED downstream chain to a company lead list. Parameterized by model
so the SAME code produces both the new (fraud_positive) and the old (all-LEIE) lead
lists for a controlled Phase-4 diff.

Reuses the production chain verbatim — score.rollup_to_company, export_leads.
resolve_names, build_final_leads.classify/normalize/taxonomy_code — only the I/O
paths change. Current production scores/leads (Model/scores, Model/output) are NOT
touched; everything is written under the given --out dir.

Chain: score all NPIs → reliability gate (score_reliable) → company rollup →
$10M + reliable-score filter → top-K → institutional screens → screened CSV.
LEIE/fraud-positive constituent counts are attached per lead (n_leie_npis already
in the rollup; n_fraud_pos_npis added) for the GATE-4 "already-caught" review;
parity with the current chain means we flag, not auto-drop.

Run:
    python -m labels.build_leads_fraud --model <m.txt> --feature-list <fl.json> --out <dir> --tag <t>
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from src.model import config
from src.model.data import log, require
from src.model.score import build_universe_matrix, rollup_to_company, PROVIDER_CONTEXT_COLS
from src.model.export_leads import resolve_names, OUT_COLS
from src.attempt_2.leads.build_final_leads import classify, normalize, taxonomy_code

LABELS = config.MODEL_DATA_DIR / "labels" / "expanded_labels.parquet"


def score_universe(model_path, fl_path):
    booster = lgb.Booster(model_file=str(model_path))
    spec = json.loads(Path(fl_path).read_text())
    df = pd.read_parquet(config.SCORED_UNIVERSE_PARQUET)
    require("npi unique in universe", df["npi"].is_unique)
    X = build_universe_matrix(df, spec)
    prov = df[PROVIDER_CONTEXT_COLS].copy()
    prov.insert(1, "model_score", booster.predict(X))
    prov["score_reliable"] = ~prov["not_scored"].fillna(True)
    prov = prov.sort_values("model_score", ascending=False).reset_index(drop=True)
    log(f"    scored {len(prov):,} NPIs | reliable {int(prov['score_reliable'].sum()):,} "
        f"| score range {prov['model_score'].min():.3f}-{prov['model_score'].max():.3f}")
    return prov


def screen(leads, prov):
    """Institutional screens — reuses build_final_leads.classify verbatim."""
    best = prov.sort_values("net_paid", ascending=False).drop_duplicates("npi").set_index("npi")
    spec = []
    for npis in leads["npi_list"]:
        members = [n for n in str(npis).split("|") if n in best.index]
        spec.append("" if not members else
                    str(best.loc[members].sort_values("net_paid", ascending=False)["primary_taxonomy"].iloc[0] or ""))
    leads = leads.copy(); leads["specialty"] = spec
    cats, toks = [], []
    for nm, sp in zip(leads["company_name"], leads["specialty"]):
        c, t = classify(normalize(nm), taxonomy_code(sp)); cats.append(c); toks.append(t)
    removed = pd.Series([c is not None for c in cats], index=leads.index)
    kept = leads.loc[~removed].copy()
    audit = leads.loc[removed].copy()
    audit["removed_category"] = [c for c in cats if c is not None]
    audit["matched"] = [t for c, t in zip(cats, toks) if c is not None]
    require("kept + removed covers every lead", len(kept) + len(audit) == len(leads))
    return kept, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--feature-list", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--top-k", type=int, default=5000)
    p.add_argument("--min-net-paid", type=float, default=10_000_000.0)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    log(f"[1/4] Scoring universe ({args.tag})")
    prov = score_universe(args.model, args.feature_list)
    prov.to_parquet(out / f"npi_scores_{args.tag}.parquet", index=False)

    log("[2/4] Company rollup")
    comp = rollup_to_company(prov)
    comp.to_parquet(out / f"company_scores_{args.tag}.parquet", index=False)

    log("[3/4] Filter ($10M + reliable score) + top-K + name resolution")
    eligible = comp[(comp["company_net_paid"] >= args.min_net_paid)
                    & (comp["company_model_score_max"] >= 0)]
    leads = eligible.sort_values(["company_model_score_max", "company_net_paid"],
                                 ascending=False).head(args.top_k).copy()
    leads = resolve_names(leads, prov[["npi", "model_score", "org_legal_name"]])
    leads["rank"] = range(1, len(leads) + 1)
    # attach fraud_positive constituent count per lead (for GATE-4 already-caught review)
    fp = set(pd.read_parquet(LABELS).query("fraud_positive")["npi"])
    leads["n_fraud_pos_npis"] = leads["npi_list"].map(
        lambda s: sum(n in fp for n in str(s).split("|")))
    require("no unresolved company names", bool(leads["company_name"].notna().all()))
    cols = OUT_COLS + ["n_fraud_pos_npis"]
    leads[cols].to_csv(out / f"leads_{args.tag}.csv", index=False)

    log("[4/4] Institutional screens")
    kept, audit = screen(leads[cols], prov)
    kept.to_csv(out / f"leads_{args.tag}_screened.csv", index=False)
    audit.to_csv(out / f"leads_{args.tag}_removed_audit.csv", index=False)
    log(f"    {len(leads):,} leads → screened {len(kept):,} (removed {len(audit):,}, "
        f"${kept['company_net_paid'].sum()/1e9:.1f}B) → {out}/leads_{args.tag}_screened.csv")


if __name__ == "__main__":
    main()
