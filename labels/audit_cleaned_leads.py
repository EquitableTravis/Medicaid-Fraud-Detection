"""
audit_cleaned_leads.py (labels) — serious evaluation of the PRODUCTION lead list
(model_leads_CLEANED.csv, the all-LEIE model's output, pre-today's work).

Five angles:
  A. Composition — counts, dollars, score bands, size, merge confidence, states.
  B. Score-band reality check — does the 12.7x-lift-only-at>=0.99 calibration hold
     on THIS list? (share of leads in each band).
  C. Institutional contamination — run TODAY's hardened screens over the list to see
     how many institutional FPs survived the screens that were live when it was built.
  D. Independent corroboration — the all-LEIE model was NEVER trained on AHCCCS/NV.
     So AHCCCS/NV membership among the leads' NPIs is a clean OUT-OF-SAMPLE check:
     do the leads hit those lists far above the universe base rate?
  E. (separate) random-sample web adjudication — run from the harness output.

Read-only. Run:
    python -m labels.audit_cleaned_leads
"""

import argparse
import re

import numpy as np
import pandas as pd

from src.model import config as c
from src.attempt_2.leads.build_final_leads import classify, normalize, taxonomy_code

CLEANED = c.MODEL_DATA_DIR / "output" / "final" / "model_leads_CLEANED.csv"
LABELS = c.MODEL_DATA_DIR / "labels" / "expanded_labels.parquet"


def log(m=""): print(m, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=int, default=25, help="random leads to dump for web adjudication")
    args = p.parse_args()

    d = pd.read_csv(CLEANED, dtype={"npi_list": str})
    n = len(d)
    log(f"\n{'='*64}\nAUDIT — model_leads_CLEANED.csv ({n:,} leads)\n{'='*64}")

    # ---- A. composition ----
    s = d["company_model_score_max"]
    log("\n[A] COMPOSITION")
    log(f"  total billing: ${d['company_net_paid'].sum()/1e9:.1f}B | "
        f"median lead ${d['company_net_paid'].median()/1e6:.1f}M | "
        f"min ${d['company_net_paid'].min()/1e6:.1f}M")
    log(f"  single-NPI leads: {(d['n_npis']==1).sum():,} ({(d['n_npis']==1).mean():.0%}) | "
        f"multi-NPI: {(d['n_npis']>1).sum():,}")
    log(f"  merge_confidence: " + ", ".join(f"{k}={v}" for k, v in d['merge_confidence'].value_counts().items()))
    log(f"  leads with >=1 already-LEIE NPI (n_leie_npis>0): {(d['n_leie_npis']>0).sum():,}")

    # state via dominant constituent NPI
    uni = pd.read_parquet(c.SCORED_UNIVERSE_PARQUET, columns=["npi", "practice_state", "net_paid"])
    st = uni.sort_values("net_paid", ascending=False).drop_duplicates("npi").set_index("npi")["practice_state"]
    d["state"] = d["npi_list"].map(lambda x: st.get(str(x).split("|")[0], ""))
    log("  top states: " + ", ".join(f"{k}={v}" for k, v in d["state"].value_counts().head(8).items()))

    # ---- B. score bands vs calibration ----
    bands = [("[0.99,1.0]", s >= 0.99), ("[0.90,0.99)", (s >= 0.90) & (s < 0.99)),
             ("[0.70,0.90)", (s >= 0.70) & (s < 0.90)), ("<0.70", s < 0.70)]
    log("\n[B] SCORE BANDS (calibration: real LEIE lift ~12.7x ONLY at >=0.99)")
    for name, mask in bands:
        log(f"  {name:14s} {int(mask.sum()):>5,} leads ({mask.mean():>4.0%})")

    # ---- C. institutional contamination via TODAY's hardened screens ----
    cats = []
    for nm, sp in zip(d["company_name"], d["specialty"]):
        cat, _ = classify(normalize(nm), taxonomy_code(sp))
        cats.append(cat)
    d["inst_screen"] = cats
    flagged = d["inst_screen"].notna()
    log(f"\n[C] INSTITUTIONAL CONTAMINATION (today's hardened screens applied to the list)")
    log(f"  leads that TODAY's screens would remove: {int(flagged.sum()):,} ({flagged.mean():.1%})")
    for k, v in d.loc[flagged, "inst_screen"].value_counts().items():
        log(f"    {k:22s} {v}")

    # ---- D. independent (AHCCCS/NV) corroboration — out-of-sample for the all-LEIE model ----
    lab = pd.read_parquet(LABELS).set_index("npi")
    sets = {k: set(lab.index[lab[k].fillna(False)]) for k in
            ["on_leie_any", "on_leie_fraud", "on_ahcccs", "on_nv"]}
    state_excl = sets["on_ahcccs"] | sets["on_nv"]
    all_lead_npis = [npi for x in d["npi_list"] for npi in str(x).split("|")]
    base_state = len(set(all_lead_npis) & state_excl) / max(1, len(set(all_lead_npis)))
    uni_state_rate = len(state_excl) / len(uni)
    def hit(npis, S): return any(npi in S for npi in str(npis).split("|"))
    d["corrob_leie_fraud"] = d["npi_list"].map(lambda x: hit(x, sets["on_leie_fraud"]))
    d["corrob_state"] = d["npi_list"].map(lambda x: hit(x, state_excl))
    d["corrob_any"] = d["corrob_leie_fraud"] | d["corrob_state"] | (d["n_leie_npis"] > 0)
    log("\n[D] ENFORCEMENT CORROBORATION (does a lead contain an excluded NPI?)")
    log(f"  >=1 fraud-relevant LEIE NPI: {int(d['corrob_leie_fraud'].sum()):,} ({d['corrob_leie_fraud'].mean():.1%})")
    log(f"  >=1 AHCCCS/NV NPI (model NEVER trained on these): {int(d['corrob_state'].sum()):,} "
        f"({d['corrob_state'].mean():.1%})")
    log(f"  >=1 ANY enforcement-list NPI: {int(d['corrob_any'].sum()):,} ({d['corrob_any'].mean():.1%})")
    log(f"  → AHCCCS/NV hit rate among lead NPIs {base_state:.3%} vs universe base {uni_state_rate:.3%} "
        f"= {base_state/max(uni_state_rate,1e-9):.1f}x lift (independent validation)")
    # does corroboration concentrate in the >=0.99 band? (validates the calibration cliff here)
    hi = s >= 0.99
    log(f"  corroboration in >=0.99 band: {d.loc[hi,'corrob_any'].mean():.1%} vs "
        f"<0.99 band: {d.loc[~hi,'corrob_any'].mean():.1%}")

    # ---- random sample for web adjudication ----
    samp = d.sample(n=args.sample, random_state=7).sort_values("new_rank")
    out = c.MODEL_DATA_DIR / "output" / "final" / "cleaned_audit_sample.csv"
    samp[["new_rank", "company_name", "state", "specialty", "company_model_score_max",
          "company_net_paid", "n_npis", "corrob_any", "inst_screen"]].to_csv(out, index=False)
    log(f"\n[E] random {args.sample} leads for web adjudication → {out}")
    for r in samp.itertuples():
        flag = (f" [INST:{r.inst_screen}]" if r.inst_screen else "") + (" [CORROB]" if r.corrob_any else "")
        log(f"  #{r.new_rank} {str(r.company_name)[:42]:42s} {r.state} {r.specialty} "
            f"${r.company_net_paid/1e6:.0f}M{flag}")


if __name__ == "__main__":
    main()
