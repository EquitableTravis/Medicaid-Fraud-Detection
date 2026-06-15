# Labels Track — Phase 4 Lead Diff + Adjudication (GATE 4)

**Date:** 2026-06-15 · Branch: `feat/labels` · **Verdict: DO NOT cut over the production
lead list as-is.** The temporal metric win is real and reproducible, but it does NOT
translate into demonstrably better *novel* leads — the newly-surfaced leads are dominated
by legitimate AZ behavioral-health providers and expose three institutional screen gaps.
This is the go/no-go gate, and the honest answer is **hold**.

## The diff (identical chain, $10M + top-5000 + screens, only the model differs)

| | new (fraud_positive) | old (all-LEIE) |
|---|---|---|
| screened leads | 4,020 ($232.4B) | 4,033 ($236.6B) |
| newly surfaced | **592** | — |
| dropped | — | **605** |
| shared | 3,428 | 3,428 |

- **Newly surfaced skew AZ** (76), then CA/MN/TX/NY — the AHCCCS behavioral-health wave the
  expanded positives teach, exactly as predicted.
- **Dropped carried 0 known-LEIE NPIs** — we lose zero *confirmed* catches; the drops are
  generic high-billers (dental, FQHC-like, home care) in NY/CA/NJ being reprioritized.
- **Only 2 of 592 newly-surfaced carry a flagged (training-positive) NPI** — so the new leads
  are genuine model *extrapolation*, not re-finds. That cuts both ways (see below).

## Adjudication of the top newly-surfaced leads (web research, 14 entities)

| lead | state | verdict |
|---|---|---|
| REFLECTION HEALTH SERVICES | AZ | **LIKELY FRAUD (confirmed on AHCCCS OIG suspension list, 11/2021)** — but one of its NPIs is already a training positive |
| EMBARK RECOVERY | AZ | institutional FP — Joint-Commission-accredited rehab since 2013 |
| SUCCEEDING AT RECOVERY | AZ | institutional FP — BayMark-owned MAT chain (Emerald Isle) |
| RIVYVE BEHAVIORAL HEALTH | AZ | institutional FP — accredited, named leadership, real facilities |
| NEW HOPE OF ARIZONA | AZ | institutional FP — long-standing children's residential |
| NEW FREEDOM OPS | AZ | borderline — 2023 AHCCCS payment hold (sued, dismissed), no indictment |
| INTENSIVE TREATMENT SYSTEMS | AZ | institutional FP — federally-regulated methadone OTP |
| P&G BEHAVIORAL HEALTH | DC | inconclusive — right size/taxonomy in a DC hot-spot, no specific signal |
| KLS BEHAVIORAL HEALTH | AZ | inconclusive — SUD residential profile, no enforcement found |
| CH MH SERVICES | AZ | inconclusive — taxonomy mismatch, unverified |
| WOODWARD FOUNDATION FOR THE DISABLED | AZ | institutional FP — DDD day program |
| ANOKA-HENNEPIN ISD#11 | MN | institutional FP — **public school district (screen gap)** |
| SALT RIVER PIMA-MARICOPA INDIAN COMMUNITY | AZ | institutional FP — **tribal government health (screen gap)** |
| DISTRICT MEDICAL GROUP | AZ | institutional FP — **academic safety-net faculty group (screen gap)** |

**Tally:** 1 confirmed fraud (and that one is a training positive's own company, not a novel
discovery) · 3 inconclusive-but-plausible · 1 borderline · ~9 legitimate/institutional.

## Why the metric win didn't become a lead win

The 42 features describe *billing shape in a peer group*, not fraud itself. The expanded label
taught the model "looks like AZ behavioral-health / SUD / DDD with high billing." It ranks the
**known** positives in that profile above clean negatives very well (PR-AUC 0.55) — but when it
**extrapolates** to unlabeled providers, it surfaces the many *legitimate* providers that share
that exact profile (accredited rehabs, MAT/methadone OTPs, DDD day programs, a school district,
a tribe, a teaching-hospital faculty group). The features that separate a real Reflection-style
fraud from a real Embark-style clinic simply aren't in the data. Concentrating the leads on a
known-fraud *sector* raises the prior, but that sector is full of legitimate providers too.

This is the same lesson as the GNN/graph-features tracks, one level up: the label change is a
genuine, honest improvement to what the model can *learn*, but it does not manufacture
discriminating signal that the features don't contain.

## Three concrete, fixable findings

1. **Screen gaps (clearly correct to fix regardless):** public school districts (LEA taxonomy
   `251300000X`), tribal-government health programs, and academic/safety-net faculty groups
   should be quarantined by the institutional screens. The expanded label pushed these into the
   top ranks where the old model never did, so the current keyword screens miss them.
2. **No confirmed catch is lost** — the drops are unconfirmed generic high-billers.
3. **The one confirmed new hit is circular** — Reflection Health is a company of a training
   positive, not a novel discovery.

## Recommendation (go/no-go)

**Do not promote the new lead list to advertising as-is.** Options, in order of my preference:

- **A — Hold + harden, then re-judge.** Add the three institutional screens, then adjudicate a
  *random* (unbiased) sample of ~30 newly-surfaced leads. Decide on real evidence, not this
  FP-seeded top sample. Cheap, and it's the honest path.
- **B — Targeted use.** Keep the current production model for the broad list; use the
  fraud_positive model only as a *secondary* ranker within AZ/NV behavioral-health, where the
  active enforcement wave makes the prior genuinely higher. Captures the upside without sending
  AZ-behavioral-health ads to legit accredited providers nationwide.
- **C — Do not ship the lead change.** Keep the labels work as a documented, reproducible
  finding (the metric win is real and worth recording); leave production leads on the current
  model.

What does NOT change either way: the temporal PR-AUC result (0.55, GATE 2) and the leak
verification (GATE 0) stand. The label is a better *training target*; it is just not, by itself,
a better *lead generator* without the screen fixes and a fair re-adjudication.
