# GNN Track — Verdict: do NOT ship the GNN; keep the tabular model

**Date:** 2026-06-15 · **Branch:** `feat/gnn-scaffold` (merged to main as a documented
negative result). **Decision: the GNN underperforms the simpler tabular model on an
identical temporal test, so the production scorer stays LightGBM. No pipeline change.**

## The honest head-to-head (same temporal split, same features)

Train on LEIE exclusions ≤2023, validate 2024, test 2025–26. Both models, same masks:

| model | val PR-AUC | val P@50 | val P@100 | test PR-AUC | test P@100 |
|---|---|---|---|---|---|
| no-graph gradient boosting | **0.195** | **0.48** | **0.40** | **0.117** | **0.31** |
| FraudSAGE GNN (best seed) | 0.145 | 0.38 | 0.20 | 0.093 | 0.22 |

GNN 3-seed range: val PR-AUC 0.105–0.145 (mean 0.121), test 0.052–0.093 (mean 0.073).
The no-graph baseline wins on **every** metric — including precision@K, the metric that
matters for lead ranking. base rate val 0.28% / test 0.38%.

## Why the GNN lost (three concrete reasons)

1. **No NPI-level structural signal.** Fraudsters are *less* connected than clean
   providers (58.5% vs 70.8% connected; mean degree 2.3 vs 4.7). The shared-owner/
   address/AO structure that flagged shell rings at the **company** grain does not
   separate fraud at the **NPI** grain — the connected nodes are disproportionately
   legitimate multi-site organizations.
2. **Over-smoothing is active harm.** A typical positive is under-connected and sits
   among clean negatives; mean-aggregation pulls its representation toward "looks
   clean." The graph doesn't just fail to help — it drags positives toward the
   negative class, which is why the GNN scored *below* the no-graph baseline.
3. **Severe overfitting / seed instability.** 270 train positives vs a 2-layer
   128-hidden SAGE → train loss ~0.001, val PR-AUC swinging 0.105–0.145 by seed.

A regularization harness exists (`train.py --tag _reg --hidden 32 --dropout 0.5
--layers 1 --wd 0.01`); not pursued because the ceiling is set by reason #1 — even a
perfectly-tuned GNN only *matches* the no-graph 0.195, never beats it.

## The bigger finding (applies beyond the GNN)

**The "0.465" we were proud of was a random-split mirage.** On the honest temporal
split — the deployment question, "predict providers excluded *next* year" — even the
strong tabular model gets only **~0.20** PR-AUC. Fraud patterns shift year to year
(e.g. the 2022–24 AZ behavioral-health wave), so any model trained on past exclusions
generalizes modestly to future ones. This re-grades the whole project's realistic
performance, independent of the GNN.

## What we keep

- **Production scorer: unchanged** (LightGBM tabular, `src/model/`).
- **This scaffolding stays** as a reusable, correct GNN pipeline (`src/gnn/`): node
  table, 5-edge graph builder with clique caps, temporal×structural splits, features,
  FraudSAGE, full-batch trainer — all gated and verified. If a **company-grain** GNN
  is tried later (where the shell-ring signal actually lives), most of this is reusable.
- **The negative result is documented** so nobody re-runs this experiment expecting
  the graph to help at NPI grain.

## If revisited

The one place graph structure plausibly pays off is **company grain**, not NPI grain —
that's where Aveanna/Pinnacle-style rings are separable. That would be a new node
universe (companies, not NPIs) and a fresh experiment, not a tweak of this one.
