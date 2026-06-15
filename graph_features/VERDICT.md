# Graph Features → Existing Model — Verdict: NEUTRAL (keep the existing model)

**Date:** 2026-06-15 · **Branch:** `feat/graph-features` · **Decision: graph features do
NOT add signal at the NPI grain. Keep the existing LightGBM unchanged — no new columns
shipped.** This independently confirms the GNN result by a totally different method
(tree features instead of message passing): graph structure does not help predict LEIE
at the NPI level.

## The probe (faithful to the plan)

The existing model is NPI-grain (→ rollup), so we augmented *it*. Same model, same
hyperparameters, same temporal split (train ≤2023, val 2024, test 2025–26); the only
change is the added feature columns. GATE 0 confirmed the harness reproduces the
existing model exactly (val PR-AUC 0.4653 vs 0.465 on its random split).

## Head-to-head (≥3 seeds, val PR-AUC; the only diff is the columns)

| variant | val PR-AUC | val seed-range | val P@100 | test PR-AUC |
|---|---|---|---|---|
| **billing (baseline)** | 0.241 | **[0.027, 0.399]** | 0.36 | 0.191 |
| billing + structure | 0.230 | [0.194, 0.281] | 0.34 | **0.196** |
| billing + proximity | 0.038 | — | 0.08 | 0.022 |
| billing + all | 0.008 | — | 0.01 | 0.010 |

## Findings

1. **Structure features (degree, component size): NEUTRAL.** Δ val PR-AUC −0.012 — *well
   inside* the baseline's seed-range noise, so no real lift by the pre-registered decision
   rule. (Mild upside: it slightly improved *test* PR-AUC, 0.196 vs 0.191, and **tightened
   the wild baseline seed-range** [0.027–0.399] → [0.194–0.281] — a stabilizer, not a
   signal source.) The honest structure-thesis test → **does not add signal.**

2. **Proximity features (dist_to_excluded / n_excluded_neighbors / comp_excluded_ratio):
   actively HARMFUL** — collapsed val PR-AUC 0.24 → 0.04. This is the leakage trap the plan
   flagged, manifesting as **anti-generalization**: because the excluded source set (LEIE
   ≤2023) ≈ the *train positives*, those features are a near-perfect *train-label echo*
   (a train positive has dist_to_excluded = 0). LightGBM splits hard on them (SHAP:
   `comp_excluded_ratio` rank 4, used heavily) — then fails on val/test, where 2024+
   positives sit far from any 2023-or-earlier exclusion (only 5,544 of 617k nodes are even
   within 4 hops of one). Temporal discipline prevented val-label leakage but couldn't fix
   that fraud doesn't cluster spatially at NPI grain. **Exclude these features.**

3. **SHAP confirms the model would *use* graph features if they helped** — `comp_size`
   ranked #1 by mean|SHAP| in the +all model — but using them didn't generalize (that model
   collapsed). High usage + no generalization = the model latching onto a train artifact,
   not signal.

## Why this matches the GNN result

Both methods now agree, from opposite directions: **graph structure carries no NPI-level
fraud signal.** The GNN over-smoothed; the tree features either no-op (structure) or
echo the train label (proximity). Root cause is the same — fraudsters are *under-connected*
at NPI grain (58.5% vs 70.8% for clean providers), so "who you're connected to" doesn't
predict "are you fraud." The shell-ring signal is real but lives at **company** grain.

## Decision (pre-registered rule applied)

> If structure-only features lift PR-AUC beyond the seed-range → adds signal, ship augmented.
> If neutral → doesn't add signal; keep the existing model.

**NEUTRAL → keep the existing LightGBM.** No pipeline change. The probe did exactly its job:
answered "does structure help?" in a few hours, cheaply and safely (the existing model was
never touched — `src/model/` is byte-for-byte unchanged), and saved a multi-week GNN rebuild
that we now have two independent reasons to expect would not pay off at this grain.

## What we keep

- Production model: **unchanged** (LightGBM, `src/model/`).
- This probe (`graph_features/`) + artifacts (`Model/graph_features/`): the identifier-graph
  feature builder, the temporal leakage discipline, and the head-to-head/SHAP/ablation
  harness — reusable if a **company-grain** model is ever built (the one grain where the
  evidence says structure might actually pay off).
