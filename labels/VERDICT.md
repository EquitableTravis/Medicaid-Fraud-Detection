# Labels Track — Verdict: ADOPT the expanded label. Biggest model win to date.

**Date:** 2026-06-15 · **Branch:** `feat/labels`. **Decision: replace the production
training label (578 all-LEIE) with the expanded `fraud_positive` set (889 = fraud-relevant
LEIE ∪ AHCCCS ∪ NV). It roughly DOUBLES honest temporal PR-AUC and eliminates the seed
fragility of the current label.** This is the first lever (after the GNN and graph-features
negatives) that actually moves the model.

## What changed about the label

| | count | |
|---|---|---|
| current label (all LEIE) | 578 | includes ~151 non-fraud exclusions (license/loan) |
| **expanded `fraud_positive`** | **889** | +462 net-new, and cleaner |
| ├ fraud-relevant LEIE (a1/a2/a3/b1/b2/b3/b7) | 420 | dropped 158 non-fraud types |
| ├ AHCCCS suspensions | 191 | 189 not on LEIE — the AZ behavioral-health wave |
| └ NV sanctions | 331 | 274 not on LEIE |

Years skew 2022–26, giving healthy temporal val/test. `labels/build_labels.py` builds the
table (`Model/labels/expanded_labels.parquet`); reuses the pursuit-pipeline AHCCCS/NV parsers.

## The head-to-head (identical temporal split, only the TRAINING label varies)

Train positives filtered to ≤2023; eval = future positives (val 2024, test 2025–26) over the
same confident-clean negatives (anomaly==0 & ~not_scored & not excluded). Same LightGBM HP as
`src/model/train.py`, early-stop on val PR-AUC, 3 seeds. `labels/train_eval_labels.py`.

**Eval target = fraud_positive** (val 139 pos, base rate 0.30%):

| variant (train label) | train+ | val PR-AUC | val P@50 | val P@100 | test PR-AUC |
|---|---|---|---|---|---|
| A · all LEIE (current) | 296 | 0.317 [0.041–0.505] | 0.59 | 0.41 | 0.326 [0.049–0.509] |
| B · fraud-LEIE only (cleaned) | 223 | 0.179 [0.005–0.356] | 0.40 | 0.28 | 0.163 [0.007–0.311] |
| **C · expanded** | **573** | **0.573 [0.559–0.582]** | **0.98** | **0.72** | **0.550 [0.547–0.556]** |

**Bias check — eval target = LEIE-fraud ONLY** (no state lists C trains on; val 89 pos):

| variant | train+ | val PR-AUC | val P@50 | val P@100 | test PR-AUC |
|---|---|---|---|---|---|
| A · all LEIE (current) | 296 | 0.248 [0.035–0.386] | 0.44 | 0.25 | 0.232 [0.037–0.338] |
| B · fraud-LEIE only | 223 | 0.119 [0.003–0.254] | 0.22 | 0.15 | 0.090 [0.004–0.201] |
| **C · expanded** | **573** | **0.427 [0.403–0.441]** | **0.66** | **0.40** | **0.341 [0.330–0.357]** |

## Three things this shows

1. **The win is real, not "teaching to the test."** C beats the current label even on the
   pure-LEIE-fraud target it shares no extra positives with (0.427 vs 0.248 val, 0.341 vs
   0.232 test). The state-enforcement positives teach fraud patterns (e.g. the AZ behavioral-
   health wave) that recur in *future* LEIE cases too.
2. **Expansion drives it, not cleaning.** B (cleaned-only, fewer positives) is *worse* than
   the current label — dropping the 158 non-fraud LEIE rows removed signal without adding any.
   The gain comes from the +462 positives (especially the 463 state-enforcement ones). C is
   both cleaned and expanded; expansion is the active ingredient.
3. **It kills seed fragility.** With ~250 train positives, one seed of A and B collapses to
   ~0.04 PR-AUC (the model memorizes noise). C's 573 positives are stable across all seeds
   (0.559–0.582). For production, that stability matters as much as the mean lift.

## Honest caveats

- Absolute PR-AUC here is NOT directly comparable to the GNN verdict's 0.195 temporal bar:
  that used a company-structural negative split and a LEIE-only target. Here negatives use a
  random hash split (they have no time dimension — "clean" is defined by anomaly==0, not by a
  date), and the target includes state lists. **Only the A-vs-C contrast is controlled** (same
  split, same negatives, same HP — only the training label moves), and that contrast is the result.
- AHCCCS/NV exclusion years are best-effort parsed from the PDFs (882/889 positives have a year).
- This validates the label as a *training target*. Adopting it in production = retrain
  `src/model` on `fraud_positive` and regenerate scores → rollup → leads (a real change to the
  lead list — done as a deliberate follow-on, not silently).

## Recommendation

Adopt C (`fraud_positive`) as the production training label. Next step: wire the expanded label
into `src/model` training, retrain, regenerate the company-rollup lead list, and diff against the
current leads (which leads move up / newly surface — expect more AZ/NV behavioral-health).
