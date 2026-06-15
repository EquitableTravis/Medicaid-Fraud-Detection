# Labels Track — Phase 0 Verification (GATE 0)

Date: 2026-06-15 · Branch: `feat/labels` · `python -m labels.verify_labels`


**GATE 0: PASS — proceed to Phase 1** (0.1 ✓ · 0.2 ✓ · 0.3 ✓)

Three leak checks on the expanded `fraud_positive` label before any production retrain. Construction note: each NPI carries a single `excl_year_all` = earliest exclusion across LEIE/AHCCCS/NV, and split = that year (train ≤2023, val 2024, test 2025-26), so an NPI lands in exactly one split.

## 0.2 Temporal dating — PASS

- train positives (≤2023): **573** · test positives (2024+ via excl_year_all): **171**
- train ∩ test positive NPIs: **0** (must be 0)
- test positives with an earlier (≤2023) source date — a dating error: **0** (must be 0)
- train positives that also carry a later (>2023) action: 26 (allowed — known-bad by 2023, and NOT in the test set, so no future-label leak)

## 0.3 Negative disjointness (company grain) — PASS

- eval-positive companies: 309 · eval-negative companies: 85266
- companies on BOTH sides of the eval: **0** (0 negative NPIs share a company_id with an eval positive)
- No company appears on both sides of the evaluation.

## 0.1 Top-50 hand-trace — PASS

- Of the top 50 test-period NPIs by score, **41 are true fraud positives** (P@50 = 0.82). Source breakdown: {'NV': 26, 'LEIE-fraud': 12, 'AHCCCS': 3}.
- Genuine 2024+ **LEIE-fraud** positives in the top 50 (not state-list echoes): **12** (need ≥5).

Hand-traced genuine LEIE-fraud leads (NPI · name · taxonomy · state · year · $net):

| NPI | name | taxonomy | state | excl_yr | net_paid |
|---|---|---|---|---|---|
| 1669733697 | DISCOVERY DIAGNOSTIC LABORATORY INC | 291U00000X | MA | 2025 | 698,071 |
| 1588709604 | NEW JOURNEYS IN RECOVERY | 251S00000X | PA | 2026 | 5,146 |
| 1508112491 | DAVID JUDD | nan | nan | 2026 | 13,617 |
| 1588784508 | ROBERT EYZAGUIRRE | 207Q00000X | CA | 2025 | 595,109 |
| 1013497452 | GOLDEN ROAD BEHAVIORAL HEALTH SVC LLC | 106S00000X | NV | 2026 | 1,628,840 |
| 1457976813 | CACTUS WREN COMMUNITY SERVICES LLC | 320600000X | AZ | 2025 | 10,345,831 |
| 1811365406 | TERA CAMPBELL | 101YM0800X | TN | 2026 | 34,949 |
| 1861603052 | EASTER WATSON | 1041C0700X | IL | 2025 | 105,560 |
