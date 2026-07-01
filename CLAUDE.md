# Project Context: medicaid-fraud-detection — Supervised Model Build

## What this repo does
A pipeline that detects suspicious Medicaid billing and outputs ranked **company-level**
fraud leads. Those leads feed Equitable Claims' advertising (detected fraud-suspect
companies → advertise to potential whistleblowers/insiders at those companies).
`select_advertise_leads.py` writes ad targets to `~/Desktop/linked in ads/`.

## CRITICAL: data lives OUTSIDE the repo
- All real data is under **`~/Desktop/Data/`** — NOT the repo's `data/` folder.
- `preclean/` = raw inputs. Never recompute features; treat feature parquets as inputs.
- Key data subfolders: `features/`, `integrated/`, `detection/tables/`, `Model/`.

## Existing pipeline (already built, do not rewrite)
Lives in `src/attempt_2/`. Three detection layers (see `leads/detect.py`):
- **Layer 1** — hard rules: billed-while-excluded (LEIE), physically-implausible rates.
- **Layer 2** — anomaly scoring on size-normalized rate features vs taxonomy peers.
  Uses robust-z = 1.4826*(x-median)/MAD within peer group, clipped ±50, signal at
  z>=3.5. `anomaly_score` = SUM of clipped z over fired features → unbounded
  **0–215 z-scale** (NOT 0–1). Stored in `detection/tables/fraud_leads.parquet`.
- **Layer 3** — low-confidence ownership track (probable excluded owner), kept separate.

Company rollup produces a separate **0–1** score (`company_anomaly_score`, mean of
peer-percentile concepts) in `Model/company_scores_full.parquet`. The per-NPI z-scale
and the company 0–1 score are DIFFERENT constructs — there is no formula converting
one to the other.

## THE NEW MODEL PLAN (supervised LightGBM, branch: feat/model-scaffold)
Goal: a supervised model at the **NPI level**, rolled up to company LATER.
- Code goes in repo `src/model/`. Model data in `~/Desktop/Data/Model/`.
- **Label:** `provider_on_leie` (provider appears on the LEIE exclusion list).
- **Algorithm:** LightGBM, binary classification.
- **Train at NPI grain**, then aggregate predictions to company afterward.

### PU (Positive-Unlabeled) learning design — "confident-clean negatives"
LEIE positives are reliable, but unlabeled != negative. So:
- Keep **ALL** positives.
- Negatives = only **LOW-anomaly** providers (confident clean).
- **Hold out** high-anomaly and unscored providers (ambiguous — neither pos nor clean-neg).

### NEVER use these as features (label leakage)
- `provider_on_leie` (it IS the label)
- all `facility_*excluded_owner*` columns
- `excluded_owner_role`
- `any_billed_after_exclusion` / `billed_after_exclusion` / `excluded_after_billing`
- any "probable excluded owner" field

### Feature source
`~/Desktop/Data/Model/provider_features.parquet` — 617,062 NPIs × 52 cols.
Includes raw volume/dollars, rate features, and peer-normalized features in two
peer bases: `_tax` (taxonomy) and `_taxstate` (taxonomy×state), each with a robust-z
(`_rz_`) and percentile (`_pct_`) variant. 578 LEIE positives (0.094% base rate).

### Evaluation (NOT accuracy)
PR-AUC, recall of held-out LEIE positives, precision@K. Base rate is ~0.1–0.2%, so
accuracy is meaningless. Report PR curves and ranked precision.

## WHAT WE JUST DID (data prep, already complete & verified)
All done with pandas/pyarrow, originals preserved. New files in `~/Desktop/Data/Model/`:

1. **`provider_features_scored.parquet`** (617,062 × 57)
   = `provider_features.parquet` LEFT JOINed on `npi` with 5 score columns from
   `detection/tables/fraud_leads.parquet`: `anomaly_score`, `n_anomaly_signals`,
   `anomaly_lead`, `not_scored`, `not_scored_reason`. 1:1 join, 0 unmatched.
   (33,977 rows have null anomaly_score — these are the `not_scored=True` providers,
   legitimately unscorable: low volume / no peer group.)

2. **`provider_features_pu.parquet`** (308,038 × 57) <- **USE THIS FOR TRAINING**
   PU-filtered from the scored file:
   - Keep ALL 578 LEIE positives (0 dropped).
   - Keep clean negatives = scored AND `anomaly_score < 0.5` -> 307,460 rows.
   - Dropped: 33,920 not_scored + 275,104 high-anomaly (>=0.5) negatives.
   - NOTE: on this data, no scores fall in (0, 0.5), so the clean negatives all have
     anomaly_score EXACTLY 0 (zero signals fired). Unambiguous clean set.
   - Resulting positive rate: 0.1876% (578 / 308,038).

(For reference, a company-level analogue exists too:
`Model/company_scores_filtered.parquet`, 388,275 rows — confident-clean companies
<=0.7 plus all 578 LEIE-positive companies. Company grain, not for NPI training.)

## Conventions / guardrails
- Backend runs LOCALLY (host Mac), not Docker.
- Do not modify the existing attempt_2 pipeline; build new code in `src/model/`.
- Treat all parquet feature files as immutable inputs; write new outputs, never
  overwrite source files.
- Remember the leakage column blocklist above when assembling the feature matrix.

## MODEL SCAFFOLD: DONE (2026-06-11, MERGED TO MAIN — `src/model/` is live)
`src/model/` is built and ran end-to-end on real data:
- `config.py` — paths, label, leakage blocklist + **detector-score exclusions**
  (`anomaly_score`/`n_anomaly_signals`/`anomaly_lead`/`not_scored`/`not_scored_reason`
  are NOT features: PU negatives were selected on `anomaly_score == 0`, so those
  columns encode the sampling design, not provider behavior).
- `data.py` — loads the PU frame (asserts 308,038 rows / 578 positives / all
  negatives anomaly_score==0), builds the shared leakage-free matrix
  (**42 features**: 57 cols − 4 identifiers − 6 leakage − 5 detector), stratified
  80/20 split (seed 42). Train and inference both go through `build_feature_matrix()`.
- `train.py` — LightGBM binary, **heavy regularization is load-bearing** (only 462
  train positives): `num_leaves=15, min_data_in_leaf=100, lambda_l2=10,
  feature_fraction=0.6, lr=0.03`, early stopping on val average_precision.
  Selected by 3-seed comparison (loose params → PR-AUC ~0.01; these → 0.35–0.47).
  **Held-out val: PR-AUC 0.465 (base rate 0.19%), ROC-AUC 0.931, P@100 0.53,
  recall@1000 63%, top-decile lift 7.9x.** Artifacts + `MODEL_REPORT.md` →
  `~/Desktop/Data/Model/artifacts/` (booster `lgbm_leie.txt`, `feature_list.json`
  incl. categorical levels, `metrics.json`, PR curve, val predictions, importances).
- `score.py` — scores all 617,062 NPIs (`provider_features_scored.parquet`),
  reapplies training categorical levels, then company rollup via
  `detection/tables/npi_to_company_map.parquet` (non-fan-out asserted).
  **`score_reliable = NOT not_scored` gate is essential**: the 33,920 unscoreable
  providers get pure missing-value extrapolation (median score 0.9996 at median
  $0 net_paid) — flagged, never ranked; company scores aggregate reliable
  constituents only. Outputs → `~/Desktop/Data/Model/scores/`
  (`provider_model_scores.parquet` w/ segment + rank_reliable,
  `company_model_scores.parquet`, top-500 CSV, `MODEL_SCORING_REPORT.md`).
- Segment separation (the PU design working): LEIE positives mean score 0.875,
  clean negatives 0.0001, held-out high-anomaly spread between (mean 0.225) —
  reliable top-1,000 = 989 high-anomaly candidates + 11 LEIE. NOTE: universe
  recall@K of LEIE is LOW and that is expected (unlabeled ≠ negative; the top
  ranks are the not-yet-caught candidates) — judge the model on held-out val.
- `lightgbm>=4.3` added to requirements.txt (installed locally for python3.13).

## LEAD LIST EXPORT (2026-06-11, `src/model/export_leads.py`, on main)
Agreed selection: **top 5,000 companies by `company_model_score_max` with
>= $10M consolidated billing** (size-defined list, NOT a score threshold — the
model's scores are uncalibrated and a "0.7" here is NOT comparable to the
unsupervised 0.70 bar). Output: `~/Desktop/Data/Model/model_leads_top5000_over10m.csv`
($354.5B billing; 16,751 companies were eligible; scores in-list run 1.0 down
to 0.008 — only ~2,687 exceed 0.7, so the tail is low-confidence padding).
Null rollup names (single-NPI companies) resolved from the best constituent
NPI's org_legal_name. Top leads: behavioral-health/treatment orgs.

## FP SCREENING APPLIED (2026-06-11, `src/model/screen_leads.py`, on main)
The validated build_final_leads screens (imported verbatim from attempt_2) ran
on the model lead list; company specialty = dominant (highest-billing)
constituent NPI's taxonomy. **Removed 967 of 5,000** (hospital_taxonomy 276,
hospital_name 196, government 192, fqhc 149, public_academic 63,
national_nonprofit 50, tribal 41) → **`model_leads_top5000_over10m_screened.csv`
(4,033 leads, $236.6B)** + `..._removed_audit.csv` (quarantine, never delete).
Notes: SOUTHCENTRAL FOUNDATION survives (no keyword matches — same behavior as
the unsupervised FinalLeads screen); 66 leads are individual NPIs with no org
name → "UNKNOWN NAME (NPI x)" (export_leads now treats blank org_legal_name as
missing; person-name resolution from NPPES is a possible follow-up).

**FINAL CUT (2026-06-11): score >= 0.90.** First cut was 0.70 (2,168 screened
leads), then tightened to 0.90 after held-out validation showed the bands
differ sharply: val precision 0.897 for score>0.9 (35/39) vs 0.40 for 0.7–0.9
(4/10, small n); all 9 known-LEIE companies in the list sat above 0.9.
`export_leads --min-score 0.9` → screen_leads →
**`~/Desktop/Data/Model/output/final/model_leads_score090_over10m_screened.csv`
= 1,816 leads, $110.7B** (+ removed_audit, 415 institutional FPs). This is THE
model handoff list. The 0.7–0.9 band (352 companies, $24.9B) is kept in the
score070 files as a second-tier reserve. NOTE: Travis reorganized the Model dir
— handoff CSVs live in `Model/output/final/`, superseded ones in
`Model/output/unimportant/`; scores/artifacts unchanged. Scores are saturated
near 1.0 (uncalibrated sigmoid; ranking valid — 2,000 distinct values in top
2,000; a raw-margin column is a pending nice-to-have).

**NPI-GRAIN LIST (2026-06-11, `src/model/export_npi_leads.py`, on main):**
same process WITHOUT company rollup, billing bar $5M: reliable score >= 0.90 &
net_paid >= $5M → 2,658 NPIs → FP screens (own org_legal_name +
primary_taxonomy; individuals can only be taxonomy-screened) removed 246 →
**`Model/output/final/model_npi_leads_score090_over5m_screened.csv` = 2,412
leads, $51.9B, 13 on-LEIE** (+ unscreened + removed_audit). Composition: 2,398
ambiguous_high_anomaly / 13 leie_positive / 1 trained-clean; 2,294 orgs + 118
individuals ("INDIVIDUAL PROVIDER (NPI x)" names). Top leads AZ-heavy
behavioral health. Cols incl. segment, provider_on_leie, practice_state.

## REPO MOVED (2026-06-11): `EquitableTravis/Medicaid-Fraud-Detection` is the repo
The repo now lives under Travis's EquitableTravis account — that is `origin`
and the source of truth (all branches + full history mirrored). The old
`travishub09/medicaid-fraud-detection` copy is retired (kept as the
`travishub09` remote in the local clone for reference only — do not push
there). `gh` is authenticated as EquitableTravis. Local clone:
`~/Desktop/medicaid-fraud-detection`. Note: the old repo's main had
"changes via PR" branch-protection rules; check whether the new repo needs
the same before assuming direct pushes to main are fine.

## MODEL vs ANOMALY BACKTEST (2026-06-11, `src/model/backtest_leie.py`, on main)
Head-to-head LEIE backtest, EXACT src/backtest methodology, identical universe
(419,342 companies scored by both; 336 fraud-relevant hits; anomaly reproduces
its published 1.99x — harness validated). Because the model TRAINED on LEIE,
headline rows are also computed held-out (companies containing the 462
train-split LEIE NPIs removed → 70 never-seen hits: val-split NPIs + names):
- **All hits: model 8.66x [CI 8.28–9.03] vs anomaly 1.99x [1.61–2.42]**
  (model row inflated by memorization — do not quote alone).
- **HELD-OUT: model 3.57x [CI 2.43–4.75, p=0.001] vs anomaly 1.71x
  [0.89–2.65, p=0.052 — not significant on this subset].** Model CI excludes 1
  and survives within billing quartiles (Q1 2.5 / Q2 1.33 / Q3 4.8 / Q4 5.0).
**VERDICT: the supervised model beats the unsupervised score on identical
ground, ~2x lift even on exclusions it never saw.** Outputs:
`~/Desktop/Data/Model/backtest/` (model_backtest.json, MODEL_BACKTEST_REPORT.md).
Caveats: held-out n=70 (wide CIs), val NPIs used for early stopping (mildly
seen), caught-fraud ground truth. config.py inputs now resolve via
`find_input()` rglob — Travis reorganizes Model/ subfolders (input/,
output/final/), never hardcode a level.

## TOP-10 LEAD RESEARCH DONE (2026-06-11, report outside repo)
Deep public-records research on the top 10 of `model_leads_CLEANED.csv` (Travis's
cleaned 1,752-row list in `Model/output/final/`; ranks 3+6 = same operator).
Report: **`~/Desktop/Data/Model/output/final/report/TOP10_LEAD_RESEARCH_REPORT.md`**
(names/details stay OUT of this public repo). Result: **4 of 9 distinct
entities corroborated by official government fraud actions** (a 10-yr state
Medicaid termination, two AHCCCS credible-allegation-of-fraud suspensions, one
pair of 2024 federal fraud charges whose alleged receipts match our flagged
total exactly) — none of the 4 on LEIE, so the model found them from billing
patterns alone. The 5 WEAK leads have identifiable FP causes: big legit
multi-site orgs (2), geocoding/mailing-address artifact, facility-billing
attribution to an individual attending NPI, high-base-rate segment.
REVISED after internal claims + feature audit: rank 7 (individual NPI)
ESCALATED — web research's benign theory was contradicted by our claims data
($15.9M of a single hospice per-diem code billed under the individual NPI);
one WEAK lead's statistical signal survives scrutiny (intensity p99.5 + extreme
yoy growth); one WEAK lead has NO ≥p99 features at all → its ~1.0 score is
segment-driven (model profiles AZ+behavioral-health) — a model bias to fix.
LEARNED: (a) WEAK = "no public corroboration", NEVER "cleared" — web research
only confirms already-caught fraud; always follow with the internal signal
audit (which features fired, code mix, velocity); (b) cross-reference lead
lists vs the AHCCCS suspension PDF to split already-caught from not-yet-caught
(ad targets must be the latter); (c) candidate features: NPI enumeration age /
operating history; (d) NEVER include individual-NPI leads in outreach without
manual review.

## PURSUIT PIPELINE (2026-06-11, `src/model/build_pursuit_pipeline.py`, on main)
Cross-referenced all 1,752 cleaned model leads against public enforcement
lists — AHCCCS suspensions PDF (212 NPIs/160 names, NPI + conservative
3-token name match), Nevada Medicaid sanctions PDF (788 NPIs), federal LEIE
(8,429 NPIs); list copies cached in `Model/enforcement_lists/` (re-pull
monthly). Result: **55 already_caught** (46 AHCCCS NPI, 2 AHCCCS name-only,
6 NV, 3 LEIE — incl. all 4 STRONG top-10 leads, evidence strings attached)
and **1,697 CLEAR re-ranked as the pursuit pipeline** →
**`Model/output/final/pursuit_pipeline.csv`** (one CSV, clear-first with
pursuit_rank, caught rows kept at bottom with evidence). Rationale: caught =
validation only (first-to-file); CLEAR = the qui tam / whistleblower-ad
targets. Run from repo root: `python -m src.model.build_pursuit_pipeline`.

## BATCH-2 LEAD RESEARCH (2026-06-11, pursuit ranks 7-16, report outside repo)
Same protocol (web agents + mandatory internal p99 feature audit) on the next
10 unresearched pursuit leads. Report:
`~/Desktop/Data/Model/output/final/report/PURSUIT_BATCH2_RESEARCH_REPORT.md`.
Verdicts: 1 STRONG (NC ABA chain — the "uncaught twin" of the publicly-audited
cohort: ~#2-biller scale, absent from all press, quota complaints, opaque
owners), 3 MODERATE (COVID-era CA lab w/ textbook profile; NC therapy group
whose 8 rate features ALL fire p99 despite a clean web profile — audit
UPGRADED it; CA FQHC look-alike w/ 30% margins + $1.26M non-clinician CEO),
6 WEAK (mostly large legit nonprofits whose 990 revenue reconciles to billing;
one defunct Philly provider; one segment-driven lab score with zero fired
features). SYSTEMATIC FINDINGS: (1) **FQHC-screen gap** — 2 HRSA FQHC
look-alikes passed the screen because OUR taxonomy extract is stale/differs
from the live registry (their current primary tax IS 261QF0400X); fix = refresh
NPPES or screen vs the HRSA look-alike site file (or flag instead of remove);
(2) the internal p99 audit is now MANDATORY — it changed 3 verdicts across two
batches; (3) single-NPI billing-scale puzzles (2 leads) need claims-grain
verification before any outreach.

## GRAPH PIPELINE (2026-06-12, `src/graph_work/`, on main)
Neo4j fraud-lead graph: shared-identifier links (authorized official / address /
phone / fax / owner / excluded person) across the 1,752 entity-resolved leads so
one-operator shells surface as connected subgraphs. LEADS, not conclusions.
- `etl/build_graph.py` 5 stages, CLI `python -m src.graph_work.etl.build_graph
  --leads <model_leads_CLEANED.csv> --nppes --owners <glob> --pecos --leie --out
  ~/Desktop/data/graphing [--no-widen]`. Stage 1 runs on --leads alone.
- `cypher/01_schema|02_load|03_leads.cypher` + README (Docker Neo4j+GDS+APOC).
- Outputs to `~/Desktop/data/graphing` (HIPAA, outside repo); only code committed.
KEY DESIGN NOTES (learned from real runs): (1) Stage-1 spine = exactly 1,752
companies / 17,805 NPI edges. (2) NPPES streamed in 200k chunks; perimeter widen
MUST be capped — unbounded it pulled 7.7M NPIs (shared infra); MAX_PERIMETER_PER_KEY=15
+ person-widen-needs-phone-tiebreaker -> 62,326 perimeter NPIs. (3) owner files key
on ENROLLMENT ID, NOT NPI — Stage 3 bridges owner->NPI via PECOS ENRLMT_ID. (4)
`_clean()` rejects literal 'nan' (dtype=str+NaN) or empty NPPES fields form bogus
rendezvous hubs. (5) reads use encoding_errors='replace' (PECOS has 0xa0 bytes).
FIRST RESULT: top shared authorized official = BUCKHALTER|MATTHEW across 15
distinct non-whitelist companies (shell-operator signal). whitelist_candidate (55
mega/govt/hospital orgs) excluded from lead queries.
DEBUG PASS 1 (code review, 2026-06-12): org_name 'nan' for 42,060 individuals
(NaN is truthy) -> _clean+fallback; perimeter isin(dict); POSSIBLY_SAME_AS ran
before owner/PECOS persons -> moved to end + empty-name guard.
DEBUG PASS 2 (LIVE-TESTED in Docker Neo4j 5.26 + GDS + APOC, 2026-06-12 — graph
fully loads & all 8 lead queries run): (1) apoc.load.csv is apoc-EXTENDED not the
bundled core -> rewrote loaders as plain LOAD CSV + ETL writes header-only
placeholders so missing-stage loads no-op; (2) perimeter NPI nodes were never
created (enriched loader used MATCH; nodes_npi.csv has only lead NPIs) -> MERGE,
npi-address edges 35,608->157,850; (3) betweenness returned nothing (directed
projection makes identity nodes sinks) -> symmetric/undirected projection +
samplingSize (4m43s->6s); (4) WCC 550-co/$45B hairball -> hub-degree filter (>20
dropped) + case-files capped <=50. GRAPH CENSUS: 78,926 NPI / 67,182 phone /
43,984 address / 13,507 person / 1,752 company / 235 org; 16,684 POSSIBLY_SAME_AS.
RESULTS: validation reconnects 75 same-operator pairs via person / 74 address / 50
phone; top operator BUCKHALTER|MATTHEW = 15 companies $3.2B; case files = Personal
Touch Home Care (4, 3 pre-flagged), Pinnacle Treatment Centers (8, 7 pre-flagged).
To view: `docker start fraud-graph` then http://localhost:7474 (neo4j/fraudgraph),
paste queries from cypher/03_leads.cypher. Container name `fraud-graph`.

**FULL GRAPH FINDINGS (2026-06-12):** all 8 lead queries + ABA deep-dive run;
report at `~/Desktop/Data/Model/output/final/report/GRAPH_FINDINGS_REPORT.md`
(raw outputs: `~/Desktop/data/graphing/_q_results.txt`). KEY: (1) top "shell"
signals are mostly PE ROLL-UPS the model flagged as separate companies — Aveanna
family (7 leads, $1.86B, BUCKHALTER=Aveanna exec), DaVita renal, BrightSpring;
treat corporate families as single review units + whitelist public-co identity
nodes. (2) Actionable small clusters: Pinnacle Treatment ×8 ($315M, 7 pre-flag),
Personal Touch ×4 ($839M), Health Acquisition+Pyramid fax pair ($1.06B, unflagged),
DC-area 6-co cluster. (3) ABA: Apogee AZ name-variant triplet (sibling bills $1.7M
@ model 0.998, below company bar — sub-threshold fragmentation pattern); Step Ahead
9 state LLCs created in 2 weeks; Highlights HQ hosts GA/VA expansion entities +
multi-state Empyrean Hospice (dossier addendum). (4) LEIE 2-hop person matches =
common-name noise (name-only matching); 3 direct-excluded leads consistent with
pursuit pipeline. (5) Top betweenness brokers via PECOS = national chains
(Walgreens/DaVita/Lincare — PAC org names not stored, ETL TODO); via NPPES-AO =
COMBS/HEINE/SWEETEN/FULLER/PETERS/PAI/BLIATOUT(=HALO CEO, corroborates batch-2)/
DRAKE — unresearched broker names are the next research batch.

## GNN TRACK (2026-06-15, `src/gnn/`, branch `feat/gnn-scaffold`)
New training track: replace the per-NPI LightGBM SCORE with a GraphSAGE GNN that
scores each NPI from its neighborhood (shared owner/PAC/address), then feed scores
into the SAME downstream pipeline (rollup → $10M → screens → dedupe). Full plan:
12 gated phases, locked decisions: anomaly_score NOT a feature (it defines the
reliable-neg class — circular); loss trains only on labeled nodes (LEIE pos ∪
reliable neg), unlabeled pass messages but zero gradient; NO Neo4j on training path
(in-memory X + edge_index); inductive GraphSAGE; temporal AND structural
(owner-group) splits. Code in `src/gnn/`, artifacts `~/Desktop/Data/Model/gnn/`,
dedicated Python 3.11 venv `.venv-gnn` (torch 2.12 + torch_geometric 2.8, MPS).
**DONE — GATE 0+1:** node table built — `Model/gnn/nodes.parquet` (617,062 NPIs,
42 leakage-free features via reused `model.data.build_feature_matrix`; positive=578
/ reliable_neg=307,460 / unlabeled=309,024; universe{pos∪neg}==PU npi set exactly).
Reuse map: `model/score.py::rollup_to_company`, `export_leads`, `screen_leads`,
`build_pursuit_pipeline` (Phase 10); `graph_work` norm helpers + `load_pecos` (edges).
NOTE: EXCLDATE only in `preclean/Caught.csv` (temporal split, Phase 5);
`model_leads_CLEANED.csv` not script-reproducible → Phase 10 targets the screened
chain.
**EDGE SET EXPANDED** (beyond the PDF's 3) after a data-leverage review: 5 core
edge types — `shared_owner` (owner files via PECOS), `shared_authorized_official`
(NPPES AO — broad; owner-files only cover 5 categories), `shared_PAC`, `shared_fax`,
`shared_address`; optional phone/mailing; SKIP taxonomy-state peer edges; DEFER
claims co-bill (Spending.csv 238M rows) to Phase 12; ADD NPI enum-date→provider-age
feature. All from one NPPES stream + PECOS + owner files.
**DONE — GATE 2+3** (`src/gnn/SCHEMA.md`, `build_edges.py`): **3,986,034 edges**
across 5 types → `Model/gnn/edges_*.npy` + `edges_all.npy`/`edge_type_all.npy` +
`nppes_node_attrs.parquet` (age, 607,863/617,062 non-null) + `edge_drop_log.csv`.
Graph healthy: largest component 30.4% (not a blob), 69.6% connected (not dust),
degree max 2,897/p99 87/p50 2; per-type clique caps (full≤cap, star≤hub, drop
above) dropped 67 hubs = the corporate roll-up infra (DaVita's WEY|SAMUEL across
1,817 NPIs, Walgreens PAC 1,907). Known operators connect (Total Renal Care,
Bayada, Pediatric Services of America=Aveanna, Pinnacle).
Re-audit (`check_edges.py`) caught owner edges 91% duplicate (pair sharing multiple
co-owners) → deduped within-type: **2.92M edges** (was 3.99M), connectivity
unchanged. Only **58.5% of 578 positives connected** (240 solo bad actors → GNN
upside concentrated on the connected set).
**DONE — GATE 4+5** (`splits.py`, `features.py`): temporal×structural split,
group=company_id (no straddles). Cutoffs cumulative-positive-driven (year
percentiles collapse on 2022-26 skew): train ≤2023 (270 pos), val 2024 (130),
test 2025-26 (178) — positives in every split. Leakage: only **3 positive↔positive
edges cross splits** (36% all-edge crossing is harmless neg/unlabeled; Phase-8
ablation is the real test). `split_masks.parquet`. Features: **X_num (617062,47)**
= 39 billing + 8 graph (provider_age, per-type degree, component size); median-
impute+standardize fit on TRAIN only, ±5 clip, no NaN/inf. X_cat (617062,3) vocab
{entity:3, taxonomy:772, state:84}. `X_num.npy`/`X_cat.npy`/`features_meta.json`.
**GATE 6+7** (`model.py`, `train.py`): inductive FraudSAGE (numeric + entity/
taxonomy/state embeddings → N×SAGEConv → logit/node), focal loss. Switched
NeighborLoader→FULL-BATCH (pyg-lib/torch-sparse no wheels for torch 2.12; graph
fits in memory, exact). HP-parameterized for tuning (--tag avoids clobber).
**KEY FINDING — GNN underperforms.** GNN val PR-AUC ~0.10-0.15, BELOW the no-graph
HistGBM on the SAME temporal split (**0.195**). The original LightGBM **0.465 was a
RANDOM-split mirage** — the honest temporal bar is ~0.20 for ANY model. Graph adds
no NPI-level fraud signal: fraudsters are LESS connected (58.5%) than clean
negatives (70.8%), so message passing OVER-SMOOTHS positives toward clean
neighbors (active harm, not just neutral). The shell-ring signal lives at COMPANY
grain, not NPI grain. Defensible negative result (plan: "ship the simpler model").
**VERDICT — MERGED TO MAIN as a documented negative result** (`src/gnn/VERDICT.md`,
branch `feat/gnn-scaffold` merged). Head-to-head, SAME temporal split, every metric:
no-graph GBM beats GNN — val PR-AUC 0.195 vs 0.145, val P@100 0.40 vs 0.20, test
0.117 vs 0.093. **DECISION: production scorer stays LightGBM (`src/model/`), no
pipeline change.** GNN scaffolding kept (full 7-gate pipeline: node table, edge
builder w/ caps, splits, features, FraudSAGE, full-batch trainer) — reusable if a
COMPANY-grain GNN is tried later (where shell-ring signal actually lives). Reg pass
not pursued (ceiling capped by no-NPI-signal). venv `.venv-gnn` (gitignored).

## GRAPH-FEATURES PROBE (2026-06-15, `graph_features/`, branch `feat/graph-features`)
Follow-up to the GNN: instead of a neural rebuild, ADD graph-structure feature columns
to the existing model and let LightGBM decide if structure helps (cheap, safe — features
can't degrade a tree). Path A (augment the NPI model; no company model exists to clone);
`src/model/` byte-for-byte untouched; reuses the GNN edges + temporal split; libs added
to `.venv-gnn` (lightgbm/networkx/shap). GATE 0: harness reproduces existing model
(0.4653 vs 0.465). **VERDICT — NEUTRAL, keep existing LightGBM** (`graph_features/VERDICT.md`):
head-to-head val PR-AUC billing 0.241 [0.027-0.399], +structure 0.230 (Δ-0.012, within
noise), +proximity 0.038 (COLLAPSED — train-label echo: excluded source set ≈ train
positives → dist=0; only 5,544/617k nodes within 4 hops of a ≤2023 exclusion), +all 0.008.
SHAP: model WOULD use graph feats (comp_size rank 1) but they don't generalize. **Two
independent methods (GNN message-passing + tree features) now agree: no NPI-grain
structural fraud signal** (fraudsters under-connected); shell-ring signal lives at COMPANY
grain. Production model unchanged; reusable harness kept for a possible company-grain model.

## LABELS TRACK (2026-06-15, `labels/` + `src/model/train_fraud.py`, MERGED to main)
Attacked the model's real bottleneck — only 578 LEIE positives, ~151 of them non-fraud
(license/loan). Built an expanded **`fraud_positive`** label (889) = fraud-relevant LEIE
(EXCLTYPE a1/a2/a3/b1/b2/b3/b7, 420) ∪ AHCCCS suspensions (191, 189 net-new) ∪ NV sanctions
(331, 274 net-new), each with an exclusion year for the temporal split (`labels/build_labels.py`
→ `Model/labels/expanded_labels.parquet`). Reuses the pursuit-pipeline AHCCCS/NV parsers.

**The metric win is real.** Head-to-head on the identical temporal split, common eval target,
3 seeds, same HP — only the training label varies (`labels/train_eval_labels.py`): all-LEIE
(current) val PR-AUC **0.317** [0.04–0.51, one seed collapses]; expanded **0.573** [0.56–0.58,
stable]. Survives the bias check (pure-LEIE-fraud target, no shared positives: 0.427 vs 0.248).
Expansion (not cleaning) drives it; it also kills the ~250-positive seed fragility.
`labels/VERDICT.md`.

**Productionized + gated per `Production_Retrain_Plan.pdf`** (`src/model/train_fraud.py` — ONE-
variable change from `train.py`: label only; `train.py`/`score.py` left byte-identical as the
runnable fallback; frame rebuilt from the full universe since 371/889 positives sit outside the
PU frame; negatives = confident-clean + **company-disjoint** from positives). GATE 0 leak checks
clean (`labels/VERIFY.md`); GATE 2 reproduces through the production path (val 0.574 / test 0.550).

**But GATE 4 says DO NOT cut over the leads** (`labels/LEAD_DIFF.md`). Controlled lead diff
(`labels/build_leads_fraud.py`, reuses the chain verbatim): 592 newly-surfaced, 605 dropped,
0 confirmed catches lost. Added 3 institutional screens (`screen_institutional_extra`: school
districts via LEA taxonomy 251300000X, tribal-by-name, academic/safety-net) → +55 FP removals.
**Unbiased re-judge of a random 30 new leads: 1 weak fraud-candidate, 16 legitimate institutions
(Albany Medical College, Tulalip Tribes, LA Free Clinic, DaVita JV, AbleNet, state-run DHS...),
13 inconclusive.** The 42 billing features can't separate legitimate from fraudulent providers
that share a billing shape — same lesson as the GNN/graph tracks, one level up. Production scorer
+ advertising leads UNCHANGED; `fraud_positive` model + `leads_new` kept under
`Model/fraud_positive/`, not promoted. The 3 new institutional screens are a keeper regardless.

## HARVEST + FORWARD TEST (2026-06-15, `src/model/{forward_test,explain_leads,finalize_handoff}.py`, MERGED to main)
After the labels track, decided (with user) to stop iterating on the model — three tracks
(GNN, graph-features, expanded label) all hit the same wall: the 42 billing features describe
billing *shape*, not fraud. So: ship the cheap wins + measure the real value.

- **3 institutional screens folded into production** (`build_final_leads.classify`, shared by
  `screen_leads` and the unsupervised FinalLeads pipeline): school districts (LEA taxonomy
  `251300000X` + name list), academic/safety-net (FACULTY / MEDICAL COLLEGE / DISTRICT MEDICAL
  GROUP), tribal gap patched (+TRIBES, INDIAN COMMUNITY — word-boundary missed "TULALIP TRIBES").
  +48 FPs removed from the production list.
- **Per-lead SHAP drivers** (`explain_leads.py`, `pred_contrib`): every lead carries its top
  driving features (utilization intensity vs taxonomy peers, paid-per-patient, taxonomy).
- **NPPES name resolution** (`finalize_handoff.py`): all 66 UNKNOWN individual-NPI leads resolved
  to person names via `provider_dim.provider_name`. No bare NPI reaches the ad team.
- **Honest score-band calibration** (`finalize_handoff.py`): LEIE lift is NOT monotonic — ~base
  rate across 0.5–0.99, then **12.7x only at ≥0.99**. So "high confidence" = ≥0.99 (1,269 leads);
  below ~0.9 leads rest on the $10M + screen filters, not the score. → `model_leads_handoff.csv`.
- **FORWARD TEST started** (`forward_test.py`): froze 2026-06-15 baseline (617k NPIs, BOTH the
  production all-LEIE and fraud_positive scores; LEIE snapshot 2026-06-02). When the next LEIE is
  re-pulled, `measure --future-leie <Caught.csv>` reports precision@K + lift on genuinely-future
  exclusions for both models — the honest production-value number AND the label tiebreaker the
  backtests couldn't give. → `Model/forward_test/`.

## NO-GEO MODEL + A/B DEEP ADJUDICATION (2026-06-16/17, `labels/`, on main)
- **No-geo model** (`labels/{ablate_geo,leads_nogeo,pipeline_nogeo}.py`): fraud_positive trained
  WITHOUT `practice_state`/`primary_taxonomy` → flags purely on billing behavior, no guilt-by-
  specialty. Detection holds (test PR-AUC 0.518 vs 0.550; P@50 1.00). Concentration barely changes
  (the AZ/behavioral skew is a data property, not a categorical-feature artifact — production list
  has the same ~6% AZ). Produced the NOVEL list (`~/Desktop/NoGeo_Model_Leads/`, 3,237 leads).
- **A-vs-B deep checks** (`labels/{audit_cleaned_leads,explain_flags}.py`): 24/25 flags are real
  peer-rate anomalies (not size, not guilt-by-specialty). Top-20 enforcement adjudication +
  forensic "do the numbers have a benign explanation" passes. KEY honest finding: the temporal
  benchmark favors the expanded model, but production (all-LEIE) leads had more NON-circular
  confirmed hits (fraud_positive re-finds its own AHCCCS training labels). Both surface real fraud
  (Numotion FCA, Capstone/United Youth convictions, AZ AHCCCS cluster). "National chain ≠ clean":
  graph operator-families include DaVita/Fresenius/ResCare with real fraud settlements.

## MODEL D — ALL-STATE EXCLUSIONS (2026-06-23, `labels/{train_model_d,pipeline_model_d}.py`, on main)
**Model D = trained on ALL exclusions: fraud-relevant LEIE ∪ all 38 state Medicaid exclusion lists**
(pulled via OpenSanctions `us_*_med_exclusions`, NPI-matched; saved
`Model/labels/all_state_exclusions_npis.csv`). 13,859 distinct national state-excluded NPIs, **2,231
in our 617k universe** → Model D positives = **2,386** (vs C's 889, A's 578; net-new +1,497).
**Wins the temporal benchmark on EVERY target incl. the neutral one** (future fraud-relevant LEIE,
which it didn't over-train on): test PR-AUC **D 0.556 vs C 0.464 vs A 0.274**, val P@50 0.91 vs 0.72
vs 0.50. Real generalization (more enforcement labels → better fraud detection), not circular.
Caveats: benchmark win ≠ lead win (the GATE-4 lesson) — still needs lead-quality + forward-test
confirmation. Lead list (filters: rollup → $10M → **score ≥ 0.90** → hardened screens → names/SHAP/
bands) → `~/Desktop/Model_D_Leads/` (4,302 leads; 4,203 novel).

## CARE-MODEL PEERING (2026-06-24, `labels/{repeer_and_train_d,pipeline_model_d_repeer,train_model_d_nogeo}.py`, on main)
Attacked the taxonomy-mismatch FP problem: the rate-anomaly peer features (`*_rz_tax`/`_pct_tax`/
`_taxstate`) z-score each provider against same-`primary_taxonomy` providers, but self-reported
taxonomy mismatches the care/reimbursement model — FQHCs coded "Family Medicine" get scored vs
fee-for-service GPs, so their bundled per-encounter rate looks anomalous from the mismatch alone.
- **Method** (`repeer_and_train_d.care_clusters`/`repeer`): MiniBatchKMeans (K=30) clusters providers
  by OPERATIONAL signature — service breadth (n_distinct_hcpcs), concentration (top_hcpcs_share,
  hcpcs_hhi), visit intensity (lines/patient), scale, tenure, entity type — NOT the payment-dollar
  magnitudes being tested. Recompute robust-z + percentile of the 4 rate metrics WITHIN cluster → 8
  `_*_cluster` cols. Feature-level effect is real: the 3 example FQHCs' paid/claim z drops
  +5.6→+1.7, +8.4→+2.6, +9.8→+3.1.
- **Benchmark** (Model D label, 3 seeds, identical temporal split, neutral/state/all targets):
  care-model-only (REPLACE the 16 tax cols with 8 cluster cols) is clearly WORSE (loses info; neutral
  test 0.431 vs 0.569). But **BOTH (ADD cluster cols on top of taxonomy) is a small, consistent WIN on
  every target and metric** — neutral test PR-AUC 0.569→0.589 & val P@50 0.91→0.95; state 0.502→0.519;
  all_excl 0.509→0.526; CIs tighter (BOTH near-separates from base). First additive feature win since A.
- **BUT the lead-list artifact goal was NOT met** (`pipeline_model_d_repeer.py`, output
  `~/Desktop/Model_D_repeer_Leads/`): re-peering only demoted the 3 named FQHCs modestly (ranks 1→8,
  4→13, 8→18); FQHC/RHC-named in top-50 unchanged (2→2), TOTAL FQHC count rose 113→125, list grew
  4,302→4,810 (+508, +634 high-confidence). Metric up, headline FP pattern still in the leads.
- **VERDICT:** keep the BOTH feature set as a modest Model D improvement; care-model peering is NOT
  the FQHC fix it was hoped to be. Per the GATE-4 lesson (benchmark win ≠ lead win) do NOT auto-promote
  — needs lead adjudication + forward-test first. `train_model_d_nogeo.py` = Model D geo-ablation
  harness (companion, not yet adjudicated).

## Next steps (not started)
0. **Model D**: freeze into the forward test (4-way A/C/no-geo/D); lead-quality adjudication before production.
   Decide whether to fold the care-model `BOTH` features into the frozen Model D variant.
1. **Run `forward_test.py measure`** once a fresh LEIE/AHCCCS pull is available (the pending result).
2. Claims-level signal (238M-row Spending.csv: co-bill / shared-patient / referral edges) — the one
   untapped source with a higher ceiling; only worth it if the forward test shows the lift is real.
3. Company-grain supervised model — where the GNN/graph tracks said shell-ring signal lives.

## LABEL PURITY AUDIT (2026-07-01, `labels/{audit_state_basis,train_label_purity}.py`, on main)
Question (Travis): are the "guaranteed fraud" positives actually fraud? **Answer: mostly no —
and that's OK for training, but not for naming.** Model D's label (fraud-relevant LEIE ∪ 30 state
lists — CLAUDE's earlier "38 lists" overstated; the artifact holds 30 states) = 2,386 in-universe
positives, of which only ~421 are strict fraud CONVICTIONS (LEIE a1/a3/b1/b7 ∪ state records with
explicit fraud text). 193 more are AZ AHCCCS credible-allegation suspensions (allegation-grade).
The other ~1,770: license actions, federal-LEIE mirrors (84 provably NON-fraud LEIE types
re-admitted via state lists — structural hole in the union), program/policy violations (KMAP,
MassHealth, NMAP, overpayment, bad debt), NV "Tier" terminations (tier semantics unpublished),
and ~900 from states that publish no reason (CA/NY/OH/SC/PA). Per-NPI evidence now durable:
`Model/labels/state_exclusion_basis.csv` (all 30 states, OpenSanctions description text; IN
dataset 404s) + `Model/labels/fraud_conviction_labels.csv` (tiered: leie_strict_fraud /
state_fraud_text / az_credible_allegation).

**Controlled 3-arm test on the newest config (no-geo + cluster features, temporal protocol,
neutral future-strict-fraud target, 3 seeds):** D_original (2,386 pos) val PR-AUC **0.667**
[0.666-0.667] / test 0.615 / P@50 0.97; D_conviction (421) **collapses** to 0.056 [0.017-0.134]
(the ~250-positive fragility again); D_convic_az (613) 0.486 / 0.413 / 0.75. Same on the
production LEIE label (172 non-fraud of 578): dropping them HURT (random-split 0.40 vs 0.50) or
tied (temporal). **DECISION: keep training on the full exclusion label (it's an EXCLUSION-RISK
model — impure-but-related positives carry signal); use the conviction tier as the evaluation
target and for lead-tier labeling, never call the training label "guaranteed fraud".** Known
caveat: part of D_original's neutral-target edge is state-exclusion→later-LEIE precedence
(already-caught-by-a-state providers count as "found"); same caveat as the labels-track
circularity note. Next: trey extra-features run (`~/Desktop/data/preclean/trey/` — Google Drive
file, needs manual download) trains D_original label + trey features vs baseline on this harness.

## TREY FEATURES — BIG WIN (2026-07-01, `labels/train_trey_features.py`, branch `trey-data-model`)
Trey's `~/Desktop/data/preclean/trey/provider_scored.parquet` (617,062 NPI-grain rows — exact
universe match — 118 cols from NEW sources: E&M coding levels, Part B/D utilization + drug/opioid
mix, Open Payments, address clustering, market saturation, PBJ staffing, hospice live-discharge,
deficiencies, behavior subscores, shell/graph structure). 63 usable feature cols after dropping
label/leakage (provider_on_exclusion, within_2_hops_of_exclusion, weak_label*, has_excluded_owner,
subscore_ownership_integrity...), detector-derived, and identity/geo/taxonomy cols.

**Controlled A/B on the exclusion-risk config (no-geo + cluster, temporal, neutral future
strict-fraud target, 3 seeds): baseline 48 feats val PR-AUC 0.667 / test 0.615 → +trey 111 feats
val 0.905 [0.894-0.916] / test 0.875 [0.870-0.881]**, P@50 0.99. Biggest single jump in the
project's history. Leak-checked: top trey features solo PR-AUC ≤0.064 (shell_score; others ≈base
rate 0.00035) — broad interaction signal, not a leaky column. Top gain: services_per_bene (75%
null — Part B/D covers ~25% of universe), shell_score, expected_net_paid, subscore_rapid_ramp,
co_location_cluster_size. Trey cols = 35.2% of total model gain. Results:
`Model/labels/trey_ab_results.json`, importances `Model/labels/trey_feature_importance.csv`.
OPEN before production cutover: (1) ask Trey how expected_net_paid/billing_residual and the
subscores were built — if any were fit/tuned against exclusion lists there's designer leakage the
column drop can't catch; (2) universe scoring + lead-list rebuild + adjudication (GATE-4 lesson:
benchmark win ≠ lead win); (3) fold into forward test.
