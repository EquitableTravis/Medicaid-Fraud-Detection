# GNN Graph Schema (Phase 2 / GATE 2)

Decided on paper before building. The hard failures are design errors caught here, not at training.

## Node type — homogeneous

One node type: **Provider (NPI)**. Node universe = the full scored set, **617,062 NPIs**, in the
stable row order of `nodes.parquet` (every edge/feature/mask array aligns to `row_index`). The
shared owner/address/etc. are **not** nodes — they collapse into direct NPI↔NPI edges (what plain
GraphSAGE consumes). Heterogeneous (typed Owner/Address/PAC nodes + R-GCN/HGT) is deferred to
Phase 12, only if the homogeneous version beats the baseline.

## Edge types — 5 core, undirected

Two NPIs are linked if they share the key. Ordered by signal-to-noise:

| # | edge type | "share…" | source | normalizer | coverage / note |
|---|-----------|----------|--------|------------|------------------|
| 1 | `shared_owner` | a disclosed CMS owner | owner files (`ASSOCIATE ID - OWNER`) → NPI via PECOS `ENRLMT_ID` | `norm_name_base` / `org_key` | highest value; **limited** (only FQHC/HHA/Hospice/Hospital/Nursing) |
| 2 | `shared_authorized_official` | the NPPES authorized official | NPPES AO last+first | `norm_name_base` | **broad** (~all org NPIs); strongest shell signal in the Neo4j work |
| 3 | `shared_PAC` | a PECOS associate id | PECOS `PECOS_ASCT_CNTL_ID` | — | high value, sparse |
| 4 | `shared_fax` | a practice fax number | NPPES practice fax | `norm_phone` | strong (answering services rarely share fax lines) |
| 5 | `shared_address` | a normalized practice address | NPPES practice address | `norm_address` (street\|city\|state\|zip5, suite split out) | strong for shell rings |

**Optional** (free from the same NPPES pass; enabled with `--with-phone` / `--with-mailing` only if
the graph is too sparse): `shared_phone` (noisier — answering services share phones) and
`shared_mailing_address` (noisier — corporate lockboxes/HQ skew toward legit roll-ups, e.g. the
DaVita-Brentwood / Aveanna-Atlanta hubs).

## Deliberate exclusions

- **SKIP `shared_taxonomy_state`** (every provider with the same taxonomy×state) — would form
  million-edge cliques that drown signal and blow memory; peer context is already in the z-score /
  percentile features. Not built.
- **DEFER `co_bill / referral`** (claims-level overlap from the 238M-row `Spending.csv`) to
  Phase 12 — highest ceiling but real cost; the homogeneous model must earn it first.

## Node-feature addition

Beyond the 42 billing features: **NPI provider age** from the NPPES enumeration date (phoenixing
signal — a new NPI at an old operator's address/owner), extracted in the same NPPES pass. Added in
Phase 4, kept additive and NaN-safe. (Plus light structural features: per-type degree, component size.)

## The mega-clique risk (named here, solved in Phase 3)

A group of *n* NPIs sharing one key → up to *n(n-1)/2* edges. One hospital chain owner (500 NPIs)
= 124,750 edges; a billing-service P.O. box can join thousands of unrelated providers. Mitigation
(Phase 3): **per-type clique caps** — full clique up to a small cap, star-connect to a representative
up to a hub cap, drop above (logged to `edge_drop_log.csv`); tighter caps on the infrastructure-prone
keys (fax/address/phone) than on the operator keys (owner/PAC/AO). Same lesson as the Neo4j
perimeter explosion (7.7M → 62k after the degree cap).

## GATE 2 — decision recorded

Homogeneous Provider nodes · 5 core edge types (owner, authorized-official, PAC, fax, address) ·
optional phone/mailing · skip taxonomy-state peer edges · defer claims co-bill · add enum-age
feature. → proceed to Phase 3.
