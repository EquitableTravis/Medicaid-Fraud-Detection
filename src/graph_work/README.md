# graph_work — Neo4j fraud-lead graph pipeline

Builds **shared-identifier links** between the ~1,752 entity-resolved fraud-lead
companies — same authorized official, address, phone, fax, owner, or excluded
person — so that "independent" companies actually run by one operator surface as
connected subgraphs.

> **Everything this produces is an investigative LEAD requiring records-level
> verification — never a conclusion of fraud.** A shared fax is a reason to pull
> records, not proof of a scheme. Person matches to the LEIE are *name-based
> leads* that must be identity-verified (DOB/address) before any use.

## What it does

| Stage | Input | Produces |
|---|---|---|
| 1 spine | `model_leads_CLEANED.csv` | `nodes_company`, `nodes_npi`, `edges_company_npi` |
| 2 NPPES | bulk NPPES (~10GB, streamed) | `nodes_npi_enriched`, `nodes_address`, `nodes_phone`, `nodes_person`, `edges_npi_address/phone/person` |
| 3 ownership | CMS owner files + PECOS | `edges_owns` (owner→NPI, bridged via PECOS enrollment) |
| 4 PECOS | PECOS enrollment | `edges_associated_with` (associate-ID person confirmation) |
| 5 LEIE | OIG LEIE (`Caught.csv`) | `flags_leie_npi`, `flags_leie_person` |

**Perimeter widening (Stage 2):** after locating the lead NPIs, a second NPPES
pass also keeps any NPI that *shares* a normalized address, phone, fax, or
authorized-official key with the lead set — the one-hop ring where hidden shells
surface. Disable with `--no-widen`.

The ring is **bounded** (`MAX_PERIMETER_PER_KEY`, default 15): a key shared by
more than that many outside NPIs is shared *infrastructure* (billing service,
office tower, hospital switchboard, common name) and contributes no perimeter
nodes — without this cap the ring balloons to millions of NPIs. Person-key
widening is further restricted to names that carry a phone tiebreaker (bare
names like `SMITH|JOHN` are too generic to widen on).

**Correctness guards baked in:**
- Phones → digits only, strip leading `1`, reject non-10-digit.
- Addresses → suite split OUT into an edge property so `STE 104` / `SUITE 104` /
  no-suite all merge on the same building key `street|city|state|zip5`; street
  abbreviations standardized.
- Persons → conservative `LAST|FIRST` base + strongest available tiebreaker
  (PECOS associate id > authorized-official phone > middle initial). Same base
  but different tiebreaker ⇒ **distinct** Person nodes plus a `POSSIBLY_SAME_AS`
  edge. Ambiguous persons are **never auto-merged** — a false merge fabricates a
  fraud link.
- `merge_confidence` is carried onto every `HAS_NPI` edge so a low-confidence
  rollup can't masquerade as a discovered link.
- Likely-legit mega-orgs (`n_npis >= 100`, government, or hospital-system flags)
  are marked `whitelist_candidate` and excluded from the lead queries so they
  don't dominate centrality.

## 1. Build the CSVs

```bash
# from the repo root. Stage 1 runs on --leads alone; later stages activate
# only when their file arg is passed.
python -m src.graph_work.etl.build_graph \
  --leads  "~/Desktop/Data/Model/output/final/model_leads_CLEANED.csv" \
  --nppes  "~/Desktop/Data/preclean/NPPES.csv" \
  --owners "~/Desktop/Data/preclean/owners/*.csv" \
  --pecos  "~/Desktop/Data/preclean/PECOS.csv" \
  --leie   "~/Desktop/Data/preclean/Caught.csv" \
  --out    ~/Desktop/data/graphing            # [--no-widen] to skip perimeter
```

All CSVs land in `~/Desktop/data/graphing`. The NPPES file is streamed in
200k-row chunks (twice when widening) — memory stays bounded; expect a few
minutes per pass.

## 2. Start Neo4j with GDS + APOC, mounting the graphing folder as import dir

```bash
docker run -d --name fraud-graph \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/fraudgraph \
  -e NEO4J_PLUGINS='["graph-data-science","apoc"]' \
  -e NEO4J_apoc_import_file_enabled=true \
  -e NEO4J_dbms_security_procedures_unrestricted='gds.*,apoc.*' \
  -v ~/Desktop/data/graphing:/var/lib/neo4j/import \
  neo4j:5.26
```

Open http://localhost:7474 (user `neo4j`, password `fraudgraph`). The CSVs are
now visible to `LOAD CSV FROM 'file:///<name>.csv'`.

The loaders use **plain `LOAD CSV`** plus only `apoc.merge.relationship` /
`apoc.do.when` (both in the bundled **apoc-core**) and the GDS procedures — no
apoc-extended needed. Wait until the plugins finish registering before loading
(the DB accepts connections a few seconds before APOC/GDS are live):

```bash
until docker exec fraud-graph cypher-shell -u neo4j -p fraudgraph \
  "SHOW PROCEDURES YIELD name WHERE name='apoc.merge.relationship' RETURN name" \
  | grep -q apoc; do sleep 3; done; echo "plugins ready"
```

## 3. Run the three Cypher files **in order**

In Neo4j Browser (or `cypher-shell`), run:

1. **`cypher/01_schema.cypher`** — uniqueness constraints + indexes. Run first.
2. **`cypher/02_load.cypher`** — idempotent MERGE loaders. Stage 2–5 sections use
   `apoc.load.csv` and no-op gracefully if a file is absent (a Stage-1-only graph
   still loads). Requires APOC.
3. **`cypher/03_leads.cypher`** — run the queries individually:
   1. **validation** — does the graph reconnect companies pre-flagged
      `possible_same_operator_family` via shared identity? (sanity check)
   2. shared authorized officials across ≥3 companies
   3. address hubs
   4. shared-fax pairs
   5. excluded-adjacent billers (≤3 hops from an excluded NPI/LEIE person)
   6. owners across multiple companies
   7. **GDS WCC** → ranked case files (cluster size, total paid, # pre-flagged)
   8. **GDS betweenness** → broker nodes bridging clusters

`cypher-shell` one-liner per file:
```bash
cat src/graph_work/cypher/01_schema.cypher | docker exec -i fraud-graph cypher-shell -u neo4j -p fraudgraph
```
(GDS queries 7–8 are best run interactively in Browser to read the results.)

## Notes
- Re-running the ETL overwrites the CSVs; re-running the loaders is idempotent
  (MERGE), so you can reload without duplications.
- HIPAA: all inputs and outputs live outside the repo under `~/Desktop`. Nothing
  here is committed except code.
