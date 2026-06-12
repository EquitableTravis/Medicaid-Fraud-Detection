// 02_load.cypher — RUN SECOND. Idempotent MERGE-based loaders.
// Every CSV must sit in the Neo4j import dir (mount ~/Desktop/data/graphing there).
// Stage 2-5 loaders use apoc.load.csv(..., {failOnError:false}) so a MISSING file
// returns zero rows (logs a warning) instead of aborting — a Stage-1-only graph
// still loads cleanly. APOC required. POSSIBLY_SAME_AS is built at the very end so
// it links persons from all of stages 2-4.

// ---------------------------------------------------------------------------
// STAGE 1 — companies, NPIs, HAS_NPI (always present)
// ---------------------------------------------------------------------------
LOAD CSV WITH HEADERS FROM 'file:///nodes_company.csv' AS r
MERGE (c:Company {company_id: r.company_id})
SET c.new_rank = toInteger(r.new_rank),
    c.company_name = r.company_name,
    c.company_model_score_max = toFloat(r.company_model_score_max),
    c.company_model_score_wmean = toFloat(r.company_model_score_wmean),
    c.company_net_paid = toFloat(r.company_net_paid),
    c.n_npis = toInteger(r.n_npis),
    c.n_leie_npis = toInteger(r.n_leie_npis),
    c.merge_confidence = r.merge_confidence,
    c.specialty = r.specialty,
    c.flag_same_operator_family = (r.flag_same_operator_family = 'True'),
    c.flag_low_linkage = (r.flag_low_linkage = 'True'),
    c.flag_verify_government = (r.flag_verify_government = 'True'),
    c.flag_verify_hospital = (r.flag_verify_hospital = 'True'),
    c.flag_contains_excluded = (r.flag_contains_excluded = 'True'),
    c.whitelist_candidate = (r.whitelist_candidate = 'True');

LOAD CSV WITH HEADERS FROM 'file:///nodes_npi.csv' AS r
MERGE (:NPI {npi: r.npi});

LOAD CSV WITH HEADERS FROM 'file:///edges_company_npi.csv' AS r
MATCH (c:Company {company_id: r.company_id})
MATCH (n:NPI {npi: r.npi})
MERGE (c)-[h:HAS_NPI]->(n)
SET h.merge_confidence = r.merge_confidence;   // carried so low-conf rollups can't masquerade as discovered links

// ---------------------------------------------------------------------------
// STAGE 2 — NPPES enrichment (optional)
// ---------------------------------------------------------------------------
CALL apoc.load.csv('file:///nodes_npi_enriched.csv', {failOnError:false}) YIELD map AS r
MATCH (n:NPI {npi: r.npi})
SET n.org_name = r.org_name, n.entity_type = r.entity_type,
    n.enum_date = r.enum_date, n.in_lead_set = (r.in_lead_set = 'True')
RETURN count(*) AS npi_enriched;

CALL apoc.load.csv('file:///nodes_address.csv', {failOnError:false}) YIELD map AS r
MERGE (:Address {address_id: r.address_id})
RETURN count(*) AS addresses;

CALL apoc.load.csv('file:///nodes_phone.csv', {failOnError:false}) YIELD map AS r
MERGE (:Phone {digits: r.digits})
RETURN count(*) AS phones;

CALL apoc.load.csv('file:///nodes_person.csv', {failOnError:false}) YIELD map AS r
MERGE (p:Person {person_id: r.person_id})
SET p.name_base = r.name_base, p.tiebreaker = r.tiebreaker,
    p.ao_phone = r.ao_phone, p.source = r.source
RETURN count(*) AS persons;

CALL apoc.load.csv('file:///edges_npi_address.csv', {failOnError:false}) YIELD map AS r
MATCH (n:NPI {npi: r.npi}) MATCH (a:Address {address_id: r.address_id})
CALL apoc.merge.relationship(n, r.rel, {}, {suite: r.suite}, a) YIELD rel
RETURN count(*) AS npi_address_edges;

CALL apoc.load.csv('file:///edges_npi_phone.csv', {failOnError:false}) YIELD map AS r
MATCH (n:NPI {npi: r.npi}) MATCH (p:Phone {digits: r.digits})
CALL apoc.merge.relationship(n, r.rel, {}, {}, p) YIELD rel
RETURN count(*) AS npi_phone_edges;

CALL apoc.load.csv('file:///edges_npi_person.csv', {failOnError:false}) YIELD map AS r
MATCH (n:NPI {npi: r.npi}) MATCH (p:Person {person_id: r.person_id})
MERGE (p)-[a:AUTHORIZED_BY]->(n) SET a.title = r.title
RETURN count(*) AS npi_person_edges;

// ---------------------------------------------------------------------------
// STAGE 3 — ownership (optional). Owner may be Person or Org.
// ---------------------------------------------------------------------------
CALL apoc.load.csv('file:///edges_owns.csv', {failOnError:false}) YIELD map AS r
MATCH (n:NPI {npi: r.npi})
CALL apoc.do.when(
  r.owner_type = 'Org',
  'MERGE (o:Org {org_id: r.owner_id}) SET o.name = r.owner_name, o.assoc_id = r.owner_assoc_id
   MERGE (o)-[rel:OWNS]->(n) SET rel.role = r.role, rel.pct = r.pct RETURN 0 AS x',
  'MERGE (o:Person {person_id: r.owner_id}) SET o.name_base = replace(r.owner_id,"PERSON:",""), o.assoc_id = r.owner_assoc_id, o.source = coalesce(o.source,"owner_file")
   MERGE (o)-[rel:OWNS]->(n) SET rel.role = r.role, rel.pct = r.pct RETURN 0 AS x',
  {r: r, n: n}) YIELD value
RETURN count(*) AS owns_edges;

// ---------------------------------------------------------------------------
// STAGE 4 — PECOS associate IDs (optional) -> ASSOCIATED_WITH
// ---------------------------------------------------------------------------
CALL apoc.load.csv('file:///edges_associated_with.csv', {failOnError:false}) YIELD map AS r
MATCH (n:NPI {npi: r.npi})
MERGE (pac:Person {person_id: 'PAC:' + r.pac_id})
SET pac.pac_id = r.pac_id, pac.name_base = r.name_base, pac.source = 'pecos'
MERGE (pac)-[a:ASSOCIATED_WITH]->(n)
SET a.enrollment_id = r.enrollment_id, a.confirms_nppes_person = (r.confirms_nppes_person = 'True')
RETURN count(*) AS associated_edges;

// ---------------------------------------------------------------------------
// STAGE 5 — LEIE flags (optional)
// ---------------------------------------------------------------------------
CALL apoc.load.csv('file:///flags_leie_npi.csv', {failOnError:false}) YIELD map AS r
MATCH (n:NPI {npi: r.npi})
SET n.leie_excluded = true, n.leie_excltype = r.EXCLTYPE, n.leie_excldate = r.EXCLDATE
RETURN count(*) AS leie_npi_flags;

CALL apoc.load.csv('file:///flags_leie_person.csv', {failOnError:false}) YIELD map AS r
MATCH (p:Person {name_base: r.name_base})
SET p.leie_name_lead = true, p.leie_excltype = r.excltype, p.leie_excldate = r.excldate
RETURN count(*) AS leie_person_leads;   // NAME-ONLY LEADS — verify identity before use

// ---------------------------------------------------------------------------
// FINALIZE — POSSIBLY_SAME_AS between persons sharing a name_base but distinct
// person_id (different tiebreaker / source: NPPES-AO vs owner-file vs PECOS).
// Run LAST so it also links the persons created by stages 3-4 — NEVER an
// auto-merge (a false merge fabricates a fraud link); empty name_base excluded.
// ---------------------------------------------------------------------------
MATCH (p1:Person)
WHERE p1.name_base IS NOT NULL AND p1.name_base <> ''
MATCH (p2:Person)
WHERE p2.name_base = p1.name_base AND p1.person_id < p2.person_id
MERGE (p1)-[:POSSIBLY_SAME_AS]-(p2);
