// 03_leads.cypher — RUN THIRD (after 01_schema then 02_load).
// Every result is an investigative LEAD requiring records-level verification.
// Run queries individually in Neo4j Browser; queries 7-8 need the GDS plugin.
// Whitelist_candidate companies (mega-orgs, govt, hospital systems) are excluded
// from link queries so they don't dominate — they connect to thousands of NPIs
// for legitimate reasons.

// ===========================================================================
// (1) VALIDATION — does the graph RECONNECT companies pre-flagged
//     possible_same_operator_family via a SHARED identity (person/address/phone)?
//     A non-trivial count means the shared-identifier approach reproduces the
//     model's same-operator hunch from independent evidence.
// ===========================================================================
MATCH (c1:Company)-[:HAS_NPI]->(:NPI)-[:AUTHORIZED_BY|LOCATED_AT|MAILS_TO|HAS_PHONE|HAS_FAX|OWNS]-(x)
      -[:AUTHORIZED_BY|LOCATED_AT|MAILS_TO|HAS_PHONE|HAS_FAX|OWNS]-(:NPI)<-[:HAS_NPI]-(c2:Company)
WHERE c1.flag_same_operator_family AND c2.flag_same_operator_family
  AND c1.company_id < c2.company_id
  AND NOT c1.whitelist_candidate AND NOT c2.whitelist_candidate
RETURN labels(x) AS via, count(DISTINCT [c1.company_id, c2.company_id]) AS company_pairs_reconnected
ORDER BY company_pairs_reconnected DESC;

// ===========================================================================
// (2) Shared AUTHORIZED OFFICIAL across >= 3 companies (classic shell signal).
// ===========================================================================
MATCH (p:Person)-[:AUTHORIZED_BY]->(:NPI)<-[:HAS_NPI]-(c:Company)
WHERE NOT c.whitelist_candidate
WITH p, collect(DISTINCT c) AS companies
WHERE size(companies) >= 3
RETURN p.person_id AS official, p.name_base AS name,
       size(companies) AS n_companies,
       [c IN companies | c.company_name] AS companies_list,
       round(reduce(s = 0.0, c IN companies | s + c.company_net_paid)) AS total_net_paid
ORDER BY n_companies DESC, total_net_paid DESC LIMIT 100;

// ===========================================================================
// (3) ADDRESS HUBS — one building tied to many companies.
// ===========================================================================
MATCH (a:Address)<-[:LOCATED_AT|MAILS_TO]-(:NPI)<-[:HAS_NPI]-(c:Company)
WHERE NOT c.whitelist_candidate
WITH a, collect(DISTINCT c) AS companies
WHERE size(companies) >= 3
RETURN a.address_id AS building, size(companies) AS n_companies,
       [c IN companies | c.company_name] AS companies_list,
       round(reduce(s = 0.0, c IN companies | s + c.company_net_paid)) AS total_net_paid
ORDER BY n_companies DESC LIMIT 100;

// ===========================================================================
// (4) SHARED-FAX pairs — a shared fax is a stronger co-operation signal than a
//     shared phone (answering services rarely share fax lines).
// ===========================================================================
MATCH (c1:Company)-[:HAS_NPI]->(:NPI)-[:HAS_FAX]->(f:Phone)<-[:HAS_FAX]-(:NPI)<-[:HAS_NPI]-(c2:Company)
WHERE c1.company_id < c2.company_id
  AND NOT c1.whitelist_candidate AND NOT c2.whitelist_candidate
RETURN f.digits AS fax, c1.company_name AS company_a, c2.company_name AS company_b,
       round(c1.company_net_paid + c2.company_net_paid) AS combined_net_paid
ORDER BY combined_net_paid DESC LIMIT 100;

// ===========================================================================
// (5) EXCLUDED-ADJACENT BILLERS — a company with net_paid > 0 within <= 3 hops
//     of an excluded NPI or an LEIE-name-lead person.
// ===========================================================================
MATCH (c:Company)-[:HAS_NPI]->(:NPI)
WHERE c.company_net_paid > 0 AND NOT c.whitelist_candidate
MATCH path = (c)-[:HAS_NPI|AUTHORIZED_BY|LOCATED_AT|MAILS_TO|HAS_PHONE|HAS_FAX|OWNS|ASSOCIATED_WITH*1..3]-(x)
WHERE (x:NPI AND x.leie_excluded) OR (x:Person AND x.leie_name_lead)
RETURN DISTINCT c.company_name AS company, round(c.company_net_paid) AS net_paid,
       labels(x) AS excluded_kind,
       coalesce(x.npi, x.person_id) AS excluded_node,
       coalesce(x.leie_excltype, '') AS excltype, length(path) AS hops
ORDER BY hops ASC, net_paid DESC LIMIT 200;

// ===========================================================================
// (6) OWNERS across MULTIPLE companies (Person or Org owner).
// ===========================================================================
MATCH (o)-[:OWNS]->(:NPI)<-[:HAS_NPI]-(c:Company)
WHERE (o:Person OR o:Org) AND NOT c.whitelist_candidate
WITH o, collect(DISTINCT c) AS companies
WHERE size(companies) >= 2
RETURN labels(o) AS owner_type, coalesce(o.name, o.name_base, o.org_id, o.person_id) AS owner,
       size(companies) AS n_companies, [c IN companies | c.company_name] AS companies_list,
       round(reduce(s = 0.0, c IN companies | s + c.company_net_paid)) AS total_net_paid
ORDER BY n_companies DESC, total_net_paid DESC LIMIT 100;

// ===========================================================================
// (7) GDS — WEAKLY CONNECTED COMPONENTS => ranked CASE FILES.
//     Project Company + the identity rendezvous nodes and the linking edges,
//     run WCC, write cluster_id back, return clusters ranked by total paid.
// ===========================================================================
CALL gds.graph.drop('leadnet', false) YIELD graphName;
CALL gds.graph.project.cypher(
  'leadnet',
  // NODES: non-whitelist companies + identity nodes
  'MATCH (c:Company) WHERE NOT c.whitelist_candidate RETURN id(c) AS id
   UNION MATCH (n:NPI) RETURN id(n) AS id
   UNION MATCH (p:Person) RETURN id(p) AS id
   UNION MATCH (a:Address) RETURN id(a) AS id
   UNION MATCH (f:Phone) RETURN id(f) AS id
   UNION MATCH (o:Org) RETURN id(o) AS id',
  // RELS: undirected linking edges
  'MATCH (c:Company)-[:HAS_NPI]->(n:NPI) WHERE NOT c.whitelist_candidate RETURN id(c) AS source, id(n) AS target
   UNION MATCH (n:NPI)-[:AUTHORIZED_BY|LOCATED_AT|MAILS_TO|HAS_PHONE|HAS_FAX]-(x) RETURN id(n) AS source, id(x) AS target
   UNION MATCH (o)-[:OWNS]->(n:NPI) RETURN id(o) AS source, id(n) AS target
   UNION MATCH (p:Person)-[:ASSOCIATED_WITH]->(n:NPI) RETURN id(p) AS source, id(n) AS target'
) YIELD graphName, nodeCount, relationshipCount;

CALL gds.wcc.write('leadnet', {writeProperty: 'cluster_id'})
YIELD componentCount, nodePropertiesWritten;

// Ranked case files: multi-company clusters by total paid + how many were pre-flagged.
MATCH (c:Company) WHERE c.cluster_id IS NOT NULL
WITH c.cluster_id AS cluster, collect(c) AS companies
WHERE size(companies) >= 2
RETURN cluster, size(companies) AS n_companies,
       round(reduce(s = 0.0, c IN companies | s + c.company_net_paid)) AS total_net_paid,
       size([c IN companies WHERE c.flag_same_operator_family]) AS n_preflagged_same_operator,
       size([c IN companies WHERE c.n_leie_npis > 0]) AS n_with_leie_npi,
       [c IN companies | c.company_name][0..15] AS sample_companies
ORDER BY total_net_paid DESC LIMIT 100;

// ===========================================================================
// (8) GDS — BETWEENNESS to surface BROKER nodes (Person/Address/Phone) that
//     bridge otherwise-separate clusters. High betweenness on a shared identity
//     = the operator/location stitching independent shells together.
// ===========================================================================
CALL gds.betweenness.stream('leadnet') YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS node, score
WHERE score > 0 AND (node:Person OR node:Address OR node:Phone OR node:Org)
RETURN labels(node) AS kind,
       coalesce(node.person_id, node.address_id, node.digits, node.org_id) AS broker,
       coalesce(node.name_base, node.name, '') AS name, round(score) AS betweenness
ORDER BY betweenness DESC LIMIT 50;
