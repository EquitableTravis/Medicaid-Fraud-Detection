// 01_schema.cypher — RUN FIRST.
// Uniqueness constraints (also create backing indexes) + helpful lookup indexes.
// Idempotent: IF NOT EXISTS on everything.

CREATE CONSTRAINT company_id   IF NOT EXISTS FOR (c:Company) REQUIRE c.company_id IS UNIQUE;
CREATE CONSTRAINT npi_id       IF NOT EXISTS FOR (n:NPI)     REQUIRE n.npi        IS UNIQUE;
CREATE CONSTRAINT address_id   IF NOT EXISTS FOR (a:Address) REQUIRE a.address_id IS UNIQUE;
CREATE CONSTRAINT phone_digits IF NOT EXISTS FOR (p:Phone)   REQUIRE p.digits     IS UNIQUE;
CREATE CONSTRAINT person_id    IF NOT EXISTS FOR (p:Person)  REQUIRE p.person_id  IS UNIQUE;
CREATE CONSTRAINT org_id       IF NOT EXISTS FOR (o:Org)     REQUIRE o.org_id     IS UNIQUE;

// Property indexes for filtering / lead queries.
CREATE INDEX company_netpaid   IF NOT EXISTS FOR (c:Company) ON (c.company_net_paid);
CREATE INDEX company_whitelist IF NOT EXISTS FOR (c:Company) ON (c.whitelist_candidate);
CREATE INDEX company_score     IF NOT EXISTS FOR (c:Company) ON (c.company_model_score_max);
CREATE INDEX npi_in_lead       IF NOT EXISTS FOR (n:NPI)     ON (n.in_lead_set);
CREATE INDEX person_namebase   IF NOT EXISTS FOR (p:Person)  ON (p.name_base);
CREATE INDEX npi_leie          IF NOT EXISTS FOR (n:NPI)     ON (n.leie_excluded);
