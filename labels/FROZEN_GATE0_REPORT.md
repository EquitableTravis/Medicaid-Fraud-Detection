# GATE 0 — frozen_2023-12 package validation

package: `/Users/traviswaters/Desktop/Data/preclean/trey/frozen_2023-12`

**FAIL** — 1 fail / 2 warn / 16 pass

- [PASS] parquet shape: (617062, 211) vs expected (617062, 211)
- [PASS] manifest n_providers: 617062 == 617062
- [PASS] npi unique: one row per NPI
- [WARN] trainable count == 126 (letter): derived 117
- [FAIL] every parquet column accounted for by manifest: unaccounted: ['priority_tier', 'priority_rank', 'rule_reasons', 'anomaly_lead_v3', 'anomaly_score_v3', 'anomaly_contributing_concepts', 'iforest_score_secondary', 'peer_basis', 'not_scored', 'not_scored_reason', 'layer3_probable_owner', 'facility_excluded_owner_n_probable', 'excluded_owner_role', 'first_service_month', 'last_service_month', 'services_per_bene', 'allowed_per_bene', 'code_concentration_hhi', 'total_services', 'total_benes', 'total_allowed', 'brand_generic_cost_ratio', 'total_claims', 'total_cost', 'dme_code_concentration', 'opioid_claims', 'op_total_dollars', 'n_manufacturers', 'jcode_paid', 'n_jcodes', 'post_deactivation_paid', 'total_paid', 'pos_bed_count', 'peer_group_key', 'nucc_grouping', 'nucc_classification', 'evidence_n_specialty_mismatch', 'evidence_n_rapid_ramp', 'evidence_n_ownership_integrity', 'evidence_n_upcoding', 'evidence_n_pharma_kickback', 'evidence_n_nemt_fraud', 'evidence_n_behavioral_health', 'evidence_n_drug_outlier', 'evidence_n_dme_ring', 'evidence_n_worthless_services', 'evidence_n_hospice_ineligibility', 'evidence_n_saturation_fraud', 'evidence_n_pill_mill', 'evidence_n_contract_pharmacy', 'evidence_n_invalid_identity', 'evidence_n_cost_report_fraud']
- [PASS] every trainable column has a vintage class: classes: {'point_in_time': 57, 'annual_capped': 36, 'reference': 0, 'current_state': 33}
- [PASS] hard/adjacent lists disjoint
- [PASS] known offenders all fenced: all fenced or absent
- [PASS] weak-supervision columns not trainable: weak cols in parquet: ['weak_label_score', 'weak_label', 'weak_label_votes']
- [PASS] leakage_adjacent count == 11 (letter): 11: ['within_2_hops_of_exclusion', 'shell_score', 'related_party_density', 'subscore_ownership_integrity', 'has_excluded_owner', 'billing_after_deactivation__peerpct', 'subscore_invalid_identity', 'dme_ineligible_referrer', 'dme_ineligible_referred_dollars', 'dme_ineligible_referred_dollars__peerpct', 'subscore_dme_ring']
- [PASS] in-time positives == manifest n_positives: 1162 vs 1162
- [PASS] forward prospective positives == 4,401 (letter): 4401
- [PASS] no NPI both prospective and pre-cutoff: 0 conflicts
- [PASS] forward positives found in universe: 1021/4401 in the 617k universe
- [PASS] pre-cutoff exclusions overlap in-time label: 922 of 1162 in-time positives matched
- [PASS] betweenness constant (drop per letter): nunique=1
- [WARN] no constant trainable columns: constants: ['betweenness']
- [PASS] non-numeric trainable columns (need encoding): none
- [PASS] no trainable column solo-AUC > 0.75 vs forward label: max: billing_surprisal = 0.645

## Top-15 solo ROC-AUC vs forward label (eligible rows)

| column | solo AUC |
|---|---|
| billing_surprisal | 0.645 |
| co_location_cluster_size | 0.623 |
| addr_provider_count | 0.622 |
| billing_emb_0 | 0.622 |
| addr_distinct_orgs | 0.621 |
| sequence_surprisal | 0.621 |
| billing_taxonomy_fit | 0.615 |
| addr_shared | 0.595 |
| addr_cluster_degree | 0.594 |
| billing_emb_5 | 0.577 |
| billing_emb_2 | 0.575 |
| tenure_months | 0.547 |
| volume_residual | 0.547 |
| billing_taxonomy_margin | 0.547 |
| subscore_saturation_fraud | 0.543 |
