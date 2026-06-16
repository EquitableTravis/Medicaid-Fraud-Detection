# Audit — `model_leads_CLEANED.csv` (the production lead list, all-LEIE model)

**Date:** 2026-06-15 · 1,752 company leads, $109B billing, all ≥$10M. Five test angles
(`labels/audit_cleaned_leads.py` + a 25-lead random web adjudication). **Verdict: mechanically
solid and genuinely enriched, but — like every list we've tested — predominantly large providers
in fraud-prone sectors; its real value is a minority of sharp anomalies plus a triage pool, not a
list of confirmed fraud.**

## A. Composition (sane)
- 1,752 leads, $109B, median $24.6M, floor $10M. 49% single-NPI, 51% multi-NPI.
- Broad national spread (NY 145, NC 144, CA 144, PA, AZ 107, VA, FL) — NOT AZ-concentrated like
  the fraud_positive model's leads.
- Only 3 leads carry an already-LEIE NPI → the list is aimed at *not-yet-caught* providers (correct).

## B. Score bands vs the calibration cliff (good)
Calibration showed real LEIE lift only at score ≥0.99. On this list:
- **≥0.99: 1,236 leads (71%)** · 0.90–0.99: 516 (29%) · below 0.90: 0.
- So 71% of the list sits in the one band that actually carries signal. Strong.

## C. Institutional contamination — today's hardened screens applied (clean)
- Only **9 of 1,752 (0.5%)** would be removed by today's hardened screens (5 school, 2 academic,
  1 school-name, 1 tribal). The screens that were live when this list was built were already good;
  the list is institutionally clean. (Contrast: the fraud_positive leads needed +48 removals.)

## D. Independent corroboration — the strongest positive (real signal)
The all-LEIE model was **never trained on AHCCCS or NV**, so those lists are a clean out-of-sample check:
- 51 leads (2.9%) contain ≥1 AHCCCS/NV-excluded NPI; 3 contain a fraud-relevant LEIE NPI; 53 (3.0%) any.
- **AHCCCS/NV hit rate among lead NPIs 0.303% vs universe base 0.084% = 3.6x lift.** The list is
  meaningfully enriched for genuine enforcement targets above chance.
- Corroboration concentrates in the ≥0.99 band (**4.0% vs 0.6%** below) — the calibration cliff
  reproduces on independent data. The score *is* doing real work at the top.

## E. Random-25 web adjudication (the sobering part)
Same protocol as the fraud_positive audit, so the rates are directly comparable:

| | fraud-candidate | legit / institutional | inconclusive |
|---|---|---|---|
| **CLEANED (this list), n=25** | **2 (8%)** | 17 (68%) | 6 (24%) |
| fraud_positive model, n=30 | 1 (3%) | 16 (53%) | 13 (43%) |

- **The 2 genuine anomalies are real value:** NPI 1588799746 = a *solo* speech-language pathologist
  (Laura Sue Veal, NM) billing **~$31M** — a severe one-clinician outlier; and CARELINK-CDPAP (NY),
  in the single highest-fraud Medicaid category (CDPAP fiscal intermediary). Plus a yellow flag:
  Unity Place (NJ) has a documented 2020 Medicaid civil overbilling settlement.
- **But 68% are clearly-legitimate established institutions** — Loma Linda faculty practice, Essentia
  St. Mary's, FQHCs (E.A. Hawse), 50–130-year nonprofits (Fraser, Emory Valley, SouthLight), national
  chains (Interim HealthCare, Pinnacle, Preferred Home Health). These are not whistleblower targets.
- The 6 inconclusive all sit in known high-fraud categories (NY social adult day care, MD behavioral
  health, AZ behavioral, VA respite) with no public signal — the genuine triage pool.

## How to read this honestly
- **The web "68% legit" is an UPPER bound on cleanliness, not proof of innocence.** A legit-looking
  operating provider is exactly what fraud looks like before it's charged (the AZ AHCCCS defendants
  looked legitimate too). Web research only catches *already-public* cases. So the real fraud rate in
  the inconclusive + "legit-but-high-fraud-category" leads is unknowable from here — which is the
  whole reason the product exists (find what's not yet public).
- **The 3.6x independent lift is the most trustworthy number** and it's a genuine positive: the list
  is enriched for real enforcement targets well above chance, and the enrichment lives in the top band.
- **The ceiling is the same one we hit all day:** the model finds "big billers in fraud-prone
  taxonomies," and most big billers there are legitimate. It cannot, from billing shape alone, tell a
  fraudulent behavioral-health org from a legitimate one — so the list is a *prioritized triage pool*,
  not a list of fraudsters.

## Bottom line
The production list is **better than its raw legit-rate suggests**: institutionally clean, 71% in the
high-signal band, 3.6x enriched on lists the model never saw, and it does surface sharp individual
anomalies (a $31M solo therapist) that are exactly the kind of lead worth pursuing. Its weakness is
that a large share are legitimate institutions that shouldn't be advertised to — addressable two ways
worth doing: (1) prioritize the ≥0.99 band + single-NPI high-$ individual outliers (where the signal
concentrates), and (2) a light "is this a large established nonprofit/chain/academic" pre-screen before
advertising. The forward test (frozen today) will give the first non-circular precision number when the
next LEIE drops.
