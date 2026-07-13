# DRAFT reply to Trey (for Travis to review — do not send as-is without reading)

Subject: Re: major new milestone — your edge reproduces on my build; three findings

Trey,

Got the package, ran it this weekend. Headline first: **your result reproduces on my
build.** I reimplemented the matched A/B from your written protocol (not your code),
grouped 5-fold CV, cluster bootstrap over match groups, three seeds. Structural
family vs none, forward label, matched on taxonomy/state/size: **ΔROC ≈ +0.05, both
CIs clear of zero** — slightly stronger than your +0.035. Full-population training
per your letter's rules also behaves: forward ROC ~0.72 group-CV OOF, and your
strict no-current_state variant costs only ~0.004 of that, so the today's-snapshot
worry barely registers at full-population grain.

Three findings you'll want to see:

1. **The edge is entirely shell_score.** related_party_density alone adds exactly
   nothing (Δ ≈ 0.000 across seeds); shell + rpd ≈ shell. And the label-adjacent
   flags add nothing on top of shell. So the claim to carry forward isn't "network
   features" plural — it's one engineered feature. Worth knowing before the $500
   embeddings run: that run's real question is whether the graph holds anything
   beyond shell_score, because nothing else in the current family does.

2. **It survives my two robustness attacks.** Dropping every forward positive banned
   within 6 months of the cutoff (investigations already in flight): edge holds at
   ~+0.04. Partialling the within_2_hops proximity channel out of shell_score:
   still ~+0.045. Both make me take the result more seriously.

3. **The one hole I can't close from my side: the graph substrate isn't frozen.**
   Exclusion nodes and owner-edge dates are vintaged, but the co-location edges are
   built from current NPPES addresses. A ring that re-formed at a shell address
   after 2023 gives a forward positive a fresh edge into the pre-2023 graph —
   visible only because the future happened. Your "both arms share the current-state
   columns" defense doesn't cover this, because the two features under test are
   themselves built from those edges and sit in only one arm. The fix is real,
   though: NBER archives monthly NPPES editions (despite the docs saying no
   historical editions exist). Rebuild the co-location layer from a ~Dec-2023
   edition and re-run the same A/B. If shell's edge holds on a truly vintaged
   graph, I'll call it clean signal with no caveats.

Also from my GATE-0 pass on the package: the parquet has **52 columns the manifest
doesn't classify at all** (post_deactivation_paid, excluded_owner_role, my own
pipeline's anomaly_score_v3, the evidence_n_* counts, raw Part B/D aggregates...).
None are hot solo against the forward label, and I quarantined them by construction
— but the manifest is supposed to be the contract, so they should each get a class
in the next cut. Relatedly, "126 trainable" in your letter is the pre-fence count;
after removing the 11 leakage_adjacent it's 117.

Agreed on the exclusions-vs-qui-tam point — that's been our experience from the
other direction all along (billing-shape features look weak against a label that
can't see billing fraud). The DOJ settlement label is the right second answer key;
send it when ready and I'll run the same protocol against it.

Next on my side: calibration and a lead-list diff against my current production
list before anything ships.

Travis
