# V2 candidate and oracle diagnostics

Correspondence is inferred from semantic/MCC/price eligibility, not annotated latent stream IDs. All eight family rows always exist. Coverage numbers below are evidence proxies, not a mathematical ceiling on V1. Official rows reuse the frozen V1 predictions; one aggregate diagnostic access is logged.

| View | F1 | Soft 3+ recall | 2+ recall | 3+ or broad recall | Singleton envelope recall | Perfect identity/current none | Perfect none/current family rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| official_frozen_v1 | 0.619493 | 0.8020 | 0.9095 | 0.8897 | 0.9774 | 0.8704 | 0.6906 |
| original | 0.655305 | 0.7883 | 0.8546 | 0.8567 | 0.9572 | 0.9090 | 0.7125 |
| valid_like | 0.637367 | 0.7912 | 0.8553 | 0.8689 | 0.9572 | 0.8913 | 0.6959 |
| test_like | 0.613157 | 0.7962 | 0.8674 | 0.8717 | 0.9572 | 0.8728 | 0.6778 |

The constrained oracle assigns the true family when proxy evidence exists and none otherwise, including perfect true-none decisions. The separate optimistic bound ignores false positives forced by uncovered cases. Neither is a realizable model. Perfect identity fixes family errors while preserving the observed none boundary; it is an optimistic relabeling headroom diagnostic, not identified continuation performance. Perfect none restores the highest scored non-none family for true recurring clients.

Full per-family recall, confidence/length/count/masking cohorts, exclusive error partitions and oracle definitions are in `v2_candidate_recall.csv`, `v2_diagnostics.json`, and `scripts/v2_diagnostics.py`.

## Learned-proposal check

The weak pair model's proposals are evaluated using the same soft
semantic/MCC/price eligibility rule, rather than counting every nonzero
family score as coverage. Among 1,403 true non-none training clients, learned
2+ proposals cover 1,197 original, 1,198 valid-like and 1,202 test-like clients
(85.32%, 85.39%, 85.67%). Learned 3+ proposals cover 1,130, 1,131 and 1,140.
Per-family and union-with-V1 recalls are in
`v2_learned_candidate_recall.csv`. These are still proxy recall measurements.

Those proposals do not pass the downstream identity gate: identity-only
mean stress F1 regresses, and the joint version increases wrong-family
errors despite a small overall gain. Candidate availability and auxiliary
pair discrimination are therefore insufficient acceptance criteria.

The selected sparse component adds evidence without deleting any V1 stream.
It complements the V1 candidates; it does not claim that every singleton is
a recurring process. Its candidate envelope is the 1/2-event evidence
already quantified above. Final downstream and removal-test outcomes are in
`v2_results.md` and `v2_error_analysis.md`.
