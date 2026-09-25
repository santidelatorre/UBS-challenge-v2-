# V2 diagnosis before modeling

Baseline verified at **0.6194934236214421 macro-F1 / 0.647 accuracy**.
All seven raw hashes match; the two cached raw-build predictions and
submissions are identical; 16 existing tests pass. This round verified those
cached artifacts, not a third clean training run. Exact tag/commit evidence
is in `v2_baseline_verification.json`.

Frozen official baseline: 353 errors. Using soft 3+ amount evidence, the
exclusive error partition is 123 candidate-absent errors (34.84%), 142
candidate-present family misrankings (40.23%), 27 candidate-present correct
family scores overridden by none (7.65%), and 61 false activations on true
none clients (17.28%). Absence and identity are associated with failure;
this partition does not establish individual causal explanations.

True non-none family evidence recall is 80.20% for soft 3+ amount candidates,
90.95% for 2+, 87.41% for broad groups, and 88.97% for existing 3+ or broad.
The permissive combined singleton envelope reaches 97.74%, but includes
ambiguous evidence and is not verified stream recall. Regular-candidate
accuracy is 74.43%; the 76 two-event-only clients score 19.74%, and the 48
singleton-only clients 4.17%.

Perfect family ranking over soft 3+ proxy evidence, with uncovered families
sent to none and all true-none clients correct, gives **0.880107** F1. A
per-class optimistic bound ignoring forced false positives is **0.904212**.
Existing broad evidence raises that oracle policy to **0.933983**. All eight
family rows already exist in V1, so its literal candidate-row oracle is
trivially 1.0: do not claim an 80% hard structural ceiling.

Perfect family identity with the observed none boundary unchanged gives an
optimistic **0.870369**. Perfect none decisions while retaining the observed
non-none family ranking gives only **0.690643**. Correcting covered families
while retaining other V1 decisions gives **0.816739**. These are oracle
relabelings for analysis only, not separately identified latent-process
accuracy or forecasts.

The central hypothesis survives in qualified form: family evidence and its
interpretation matter much more than a none-only correction. Missing 3+
evidence is serious, but misranking among covered families is slightly larger.
Candidate generation is not shown to cap performance below 0.80. Start with
complementary sparse evidence, then learned identity if it fails to transfer.
Do not attribute every wrong-family error to stream membership: target
competition, continuation and ambiguous identity are entangled.

Train OOF and both fixed stress views show similar error partitions. This
supports using them to test the mechanism, without assuming the artificial
noise exactly reproduces official corruption. Full tables and per-family
recalls are in `v2_diagnostics.json` and `v2_candidate_recall.csv`.
