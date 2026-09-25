# V2 leakage and provenance audit

- Reference tag `baseline-v1-0.619493` remains at commit
  `e4aa4c58175242e198cefd32d6ac4558145523af`. V1 source files, historical
  reports, and submission are preserved. V2 source is additive. The
  experiment ledgers retain their exact V1 prefix and append V2 records.
- Baseline verification recomputed official metrics from frozen predictions
  and raw labels, checked all raw hashes and the equality of the two cached
  runs. This was logged as cached verification, not a third clean rebuild.
- The sparse feature function takes histories, V1 features and unlabeled
  price profiles. It does not accept targets. Every client has all eight
  candidate rows; no difficult clients/classes are excluded. IDs are join
  keys, never predictive columns.
- V2 supervised fitting uses the 2,000 train clients only. Five-fold splits
  match V1 for each of seeds 42, 17 and 2026. All three augmented histories
  of each training client remain inside that client's training fold.
  Test labels are unavailable. Official validation labels never enter
  fitting, learned representations, priors or threshold calibration.
- The weak pair model uses only the disjoint 10,000 pretraining clients.
  Its 8,000 fitting clients and 2,000 threshold-calibration clients are
  disjoint. Anchor rules use clean pre-cutoff histories; corrupted pair
  features do not necessarily satisfy those rules. Calibration reuses weak
  auxiliary labels, so its precision is not an independent final test.
  Pair models and proposals were rejected after challenge-side CV.
- The amount/MCC/text/refund functions are label-free at deployment. Oracle
  labels and true-family selections appear only in diagnostics. No oracle
  field is appended to a training feature matrix.
- The removal benchmark deliberately uses train labels to select a recovered
  true-family stream, then removes card observations while keeping targets
  fixed. It is evaluation-only simulation, not observed naturally sparse
  performance, and not a rule used on official/test clients. Other refunds
  and transactions remain visible. The fresh corruption seed is 91711;
  fitting augmentation uses 2026.
- Strong-cohort and sparse-cohort gates were declared before V2 evaluation.
  One fixed blend was added after first-seed evidence of strong-case
  displacement and before multi-seed/official checks. All attempted variants
  and all seeds are reported; no seed was discarded.
- A new official evaluation requires a committed freeze and passing fresh
  stress receipt. Probabilities are serialized before the label loader is
  called. Hashes and purpose are appended to the holdout access log.
  Official aggregate diagnostics mean this is not a pristine unseen holdout.
- Raw V2 builds explicitly bypass transaction caches and relearn price
  profiles from raw unlabeled histories. The embedded V1 model is refit too;
  prediction checks compare it against the frozen V1 probabilities. Final
  reproduction outcomes and hashes are in `v2_results.md` and
  `v2_reproduction.json` when available.
- Tests cover sparse gating, currency/client separation, event ordering,
  ID-renaming invariance, ignored label columns, pair symmetry and weak-label
  isolation, in addition to the 16 frozen reference tests.

Residual limitations: semantic/price candidate coverage is a proxy for
unknown stream identity; synthetic corruption is not a verified generator;
OOF estimates were used for model selection; official validation has prior
research exposure. A bootstrap interval conditions on this selected model
and sample and cannot remove that selection uncertainty.
