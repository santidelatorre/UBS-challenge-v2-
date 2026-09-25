# V2 results

**Retain V1: official macro-F1 0.619493. V2 official result: not evaluated. Accepted-result delta: +0.000000.**

V2 found a promising sparse-stream signal, but the selected candidate failed a predeclared robustness gate. No V2 improvement over the official baseline is claimed. No new official model evaluation or submission was made. The 0.80 ambition remains unmet.

The frozen train-side candidate gives 37.5% weight to the new sparse experts, 37.5% to V1 compact experts and 25% to V1 legacy experts. All seeds (42, 17, 2026) are retained.

| Evaluation | V1 F1 | V2 candidate F1 | Delta |
|---|---:|---:|---:|
| Fixed original OOF | 0.655305 | 0.681629 | +0.026323 |
| Fixed valid_like OOF | 0.637367 | 0.647437 | +0.010070 |
| Fixed test_like OOF | 0.613157 | 0.622491 | +0.009334 |
| fresh_valid_like OOF | 0.649855 | 0.668409 | +0.018554 |
| fresh_test_like OOF | 0.613343 | 0.621718 | +0.008375 |
| keep4 OOF | 0.627630 | 0.642327 | +0.014697 |
| keep3 OOF | 0.581895 | 0.598704 | +0.016809 |
| keep2 OOF | 0.527134 | 0.559992 | +0.032858 |
| keep1 OOF | 0.298328 | 0.433160 | +0.134832 |

All full-sample F1 comparisons above improve. Nevertheless, at four visible events the affected 901-client cohort loses ten correct predictions (653 to 643): accuracy 72.4750% to 71.3651%, a **1.1099-point loss** against a fixed one-point tolerance. That narrowly fails the gate. We did not change the tolerance, blend, features or selected variant after seeing the fresh-view results, and did not use official validation to rescue it.

The original/valid-like/test-like direct sparse model passed screening in all three seeds. Its fixed conservative blend improved mean fixed-stress F1 by 0.009702 and sparse accuracy by 6.55 points while preserving the original strong cohort within tolerance. Fresh-corruption mean F1 improved by 0.013465. Controlled one- and two-event recovery improved; four-event displacement remains the unresolved mechanism. This is useful train-side evidence, not proof of official transfer.

The learned pair model used all 10,000 unlabeled clients for fitting/calibration structure, but failed downstream identity checks despite auxiliary AUC 0.99783. The missing-candidate-indicator ablation also failed, supporting the value of the new sparse transaction evidence. No survival or neural model was trained. See `v2_research_decisions.md`.

One complete V2 raw-data build refit all 15 estimators in 486.8 seconds. Its embedded V1 price profiles and all nine estimator weight serializations match the frozen V1 artifact exactly. Initial verification also checked the two cached V1 probability files and their 0.619493 score. The rejected V2 candidate was not evaluated officially or rebuilt twice; another identical fit would not resolve its failed robustness gate. Hashes and exact configuration are in `v2_final_decision.json` and `v2_reproduction.json`.

The baseline tag remains `baseline-v1-0.619493` at `e4aa4c58175242e198cefd32d6ac4558145523af`; work remains on `research/v2-stream-identity`. V1 source and submission are unchanged. Retained submission: `submissions/submission.csv`, not uploaded.

Research entry points:

```powershell
python scripts/v2_diagnostics.py
python scripts/v2_experiments.py --kind sparse --seed 42
python scripts/v2_experiments.py --kind sparse --seed 17
python scripts/v2_experiments.py --kind sparse --seed 2026
python scripts/v2_pair_learning.py
python scripts/v2_experiments.py --kind pair --seed 42
python scripts/v2_experiments.py --kind gate_only --seed 42
python scripts/v2_selection.py
python scripts/v2_stress.py build
python scripts/v2_stress.py predict
python scripts/v2_pipeline.py train --output outputs/new_v2_research_build --device cuda
python -m pytest -q
```

Research experiment IDs and output directories are immutable; the commands above describe dependency order on a fresh research checkout with V1 research caches present. Do not rerun completed experiment IDs in place. The raw training command bypasses feature caches. The official evaluation runner rejects the failed fresh-stress receipt; there is no V2 official freeze. Reports can be regenerated with `python scripts/v2_report.py` and `python scripts/v2_ledger.py`.
