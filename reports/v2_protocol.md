# V2 research protocol, declared before new model evaluation

Reference: `baseline-v1-0.619493`, peeled commit
`e4aa4c58175242e198cefd32d6ac4558145523af`; work stays on
`research/v2-stream-identity`. Baseline official macro-F1 is
0.6194934236214421. V1 reports, package files and submission are preserved.

First verify cached baseline runs against pinned raw inputs and labels; do
not call this a new clean training run. One logged access reuses frozen V1
predictions for aggregate oracle analysis. No V2 model is evaluated there.

Research choices, in order of expected value:

1. Sparse complementary evidence: retain V1 strong streams; add family-relative
   single-event and pair evidence only in family rows lacking a 3+ amount
   stream. High value, small implementation/training cost.
2. Learned identity/proposals: conservative stream anchors from the disjoint
   unlabeled clients, corrupted observations and within-client hard negatives.
   Proceed if diagnostics show identity errors and the cheap sparse test does
   not adequately explain them. Compare new candidate representation, not
   generic boosting engines or tuning grids.
3. Continuation supervision: historical cutoffs and future recurrence of
   previously discovered streams, only if none headroom and preceding results
   justify the extra target-transfer risk.
4. A learned set model has lower priority and requires evidence that simpler
   identity learning transfers. No large neural search is authorized by this
   protocol's evidence.

All official targets are excluded from features, auxiliary fitting and
training. Supervised challenge fitting uses only 2,000 training clients.
Augmented copies remain in their client's fold. Original/valid-like/test-like
views use V1's fixed corruption seed 2026. New auxiliary models use only
unlabeled clients, with client-separated auxiliary evaluation. Auxiliary
metrics never count as challenge metrics.

Screen each conceptual component on the five V1 folds for seed 42 with the
unchanged compact learning parameters. Cache lineage must be explicit:
`compact_*` files were last overwritten by the V1 min-count-2 experiment and
must NOT be treated as V1 compact features. Reconstruct V1 compact features
from the min-count-3 ranking matrices and matching payment context, or raw
data. First reproduce the matched control's recorded predictions.

Advance a component to seed confirmation if mean valid/test stress macro-F1
improves by at least 0.003 on the matched control, original macro-F1 loses at
most 0.010, and its intended mechanism improves (for sparse evidence: mean
sparse-cohort accuracy improves by at least 0.020). Report all trials and all
seeds. Do not use the official holdout to rescue a failed component.

Before a new official evaluation require the candidate, averaged over seeds
42, 17 and 2026, to beat the full V1 reference by at least 0.005 mean stress
macro-F1, lose at most 0.010 original macro-F1, and have positive stress gain
in at least two matched seed comparisons. Sparse candidates must improve
sparse accuracy by at least 0.030 on average and lose at most 0.010 accuracy
in strong-candidate clients. For a pivoted identity/continuation candidate,
require reduced corresponding family/none errors instead of sparse rescue.

Also check a fresh, evaluation-only corruption seed (91711) and controlled
stream observation removal to 4/3/2/1 events before official access. Removal
uses train-side proxy membership, explicitly labeled as simulation. Do not
retune on that fresh view. A material reversal rejects the candidate.

Only candidates passing these gates get a committed `v2_freeze_N.md` with
exact configuration, code commit, evidence and acceptance rule. Write
predictions before reading official labels, and append the access log. A
win requires official F1 above V1; report paired uncertainty and do not claim
statistical certainty from a small gain. Rebuild a winner from raw data and
check reproducibility, IDs and submission contract. If no component passes,
retain V1 and finish the round with explicit negative results and next
hypotheses. No new official score is necessary for a rejected train-side
candidate.
