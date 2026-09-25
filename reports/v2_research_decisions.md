# V2 research decisions

## Diagnosis and scope

The three-event coverage deficit matters, but does not create a hard 0.80
ceiling. All eight family rows exist, broad candidates cover many missing
amount cases, and perfect-none headroom is much smaller than optimistic
perfect-identity headroom. See `v2_initial_diagnosis.md` for the numerical
falsification exercise and its limitations.

## Experiments and decisions

1. **Gated sparse expert: retain as complementary evidence.** New 1/2-event
   amount components expose family-relative semantic, MCC, price, recency,
   cadence, currency-separated and refund evidence. New features exist only
   for rows without a V1 3+ amount candidate. V1 stream construction is
   unchanged. Five-fold controls reproduce recorded V1 probabilities to
   floating-point precision. Across all seeds, mean stress gains for the
   sparse-aware none variant are +0.024818, +0.026547 and +0.034161.
2. **Identity-only ablation: useful but weaker.** Keeping V1 none detection
   while changing family ranking improves each seed's stress score by about
   0.021. Sparse evidence therefore helps family discrimination independently
   of its benefit to rejection. Do not interpret the joint gain solely as a
   better family classifier.
3. **Direct replacement: reject under the strong-cohort gate.** The three-seed
   joint replacement improves full-ensemble stress F1 by +0.024419, but loses
   1.44 percentage points in valid-like strong-candidate accuracy. That
   violates the predeclared one-point limit. A fixed 50% blend was declared
   after the first-seed tradeoff, before multi-seed confirmation and official
   access. No continuous weight tuning was performed.
4. **Conservative blend: selected on existing train views.** Weights are
   37.5% sparse, 37.5% original compact and 25% original legacy. Mean stress
   gain is +0.009702; sparse accuracy gains 6.55 points; all strong-cohort
   changes remain within the gate. Original OOF gains +0.026323. All seeds
   are retained. Fresh-view and official outcomes are recorded in
   `v2_results.md`; these may still refute transfer.
5. **Weak pair identity: reject.** From 10,000 disjoint unlabeled clients,
   7,319 conservative anchors yielded 272,633 positive/hard-negative pairs.
   A small LightGBM model fitted on 8,000 clients and calibrated on 2,000
   scores AUC 0.99783 on fresh-corrupted weak pairs. This is an auxiliary
   diagnostic with biased weak labels, not true stream accuracy. Proposals
   use calibrated edges and constrained graph unions. Identity-only
   downstream mean stress F1 loses 0.004562 and adds five wrong-family
   errors on average. The joint-none variant gains 0.004833 but adds 18
   wrong-family errors: its mechanism gate fails. No seed expansion or
   official access follows. More permissive candidate recall alone does not
   establish useful identity.

## Revised theory

Small streams contain usable information that was largely hidden by the
three-event representation. A global two-event replacement (V1's failed
experiment) differs from complementary evidence trained alongside intact
strong streams. The benefit has a measurable cost: new alternatives can
displace correct strong-family predictions. Conservative composition
addresses that tradeoff without pretending strong cases are untouched.

Learning the conservative amount/semantic weak-label task nearly perfectly
did not solve family discrimination. The anchors favor already easy streams,
and their negatives do not fully capture ambiguous competitors in the
official target. That falsifies this particular pair-supervision program,
not every learned-identity approach. Auxiliary success is inadequate evidence
for a larger network.

## Next hypotheses, ranked by expected value

1. Family-conditioned identity on ambiguous, same-family/overlapping-price
   hard negatives, with a separate auxiliary evaluation that tests departures
   from the anchor definitions. Keep a downstream transfer gate.
2. Continuation on already recovered candidates, using several historical
   cutoffs and explicit cancellation/refund trajectories. The current
   none-only oracle and V1's failed auxiliary transfer limit expected value;
   do not assume observed historical recurrence equals the official target.
3. An independent shifted labeled evaluation set or authoritative generator
   clarification would reduce selection uncertainty more than another small
   holdout-informed adjustment.
4. A small set/sequence model remains deferred: the cheap learned-pair model
   did not establish the transfer evidence needed to justify it.

No survival model or neural model was executed in this focused round. They
are deferred hypotheses, not negative experimental results. We did not
repeat generic classifier/seed/threshold grids. Official validation cannot
be used to change the frozen blend after seeing its score.
