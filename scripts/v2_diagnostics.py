"""Verify V1 and decompose frozen errors. No fitted V2 predictor or oracle features.

Candidate correspondence is a proxy, not ground-truth stream identity. The
operational model always has eight rows, so proxy coverage is not its ceiling.
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ubs_recurrence.data import ROOT, LABELS, PREDICTION, transactions, aligned_target
from ubs_recurrence.evaluation import metrics
from ubs_recurrence.augmentation import corrupt_transactions, BACKGROUND
from ubs_recurrence.features import normalize, PATTERNS, MCC
from ubs_recurrence.price_prior import price_support
from ubs_recurrence.streams import extract_streams, family_features


OUT = ROOT / "outputs/v2_diagnostics"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify():
    assert git("branch", "--show-current") == "research/v2-stream-identity"
    base = git("rev-parse", "baseline-v1-0.619493^{commit}")
    assert base == "e4aa4c58175242e198cefd32d6ac4558145523af"
    manifest = json.loads((ROOT / "reports/data_manifest.json").read_text())
    for name, receipt in manifest.items():
        assert sha(ROOT / "data/raw" / name) == receipt["sha256"], name
    a = ROOT / "outputs/final_eval_a/probabilities.csv"
    b = ROOT / "outputs/final_eval_b/probabilities.csv"
    assert a.read_bytes() == b.read_bytes()
    assert sha(a) == "757b5fc2d4e71e585f232dae28f3a509078083ba40b8d2d49338ed97b376613e"
    for name in ["evidence_features.parquet", "evidence_streams.parquet"]:
        pd.testing.assert_frame_equal(pd.read_parquet(a.parent / name), pd.read_parquet(b.parent / name))
    assert (ROOT / "outputs/final_submission/submission.csv").read_bytes() == (ROOT / "outputs/final_submission_b/submission.csv").read_bytes()
    assert (ROOT / "submissions/submission.csv").read_bytes() == (ROOT / "outputs/final_submission/submission.csv").read_bytes()
    # One explicitly logged cached-baseline reanalysis; no new candidate selection.
    entry = dict(timestamp=datetime.now(timezone.utc).isoformat(), batch="v2_baseline_diagnostics",
                 purpose="Verify frozen V1 cached predictions against raw labels; aggregate candidate/oracle diagnostics only",
                 predictions_frozen_before_labels=True, prediction_sha256=sha(a),
                 new_model_evaluation=False, clean_rebuild_this_round=False)
    if not OUT.exists():
        with (ROOT / "reports/holdout_access_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    OUT.mkdir(exist_ok=True)
    p = pd.read_csv(a)
    y = aligned_target(p.client_id, "valid", allow_holdout=True)
    q = p[["p_" + f for f in LABELS]].to_numpy()
    m = metrics(y, q)
    assert abs(m["macro_f1"] - .6194934236214421) < 1e-14
    assert m["accuracy"] == .647
    assert np.array_equal(np.array(LABELS)[q.argmax(1)], p[PREDICTION])
    evidence = json.loads((ROOT / "reports/reproduction.json").read_text())
    receipt = dict(branch=git("branch", "--show-current"), starting_commit=base,
                   audit_commit=git("rev-parse", "HEAD"), tag="baseline-v1-0.619493",
                   tag_object=git("rev-parse", "baseline-v1-0.619493"), tag_commit=base,
                   raw_hashes_verified=manifest, macro_f1=m["macro_f1"], accuracy=m["accuracy"],
                   prediction_sha256=sha(a), cached_runs_identical=True,
                   clean_rebuild_this_round=False, previous_clean_rebuild_evidence=evidence,
                   policy="V1 source, tag, submission and historical reports remain unchanged; V2 scripts are additive.")
    (ROOT / "reports/v2_baseline_verification.json").write_text(json.dumps(receipt, indent=2))
    return p.client_id.to_numpy(), y, q


def f1(y, pred):
    return float(f1_score(y, pred, labels=range(8), average="macro", zero_division=0))


def cohort(y, pred, mask):
    return dict(n=int(mask.sum()), errors=int((y[mask] != pred[mask]).sum()),
                accuracy=float((y[mask] == pred[mask]).mean()) if mask.any() else None,
                macro_f1_all8=f1(y[mask], pred[mask]) if mask.any() else None)


def diagnose(name, ids, y, p, streams, two, raw, profiles):
    index = pd.MultiIndex.from_product([ids, LABELS], names=["client_id", "family"])
    soft = family_features(streams, ids, profiles).reindex(index).fillna(-999)
    hard = family_features(streams, ids).reindex(index).fillna(-999)
    two = two.reindex(index).fillna(-999)
    shape = (len(ids), 8)
    cov = {
        "hard_amount3": (hard.amount0_count.to_numpy().reshape(shape) >= 3),
        "soft_amount3": (soft.amount0_count.to_numpy().reshape(shape) >= 3),
        "soft_amount2": (two.amount0_count.to_numpy().reshape(shape) >= 2),
        "broad2": (soft.broad0_count.to_numpy().reshape(shape) >= 2),
    }
    # A permissive singleton evidence envelope, not an assertion of recurrence.
    d = raw[(raw.type == "card_payment") & (raw.direction == "out")].copy()
    d["norm"] = d.description.map(normalize)
    d = d[~d.norm.str.contains(BACKGROUND, regex=True)]
    singleton = pd.DataFrame(False, index=ids, columns=LABELS)
    for f in LABELS[:-1]:
        semantic = d.norm.str.contains(PATTERNS[f].replace("(", "(?:"), regex=True)
        plausible = (semantic | (d.mcc == MCC[f])) & (price_support(d.amount, profiles[f]) > .2)
        singleton.loc[d.loc[plausible, "client_id"].unique(), f] = True
    cov["single_plausible"] = singleton.to_numpy()
    cov["combined3_broad"] = cov["soft_amount3"] | cov["broad2"]
    cov["combined2_broad_single"] = cov["soft_amount2"] | cov["broad2"] | cov["single_plausible"]
    row = np.arange(len(ids)); pred = p.argmax(1); bestfamily = p[:, :7].argmax(1)
    non = y < 7; err = y != pred
    has = cov["soft_amount3"][row, y]
    cases = {
        "candidate_absent_error": err & non & ~has,
        "candidate_present_family_misranked": err & non & has & (bestfamily != y),
        "candidate_present_correct_family_none_override": err & non & has & (bestfamily == y),
        "true_none_false_activation": err & ~non,
    }
    assert sum(m.sum() for m in cases.values()) == err.sum()
    decomp = {k:dict(count=int(v.sum()), fraction_of_errors=float(v.sum()/err.sum())) for k,v in cases.items()}
    t = soft.iloc[row * 8 + y].reset_index(drop=True)
    count = t.amount0_count.to_numpy(); gap = t.amount0_gap_cv.to_numpy()
    has2 = cov["soft_amount2"][row,y]; single = cov["single_plausible"][row,y]
    masks = {
        "C1_amount3": non & has,
        "C1_regular": non & has & (gap <= .25) & (gap >= 0),
        "C2_amount2_only": non & ~has & has2,
        "C3_single_only": non & ~has & ~has2 & single,
        "no_plausible_proxy": non & ~has & ~has2 & ~single,
        "C4_present_wrong_family": non & has & (pred < 7) & err,
        "C5_family_to_none": non & (pred == 7),
        "C6_none_to_family": ~non & (pred < 7),
        "C7_music_software_streaming": np.isin(y, [4,5,6]),
    }
    for lo, hi in [(3,4),(4,6),(6,10),(10,1000)]:
        masks[f"length_{lo}_to_{hi-1}"] = non & (count >= lo) & (count < hi)
    for col, bins in [("amount0_amount_cv",[0,.01,.035,.1,100]),
                      ("amount0_own_semantic",[0,.01,.25,.5,1.01]),
                      ("amount0_own_mcc",[0,.25,.5,.9,1.01]),
                      ("amount0_generic_fraction",[0,.01,.25,.5,1.01])]:
        vals=t[col].to_numpy()
        for lo,hi in zip(bins[:-1], bins[1:]):
            masks[f"{col}_{lo}_{hi}"] = non & has & (vals >= lo) & (vals < hi)
    nc = cov["soft_amount3"][:,:7].sum(1)
    for lo,hi in [(0,1),(1,2),(2,4),(4,8)]:
        masks[f"candidate_families_{lo}_{hi-1}"] = (nc >= lo) & (nc < hi)
    oracles = {}
    candidate_rows=[]
    for method, available in cov.items():
        covered = available[row,y] & non
        oracle = np.where(covered | ~non, y, 7)
        recalls=[]
        for fi,f in enumerate(LABELS[:-1]):
            m=y==fi; r=float(available[m,fi].mean()); recalls.append(r)
            candidate_rows.append(dict(view=name,method=method,family=f,n=int(m.sum()),covered=int(available[m,fi].sum()),recall=r))
        # This upper bound ignores unavoidable false positives when evidence is absent.
        upper=(1+sum(2*r/(1+r) for r in recalls))/8
        oracles[method]=dict(non_none_recall=float(covered.sum()/non.sum()),
                             proxy_constrained_oracle_f1=f1(y,oracle),
                             optimistic_f1_upper_bound=float(upper))
    identity = pred.copy(); identity[non & (pred<7)] = y[non & (pred<7)]
    perfectnone = np.where(non, bestfamily, 7)
    repair_covered=pred.copy(); repair_covered[has & non]=y[has & non]
    res=dict(view=name, baseline_macro_f1=f1(y,pred), errors=int(err.sum()), decomposition=decomp,
             overlapping_error_types={"wrong_family":int((err & non & (pred<7)).sum()),"family_to_none":int((non & (pred==7)).sum()),"none_to_family":int((~non & (pred<7)).sum())},
             candidates=oracles, perfect_identity_keep_current_none_boundary=f1(y,identity),
             perfect_none_keep_family_ranking=f1(y,perfectnone),
             repair_covered_keep_other_predictions=f1(y,repair_covered),
             all_eight_family_rows_formal_oracle=1.0,
             cohorts={k:cohort(y,pred,m) for k,m in masks.items()})
    (OUT/f"{name}.json").write_text(json.dumps(res,indent=2))
    return res,candidate_rows


def main():
    ids,y,p=verify()
    profiles=json.loads((ROOT/"data/cache/unlabeled_price_profiles.json").read_text())
    raw=transactions("valid",use_cache=False)
    streams=pd.read_parquet(ROOT/"outputs/final_eval_a/evidence_streams.parquet")
    two=family_features(extract_streams(raw,.035,True,min_count=2),ids,profiles)
    results=[]; rows=[]
    r,c=diagnose("official_frozen_v1",ids,y,p,streams,two,raw,profiles);results.append(r);rows.extend(c)
    print(json.dumps(r,indent=2),flush=True)
    df=transactions("train",use_cache=False)
    for scenario in ["original","valid_like","test_like"]:
        pred=pd.read_csv(ROOT/f"outputs/experiments/selection_compact_3seed_plus25pct_v1_{scenario}/predictions.csv")
        ids=pred.client_id.to_numpy();y=aligned_target(ids);p=pred[["p_"+f for f in LABELS]].to_numpy()
        path="streams_0.035_filtered_train.parquet" if scenario=="original" else f"streams_train_noise{scenario}_s2026.parquet"
        s=pd.read_parquet(ROOT/"data/cache"/path)
        two=pd.read_parquet(ROOT/f"data/cache/sparse_ranking_{scenario}_train.parquet")
        raw=df if scenario=="original" else corrupt_transactions(df,scenario,2026)
        r,c=diagnose(scenario,ids,y,p,s,two,raw,profiles);results.append(r);rows.extend(c)
        print(scenario,r["baseline_macro_f1"],r["decomposition"],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/"reports/v2_candidate_recall.csv",index=False)
    (ROOT/"reports/v2_diagnostics.json").write_text(json.dumps(results,indent=2))
    lines=["# V2 candidate and oracle diagnostics\n\n",
           "Correspondence is inferred from semantic/MCC/price eligibility, not annotated latent stream IDs. All eight family rows always exist. Coverage numbers below are evidence proxies, not a mathematical ceiling on V1. Official rows reuse the frozen V1 predictions; one aggregate diagnostic access is logged.\n\n",
           "| View | F1 | Soft 3+ recall | 2+ recall | 3+ or broad recall | Singleton envelope recall | Perfect identity/current none | Perfect none/current family rank |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"]
    for r in results:
        c=r["candidates"]
        lines.append(f"| {r['view']} | {r['baseline_macro_f1']:.6f} | {c['soft_amount3']['non_none_recall']:.4f} | {c['soft_amount2']['non_none_recall']:.4f} | {c['combined3_broad']['non_none_recall']:.4f} | {c['combined2_broad_single']['non_none_recall']:.4f} | {r['perfect_identity_keep_current_none_boundary']:.4f} | {r['perfect_none_keep_family_ranking']:.4f} |\n")
    lines.append("\nThe constrained oracle assigns the true family when proxy evidence exists and none otherwise, including perfect true-none decisions. The separate optimistic bound ignores false positives forced by uncovered cases. Neither is a realizable model. Perfect identity fixes family errors while preserving the observed none boundary; it is an optimistic relabeling headroom diagnostic, not identified continuation performance. Perfect none restores the highest scored non-none family for true recurring clients.\n\nFull per-family recall, confidence/length/count/masking cohorts, exclusive error partitions and oracle definitions are in `v2_candidate_recall.csv`, `v2_diagnostics.json`, and `scripts/v2_diagnostics.py`.\n")
    (ROOT/"reports/v2_candidate_recall.md").write_text("".join(lines))


if __name__=="__main__":
    main()
