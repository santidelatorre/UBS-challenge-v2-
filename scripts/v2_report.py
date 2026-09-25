"""Generate final V2 evidence from immutable executed predictions and receipts."""
import v2_bootstrap
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from ubs_recurrence.data import ROOT,LABELS,PREDICTION,validate_submission
from ubs_recurrence.evaluation import metrics


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_bootstrap(y,a,b):
    rng=np.random.default_rng(96025);delta=[]
    for _ in range(3000):
        ix=rng.integers(0,len(y),len(y));scores=[]
        for p in [a,b]:
            cm=np.bincount(y[ix]*8+p[ix],minlength=64).reshape(8,8)
            den=cm.sum(0)+cm.sum(1);scores.append(np.divide(2*cm.diagonal(),den,out=np.zeros(8),where=den>0).mean())
        delta.append(scores[1]-scores[0])
    return np.quantile(delta,[.025,.975]).tolist()


def rejected_report(selection,fresh):
    """A failed predeclared gate is not repaired by another holdout query."""
    selected=selection["selected"]
    old=joblib.load(ROOT/"outputs/final_run_a/model.joblib")
    built=joblib.load(ROOT/"outputs/v2_run_a/model.joblib")
    a=old['model'];b=built['model'].baseline_
    checks=dict(profiles_equal=old['profiles']==built['profiles'],compact_columns_equal=a.columns_==b.columns_,
                training_ids_equal=bool(np.array_equal(a.training_ids_,b.training_ids_)),
                compact_models_equal=[x.booster_.model_to_string()==z.booster_.model_to_string() for pair,other in zip(a.models_,b.models_) for x,z in zip(pair,other)],
                legacy_models_equal=[x.booster_.model_to_string()==z.booster_.model_to_string() if hasattr(x,'booster_') else bytes(x.get_booster().save_raw(raw_format='json'))==bytes(z.get_booster().save_raw(raw_format='json')) for (x,_,_),(z,_,_) in zip(a.legacy_models_,b.legacy_models_)])
    assert checks['profiles_equal'] and checks['training_ids_equal'] and checks['compact_columns_equal']
    assert all(checks['compact_models_equal']) and all(checks['legacy_models_equal'])
    (ROOT/"reports/v2_embedded_baseline_rebuild.json").write_text(json.dumps(checks,indent=2))
    decision=dict(v1_official_macro_f1=.6194934236214421,v2_official_macro_f1=None,
                  accepted_result_macro_f1=.6194934236214421,accepted_delta=0.,
                  status="retain_v1_candidate_rejected_before_official_access",official_candidate_accesses=0,
                  reason="keep4 removal-cohort loss exceeds the predeclared 0.010 accuracy tolerance",
                  keep4_baseline_correct=653,keep4_candidate_correct=643,keep4_clients=901,
                  keep4_accuracy_delta=643/901-653/901,gate_tolerance=-.010,
                  mean_fresh_stress_gain=fresh['fresh_mean_gain'],submission="submissions/submission.csv",
                  candidate_configuration=selected,clean_build=built['metadata'],baseline_weights_rebuilt_identically=checks,
                  candidate_model_sha256=sha(ROOT/"outputs/v2_run_a/model.joblib"),
                  predictive_source_commit="ff18f792b5881c67a33fba09a53ea9b91f7776be",
                  second_candidate_rebuild=False,second_rebuild_reason="Rejected before official evaluation; another identical fit would not repair the failed gate")
    (ROOT/"reports/v2_final_decision.json").write_text(json.dumps(decision,indent=2))
    (ROOT/"reports/v2_reproduction.json").write_text(json.dumps({k:v for k,v in decision.items() if k!='candidate_configuration'},indent=2))
    lines=["# V2 results\n\n",
           "**Retain V1: official macro-F1 0.619493. V2 official result: not evaluated. Accepted-result delta: +0.000000.**\n\n",
           "V2 found a promising sparse-stream signal, but the selected candidate failed a predeclared robustness gate. No V2 improvement over the official baseline is claimed. No new official model evaluation or submission was made. The 0.80 ambition remains unmet.\n\n",
           "The frozen train-side candidate gives 37.5% weight to the new sparse experts, 37.5% to V1 compact experts and 25% to V1 legacy experts. All seeds (42, 17, 2026) are retained.\n\n",
           "| Evaluation | V1 F1 | V2 candidate F1 | Delta |\n|---|---:|---:|---:|\n"]
    for v in ['original','valid_like','test_like']:
        x=selection['baseline'][v]['macro_f1'];z=selected['views'][v]['macro_f1'];lines.append(f"| Fixed {v} OOF | {x:.6f} | {z:.6f} | {z-x:+.6f} |\n")
    for v in ['fresh_valid_like','fresh_test_like','keep4','keep3','keep2','keep1']:
        x=fresh[v]['baseline']['macro_f1'];z=fresh[v]['candidate']['macro_f1'];lines.append(f"| {v} OOF | {x:.6f} | {z:.6f} | {z-x:+.6f} |\n")
    lines.append("\nAll full-sample F1 comparisons above improve. Nevertheless, at four visible events the affected 901-client cohort loses ten correct predictions (653 to 643): accuracy 72.4750% to 71.3651%, a **1.1099-point loss** against a fixed one-point tolerance. That narrowly fails the gate. We did not change the tolerance, blend, features or selected variant after seeing the fresh-view results, and did not use official validation to rescue it.\n\n")
    lines.append("The original/valid-like/test-like direct sparse model passed screening in all three seeds. Its fixed conservative blend improved mean fixed-stress F1 by 0.009702 and sparse accuracy by 6.55 points while preserving the original strong cohort within tolerance. Fresh-corruption mean F1 improved by 0.013465. Controlled one- and two-event recovery improved; four-event displacement remains the unresolved mechanism. This is useful train-side evidence, not proof of official transfer.\n\n")
    lines.append("The learned pair model used all 10,000 unlabeled clients for fitting/calibration structure, but failed downstream identity checks despite auxiliary AUC 0.99783. The missing-candidate-indicator ablation also failed, supporting the value of the new sparse transaction evidence. No survival or neural model was trained. See `v2_research_decisions.md`.\n\n")
    lines.append("One complete V2 raw-data build refit all 15 estimators in 486.8 seconds. Its embedded V1 price profiles and all nine estimator weight serializations match the frozen V1 artifact exactly. Initial verification also checked the two cached V1 probability files and their 0.619493 score. The rejected V2 candidate was not evaluated officially or rebuilt twice; another identical fit would not resolve its failed robustness gate. Hashes and exact configuration are in `v2_final_decision.json` and `v2_reproduction.json`.\n\n")
    lines.append("The baseline tag remains `baseline-v1-0.619493` at `e4aa4c58175242e198cefd32d6ac4558145523af`; work remains on `research/v2-stream-identity`. V1 source and submission are unchanged. Retained submission: `submissions/submission.csv`, not uploaded.\n\n")
    lines.append("Research entry points:\n\n```powershell\npython scripts/v2_diagnostics.py\npython scripts/v2_experiments.py --kind sparse --seed 42\npython scripts/v2_experiments.py --kind sparse --seed 17\npython scripts/v2_experiments.py --kind sparse --seed 2026\npython scripts/v2_pair_learning.py\npython scripts/v2_experiments.py --kind pair --seed 42\npython scripts/v2_experiments.py --kind gate_only --seed 42\npython scripts/v2_selection.py\npython scripts/v2_stress.py build\npython scripts/v2_stress.py predict\npython scripts/v2_pipeline.py train --output outputs/new_v2_research_build --device cuda\npython -m pytest -q\n```\n\nResearch experiment IDs and output directories are immutable; the commands above describe dependency order on a fresh research checkout with V1 research caches present. Do not rerun completed experiment IDs in place. The raw training command bypasses feature caches. The official evaluation runner rejects the failed fresh-stress receipt; there is no V2 official freeze. Reports can be regenerated with `python scripts/v2_report.py` and `python scripts/v2_ledger.py`.\n")
    (ROOT/"reports/v2_results.md").write_text(''.join(lines))
    # Challenge-OOF cohorts retain exact V1 3+/2+ eligibility definitions.
    rows=[];perclass=[];errors=[]
    from ubs_recurrence.data import transactions
    from ubs_recurrence.augmentation import corrupt_transactions,BACKGROUND
    from ubs_recurrence.features import normalize,PATTERNS,MCC
    from ubs_recurrence.price_prior import price_support
    raw=transactions('train',use_cache=False)
    profiles=built['profiles']
    for v in ['original','valid_like','test_like']:
        oldp=pd.read_csv(ROOT/f'outputs/experiments/selection_compact_3seed_plus25pct_v1_{v}/predictions.csv')
        newp=pd.read_csv(ROOT/f'outputs/experiments/v2_selection_sparse_joint_w0.375_{v}/predictions.csv').set_index('client_id').loc[oldp.client_id]
        ids=oldp.client_id.to_numpy();y=oldp.true_label.map({f:i for i,f in enumerate(LABELS)}).to_numpy()
        cols=['p_'+f for f in LABELS];pa=oldp[cols].to_numpy();pb=newp[cols].to_numpy();a=pa.argmax(1);b=pb.argmax(1)
        features=pd.read_parquet(ROOT/f'data/cache/v2/baseline_{v}.parquet')
        two=pd.read_parquet(ROOT/f'data/cache/sparse_ranking_{v}_train.parquet').reindex(features.index)
        n=np.arange(len(ids));strong=features.amount0_count.to_numpy().reshape(-1,8)[n,y]>=3
        has2=two.amount0_count.to_numpy().reshape(-1,8)[n,y]>=2;non=y<7
        d=raw if v=='original' else corrupt_transactions(raw,v,2026)
        d=d[(d.type=='card_payment')&(d.direction=='out')].copy();d['norm']=d.description.map(normalize)
        d=d[~d.norm.str.contains(BACKGROUND,regex=True)];single=np.zeros(len(ids),bool)
        for fi,f in enumerate(LABELS[:-1]):
            ok=(d.norm.str.contains(PATTERNS[f].replace('(','(?:'),regex=True)|(d.mcc==MCC[f]))&(price_support(d.amount,profiles[f])>.2)
            available=set(d.loc[ok,'client_id']);single|=(y==fi)&np.array([cid in available for cid in ids])
        masks={'C1_3plus':non&strong,'C2_only2':non&~strong&has2,'C3_single_only':non&~strong&~has2&single,
               'no_plausible_proxy':non&~strong&~has2&~single,'C4_V1_present_wrong_family':non&strong&(a<7)&(a!=y),
               'C5_V1_family_to_none':non&(a==7),'C6_V1_none_to_family':(~non)&(a<7),'C7_music_software_streaming':np.isin(y,[4,5,6]),'true_none':~non}
        for name,mask in masks.items():
            rows.append(dict(view=v,cohort=name,n=int(mask.sum()),v1_accuracy=float((a[mask]==y[mask]).mean()) if mask.any() else None,v2_accuracy=float((b[mask]==y[mask]).mean()) if mask.any() else None,rescued=int(((a!=y)&(b==y)&mask).sum()),lost=int(((a==y)&(b!=y)&mask).sum())))
        ma=metrics(y,pa);mb=metrics(y,pb)
        for f in LABELS:perclass.append(dict(view=v,family=f,v1_f1=ma['classification_report'][f]['f1-score'],v2_f1=mb['classification_report'][f]['f1-score']))
        for model,pred in [('V1',a),('V2 candidate',b)]:
            errors.append(dict(view=v,model=model,wrong_family=int((non&(pred<7)&(pred!=y)).sum()),family_to_none=int((non&(pred==7)).sum()),none_to_family=int(((~non)&(pred<7)).sum())))
    pd.DataFrame(rows).to_csv(ROOT/'reports/v2_error_cohorts.csv',index=False)
    pd.DataFrame(perclass).to_csv(ROOT/'reports/v2_per_class_oof.csv',index=False)
    pd.DataFrame(errors).to_csv(ROOT/'reports/v2_error_counts.csv',index=False)
    text=['# V2 error analysis\n\nNo V2 official error analysis exists because the candidate failed its predeclared gate before official access. This report compares complete client OOF predictions and fixed stress views; no cohort is removed from overall F1. Initial official V1 diagnostics remain in `v2_diagnostics.json`.\n\n',
          '| View | Cohort | n | V1 accuracy | V2 accuracy | Rescued | Lost |\n|---|---|---:|---:|---:|---:|---:|\n']
    for r in rows:
        if r['v1_accuracy'] is not None:text.append(f"| {r['view']} | {r['cohort']} | {r['n']} | {r['v1_accuracy']:.4f} | {r['v2_accuracy']:.4f} | {r['rescued']} | {r['lost']} |\n")
    text.append('\nC4/C5/C6 are defined by V1 errors so their rescue rates can be compared on fixed clients. C1/C2/C3 use inferred semantic/MCC/price eligibility, not annotated true stream membership.\n\n| View | Model | Wrong family | Family to none | None to family |\n|---|---|---:|---:|---:|\n')
    for r in errors:text.append(f"| {r['view']} | {r['model']} | {r['wrong_family']} | {r['family_to_none']} | {r['none_to_family']} |\n")
    text.append('\nControlled removal on 901 selected train clients (same target, latest k selected-stream card events retained, other events/refunds unchanged):\n\n| Events | V1 accuracy | V2 accuracy | Delta points |\n|---|---:|---:|---:|\n')
    for k in [4,3,2,1]:
        v=fresh[f'keep{k}'];x=v['baseline']['cohorts']['removal_cohort']['accuracy'];z=v['candidate']['cohorts']['removal_cohort']['accuracy'];text.append(f'| {k} | {x:.4f} | {z:.4f} | {100*(z-x):+.4f} |\n')
    text.append('\nOne-event recovery rises from 13.65% to 32.96%, but four-event displacement violates the fixed gate. Full-sample removal F1 includes clients whose streams were not removed; the affected-cohort table is necessary to see that tradeoff. This is why a global F1 win alone was not accepted.\n\n| View | Family | V1 F1 | V2 F1 |\n|---|---|---:|---:|\n')
    for r in perclass:text.append(f"| {r['view']} | {r['family']} | {r['v1_f1']:.4f} | {r['v2_f1']:.4f} |\n")
    text.append('\nThese estimates were used for research decisions and are not independent claims of deployment improvement. Next work should address displacement of plausible existing streams and use a new independent stress design; it should not relabel this failed gate as passing.\n')
    (ROOT/'reports/v2_error_analysis.md').write_text(''.join(text))
    print(json.dumps({k:v for k,v in decision.items() if k not in ['candidate_configuration','clean_build','baseline_weights_rebuilt_identically']},indent=2))


def main():
    selection=json.loads((ROOT/"reports/v2_selection.json").read_text());sel=selection["selected"]
    fresh=json.loads((ROOT/"reports/v2_fresh_stress.json").read_text())
    a=ROOT/"outputs/v2_eval_a";b=ROOT/"outputs/v2_eval_b"
    if not (a/"metrics.json").exists():
        assert not fresh['no_material_reversal'],"A passing candidate still requires its frozen official evaluation"
        rejected_report(selection,fresh)
        return
    m=json.loads((a/"metrics.json").read_text());p=pd.read_csv(a/"probabilities.csv")
    cols=["p_"+f for f in LABELS];ids=p.client_id.to_numpy()
    old=pd.read_csv(ROOT/"outputs/experiments/final_final_eval_a/predictions.csv").set_index("client_id").loc[ids]
    y=old.true_label.map({f:i for i,f in enumerate(LABELS)}).to_numpy()
    q=p[cols].to_numpy();v1=old[cols].to_numpy();yp=q.argmax(1);op=v1.argmax(1)
    assert metrics(y,q)["macro_f1"]==m["macro_f1"]
    diff=m["macro_f1"]-.6194934236214421;ci=paired_bootstrap(y,op,yp)
    reproduction=dict(clean_raw_build_a=True,clean_raw_build_b=(ROOT/"outputs/v2_run_b/metadata.json").exists(),
                      probability_sha_a=sha(a/"probabilities.csv"),official_macro_f1=m["macro_f1"],delta_from_v1=diff,
                      paired_bootstrap_delta_95_ci=ci,bootstrap_seed=96025,bootstrap_resamples=3000,
                      bootstrap_limit="Conditional on fitted models and this exposed validation set; excludes selection uncertainty")
    if (b/"probabilities.csv").exists():
        p2=pd.read_csv(b/"probabilities.csv");assert np.array_equal(p.client_id,p2.client_id)
        delta=float(np.max(np.abs(p[cols].to_numpy()-p2[cols].to_numpy())))
        assert delta==0 and np.array_equal(p[PREDICTION],p2[PREDICTION])
        reproduction.update(maximum_probability_difference=delta,all_predictions_identical=True,probability_sha_b=sha(b/"probabilities.csv"))
    if (ROOT/"outputs/v2_submission_b/submission.csv").exists():
        sa=ROOT/"outputs/v2_submission_a/submission.csv";sb=ROOT/"outputs/v2_submission_b/submission.csv"
        assert sa.read_bytes()==sb.read_bytes()
        sub=pd.read_csv(sa);validate_submission(sub)
        reproduction.update(test_submissions_byte_identical=True,submission_sha256=sha(sa),submission_rows=len(sub),exact_ids=True,legal_labels=True)
        if diff>0:
            dest=ROOT/"submissions/submission_v2.csv";dest.write_bytes(sa.read_bytes());validate_submission(pd.read_csv(dest))
            (ROOT/"submissions/validation_v2.json").write_text(json.dumps(reproduction,indent=2))
    (ROOT/"reports/v2_reproduction.json").write_text(json.dumps(reproduction,indent=2))
    lines=["# V2 results\n\n",f"**V1 official macro-F1: 0.619493. V2: {m['macro_f1']:.6f}. Delta: {diff:+.6f}. Accuracy: {m['accuracy']:.6f}.**\n\n",
           "The 0.80 ambition remains unmet. This is a frozen external comparison after train-side gates, not an untouched holdout or hidden-test score. ",
           f"The paired 3,000-client-bootstrap interval for the F1 difference is [{ci[0]:+.4f}, {ci[1]:+.4f}]; it does not account for model-selection uncertainty.\n\n",
           "The model preserves V1 stream construction and adds family-relative evidence from single events and pairs only where a family lacks a 3+ amount candidate. Weights are 37.5% sparse compact, 37.5% V1 compact and 25% V1 legacy; all three seeds are retained. No weights or thresholds were changed after official access.\n\n",
           "| Evaluation | V1 F1 | V2 F1 | Delta |\n|---|---:|---:|---:|\n"]
    for v in ["original","valid_like","test_like"]:
        x=selection['baseline'][v]['macro_f1'];z=sel['views'][v]['macro_f1'];lines.append(f"| Fixed {v} OOF | {x:.6f} | {z:.6f} | {z-x:+.6f} |\n")
    for v in ["fresh_valid_like","fresh_test_like"]:
        x=fresh[v]['baseline']['macro_f1'];z=fresh[v]['candidate']['macro_f1'];lines.append(f"| {v} OOF | {x:.6f} | {z:.6f} | {z-x:+.6f} |\n")
    lines.append(f"| Official validation | 0.619493 | {m['macro_f1']:.6f} | {diff:+.6f} |\n")
    lines.append("\nThe direct sparse replacement was rejected for strong-cohort regression. The weak pair model was rejected because auxiliary success did not reduce challenge wrong-family errors. No continuation or neural model was trained in this round. See `v2_research_decisions.md`.\n\n")
    if reproduction.get("all_predictions_identical"):
        lines.append("Two independent raw-data fits reproduce every official probability and prediction exactly. Their embedded V1 models also reproduce the frozen V1 probabilities. ")
    if reproduction.get("test_submissions_byte_identical"):
        lines.append("The two test submissions are byte-identical and pass exact-ID/legal-label checks. The new candidate is `submissions/submission_v2.csv`; V1 remains at `submissions/submission.csv`. Neither has been uploaded. ")
    lines.append("\n\nReproduce from the pinned raw inputs:\n\n```powershell\npython -X utf8 scripts/v2_pipeline.py train --output outputs/my_v2 --device cuda\npython -X utf8 scripts/v2_pipeline.py evaluate --model outputs/my_v2/model.joblib --output outputs/my_v2_eval --freeze reports/v2_freeze_1.md --purpose \"Independent frozen V2 reproduction\"\npython -X utf8 scripts/v2_pipeline.py submit --model outputs/my_v2/model.joblib --output outputs/my_v2_submission\npython -m pytest -q\n```\n\nEvaluation explicitly accesses official labels and logs it. Full training starts from raw JSONL, relearns unlabeled price priors and refits all 15 estimators. It requires `v2_selection.json`; evaluation also requires the committed freeze and fresh-stress receipt. Use new output directories; existing evidence is not overwritten. Research CV scripts additionally depend on V1 research caches, unlike this raw-data runner.\n")
    (ROOT/"reports/v2_results.md").write_text("".join(lines))
    base=pd.read_parquet(a/"baseline_evidence.parquet");extra=pd.read_parquet(a/"evidence_features.parquet")
    trueindex=pd.MultiIndex.from_arrays([ids,np.array(LABELS)[y]],names=base.index.names)
    ev=base.reindex(trueindex);sp=extra.reindex(trueindex);non=y<7;strong=ev.amount0_count.to_numpy()>=3
    masks={"all":np.ones(len(y),bool),"C1_true_family_3plus":non&strong,
           "C1_regular":non&strong&(ev.amount0_gap_cv.to_numpy()>=0)&(ev.amount0_gap_cv.to_numpy()<=.25),
           "sparse_true_family":non&~strong,
           "sparse_best_proposal_pair":non&~strong&(sp.sparse0_count.to_numpy()==2),
           "sparse_best_proposal_single":non&~strong&(sp.sparse0_count.to_numpy()==1),
           "C7_music_software_streaming":np.isin(y,[4,5,6]),"true_none":~non}
    rows=[]
    for name,mask in masks.items():
        rows.append(dict(cohort=name,n=int(mask.sum()),v1_accuracy=float((op[mask]==y[mask]).mean()),v2_accuracy=float((yp[mask]==y[mask]).mean()),rescued=int(((op!=y)&(yp==y)&mask).sum()),lost=int(((op==y)&(yp!=y)&mask).sum())))
    pd.DataFrame(rows).to_csv(ROOT/"reports/v2_error_cohorts.csv",index=False)
    e=["# V2 frozen error analysis\n\n", "This analysis follows the frozen official evaluation and does not change the predictor. Cohorts overlap. Pair/single rows refer to the best new sparse proposal and differ from the eligibility-envelope cohorts in the initial oracle audit.\n\n",
       "| Cohort | n | V1 accuracy | V2 accuracy | Rescued | Lost |\n|---|---:|---:|---:|---:|---:|\n"]
    for r in rows:e.append(f"| {r['cohort']} | {r['n']} | {r['v1_accuracy']:.4f} | {r['v2_accuracy']:.4f} | {r['rescued']} | {r['lost']} |\n")
    e.append("\n| Error type | V1 | V2 |\n|---|---:|---:|\n")
    for name,fun in [("wrong recurring family",lambda p:non&(p<7)&(p!=y)),("family to none",lambda p:non&(p==7)),("none to family",lambda p:(~non)&(p<7)),("candidate present but wrong family wins",lambda p:non&strong&(p<7)&(p!=y))]:
        e.append(f"| {name} | {int(fun(op).sum())} | {int(fun(yp).sum())} |\n")
    e.append("\n| Family | V1 F1 | V2 F1 | Delta |\n|---|---:|---:|---:|\n");oldm=metrics(y,v1)
    for f in LABELS:
        a1=oldm['classification_report'][f]['f1-score'];a2=m['classification_report'][f]['f1-score'];e.append(f"| {f} | {a1:.4f} | {a2:.4f} | {a2-a1:+.4f} |\n")
    e.append("\nControlled removal benchmark (901 train clients; proxy membership selected using training labels only; fitted models never see these removal views):\n\n| Visible selected-stream events | V1 accuracy | V2 accuracy |\n|---|---:|---:|\n")
    for k in [4,3,2,1]:
        v=fresh[f'keep{k}'];a1=v['baseline']['cohorts']['removal_cohort']['accuracy'];a2=v['candidate']['cohorts']['removal_cohort']['accuracy'];e.append(f"| {k} | {a1:.4f} | {a2:.4f} |\n")
    e.append("\nThe model still fails many sparse and ambiguous-family clients. Neither a visible candidate nor improved synthetic removal performance proves it is the process chosen by the undisclosed official target generator. Cohort gains support the representation mechanism but do not identify every failure causally.\n")
    (ROOT/"reports/v2_error_analysis.md").write_text("".join(e))
    print(json.dumps(reproduction,indent=2))


if __name__=="__main__":main()
