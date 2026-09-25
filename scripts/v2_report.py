"""Generate final V2 evidence from immutable executed predictions and receipts."""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
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


def main():
    selection=json.loads((ROOT/"reports/v2_selection.json").read_text());sel=selection["selected"]
    fresh=json.loads((ROOT/"reports/v2_fresh_stress.json").read_text())
    a=ROOT/"outputs/v2_eval_a";b=ROOT/"outputs/v2_eval_b"
    assert (a/"metrics.json").exists(),"Use a negative-result report if fresh stress rejects the candidate"
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
