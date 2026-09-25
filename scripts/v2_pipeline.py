"""Raw-data V2 train/predict/evaluate. Official access requires a freeze file."""
import v2_bootstrap
import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from ubs_recurrence.data import ROOT,LABELS,PREDICTION,transactions,aligned_target,validate_submission
from ubs_recurrence.augmentation import corrupt_transactions
from ubs_recurrence.model import build_features
from ubs_recurrence.v2_sparse import sparse_features
from ubs_recurrence.v2_model import SparseForecaster
from ubs_recurrence.price_prior import learn_price_profiles
from ubs_recurrence.evaluation import metrics,record,source_fingerprint


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_check():
    manifest=json.loads((ROOT/"reports/data_manifest.json").read_text())
    for name,r in manifest.items():assert sha(ROOT/"data/raw"/name)==r["sha256"],name
    return manifest


def train(args):
    start=time.perf_counter();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    manifest=manifest_check();selected=json.loads((ROOT/"reports/v2_selection.json").read_text())["selected"]
    assert selected and selected["gate"]["passed"]
    df=transactions("train",use_cache=False);profiles=learn_price_profiles(use_cache=False)
    base=[];expanded=[];legacy=[]
    for view in ["original","valid_like","test_like"]:
        raw=df if view=="original" else corrupt_transactions(df,view,2026)
        x,_,old=build_features(raw,profiles,return_legacy=True)
        sp=sparse_features(raw,x,profiles)
        base.append(x);expanded.append(pd.concat([x,sp],axis=1).fillna(-999));legacy.append(old)
        print("Raw features",view,x.shape,sp.shape,flush=True)
    ids=base[0].index.get_level_values(0).unique().to_numpy();y=aligned_target(ids)
    model=SparseForecaster(selected["sparse_weight"],selected["variant"],args.device).fit(base,expanded,y,legacy)
    metadata=dict(created_at=datetime.now(timezone.utc).isoformat(),git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                  source_sha256=source_fingerprint(),raw_manifest=manifest,clean_raw_build=True,training_clients=len(ids),training_split="train only",
                  seeds=[42,17,2026],sparse_weight=selected["sparse_weight"],variant=selected["variant"],legacy_weight=.25,
                  corruption_seed=2026,seconds=time.perf_counter()-start,device=args.device,official_labels_used=False)
    joblib.dump(dict(model=model,profiles=profiles,metadata=metadata),out/"model.joblib")
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2),flush=True)


def predict(args):
    start=time.perf_counter();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    manifest_check();artifact=joblib.load(args.model)
    split="test" if args.command=="submit" else "valid"
    if split=="valid":
        assert args.freeze and Path(args.freeze).exists(),"Official evaluation requires a committed freeze"
        status=json.loads((ROOT/"reports/v2_fresh_stress.json").read_text())
        assert status["no_material_reversal"]
    raw=transactions(split,use_cache=False);profiles=artifact["profiles"]
    x,s,legacy=build_features(raw,profiles,return_legacy=True)
    expanded=pd.concat([x,sparse_features(raw,x,profiles)],axis=1).fillna(-999)
    p=artifact["model"].predict_proba(x,expanded,legacy);ids=x.index.get_level_values(0).unique().to_numpy()
    table=pd.DataFrame(p,columns=["p_"+f for f in LABELS]);table.insert(0,"client_id",ids);table[PREDICTION]=np.array(LABELS)[p.argmax(1)]
    table.to_csv(out/"probabilities.csv",index=False);x.to_parquet(out/"baseline_evidence.parquet");expanded.to_parquet(out/"evidence_features.parquet")
    # Cross-check the new raw build's embedded V1 pipeline, not just its score.
    baseline=artifact["model"].baseline_.predict_proba(x,legacy)
    reference=ROOT/("outputs/final_submission/probabilities.csv" if split=="test" else "outputs/final_eval_a/probabilities.csv")
    if reference.exists():
        old=pd.read_csv(reference).set_index("client_id").loc[ids,["p_"+f for f in LABELS]].to_numpy()
        delta=float(np.max(np.abs(old-baseline)));assert delta<1e-10,delta
        print("Embedded raw-rebuilt V1 matches frozen probabilities",delta,flush=True)
    if split=="test":
        sample=pd.read_csv(ROOT/"data/raw/sample_submission.csv")
        sub=table.set_index("client_id")[[PREDICTION]].reindex(sample.client_id).reset_index();validate_submission(sub,sample)
        sub.to_csv(out/"submission.csv",index=False);validate_submission(pd.read_csv(out/"submission.csv"),sample)
        receipt=dict(rows=len(sub),exact_ids=True,legal_labels=True,sha256=sha(out/"submission.csv"),uploaded=False)
        (out/"submission_validation.json").write_text(json.dumps(receipt,indent=2));print(receipt,flush=True)
    else:
        entry=dict(timestamp=datetime.now(timezone.utc).isoformat(),batch=out.name,purpose=args.purpose,
                   freeze=args.freeze,freeze_sha256=sha(Path(args.freeze)),model_sha256=sha(Path(args.model)),prediction_sha256=sha(out/"probabilities.csv"),predictions_frozen_before_labels=True)
        with (ROOT/"reports/holdout_access_log.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(entry)+"\n")
        y=aligned_target(ids,"valid",allow_holdout=True)
        r=record("v2_"+out.name,ids,y,p,runtime=time.perf_counter()-start,split="official_validation",metadata=dict(hypothesis="Complementary sparse representation transfers under official corruption",baseline_comparison=.6194934236214421,features="V1 plus gated single/pair evidence",model="SparseForecaster",parameters=artifact["metadata"],seed=[42,17,2026],protocol=args.purpose,status="frozen_evaluation",conclusion="Compare against immutable V1",candidate_generation="V1 unchanged, sparse 1/2-event components",stream_identity="soft family semantic/MCC/price",continuation_model="blend of original/sparse-aware none",candidate_recall="V1 evidence retained; sparse envelope from diagnostics"))
        (out/"metrics.json").write_text(json.dumps(r,indent=2))
        print("Official macro-F1",r["macro_f1"],"delta",r["macro_f1"]-.6194934236214421,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=["train","evaluate","submit"]);ap.add_argument("--output",required=True);ap.add_argument("--model");ap.add_argument("--device",default="cuda",choices=["cuda","cpu"]);ap.add_argument("--freeze");ap.add_argument("--purpose",default="Frozen V2 external evaluation; no holdout fitting or calibration")
    args=ap.parse_args();train(args) if args.command=="train" else predict(args)
