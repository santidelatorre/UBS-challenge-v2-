"""Predeclared sparse ensemble comparison using train-side evidence only."""
import json
import time
import numpy as np
import pandas as pd
from ubs_recurrence.data import ROOT,LABELS,aligned_target
from ubs_recurrence.evaluation import record,metrics
from v2_experiments import VIEWS,base_views,cohort_metrics


def load(name,ids=None):
    d=pd.read_csv(ROOT/f"outputs/experiments/{name}/predictions.csv").set_index("client_id")
    if ids is not None:d=d.loc[ids]
    return d.index.to_numpy(),d[["p_"+f for f in LABELS]].to_numpy()


def main():
    start=time.perf_counter();base=base_views();results={};controls={};seeds=[42,17,2026]
    for view in VIEWS:
        ids,reference=load("selection_compact_3seed_plus25pct_v1_"+view);y=aligned_target(ids)
        _,old=load("ensemble_equal3_"+view,ids)
        comp=np.mean([load(f"v2_control_{view}_s{s}",ids)[1] for s in seeds],axis=0)
        assert np.max(np.abs(reference-(.75*comp+.25*old)))<1e-10
        controls[view]={"macro_f1":metrics(y,reference)["macro_f1"],"cohorts":cohort_metrics(y,reference,base[view])}
        for variant in ["sparse_identity","sparse_joint"]:
            new=np.mean([load(f"v2_{variant}_{view}_s{s}",ids)[1] for s in seeds],axis=0)
            for weight in [.75,.375]:
                name=f"{variant}_w{weight:g}";p=weight*new+(.75-weight)*comp+.25*old
                cm=cohort_metrics(y,p,base[view])
                r=record(f"v2_selection_{name}_{view}",ids,y,p,runtime=time.perf_counter()-start,metadata=dict(hypothesis="One fixed V1 blend limits displacement of strong streams while preserving sparse rescue",baseline_comparison="Full frozen V1 OOF",features=variant,model="three-seed sparse plus original compact and fixed 25% legacy",parameters={"sparse_weight":weight,"compact_weight":.75-weight,"legacy_weight":.25},seed=seeds,protocol="v2_protocol.md predeclared direct/50% blend; all seeds retained; no holdout access",status="selection_comparison",conclusion="Apply full train-side gates before fresh stress",cohorts=cm,candidate_generation="V1 plus gated single/pair evidence",stream_identity="soft family confidence",continuation_model="baseline none" if variant.endswith("identity") else "sparse-aware none",candidate_recall="Proxy envelope in v2_diagnostics.json"))
                results.setdefault(name,{"variant":variant,"sparse_weight":weight,"views":{}})["views"][view]={"macro_f1":r["macro_f1"],"cohorts":cm}
    for name,r in results.items():
        gain=np.mean([r["views"][v]["macro_f1"]-controls[v]["macro_f1"] for v in VIEWS[1:]])
        sparse=np.mean([r["views"][v]["cohorts"]["sparse"]["accuracy"]-controls[v]["cohorts"]["sparse"]["accuracy"] for v in VIEWS[1:]])
        strong={v:r["views"][v]["cohorts"]["strong"]["accuracy"]-controls[v]["cohorts"]["strong"]["accuracy"] for v in VIEWS}
        clean=r["views"]["original"]["macro_f1"]-controls["original"]["macro_f1"]
        seedg=[]
        for s in seeds:
            rr=json.loads((ROOT/f"reports/v2_sparse_s{s}.json").read_text())
            seedg.append(rr[r["variant"]]["screen_gate"]["mean_stress_gain"])
        r["gate"]={"mean_stress_gain":float(gain),"sparse_accuracy_gain":float(sparse),"strong_accuracy_deltas":strong,"original_gain":clean,"matched_seed_stress_gains":seedg,
                   "passed":bool(gain>=.005 and sparse>=.030 and min(strong.values())>=-.010 and clean>=-.010 and sum(x>0 for x in seedg)>=2)}
    passing=[r for r in results.values() if r["gate"]["passed"]]
    selected=max(passing,key=lambda r:r["gate"]["mean_stress_gain"]) if passing else None
    out=dict(baseline=controls,candidates=results,selected=selected,decision="No official access until fresh stress passes; no fresh-view retuning")
    (ROOT/"reports/v2_selection.json").write_text(json.dumps(out,indent=2))
    print(json.dumps({k:r["gate"] for k,r in results.items()},indent=2));print("SELECTED",None if selected is None else (selected["variant"],selected["sparse_weight"]))


if __name__=="__main__":main()
