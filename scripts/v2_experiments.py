"""Matched V1/V2 client CV. All official labels remain outside this runner."""
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker,LGBMClassifier
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold

from ubs_recurrence.data import ROOT,LABELS,transactions,aligned_target
from ubs_recurrence.augmentation import corrupt_transactions
from ubs_recurrence.compact import compact_features,global_features
from ubs_recurrence.model import parameters,none_features
from ubs_recurrence.evaluation import metrics,record,source_fingerprint
from ubs_recurrence.v2_sparse import sparse_features
from ubs_recurrence.v2_identity import learned_groups,identity_features

VIEWS=["original","valid_like","test_like"]
CACHE=ROOT/"data/cache/v2"


def base_views():
    CACHE.mkdir(exist_ok=True)
    df=transactions("train",use_cache=False);out={}
    for view in VIEWS:
        path=CACHE/f"baseline_{view}.parquet"
        if path.exists():out[view]=pd.read_parquet(path);continue
        d=df if view=="original" else corrupt_transactions(df,view,2026)
        rank=pd.read_parquet(ROOT/("data/cache/soft_ranking_original.parquet" if view=="original" else f"data/cache/soft_ranking_train_noise{view}_s2026.parquet"))
        ids=rank.index.get_level_values(0).unique()
        x=compact_features(rank,global_features(d,ids))
        ctx=pd.read_parquet(ROOT/f"data/cache/payment_context_{view}_train.parquet")
        out[view]=pd.concat([x,ctx.reindex(x.index)],axis=1).fillna(-999)
        out[view].to_parquet(path)
    return out


def extras(kind,base):
    df=transactions("train",use_cache=False)
    profiles=json.loads((ROOT/"data/cache/unlabeled_price_profiles.json").read_text())
    out={}
    for view in VIEWS:
        path=CACHE/f"{kind}_{view}.parquet"
        if path.exists():out[view]=pd.read_parquet(path);continue
        d=df if view=="original" else corrupt_transactions(df,view,2026)
        if kind=="sparse":x=sparse_features(d,base[view],profiles)
        elif kind=="gate_only":
            x=pd.DataFrame({"sparse_eligible":((base[view].amount0_count<3)&(base[view].index.get_level_values("family")!="none")).astype(int)},index=base[view].index)
        elif kind=="pair":
            auxiliary=joblib.load(ROOT/"outputs/v2_pair_learning/model.joblib")
            streams=learned_groups(d,auxiliary["model"],auxiliary["threshold"])
            streams.to_parquet(CACHE/f"pair_streams_{view}.parquet")
            x=identity_features(streams,base[view],profiles)
        else:raise ValueError(kind)
        x.to_parquet(path);out[view]=x
        print("Built",kind,view,x.shape,flush=True)
    return out


def cohort_metrics(y,p,base):
    pred=p.argmax(1);cnt=base.amount0_count.to_numpy().reshape(-1,8)
    strong=cnt[np.arange(len(y)),y]>=3
    masks={"strong":(y<7)&strong,"sparse":(y<7)&~strong,"none":y==7,
           "music_software_streaming":np.isin(y,[4,5,6])}
    out={k:{"n":int(m.sum()),"accuracy":float((pred[m]==y[m]).mean()),
            "macro_f1":metrics(y[m],p[m])["macro_f1"]} for k,m in masks.items()}
    out["wrong_family_errors"]=int(((y<7)&(pred<7)&(pred!=y)).sum())
    out["family_to_none_errors"]=int(((y<7)&(pred==7)).sum())
    out["none_to_family_errors"]=int(((y==7)&(pred<7)).sum())
    return out


def hier(q,pn):
    q=q.copy();q[:,:7]=q[:,:7]/np.maximum(q[:,:7].sum(1,keepdims=True),1e-12)*(1-pn[:,None]);q[:,7]=pn
    return q


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--kind",choices=["sparse","pair","gate_only"],default="sparse");ap.add_argument("--seed",type=int,default=42);args=ap.parse_args()
    start=time.perf_counter();base=base_views();extra=extras(args.kind,base)
    ids=base["original"].index.get_level_values(0).unique().to_numpy();y=aligned_target(ids)
    variants={"control":base,args.kind:{v:pd.concat([base[v],extra[v]],axis=1).fillna(-999) for v in VIEWS}}
    preds={k:{v:np.zeros((len(ids),8)) for v in VIEWS} for k in ["control",args.kind+"_identity",args.kind+"_joint"]}
    target=(np.repeat(y,8)==np.tile(np.arange(8),len(y))).astype(int)
    aggregate={k:{v:none_features(x,ids) for v,x in views.items()} for k,views in variants.items()}
    artifact=ROOT/f"outputs/v2_models/{args.kind}_s{args.seed}";artifact.mkdir(parents=True,exist_ok=False)
    for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=args.seed).split(ids,y)):
        it=(tr[:,None]*8+np.arange(8)).ravel();iv=(va[:,None]*8+np.arange(8)).ravel();fit={};pv={}
        for kind,views in variants.items():
            cols=list(views["original"].columns)
            X=pd.concat([views[v].reindex(columns=cols).iloc[it] for v in VIEWS],ignore_index=True).fillna(-999)
            rank=LGBMRanker(**parameters(args.seed),objective="lambdarank");rank.fit(X,np.tile(target[it],3),group=np.full(len(tr)*3,8))
            nx=pd.concat([aggregate[kind][v].iloc[tr] for v in VIEWS],ignore_index=True)
            none=LGBMClassifier(**parameters(args.seed));none.fit(nx,np.tile(y[tr]==7,3))
            fit[kind]=(rank,none,cols,list(nx.columns));pv[kind]={}
            for v in VIEWS:
                q=softmax(rank.predict(views[v].reindex(columns=cols).iloc[iv]).reshape(-1,8),axis=1)
                pn=none.predict_proba(aggregate[kind][v].iloc[va])[:,1]
                pv[kind][v]=(q,pn)
        for v in VIEWS:
            qb,pb=pv["control"][v];qe,pe=pv[args.kind][v]
            preds["control"][v][va]=hier(qb,pb)
            preds[args.kind+"_identity"][v][va]=hier(qe,pb)
            preds[args.kind+"_joint"][v][va]=hier(qe,pe)
        joblib.dump(dict(models=fit,train_ids=ids[tr],valid_ids=ids[va],seed=args.seed,fold=fold),artifact/f"fold{fold}.joblib")
        print("fold",fold,{k:{v:round(metrics(y[va],p[va])["macro_f1"],4) for v,p in vp.items()} for k,vp in preds.items()},flush=True)
    result={}
    for kind,views in preds.items():
        result[kind]={}
        for v,p in views.items():
            if kind=="control":
                old=pd.read_csv(ROOT/f"outputs/experiments/compact_l15_hierTrue_payments_{v}_s{args.seed}/predictions.csv").set_index("client_id").loc[ids,["p_"+f for f in LABELS]].to_numpy()
                diff=float(np.max(np.abs(old-p)));assert diff<1e-10,(v,diff)
                print("Matched control reproduced",v,"maxdiff",diff,flush=True)
            cm=cohort_metrics(y,p,base[v])
            hypothesis="Complement strong streams with family-conditioned single/pair evidence only when 3+ amount evidence is absent" if args.kind=="sparse" else "Learned pair compatibility from independent weak anchors repairs stream membership beyond a fixed amount radius"
            if args.kind=="gate_only":hypothesis="Ablation: distinguish new transaction evidence from merely exposing the redundant missing-candidate indicator"
            meta=dict(hypothesis=hypothesis,baseline_comparison=f"V1 compact seed {args.seed}, exactly reproduced",
                      features=kind,model="unchanged compact LightGBM ranker and none model",parameters=parameters(args.seed),seed=args.seed,
                      candidate_generation="V1 preserved; additional 1/2-event amount components with family-relative semantic/MCC/price/cadence/refund evidence" if args.kind=="sparse" else "V1 plus weak pair graph proposals; calibrated edges, constrained maximum-spanning unions; soft family features",
                      stream_identity="soft semantic/MCC/unlabeled price; no hard assignment",continuation_model="V1 none" if "identity" in kind or kind=="control" else "V1 none architecture with added sparse evidence",
                      protocol="five-fold client CV, all three corruption views grouped; no V2 official access",cohorts=cm,
                      candidate_recall="See v2_diagnostics.json; extra sparse features do not delete V1 candidates",status="screened",conclusion="Apply predeclared v2_protocol.md gate; scores are exploratory")
            if args.kind=="gate_only":
                meta.update(candidate_generation="V1 unchanged; no new transaction evidence",stream_identity="V1 unchanged",continuation_model="V1 with redundant indicator only",status="mechanism_ablation")
            eid=f"v2_{kind}_{v}_s{args.seed}" if args.kind=="sparse" or kind!="control" else f"v2_{args.kind}_control_{v}_s{args.seed}"
            r=record(eid,ids,y,p,metadata=meta,runtime=time.perf_counter()-start)
            # V1's generated table has a disposition column; append in that format.
            ledger=ROOT/"reports/experiment_log.md"
            lines=ledger.read_text(encoding="utf-8").splitlines()
            lines[-1]=lines[-1].rsplit(" | ",1)[0]+" | screened | "+hypothesis+" |"
            ledger.write_text("\n".join(lines)+"\n",encoding="utf-8")
            result[kind][v]={"macro_f1":r["macro_f1"],"cohorts":cm}
    for kind in [args.kind+"_identity",args.kind+"_joint"]:
        gain=np.mean([result[kind][v]["macro_f1"]-result["control"][v]["macro_f1"] for v in VIEWS[1:]])
        sparse=np.mean([result[kind][v]["cohorts"]["sparse"]["accuracy"]-result["control"][v]["cohorts"]["sparse"]["accuracy"] for v in VIEWS[1:]])
        original=result[kind]["original"]["macro_f1"]-result["control"]["original"]["macro_f1"]
        fewer_wrong=np.mean([result["control"][v]["cohorts"]["wrong_family_errors"]-result[kind][v]["cohorts"]["wrong_family_errors"] for v in VIEWS[1:]])
        mechanism=(sparse>=.020) if args.kind=="sparse" else (fewer_wrong>0)
        result[kind]["screen_gate"]={"mean_stress_gain":float(gain),"sparse_accuracy_gain":float(sparse),"fewer_wrong_family":float(fewer_wrong),"original_gain":original,"passed":bool(gain>=.003 and mechanism and original>=-.010)}
    (ROOT/f"reports/v2_{args.kind}_s{args.seed}.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v.get("screen_gate") for k,v in result.items()},indent=2),flush=True)


if __name__=="__main__":main()
