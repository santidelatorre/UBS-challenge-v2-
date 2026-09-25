"""Fresh corruption and controlled missing-observation diagnostics on train only.

Models are the saved client-OOF fits; no model is fitted on a stress target.
Target labels select a proxy stream only in the explicit removal diagnostic.
"""
import v2_bootstrap
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMRanker
from xgboost import XGBRanker

from ubs_recurrence.data import ROOT,LABELS,transactions,aligned_target
from ubs_recurrence.augmentation import corrupt_transactions,BACKGROUND
from ubs_recurrence.features import normalize
from ubs_recurrence.streams import amount_components
from ubs_recurrence.model import build_features,none_features
from ubs_recurrence.v2_sparse import sparse_features
from ubs_recurrence.evaluation import metrics,record
from v2_experiments import base_views,extras,hier,cohort_metrics

CACHE=ROOT/"data/cache/v2_stress"
SEEDS=[42,17,2026]


def original_legacy():
    result={}
    for v in ["original","valid_like","test_like"]:
        hard="ranking_clocksFalse_idTrue_train.parquet" if v=="original" else f"ranking_train_noise{v}_s2026.parquet"
        soft="soft_ranking_original.parquet" if v=="original" else f"soft_ranking_train_noise{v}_s2026.parquet"
        result[v]={"hard":pd.read_parquet(ROOT/"data/cache"/hard),"soft":pd.read_parquet(ROOT/"data/cache"/soft)}
    return result


def removal_members(df,base,ids,y):
    d=df[(df.type=="card_payment")&(df.direction=="out")].copy()
    d=d[~d.description.map(normalize).str.contains(BACKGROUND,regex=True)]
    groups={}
    for (cid,cur),g in d.groupby(["client_id","currency"],sort=True):
        for comp in amount_components(g.amount.to_numpy(),.035,3):
            s=g.iloc[comp]
            if len(s)>=5:groups.setdefault(cid,[]).append(s)
    chosen={}
    for cid,yi in zip(ids,y):
        if yi==7:continue
        r=base.loc[(cid,LABELS[yi])]
        if r.amount0_count<5 or not 0<=r.amount0_gap_cv<=.35:continue
        possible=[g for g in groups.get(cid,[]) if len(g)==r.amount0_count and abs(g.amount.median()-r.amount0_amount_median)<1e-7]
        if len(possible)==1:chosen[cid]=possible[0].sort_values("timestamp").index.to_numpy()
    return chosen


def build():
    CACHE.mkdir(exist_ok=True);base=base_views();sp=extras("sparse",base);legacy=original_legacy()
    raw=transactions("train",use_cache=False);ids=base["original"].index.get_level_values(0).unique().to_numpy();y=aligned_target(ids)
    profiles=json.loads((ROOT/"data/cache/unlabeled_price_profiles.json").read_text())
    members=removal_members(raw,base["original"],ids,y)
    (CACHE/"removal_clients.json").write_text(json.dumps(sorted(members)))
    print("Removal diagnostic clients",len(members),flush=True)
    scenarios=["fresh_valid_like","fresh_test_like","keep4","keep3","keep2","keep1"]
    for name in scenarios:
        complete=CACHE/f"{name}_complete.json"
        if complete.exists():continue
        if name.startswith("fresh"):
            d=corrupt_transactions(raw,name.removeprefix("fresh_"),91711)
        else:
            keep=int(name[-1]);remove=np.concatenate([ix[:-keep] for ix in members.values()])
            d=raw.drop(index=remove);d=d[d.client_id.isin(members)].copy()
        x,_,old=build_features(d,profiles,return_legacy=True)
        extra=sparse_features(d,x,profiles)
        if name.startswith("keep"):
            for dest,part in [(base["original"].copy(),x)]:
                dest.loc[part.index,part.columns]=part;x=dest
            dest=sp["original"].copy();dest.loc[extra.index,extra.columns]=extra;extra=dest.fillna(-999)
            for k in old:
                dest=legacy["original"][k].copy();dest.loc[old[k].index,old[k].columns]=old[k];old[k]=dest.fillna(-999)
        x.to_parquet(CACHE/f"{name}_base.parquet");extra.to_parquet(CACHE/f"{name}_sparse.parquet")
        for k,m in old.items():m.to_parquet(CACHE/f"{name}_{k}.parquet")
        complete.write_text(json.dumps(dict(scenario=name,clients=len(ids),changed_clients=len(ids) if name.startswith("fresh") else len(members),
                                           target_usage="none in input generation" if name.startswith("fresh") else "train-side diagnostic only: select a recovered true-family stream, retain latest k events; not used for fitting",
                                           note="Hidden card observations only; other payments/refunds remain observed. Target stays fixed.")))
        print("Built stress",name,flush=True)


def predict():
    start=time.perf_counter();base=base_views();ids=base["original"].index.get_level_values(0).unique().to_numpy();y=aligned_target(ids)
    names=["fresh_valid_like","fresh_test_like","keep4","keep3","keep2","keep1"]
    matrices={v:pd.read_parquet(CACHE/f"{v}_base.parquet") for v in names}
    augmented={v:pd.concat([x,pd.read_parquet(CACHE/f"{v}_sparse.parquet")],axis=1) for v,x in matrices.items()}
    predictions={v:{k:[] for k in ["control","sparse_identity","sparse_joint"]} for v in names}
    for seed in SEEDS:
        p={v:{k:np.zeros((len(ids),8)) for k in predictions[v]} for v in names}
        for fold in range(5):
            saved=joblib.load(ROOT/f"outputs/v2_models/sparse_s{seed}/fold{fold}.joblib")
            va=pd.Index(ids).get_indexer(saved["valid_ids"]);assert (va>=0).all()
            assert not set(saved["train_ids"])&set(saved["valid_ids"])
            iv=(va[:,None]*8+np.arange(8)).ravel()
            for v in names:
                prob={}
                for k,frames in [("control",matrices),("sparse",augmented)]:
                    rank,none,cols,ncols=saved["models"][k];x=frames[v].reindex(columns=cols).fillna(-999)
                    q=softmax(rank.predict(x.iloc[iv]).reshape(-1,8),axis=1)
                    nx=none_features(x,ids).reindex(columns=ncols).fillna(-999)
                    pn=none.predict_proba(nx.iloc[va])[:,1];prob[k]=(q,pn)
                qb,pb=prob["control"];qs,ps=prob["sparse"]
                p[v]["control"][va]=hier(qb,pb);p[v]["sparse_identity"][va]=hier(qs,pb);p[v]["sparse_joint"][va]=hier(qs,ps)
        for v in names:
            for k in p[v]:predictions[v][k].append(p[v][k])
        print("Stress compact predictions seed",seed,flush=True)
    oldtrain=original_legacy();oldstress={v:{k:pd.read_parquet(CACHE/f"{v}_{k}.parquet") for k in ["hard","soft"]} for v in names}
    oldp={v:np.zeros((len(ids),8)) for v in names};oldoriginal=np.zeros((len(ids),8))
    target=(np.repeat(y,8)==np.tile(np.arange(8),len(ids))).astype(int)
    for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(ids,y)):
        it=(tr[:,None]*8+np.arange(8)).ravel();iv=(va[:,None]*8+np.arange(8)).ravel()
        for kind in ["hard","soft","xgb"]:
            source="soft" if kind=="xgb" else kind;cols=list(oldtrain["original"][source].columns)
            cache=ROOT/f"outputs/v2_models/legacy_fold{fold}_{kind}.joblib"
            if cache.exists():model=joblib.load(cache)
            else:
                x=pd.concat([oldtrain[v][source].reindex(columns=cols).iloc[it] for v in oldtrain],ignore_index=True).fillna(-999)
                if kind=="xgb":
                    model=XGBRanker(n_estimators=500,max_depth=4,learning_rate=.04,min_child_weight=10,subsample=.85,colsample_bytree=.9,reg_lambda=10,n_jobs=4,random_state=42,objective="rank:pairwise",tree_method="hist",device="cuda",lambdarank_pair_method="mean",lambdarank_num_pair_per_sample=4)
                    x=x.astype(np.float32)
                else:model=LGBMRanker(n_estimators=450,num_leaves=15,learning_rate=.035,min_child_samples=105,colsample_bytree=.9,reg_lambda=5,n_jobs=4,verbosity=-1,random_state=42,objective="lambdarank")
                model.fit(x,np.tile(target[it],3),group=np.full(len(tr)*3,8));joblib.dump(model,cache)
            oldoriginal[va]+=softmax(model.predict(oldtrain["original"][source].reindex(columns=cols).iloc[iv]).reshape(-1,8),axis=1)/3
            for v in names:oldp[v][va]+=softmax(model.predict(oldstress[v][source].reindex(columns=cols).iloc[iv].fillna(-999)).reshape(-1,8),axis=1)/3
        print("Legacy stress fold",fold,flush=True)
    known=pd.read_csv(ROOT/"outputs/experiments/ensemble_equal3_original/predictions.csv").set_index("client_id").loc[ids,["p_"+f for f in LABELS]].to_numpy()
    assert np.max(np.abs(oldoriginal-known))<1e-6,np.max(np.abs(oldoriginal-known))
    removal=set(json.loads((CACHE/"removal_clients.json").read_text()));mask=np.array([cid in removal for cid in ids])
    result={}
    selected=json.loads((ROOT/"reports/v2_selection.json").read_text())["selected"]
    assert selected is not None,"No candidate passed fixed-view selection"
    for v in names:
        compact={k:np.mean(x,axis=0) for k,x in predictions[v].items()}
        baseline=.75*compact["control"]+.25*oldp[v]
        variant=selected["variant"];weight=selected["sparse_weight"]
        candidate=weight*compact[variant]+(.75-weight)*compact["control"]+.25*oldp[v]
        result[v]={}
        for k,p in [("baseline",baseline),("candidate",candidate)]:
            cm=cohort_metrics(y,p,base["original"] if v.startswith("keep") else matrices[v])
            if v.startswith("keep"):
                cm["removal_cohort"]={"n":int(mask.sum()),"accuracy":float((y[mask]==p[mask].argmax(1)).mean()),"macro_f1":metrics(y[mask],p[mask])["macro_f1"]}
            r=record(f"v2_stress_{k}_{v}",ids,y,p,metadata=dict(hypothesis="Frozen train-side selected sparse component transfers to fresh corruption and controlled missing events",features="same frozen sparse features",model=k,parameters=selected,seed=SEEDS,protocol="Saved client-OOF models; evaluation-only fresh seed 91711/removal; no retuning",status="diagnostic",conclusion="Compare to fixed V1; no official labels",cohorts=cm,candidate_generation="V1 plus gated sparse evidence",stream_identity="soft family evidence",continuation_model="as selected",baseline_comparison="matched full V1",candidate_recall="tracked via evidence cohorts"),runtime=time.perf_counter()-start,split="oof_fresh_stress")
            result[v][k]={"macro_f1":r["macro_f1"],"cohorts":cm}
    result["fresh_mean_gain"]=float(np.mean([result[v]["candidate"]["macro_f1"]-result[v]["baseline"]["macro_f1"] for v in names[:2]]))
    result["no_material_reversal"]=bool(result["fresh_mean_gain"]>0 and all(result[v]["candidate"]["cohorts"]["removal_cohort"]["accuracy"]>=result[v]["baseline"]["cohorts"]["removal_cohort"]["accuracy"]-.01 for v in names[2:]))
    (ROOT/"reports/v2_fresh_stress.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("action",choices=["build","predict"]);args=ap.parse_args()
    build() if args.action=="build" else predict()
