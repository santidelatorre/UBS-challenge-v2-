"""Fit weak pair compatibility using disjoint unlabeled clients only."""
import v2_bootstrap
import json
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve
from ubs_recurrence.data import ROOT,transactions
from ubs_recurrence.augmentation import corrupt_transactions
from ubs_recurrence.evaluation import source_fingerprint
from ubs_recurrence.v2_identity import event_arrays,weak_pairs,pair_features


def main():
    start=time.perf_counter();out=ROOT/"outputs/v2_pair_learning";out.mkdir(exist_ok=False)
    raw=transactions("unlabeled_pretrain",use_cache=False)
    clean=raw[(raw.type=="card_payment")&(raw.direction=="out")].reset_index(drop=True)
    l,r,y,clients,anchors=weak_pairs(clean)
    ids=np.sort(clean.client_id.unique());rng=np.random.default_rng(6113);validids=set(rng.choice(ids,2000,replace=False))
    va=np.array([c in validids for c in clients]);tr=~va
    assert not set(clients[tr])&set(clients[va])
    matrices=[]
    for view in ["original","valid_like","test_like"]:
        noisy=clean if view=="original" else corrupt_transactions(clean,view,6113)
        # Data loader sorting and corruption sorting agree; assert event alignment.
        assert np.array_equal(noisy.client_id,clean.client_id) and np.array_equal(noisy.timestamp,clean.timestamp)
        z=event_arrays(noisy)
        if view!="original":
            # Controlled amount drift lets positive inputs leave their anchor radius.
            z["amount"]=z["amount"]*np.exp(np.random.default_rng(6113+(view=="test_like")).normal(0,.025,len(clean)))
        x=pair_features(z,l,r);matrices.append(x)
        print(view,x.shape,"positive",int(y.sum()),flush=True)
    params=dict(n_estimators=180,num_leaves=15,learning_rate=.04,min_child_samples=150,reg_lambda=10,n_jobs=4,verbosity=-1,random_state=6113)
    model=LGBMClassifier(**params);model.fit(pd.concat([x.loc[tr] for x in matrices],ignore_index=True),np.tile(y[tr],3))
    # A fresh corruption seed supplies threshold calibration only on auxiliary clients.
    noisy=corrupt_transactions(clean,"test_like",91711);z=event_arrays(noisy)
    z["amount"]*=np.exp(np.random.default_rng(91711).normal(0,.025,len(clean)))
    x=pair_features(z,l[va],r[va]);p=model.predict_proba(x)[:,1]
    prec,rec,th=precision_recall_curve(y[va],p)
    eligible=np.flatnonzero((prec[:-1]>=.95)&(th>=.5))
    threshold=float(th[eligible[np.argmax(rec[eligible])]]) if len(eligible) else .99
    pred=p>=threshold
    receipt=dict(auxiliary_only=True,challenge_labels_used=False,source_clients=len(ids),fit_clients=len(ids)-len(validids),calibration_clients=len(validids),
                 pairs=len(y),positive_pairs=int(y.sum()),negative_pairs=int((y==0).sum()),anchors=anchors,
                 heldout_fresh_corruption_auc=float(roc_auc_score(y[va],p)),heldout_fresh_corruption_ap=float(average_precision_score(y[va],p)),
                 threshold=threshold,threshold_rule="maximum weak-label recall subject to precision >= .95 and threshold >= .5, held-out auxiliary clients",
                 auxiliary_precision=float(y[va][pred].mean()) if pred.any() else 0.,auxiliary_recall=float(pred[y[va]==1].mean()),
                 parameters=params,seed=6113,runtime_seconds=time.perf_counter()-start,source_sha256=source_fingerprint(),
                 limitations="Weak stream IDs are biased to regular stable-price anchored streams. Auxiliary calibration is not independent evaluation after threshold selection. No true stream IDs exist; downstream challenge CV is decisive.")
    # Keep the 8,000-client model: do not train on calibration clients afterward.
    joblib.dump(dict(model=model,threshold=threshold,receipt=receipt),out/"model.joblib")
    (ROOT/"reports/v2_pair_learning.json").write_text(json.dumps(receipt,indent=2))
    pd.Series(model.feature_importances_,index=matrices[0].columns).sort_values(ascending=False).to_csv(out/"importance.csv")
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=="__main__":main()
