"""Conservative weak stream membership and learned pair proposals.

Auxiliary labels are structural hypotheses from independent histories. They
are never challenge labels. Corrupted pair inputs need not satisfy the rules
that produced clean anchors, which tests recovery beyond the anchor rules.
"""
from functools import lru_cache
import numpy as np
import pandas as pd

from .data import LABELS,CUTOFF
from .features import normalize,PATTERNS,MCC,GENERIC
from .templates import canonical
from .augmentation import BACKGROUND
from .streams import amount_components,fast_stats
from .price_prior import price_support


def event_arrays(df):
    norm=df.description.map(normalize).to_numpy()
    canon=np.array([canonical(s) for s in norm])
    semantic=np.stack([pd.Series(norm).str.contains(PATTERNS[f].replace("(","(?:"),regex=True) for f in LABELS[:-1]],axis=1).astype(float)
    return dict(amount=df.amount.to_numpy(),days=(df.timestamp-CUTOFF).dt.total_seconds().to_numpy()/86400,
                desc=norm,canon=canon,mcc=df.mcc.to_numpy(),currency=df.currency.to_numpy(),semantic=semantic,
                dom=df.timestamp.dt.day.to_numpy(),dow=df.timestamp.dt.dayofweek.to_numpy(),fee=df.fee.to_numpy(),
                generic=np.isin(canon,GENERIC),background=pd.Series(norm).str.contains(BACKGROUND,regex=True).to_numpy())


@lru_cache(maxsize=200000)
def text_sim(a,b):
    ta=set(a.split());tb=set(b.split())
    ca={a[i:i+3] for i in range(max(len(a)-2,1))};cb={b[i:i+3] for i in range(max(len(b)-2,1))}
    return len(ta&tb)/max(len(ta|tb),1),len(ca&cb)/max(len(ca|cb),1)


def pair_features(z,left,right):
    a=z["amount"];t=z["days"];gap=np.abs(t[left]-t[right]);lr=np.abs(np.log(a[left]/a[right]))
    sl=z["semantic"][left];sr=z["semantic"][right]
    sims=np.array([text_sim(*sorted((x,y))) for x,y in zip(z["canon"][left],z["canon"][right])]).reshape(-1,2)
    x={"log_amount_difference":lr,"relative_amount_difference":1-np.exp(-lr),
       "log_amount_min":np.log(np.minimum(a[left],a[right])),"log_amount_max":np.log(np.maximum(a[left],a[right])),
       "gap":gap,"same_mcc":(z["mcc"][left]==z["mcc"][right]).astype(float),
       "same_currency":(z["currency"][left]==z["currency"][right]).astype(float),
       "token_jaccard":sims[:,0],"char3_jaccard":sims[:,1],
       "same_description":(z["canon"][left]==z["canon"][right]).astype(float),
       "semantic_overlap":(sl*sr).sum(1),"semantic_conflict":((sl.sum(1)>0)&(sr.sum(1)>0)&((sl*sr).sum(1)==0)).astype(float),
       "semantic_known_min":np.minimum(sl.sum(1),sr.sum(1)),"semantic_known_max":np.maximum(sl.sum(1),sr.sum(1)),
       "generic_count":z["generic"][left].astype(float)+z["generic"][right],
       "background_count":z["background"][left].astype(float)+z["background"][right],
       "day_of_month_difference":np.abs(z["dom"][left]-z["dom"][right]),
       "same_weekday":(z["dow"][left]==z["dow"][right]).astype(float),
       "fee_difference":np.abs(z["fee"][left]-z["fee"][right]),
       "recent_age":-np.maximum(t[left],t[right])}
    for period in [7,14,28,30.4375,60,90]:
        r=np.mod(gap,period);x[f"gap_phase_{period}"]=np.minimum(r,period-r)
    for f in LABELS[:-1]:
        fi=LABELS.index(f)
        x["semantic_"+f]=sl[:,fi]+sr[:,fi]
        x["mcc_"+f]=(z["mcc"][left]==MCC[f]).astype(float)+(z["mcc"][right]==MCC[f])
    return pd.DataFrame(x,dtype=np.float32)


def weak_anchors(clean):
    """High-precision proxy IDs; rejects ambiguous/irregular amount collisions."""
    z=event_arrays(clean);sid=np.full(len(clean),-1);family=np.full(len(clean),-1);next_id=0
    for _,g in clean.groupby(["client_id","currency"],sort=True):
        ix=g.index.to_numpy()
        for component in amount_components(z["amount"][ix],.035,3):
            s=ix[component];s=s[np.argsort(z["days"][s])]
            gap=np.diff(z["days"][s]);sem=z["semantic"][s].mean(0);f=int(sem.argmax())
            if len(s)<4 or np.ptp(z["days"][s])<60 or not 7<=np.median(gap)<=95:continue
            if np.std(gap)/max(np.mean(gap),1)>.35 or sem[f]<.45 or max(np.delete(sem,f))>.10:continue
            if (z["mcc"][s]==MCC[LABELS[f]]).mean()<.6 or z["background"][s].any():continue
            sid[s]=next_id;family[s]=f;next_id+=1
    return sid,family,z


def weak_pairs(clean,seed=6113,max_each=30):
    sid,fam,z=weak_anchors(clean);rng=np.random.default_rng(seed);ls=[];rs=[];ys=[];cs=[]
    # Clear event semantics supply hard negatives even when amount groups collide.
    sem=z["semantic"];clear=(sem.sum(1)==1);ef=sem.argmax(1)
    for cid,g in clean.groupby("client_id",sort=True):
        ix=g.index.to_numpy();i,j=np.triu_indices(len(ix),1);l=ix[i];r=ix[j]
        gap=np.abs(z["days"][l]-z["days"][r]);ratio=np.abs(np.log(z["amount"][l]/z["amount"][r]))
        possible=(z["currency"][l]==z["currency"][r])&(gap>=3)&(gap<=180)&(ratio<.6)
        pos=possible&(sid[l]>=0)&(sid[l]==sid[r])
        neg=possible&(((fam[l]>=0)&(fam[r]>=0)&(fam[l]!=fam[r])) |
                      (clear[l]&clear[r]&(ef[l]!=ef[r])) |
                      ((sid[l]>=0)&z["background"][r]) | ((sid[r]>=0)&z["background"][l]))
        # Reject any contradictory label; noisy semantic outliers in a conservative
        # positive cannot simultaneously become a weak negative.
        neg &= ~pos
        for mask,label in [(pos,1),(neg,0)]:
            chosen=np.flatnonzero(mask)
            if len(chosen)>max_each:chosen=rng.choice(chosen,max_each,replace=False)
            ls.extend(l[chosen]);rs.extend(r[chosen]);ys.extend([label]*len(chosen));cs.extend([cid]*len(chosen))
    return np.array(ls),np.array(rs),np.array(ys),np.array(cs),dict(anchor_streams=int(len(set(sid)-{-1})),anchor_events=int((sid>=0).sum()))


def learned_groups(df,model,threshold):
    card=df[(df.type=="card_payment")&(df.direction=="out")].sort_values(["client_id","timestamp"],kind="stable").reset_index(drop=True)
    z=event_arrays(card);ls=[];rs=[]
    for _,g in card.groupby(["client_id","currency"],sort=True):
        ix=g.index.to_numpy();i,j=np.triu_indices(len(ix),1);l=ix[i];r=ix[j]
        gap=np.abs(z["days"][l]-z["days"][r]);delta=np.abs(np.log(z["amount"][l]/z["amount"][r]))
        valid=(gap>=3)&(gap<=180)&(delta<.6)
        ls.extend(l[valid]);rs.extend(r[valid])
    l=np.array(ls,dtype=int);r=np.array(rs,dtype=int)
    p=model.predict_proba(pair_features(z,l,r))[:,1]
    parent=np.arange(len(card));size=np.ones(len(card),dtype=int)
    amin=z["amount"].copy();amax=amin.copy()
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return a
    # Maximum spanning union with an amount diameter constraint limits chaining.
    for k in np.argsort(-p,kind="stable"):
        if p[k]<threshold:break
        a=find(l[k]);b=find(r[k])
        if a==b:continue
        if np.log(max(amax[a],amax[b])/min(amin[a],amin[b]))>.25:continue
        if size[a]<size[b]:a,b=b,a
        parent[b]=a;size[a]+=size[b];amin[a]=min(amin[a],amin[b]);amax[a]=max(amax[a],amax[b])
    groups={}
    for i in range(len(card)):groups.setdefault(find(i),[]).append(i)
    out=[]
    for ix in groups.values():
        if len(ix)<2:continue
        s=np.array(ix);stats=fast_stats(z["amount"][s],z["days"][s],z["desc"][s],z["mcc"][s],z["semantic"][s],z["dom"][s],z["dow"][s],z["currency"][s])
        stats.update(client_id=card.iloc[s[0]].client_id,currency=z["currency"][s[0]])
        out.append(stats)
    return pd.DataFrame(out)


def identity_features(streams,base,profiles):
    grouped={cid:g for cid,g in streams.groupby("client_id")};rows=[]
    stats=["count","amount_median","amount_cv","amount_mad","last_age","first_age","span","gap_median","gap_cv","gap_last","gap_recent","next_median","overdue_ratio","linear_period","linear_residual","generic_fraction","description_top_fraction"]
    for (cid,f),_ in base.iterrows():
        row={}
        if f=="none" or cid not in grouped:rows.append(row);continue
        g=grouped[cid].copy()
        prior=np.array([price_support(g.amount_median.to_numpy(),profiles[o]) for o in LABELS[:-1]]).T
        sem=g[["semantic_"+o for o in LABELS[:-1]]].to_numpy()
        mcc=g[["mcc_"+o for o in LABELS[:-1]]].to_numpy()
        support=(3*sem+mcc+.05)*prior;post=support/np.maximum(support.sum(1,keepdims=True),1e-12)
        fi=LABELS.index(f);weight=post[:,fi]*g["count"].to_numpy()*np.exp(-g.last_age.to_numpy()/60)
        keep=prior[:,fi]>.02
        order=np.argsort(-np.where(keep,weight,-1),kind="stable")
        row["learned_proposals"]=int(keep.sum());row["learned_support_sum"]=float(weight[keep].sum())
        for rank,k in enumerate([k for k in order if keep[k]][:2]):
            s=g.iloc[k];prefix=f"learned{rank}_"
            row.update({prefix+c:float(s[c]) for c in stats})
            row.update({prefix+"semantic":sem[k,fi],prefix+"mcc":mcc[k,fi],prefix+"price_support":prior[k,fi],prefix+"identity_soft":post[k,fi],prefix+"other_semantic":max(np.delete(sem[k],fi))})
        rows.append(row)
    return pd.DataFrame(rows,index=base.index).fillna(-999)
