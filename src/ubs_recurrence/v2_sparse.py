"""Additive sparse evidence; V1 extraction and source remain frozen.

Only family rows without a V1 3+ amount candidate receive sparse features.
These are possible events/pairs, never asserted ground-truth subscriptions.
"""
from collections import Counter
import numpy as np
import pandas as pd

from .data import CUTOFF, LABELS
from .augmentation import BACKGROUND
from .features import normalize, PATTERNS, MCC, GENERIC
from .templates import canonical
from .price_prior import price_support


def sparse_features(df, base, profiles):
    d = df[(df.type == "card_payment") & (df.direction == "out")].copy()
    d["norm"] = d.description.map(normalize)
    d = d[~d.norm.str.contains(BACKGROUND, regex=True)].copy()
    d["age"] = (CUTOFF - d.timestamp).dt.total_seconds() / 86400
    d["canonical"] = d.description.map(canonical)
    semantic = np.stack([d.norm.str.contains(PATTERNS[f].replace("(", "(?:"), regex=True) for f in LABELS[:-1]], axis=1)
    d["pos"] = np.arange(len(d))
    a=d.amount.to_numpy(); age=d.age.to_numpy(); mcc=d.mcc.to_numpy()
    desc=d.norm.to_numpy(); canon=d.canonical.to_numpy(); fee=d.fee.to_numpy()
    dom=d.timestamp.dt.day.to_numpy(); dow=d.timestamp.dt.dayofweek.to_numpy()
    refunds={k:g for k,g in df[df.type=="refund"].groupby(["client_id","currency"])}
    candidates={}
    for (cid, currency),g in d.groupby(["client_id","currency"],sort=True):
        ix=g.pos.to_numpy(); order=ix[np.argsort(a[ix],kind="stable")]
        cuts=np.flatnonzero(np.diff(np.log(a[order]))>.035)+1
        for s in np.split(order,cuts):
            if len(s)>2:continue
            s=s[np.argsort(age[s])[::-1]]; n=len(s); med=float(np.median(a[s])); gap=float(np.ptp(age[s]))
            ownsem=semantic[s].mean(0); ownmcc=np.array([(mcc[s]==MCC[f]).mean() for f in LABELS[:-1]])
            prior=np.array([float(price_support(med,profiles[f])) for f in LABELS[:-1]])
            support=(3*ownsem+ownmcc+.05)*prior
            posterior=support/max(support.sum(),1e-12)
            r=refunds.get((cid,currency))
            if r is None: nr=0; ra=999.; after=0
            else:
                matched=r[np.abs(np.log(r.amount/med))<.04]
                nr=len(matched); ra=float((CUTOFF-matched.timestamp.max()).total_seconds()/86400) if nr else 999.
                after=int(nr>0 and 0<=age[s[-1]]-ra<=10)
            tok=[set(x.split()) for x in canon[s]]
            v=dict(count=n,amount=med,amount_cv=float(a[s].std()/a[s].mean()),
                   last_age=float(age[s].min()),first_age=float(age[s].max()),gap=gap,
                   cadence_error=float(np.min(np.abs(gap-np.array([7,14,28,30,60,90])))) if n==2 else 999.,
                   next_month=30-age[s].min(),next_pair=gap-age[s].min() if n==2 else 999.,
                   overdue_pair=age[s].min()/max(gap,1) if n==2 else 999.,
                   same_mcc=int(len(set(mcc[s]))==1),same_description=int(len(set(canon[s]))==1),
                   token_overlap=len(set.intersection(*tok))/max(len(set.union(*tok)),1),
                   dom_distance=float(np.ptp(dom[s])),same_weekday=int(len(set(dow[s]))==1),
                   generic_fraction=float(np.isin(desc[s],GENERIC).mean()),
                   fee_fraction=float((fee[s]>0).mean()),refund_count=nr,refund_last_age=ra,refund_after=after)
            candidates.setdefault(cid,[]).append((v,ownsem,ownmcc,prior,posterior))
    rows=[]
    for (cid,f),br in base.iterrows():
        if f=="none" or br.amount0_count>=3:
            rows.append({"sparse_eligible":0});continue
        fi=LABELS.index(f); eligible=[]
        for v,sem,mc,pr,post in candidates.get(cid,[]):
            # Soft evidence allows price-compatible masked events, but does not
            # turn every background payment into an asserted family stream.
            if pr[fi]<.02:continue
            z={**v,"semantic":sem[fi],"mcc":mc[fi],"price_support":pr[fi],
               "identity_soft":post[fi],"identity_margin":post[fi]-max(np.delete(post,fi)),
               "other_semantic":max(np.delete(sem,fi)),
               "price_position":(v["amount"]-profiles[f]["low"])/(profiles[f]["high"]-profiles[f]["low"])}
            strength=post[fi]*v["count"]*np.exp(-v["last_age"]/60)
            eligible.append((strength,z))
        eligible.sort(key=lambda z:-z[0])
        row={"sparse_eligible":1,"sparse_proposals":len(eligible),
             "sparse_singletons":sum(z[1]["count"]==1 for z in eligible),
             "sparse_pairs":sum(z[1]["count"]==2 for z in eligible),
             "sparse_identity_sum":sum(z[1]["identity_soft"] for z in eligible),
             "sparse_recent_support":sum(z[0] for z in eligible)}
        for rank,(_,z) in enumerate(eligible[:2]):
            row.update({f"sparse{rank}_{k}":v for k,v in z.items()})
        rows.append(row)
    return pd.DataFrame(rows,index=base.index).fillna(-999)
