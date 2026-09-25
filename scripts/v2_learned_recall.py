"""Comparable soft eligibility recall for learned pair proposals, train only."""
import v2_bootstrap
import json
import numpy as np
import pandas as pd
from ubs_recurrence.data import ROOT,LABELS,aligned_target
from ubs_recurrence.price_prior import price_support


def main():
    profile=json.loads((ROOT/"data/cache/unlabeled_price_profiles.json").read_text());rows=[]
    for view in ["original","valid_like","test_like"]:
        base=pd.read_parquet(ROOT/f"data/cache/v2/baseline_{view}.parquet");ids=base.index.get_level_values(0).unique().to_numpy();y=aligned_target(ids)
        s=pd.read_parquet(ROOT/f"data/cache/v2/pair_streams_{view}.parquet")
        score=np.stack([(3*s['semantic_'+f].to_numpy()+s['mcc_'+f].to_numpy()+.05)*price_support(s.amount_median,profile[f]) for f in LABELS[:-1]],axis=1)
        for fi,f in enumerate(LABELS[:-1]):
            for n in [2,3]:
                good=(score[:,fi]>=.4*score.max(1))&(score[:,fi]>.02)&(s['count'].to_numpy()>=n)
                available=set(s.loc[good,'client_id']);m=y==fi
                covered=np.array([cid in available for cid in ids])
                original=base.amount0_count.to_numpy().reshape(-1,8)[:,fi]>=3
                rows.append(dict(view=view,family=f,min_count=n,n=int(m.sum()),covered=int((covered&m).sum()),recall=float(covered[m].mean()),combined_v1_recall=float((covered|original)[m].mean())))
    pd.DataFrame(rows).to_csv(ROOT/"reports/v2_learned_candidate_recall.csv",index=False)
    print(pd.DataFrame(rows).groupby(['view','min_count'])[['n','covered']].sum().to_string())


if __name__=="__main__":main()
