import numpy as np
import pandas as pd
from ubs_recurrence.data import LABELS
from ubs_recurrence.v2_sparse import sparse_features


def test_sparse_gating_identity_and_currency_separation():
    ids=["a","b"]
    base=pd.DataFrame({"amount0_count":-999},index=pd.MultiIndex.from_product([ids,LABELS],names=["client_id","family"]))
    base.loc[("a","music"),"amount0_count"]=4
    rows=[]
    for cid in ids:
        for date,curr in [("2025-11-15","CHF"),("2025-12-15","EUR")]:
            rows.append(dict(client_id=cid,timestamp=pd.Timestamp(date,tz="UTC"),amount=12.,currency=curr,
                             type="card_payment",direction="out",mcc="5812",description="audio streaming",fee=0.))
    raw=pd.DataFrame(rows)
    profiles={f:{"low":5.,"high":30.} for f in LABELS[:-1]}
    x=sparse_features(raw,base,profiles)
    assert x.index.equals(base.index)
    assert x.loc[("a","music"),"sparse_eligible"]==0
    assert (x.xs("none",level="family").sparse_eligible==0).all()
    assert x.loc[("b","music"),"sparse_pairs"]==0
    assert x.loc[("b","music"),"sparse_singletons"]==2
    assert x.loc[("b","music"),"sparse0_semantic"]==1
    # Labels, row order and identifier spelling cannot supply predictive signal.
    shuffled=raw.sample(frac=1,random_state=71).assign(target_next_recurring_merchant="gym")
    pd.testing.assert_frame_equal(x,sparse_features(shuffled,base,profiles))
    renamed=raw.assign(client_id=raw.client_id.map({"a":"x","b":"z"}))
    renamed_base=base.rename(index={"a":"x","b":"z"},level="client_id")
    z=sparse_features(renamed,renamed_base,profiles)
    np.testing.assert_equal(x.to_numpy(),z.to_numpy())
