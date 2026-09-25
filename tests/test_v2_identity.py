import numpy as np
import pandas as pd
from ubs_recurrence.v2_identity import event_arrays,pair_features,weak_pairs,learned_groups


def sample():
    return pd.DataFrame([dict(client_id="a",timestamp=pd.Timestamp(t,tz="UTC"),amount=a,currency=c,
                              description="audio streaming",mcc="5812",type="card_payment",direction="out",fee=0.)
                         for t,a,c in [("2025-08-15",12.,"CHF"),("2025-09-15",12.1,"CHF"),
                                       ("2025-10-15",12.,"CHF"),("2025-11-15",12.1,"CHF"),
                                       ("2025-10-15",12.,"EUR"),("2025-11-15",12.1,"EUR")]])


def test_pair_symmetry_and_weak_label_isolation():
    d=sample();z=event_arrays(d);l=np.array([0,2,4]);r=np.array([1,3,5])
    pd.testing.assert_frame_equal(pair_features(z,l,r),pair_features(z,r,l))
    a=weak_pairs(d);b=weak_pairs(d.assign(target_next_recurring_merchant="none"))
    for x,y in zip(a[:4],b[:4]):np.testing.assert_array_equal(x,y)
    assert a[-1]["anchor_streams"]==1
    assert a[2].sum()==6


def test_proposals_do_not_join_currencies_or_clients():
    class Compatible:
        def predict_proba(self,x):
            return np.tile([.01,.99],(len(x),1))
    d=sample();d=pd.concat([d,d.assign(client_id="b")],ignore_index=True)
    groups=learned_groups(d,Compatible(),.9)
    assert len(groups)==4
    assert set(groups.currency)=={"CHF","EUR"}
    assert set(groups.client_id)=={"a","b"}
    assert sorted(groups["count"])==[2,2,4,4]
