"""Frozen V2 composition; does not change the reference FamilyForecaster."""
from dataclasses import dataclass
import numpy as np
from scipy.special import softmax
from .model import FamilyForecaster,none_features


@dataclass
class SparseForecaster:
    sparse_weight: float=.375
    variant: str="sparse_joint"
    device: str="cuda"

    def fit(self,base_views,sparse_views,y,legacy_views):
        self.baseline_=FamilyForecaster(device=self.device).fit(base_views,y,legacy_views)
        self.sparse_=FamilyForecaster(legacy_weight=0,device=self.device).fit(sparse_views,y)
        return self

    def predict_proba(self,base,sparse,legacy):
        ids=base.index.get_level_values(0).unique().to_numpy()
        qb=self.baseline_.predict_proba(base,legacy)
        qs=self.sparse_.predict_proba(sparse)
        # Recover the compact contribution without fitting or altering V1.
        core=[]
        x=base.reindex(columns=self.baseline_.columns_).fillna(-999)
        nx=none_features(x,ids).reindex(columns=self.baseline_.none_columns_).fillna(-999)
        for rank,none in self.baseline_.models_:
            q=softmax(rank.predict(x).reshape(-1,8),axis=1);pn=none.predict_proba(nx)[:,1]
            q[:,:7]=q[:,:7]/q[:,:7].sum(1,keepdims=True)*(1-pn[:,None]);q[:,7]=pn;core.append(q)
        compact=np.mean(core,axis=0)
        if self.variant!="sparse_joint":
            raise ValueError("Only the selected joint variant is frozen for deployment")
        p=qb+self.sparse_weight*(qs-compact)
        assert np.isfinite(p).all() and (p>=-1e-12).all() and np.allclose(p.sum(1),1)
        return p
