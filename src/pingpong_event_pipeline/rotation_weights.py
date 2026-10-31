"""Cluster-balanced correspondence mass and exactly weighted robust losses."""
from collections import Counter
import numpy as np


def correspondence_weights(pairs, mode='uniform'):
    if mode not in {'uniform','cluster_equal'}:raise ValueError('unknown correspondence weighting mode')
    if mode=='uniform':return np.ones(len(pairs),float)
    keys=[str(a.get('track_id',segment)) for segment,a,b,dt in pairs]
    counts=Counter(keys)
    weights=np.array([1/counts[k] for k in keys],float)
    return weights/weights.mean() if len(weights) else weights


def weighted_soft_l1(weights):
    """Return SciPy loss derivatives for sum(w*rho(residual^2))."""
    w=np.asarray(weights,float).reshape(-1)
    def loss(z):
        root=np.sqrt(1+z)
        return np.vstack([2*w*(root-1),w/root,-.5*w/(root**3)])
    return loss


def weighted_median(values,weights):
    values=np.asarray(values,float);weights=np.asarray(weights,float)
    order=np.argsort(values,kind='stable')
    i=np.searchsorted(np.cumsum(weights[order]),weights.sum()*.5,side='left')
    return float(values[order[min(i,len(order)-1)]])
