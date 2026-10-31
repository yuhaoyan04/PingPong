"""Texture recurrence diagnostics independent of YOLO IDs and SO(3) radius.

Existing crops still depend on ball-centre/scale normalization. Appearance
periods can be harmonics or multiple turns; none is accepted as physical spin.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def texture_feature(image, crop_scale=1.2, side=48, interior_fraction=.78):
    smooth = cv2.GaussianBlur(image, (0,0), 2)
    small = cv2.resize(smooth, (side,side), interpolation=cv2.INTER_AREA).astype(float)
    y,x = np.mgrid[:side,:side]; centre=(side-1)/2
    mask = (x-centre)**2+(y-centre)**2 < (side/2/crop_scale*interior_fraction)**2
    v=small[mask]; v-=v.mean()
    return v/max(1e-9, np.linalg.norm(v))


def recurrence(features, timestamps_us, min_pairs=16, max_lag_frames=160, min_lag_frames=6):
    times=np.asarray(timestamps_us,np.int64); f=np.asarray(features,float)
    if len(times)!=len(f) or len(times)<2 or np.any(np.diff(times)<=0):
        raise ValueError('need matching features and increasing timestamps')
    steps=np.diff(times)
    if not np.all(steps==steps[0]):
        raise ValueError('recurrence requires uniform sampling; do not bridge missing frames')
    # Normalize here too so synthetic/alternate features have the same meaning.
    norms=np.linalg.norm(f,axis=1,keepdims=True)
    f=f/np.maximum(norms,1e-9); matrix=np.clip(f@f.T,-1,1)
    lags=np.arange(min_lag_frames,min(len(times)-min_pairs,max_lag_frames)+1)
    scores=np.array([float(np.diag(matrix,k).mean()) for k in lags])
    smooth=gaussian_filter1d(scores,1) if len(scores) else scores
    indices,properties=find_peaks(smooth,prominence=.04,distance=8)
    peaks=[]
    for i,prominence in zip(indices,properties['prominences']):
        lag=int(lags[i]); delay=lag*int(steps[0])
        peaks.append({'lag_frames':lag,'delay_us':delay,'equivalent_1_over_delay_rps':1e6/delay,
                      'mean_correlation':float(smooth[i]),'prominence':float(prominence),'pairs':len(times)-lag})
    return {'step_us':int(steps[0]),'lags':lags,'scores':scores,'smoothed':smooth,'matrix':matrix,'peaks':peaks}


def common_peaks(left, right):
    """Retain all plausible repeat branches; do not select one using fitted RPS."""
    matches=[]
    for a in left:
        for b in right:
            period=(a['delay_us']+b['delay_us'])/2
            if min(a['mean_correlation'],b['mean_correlation'])>=.2 and abs(a['delay_us']-b['delay_us'])<=max(1000,.08*period):
                matches.append({'F3E_delay_us':a['delay_us'],'F59_delay_us':b['delay_us'],
                                'mean_repeat_delay_us':period,'equivalent_1_over_delay_rps':1e6/period,
                                'minimum_view_correlation':min(a['mean_correlation'],b['mean_correlation']),
                                'physical_spin_accepted':False})
    return sorted(matches,key=lambda r:r['mean_repeat_delay_us'])
