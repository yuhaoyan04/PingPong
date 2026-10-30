"""Estimate full-rotation period from recurrence of ball-centred event texture."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from scipy.signal import find_peaks
from estimate_event_plane_spin import normalized_events

def frame(x,y,p,bins,radius):
 out=np.zeros((2,bins,bins),np.float32)
 ix=np.floor((x+radius)*bins/(2*radius)).astype(int);iy=np.floor((y+radius)*bins/(2*radius)).astype(int)
 ok=(ix>=0)&(ix<bins)&(iy>=0)&(iy<bins)
 np.add.at(out,(p[ok].astype(int),iy[ok],ix[ok]),1)
 for k in range(2):
  out[k]=cv2.GaussianBlur(out[k],(0,0),1.0)
  out[k]-=cv2.GaussianBlur(out[k],(0,0),4.0)
 v=out.ravel();v-=v.mean();norm=np.linalg.norm(v)
 return v/norm if norm>1e-8 else np.zeros_like(v)

def main():
 p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--path',required=True);p.add_argument('--output',required=True)
 p.add_argument('--inner-radius',type=float,default=.62);p.add_argument('--step-us',type=int,default=500)
 p.add_argument('--history-us',type=int,default=1200);p.add_argument('--bins',type=int,default=48)
 p.add_argument('--min-rps',type=float,default=8);p.add_argument('--max-rps',type=float,default=80);p.add_argument('--minimum-events',type=int,default=120);a=p.parse_args()
 x,y,t,pol=normalized_events(a.raw,a.path,a.inner_radius);times=np.arange(int(t.min())+a.history_us,int(t.max()),a.step_us,dtype=np.int64)
 frames=[];counts=[]
 for now in times:
  take=(t>now-a.history_us)&(t<=now);counts.append(int(take.sum()));frames.append(frame(x[take],y[take],pol[take],a.bins,a.inner_radius))
 F=np.asarray(frames);counts=np.asarray(counts);minlag=max(1,int(round(1e6/a.max_rps/a.step_us)));maxlag=min(len(F)-2,int(round(1e6/a.min_rps/a.step_us)))
 curve=[];support=[]
 for lag in range(minlag,maxlag+1):
  ok=(counts[:-lag]>=a.minimum_events)&(counts[lag:]>=a.minimum_events)
  sims=np.sum(F[:-lag][ok]*F[lag:][ok],axis=1) if ok.any() else np.empty(0)
  curve.append(float(np.median(sims)) if len(sims) else -1);support.append(int(len(sims)))
 curve=np.asarray(curve);lags=np.arange(minlag,maxlag+1);distance=max(1,int(round(2000/a.step_us)))
 peaks,_=find_peaks(curve,distance=distance,prominence=.002)
 order=peaks[np.argsort(curve[peaks])[::-1]] if len(peaks) else []
 candidates=[]
 for i in order[:10]:
  period=lags[i]*a.step_us*1e-6;candidates.append({'period_ms':1000*period,'rps':1/period,'rpm':60/period,'similarity':float(curve[i]),'support':support[i]})
 report={'status':'recurrence_candidates_require_stereo_and_harmonic_check','duration_s':float((t.max()-t.min())*1e-6),'frames':len(F),'top_candidates':candidates,'parameters':vars(a)}
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf8')
 np.savez(out.with_suffix('.npz'),lag_ms=lags*a.step_us/1000,similarity=curve,support=support)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
