"""Marker-ID-free spin proxy from sub-millisecond event-texture shifts."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np
from estimate_event_plane_spin import normalized_events


def render(x, y, polarity, bins, radius):
    image = np.zeros((bins, bins), np.float32)
    ix = np.floor((x + radius) * bins / (2*radius)).astype(np.int32)
    iy = np.floor((y + radius) * bins / (2*radius)).astype(np.int32)
    valid = (ix >= 0) & (ix < bins) & (iy >= 0) & (iy < bins)
    # Activity is more stable than signed polarity under fast contrast reversal.
    np.add.at(image, (iy[valid], ix[valid]), 1.0)
    image = cv2.GaussianBlur(image, (0, 0), .8)
    image -= cv2.GaussianBlur(image, (0, 0), 5.0)
    return image


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',required=True); p.add_argument('--path',required=True)
    p.add_argument('--output',required=True); p.add_argument('--slice-us',type=int,default=500)
    p.add_argument('--bins',type=int,default=64); p.add_argument('--inner-radius',type=float,default=.72)
    p.add_argument('--minimum-events',type=int,default=80)
    a=p.parse_args()
    x,y,t,pol=normalized_events(a.raw,a.path,a.inner_radius)
    edge=np.arange(int(t.min()),int(t.max())+a.slice_us,a.slice_us,dtype=np.int64)
    frames=[]
    for lo,hi in zip(edge[:-1],edge[1:]):
        take=(t>=lo)&(t<hi)
        frames.append((render(x[take],y[take],pol[take],a.bins,a.inner_radius),int(take.sum()),lo))
    window=cv2.createHanningWindow((a.bins,a.bins),cv2.CV_32F)
    samples=[]; scale=2*a.inner_radius/a.bins; dt=a.slice_us*1e-6
    for first,second in zip(frames[:-1],frames[1:]):
        if min(first[1],second[1])<a.minimum_events: continue
        shift,response=cv2.phaseCorrelate(first[0],second[0],window)
        # A central sphere point obeys vx=omega_y and vy=-omega_x.
        vx,vy=shift[0]*scale/dt,shift[1]*scale/dt
        omega=np.array([-vy,vx,0.0]); rpm=np.linalg.norm(omega)*60/(2*np.pi)
        samples.append({'time_us':int(first[2]+a.slice_us),'events':int(first[1]+second[1]),
                        'response':float(response),'shift_bins':[float(shift[0]),float(shift[1])],
                        'omega_xy_rad_s':omega[:2].tolist(),'rpm_xy':float(rpm)})
    good=[s for s in samples if s['response']>=.08 and np.hypot(*s['shift_bins'])<=a.bins*.25]
    vec=np.array([s['omega_xy_rad_s'] for s in good]) if good else np.empty((0,2))
    median=np.median(vec,axis=0) if len(vec) else np.zeros(2)
    rpms=np.array([s['rpm_xy'] for s in good]) if good else np.empty(0)
    report={'purpose':'marker_id_free_event_texture_shift_spin_proxy','samples':samples,
      'accepted_samples':len(good),'total_samples':len(samples),
      'omega_xy_median_rad_s':median.tolist(),
      'rpm_xy_median_vector':float(np.linalg.norm(median)*60/(2*np.pi)),
      'rpm_xy_median_magnitude':float(np.median(rpms)) if len(rpms) else None,
      'rpm_xy_iqr':([float(np.percentile(rpms,25)),float(np.percentile(rpms,75))] if len(rpms) else None),
      'parameters':vars(a),'status':'proxy_excludes_optical_axis_component'}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
