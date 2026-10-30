"""Estimate sphere rotation from local Surface-of-Active-Events planes."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from estimate_event_plane_spin import normalized_events

def collect(x,y,t,pol,bins=64,radius=.72,neighbourhood=2,max_age_us=1200,
            min_neighbours=8,max_residual_us=140,sample_every=2,
            fit_translation=False):
    sae=np.full((2,bins,bins),-10**15,np.int64)
    sae_x=np.zeros((2,bins,bins),np.float64)
    sae_y=np.zeros((2,bins,bins),np.float64)
    rows=[]; diagnostics=[]
    gx=np.floor((x+radius)*bins/(2*radius)).astype(int)
    gy=np.floor((y+radius)*bins/(2*radius)).astype(int)
    for i in range(len(t)):
        px,py,p=int(gx[i]),int(gy[i]),int(pol[i])
        if px<0 or px>=bins or py<0 or py>=bins: continue
        sae[p,py,px]=t[i]; sae_x[p,py,px]=x[i]; sae_y[p,py,px]=y[i]
        if i%sample_every: continue
        x0=max(0,px-neighbourhood);x1=min(bins,px+neighbourhood+1)
        y0=max(0,py-neighbourhood);y1=min(bins,py+neighbourhood+1)
        patch=sae[p,y0:y1,x0:x1]; yy,xx=np.indices(patch.shape)
        age=t[i]-patch; valid=(age>=0)&(age<=max_age_us)
        if valid.sum()<min_neighbours: continue
        xn=sae_x[p,y0:y1,x0:x1][valid]
        yn=sae_y[p,y0:y1,x0:x1][valid]
        tn=(patch[valid].astype(float)-t[i])*1e-6
        A=np.column_stack((xn-x[i],yn-y[i],np.ones(len(xn))))
        if np.linalg.cond(A)>100: continue
        coef=np.linalg.lstsq(A,tn,rcond=None)[0]
        residual=np.median(np.abs(tn-A@coef))*1e6
        if residual>max_residual_us: continue
        a,b=coef[:2]
        if not 2e-4<np.hypot(a,b)<.15: continue
        z=np.sqrt(max(1e-6,1-x[i]**2-y[i]**2))
        # Unknowns are wx,wy,wz plus residual normalised image translation ux,uy.
        row=[-b*z,a*z,-a*y[i]+b*x[i]]
        if fit_translation: row += [a,b]
        rows.append(row)
        diagnostics.append(residual)
    width=5 if fit_translation else 3
    return np.asarray(rows,float).reshape(-1,width),np.asarray(diagnostics)

def solve(A,max_rpm,seed=91,iterations=10000,threshold=.25):
    dimensions=A.shape[1]
    if len(A)<30:return np.zeros(dimensions),np.zeros(len(A),bool),None
    rng=np.random.default_rng(seed);limit=max_rpm*2*np.pi/60;best=None
    for _ in range(iterations):
        ids=rng.choice(len(A),dimensions,replace=False)
        try:q=np.linalg.solve(A[ids],np.ones(dimensions))
        except np.linalg.LinAlgError:continue
        if not np.isfinite(q).all() or np.linalg.norm(q[:3])>limit:continue
        r=np.abs(A@q-1); inside=r<threshold
        key=(inside.sum(),-np.median(r[inside]) if inside.any() else -np.inf)
        if best is None or key>best[0]:best=(key,q,inside)
    if best is None or best[2].sum()<20:return np.zeros(5),np.zeros(len(A),bool),None
    q=best[1];inside=best[2]
    for _ in range(8):
        r=A@q-1;s=1.4826*np.median(np.abs(r[inside]-np.median(r[inside])))+.02
        w=np.minimum(1,1.5*s/np.maximum(np.abs(r),1e-8)); w[~inside]*=.1
        root=np.sqrt(w);q=np.linalg.lstsq(A*root[:,None],root,rcond=None)[0]
        if np.linalg.norm(q[:3])>limit:
            q[:3] *= limit/np.linalg.norm(q[:3])
        inside=np.abs(A@q-1)<threshold
    return q,inside,np.abs(A@q-1)

def main():
 p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--path',required=True);p.add_argument('--output',required=True)
 p.add_argument('--inner-radius',type=float,default=.72);p.add_argument('--bins',type=int,default=64);p.add_argument('--max-age-us',type=int,default=1200)
 p.add_argument('--maximum-rpm',type=float,default=12000);p.add_argument('--sample-every',type=int,default=2);a=p.parse_args()
 x,y,t,pol=normalized_events(a.raw,a.path,a.inner_radius)
 A,plane_res=collect(x,y,t,pol,a.bins,a.inner_radius,max_age_us=a.max_age_us,sample_every=a.sample_every,fit_translation=False)
 q,inside,res=solve(A,a.maximum_rpm);mag=np.linalg.norm(q[:3])
 report={'purpose':'sae_plane_rigid_sphere_rotation','events_inside_ball':len(x),'local_planes':len(A),'inlier_planes':int(inside.sum()),
  'inlier_fraction':float(inside.mean()) if len(inside) else 0,'rpm':float(mag*60/(2*np.pi)) if mag else None,
  'axis_camera':(q[:3]/mag).tolist() if mag else None,'residual_translation_normalized_s':None,
  'plane_residual_us_median':float(np.median(plane_res)) if len(plane_res) else None,
  'equation_residual_median':float(np.median(res[inside])) if res is not None and inside.any() else None,'parameters':vars(a),
  'status':'candidate_requires_dual_camera_and_parameter_stability'}
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
