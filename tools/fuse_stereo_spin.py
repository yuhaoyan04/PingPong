"""Fuse two SAE spin reports with explicit fail-closed quality gates."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--master',required=True);p.add_argument('--slave',required=True);p.add_argument('--calibration',required=True);p.add_argument('--output',required=True)
 p.add_argument('--max-rpm-disagreement',type=float,default=.25);p.add_argument('--max-axis-angle-deg',type=float,default=25);p.add_argument('--minimum-inliers',type=int,default=150);a=p.parse_args()
 m=json.load(open(a.master,encoding='utf8'));s=json.load(open(a.slave,encoding='utf8'));R=np.load(a.calibration)['R']
 am=np.asarray(m['axis_camera'],float); ass=R.T@np.asarray(s['axis_camera'],float)
 if am@ass<0:ass=-ass
 angle=float(np.degrees(np.arccos(np.clip(am@ass,-1,1))))
 rm,rs=float(m['rpm']),float(s['rpm']); disagreement=abs(rm-rs)/max(1.,(rm+rs)/2)
 gates={'master_inliers':m['inlier_planes']>=a.minimum_inliers,'slave_inliers':s['inlier_planes']>=a.minimum_inliers,
        'rpm_agreement':disagreement<=a.max_rpm_disagreement,'axis_agreement':angle<=a.max_axis_angle_deg,
        'finite_positive':bool(np.isfinite([rm,rs]).all() and min(rm,rs)>0)}
 valid=all(gates.values());weights=np.array([m['inlier_planes'],s['inlier_planes']],float);weights/=weights.sum()
 axis=weights[0]*am+weights[1]*ass;axis/=np.linalg.norm(axis)
 proxy=float(weights@[rm,rs]) if valid else None
 report={'status':'valid_relative_spin_signal_uncalibrated' if valid else 'rejected_quality_gate','rpm':None,
         'rpm_proxy_uncalibrated':proxy,
         'axis_master_camera':axis.tolist() if valid else None,'master_rpm':rm,'slave_rpm':rs,
         'rpm_relative_disagreement':disagreement,'axis_angle_deg':angle,'gates':gates,
         'quality':'relative_strength_and_axis_only_requires_known-speed_calibration' if valid else 'invalid','parameters':vars(a)}
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
