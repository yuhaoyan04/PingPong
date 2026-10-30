"""Run the accepted coarse stereo spin pipeline on one tracked ball clip."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path

def run(command):
    subprocess.run([sys.executable,*map(str,command)],check=True)

def main():
 p=argparse.ArgumentParser(description='双目事件球粗略转速离线解算（带拒绝门控）')
 p.add_argument('--master-raw',required=True);p.add_argument('--slave-raw',required=True)
 p.add_argument('--master-path',required=True);p.add_argument('--slave-path',required=True)
 p.add_argument('--calibration',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
 root=Path(__file__).parent;out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 reports=[]
 for camera,raw,path in [('master',a.master_raw,a.master_path),('slave',a.slave_raw,a.slave_path)]:
  report=out/f'{camera}_sae_spin.json';reports.append(report)
  run([root/'estimate_spin_by_sae.py','--raw',raw,'--path',path,'--output',report,
       '--bins','56','--max-age-us','900','--sample-every','7'])
 run([root/'fuse_stereo_spin.py','--master',reports[0],'--slave',reports[1],
      '--calibration',a.calibration,'--output',out/'stereo_spin.json'])
 print(f'Final report: {out / "stereo_spin.json"}')
if __name__=='__main__':main()
