"""Find candidate spin frequencies from periodic logo-event visibility."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt,find_peaks
from estimate_event_plane_spin import normalized_events

def main():
 p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--path',required=True);p.add_argument('--output',required=True)
 p.add_argument('--inner-radius',type=float,default=.62);p.add_argument('--bin-us',type=int,default=100)
 p.add_argument('--min-rps',type=float,default=5);p.add_argument('--max-rps',type=float,default=120);a=p.parse_args()
 x,y,t,pol=normalized_events(a.raw,a.path,a.inner_radius)
 edges=np.arange(int(t.min()),int(t.max())+a.bin_us,a.bin_us,dtype=np.int64)
 counts=np.histogram(t,bins=edges)[0].astype(float);fs=1e6/a.bin_us
 signal=np.sqrt(counts+.25);sos=butter(3,[a.min_rps,a.max_rps],btype='bandpass',fs=fs,output='sos')
 filtered=sosfiltfilt(sos,signal)
 nfft=1
 while nfft<16*len(filtered):nfft*=2
 spectrum=np.abs(np.fft.rfft(filtered*np.hanning(len(filtered)),n=nfft))**2
 freq=np.fft.rfftfreq(nfft,1/fs);valid=(freq>=a.min_rps)&(freq<=a.max_rps)
 peaks,_=find_peaks(spectrum[valid],distance=max(1,int(2/(freq[1]-freq[0]))))
 vf=freq[valid];vp=spectrum[valid];order=peaks[np.argsort(vp[peaks])[::-1]] if len(peaks) else []
 candidates=[]
 floor=float(np.median(vp)+1e-12)
 for i in order[:8]:candidates.append({'rps':float(vf[i]),'rpm':float(60*vf[i]),'power_over_median':float(vp[i]/floor)})
 report={'status':'frequency_candidates_not_ground_truth','duration_s':float((t.max()-t.min())*1e-6),'logo_events':len(t),
         'top_candidates':candidates,'frequency_resolution_rps':float(fs/nfft),'parameters':vars(a)}
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf8')
 np.savez(out.with_suffix('.npz'),counts=counts,filtered=filtered,frequency=freq,power=spectrum)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
