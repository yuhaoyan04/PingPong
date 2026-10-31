"""Validate observation preservation, raw-data residuals and smoothing media."""
import argparse
import json
import sys
import zipfile
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pingpong_event_pipeline.marker_spin import read_tracks,make_pairs
from pingpong_event_pipeline.spin import _rotate_vectors
from pingpong_event_pipeline.motion import write_csv


def finalize(output,tracks,dataset):
    output=Path(output)
    report=json.loads((output/'smoothing_report.json').read_text())
    raw,_,_=read_tracks(tracks,dataset)
    smoothed,_,_=read_tracks(output/'smoothed_rotation_tracks.csv',dataset)
    flags=[];corrections=[]
    for key,run in raw.items():
        other=smoothed[key]
        assert len(run)==len(other)
        for a,b in zip(run,other):
            for field in ['camera','group','segment_id','track_id','timestamp_us','frame_index','image','source','track_support']:
                assert a[field]==b[field],(key,field)
            for field in ['x1','y1','x2','y2']:assert float(a[field])==float(b[field])
            corrections.append(float(b['smoothing_displacement_px']))
        for mode,rows in [('before',run),('after',other)]:
            p=np.array([r['point'] for r in rows]);v=np.diff(p,axis=0)
            if len(v)<3:continue
            ref=(v[:-2]+v[2:])/2;mid=v[1:-1];norm=np.linalg.norm(ref,axis=1)*np.linalg.norm(mid,axis=1)
            for i in np.flatnonzero((norm>.2)&(np.sum(ref*mid,axis=1)<-.3*norm)):
                r=rows[int(i)+2]
                flags.append({'mode':mode,'camera':key[0],'group':key[1],'segment_id':key[2],
                              'frame_index':r['frame_index'],'timestamp_us':r['timestamp'],
                              'all_four_context_points_ring_supported':all(x['track_support']=='ring_supported' for x in rows[int(i):int(i)+4]),
                              'track_support':r['track_support'],'source':r['source']})
    write_csv(output/'local_reverse_diagnostics.csv',flags)
    report['reliable_context_reverse_steps']={mode:sum(r['mode']==mode and r['all_four_context_points_ring_supported'] for r in flags) for mode in ['before','after']}
    report['correction_percentiles_px']={str(q):float(np.quantile(corrections,q)) for q in [.5,.9,.95,.99,1]}
    before=json.loads((output/'before_spin.json').read_text());after=json.loads((output/'after_spin.json').read_text())
    bmap={s['serve']:s for s in before['serves']};amap={s['serve']:s for s in after['serves']}
    residuals=[]
    for name,s in bmap.items():
        for wb,wa in zip(s['windows'],amap[name]['windows']):
            assert wb['window_id']==wa['window_id']
            for source in ['JOINT','F3E','F59']:
                xb=wb if source=='JOINT' else wb['mono'][source]
                xa=wa if source=='JOINT' else wa['mono'][source]
                if not xb['numerically_usable'] or not xa['numerically_usable']:continue
                for camera in (['F3E','F59'] if source=='JOINT' else [source]):
                    runs=[(key[2],r) for key,r in raw.items() if key[0]==camera and key[1]==f'{name}_{camera}']
                    pairs=make_pairs(runs,wb['start_us'],wb['end_us'],int(before['config'].get('maximum_pair_us',2000)))
                    if source=='JOINT':
                        lo,hi=wb['common_support']['start_us'],wb['common_support']['end_us']
                        pairs=[p for p in pairs if p[1]['timestamp']>=lo and p[2]['timestamp']<=hi]
                    if not pairs:continue
                    a=np.array([p[1]['sphere'] for p in pairs]);b=np.array([p[2]['sphere'] for p in pairs]);dt=np.array([p[3] for p in pairs])
                    radii=np.array([p[1]['radius'] for p in pairs])
                    row={'serve':name,'window_id':wb['window_id'],'source':source,'camera':camera,'raw_pairs':len(pairs)}
                    for mode,fit in [('before',xb),('after',xa)]:
                        w=np.array(fit['omega_camera_rad_s'])
                        if source=='JOINT' and camera=='F59':w=np.array(s['R_F59_to_F3E']).T@w
                        errors=np.linalg.norm(_rotate_vectors(a,w,dt)[:,:2]-b[:,:2],axis=1)*radii
                        row[mode+'_rms_against_original_pairs_px']=float(np.sqrt(np.mean(errors**2)))
                    residuals.append(row)
    write_csv(output/'original_observation_reprojection.csv',residuals)
    frames=json.loads((output/'frame_values.json').read_text())['serves'];videos=[]
    if report.get('videos_generated',True):
        for s in frames:
            file=output/'videos'/f"{s['serve']}_smoothing.mp4"
            cap=cv2.VideoCapture(str(file));n=0
            while True:
                ok,frame=cap.read()
                if not ok:break
                assert frame.shape[:2]==(1040,1280);n+=1
            cap.release();assert n==s['frames'],(file,n,s['frames'])
            videos.append({'file':str(file.relative_to(output)),'decoded_frames':n,'passed':True})
    report['raw_observation_residual_note']='Both fits reproject the SAME original unsmoothed point pairs. These observations are not independent ground truth or held-out data; lower smooth-data residual/roughness alone does not establish physical accuracy.'
    (output/'smoothing_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    validation={'passed':True,'observation_ids_times_sources_boxes_unchanged':True,'videos':videos,
                'input_observations':report['observations'],'output_observations':sum(map(len,smoothed.values())),
                'physical_accuracy_validated':False}
    (output/'validation.json').write_text(json.dumps(validation,indent=2))
    page=(output/'review.html').read_text()
    note=f"<p>95% 的中心修正不超过 {report['correction_percentiles_px']['0.95']:.2f} 个裁剪像素；可靠球轮廓上下文内的局部反向步诊断 {report['reliable_context_reverse_steps']['before']} → {report['reliable_context_reverse_steps']['after']}。球轮廓回退区保留原轨迹，不追加平滑，也不参与转速点对。诊断计数不是真实错误数。<a href=\"local_reverse_diagnostics.csv\">逐处记录</a> · <a href=\"original_observation_reprojection.csv\">两种模型对同一原始观测的残差</a>（没有独立真值）。</p>"
    if 'data-final-validation' not in page:
        page=page.replace('</h1>','</h1><div data-final-validation>'+note+'</div>',1)
        page=page.replace('</video>','</video><button onclick="step(this,-1)">上一帧</button> <button onclick="step(this,1)">下一帧</button>')
        page=page.replace('</html>','''<script>function step(button,d){const v=button.closest('section').querySelector('video');v.pause();let n=Math.floor(v.currentTime*25+.0001)+d;v.currentTime=Math.max(0,Math.min((n+.1)/25,v.duration-1/25));}</script></html>''')
        (output/'review.html').write_text(page,encoding='utf-8')
    for f in output.rglob('*.log'):
        if f.stat().st_size==0:f.unlink()
    with zipfile.ZipFile(output/'smoothing_review_bundle.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for f in sorted(output.rglob('*')):
            if f.is_file() and f.name!='smoothing_review_bundle.zip':z.write(f,f.relative_to(output))
    print(json.dumps({'videos':len(videos),'reliable_reverse_step_diagnostics':report['reliable_context_reverse_steps'],
                      'bundle_bytes':(output/'smoothing_review_bundle.zip').stat().st_size},ensure_ascii=False,indent=2))
    return validation


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tracks',type=Path,default=Path('output/latest/tracking/rotation_tracks_all.csv'))
    p.add_argument('--dataset',type=Path,default=Path('data/server_marker_two_camera_train_v1'))
    a=p.parse_args();finalize(a.output,a.tracks,a.dataset)
