"""Reproducible whole-chain diagnostics; never changes labels or production weights.

Run after tools/yolo/audit_group_holdout.py for both cameras. New candidate
rotation fits use exactly the baseline smoothed coordinates, radius, windows,
and camera orientations. Human review selections are proposals, not labels.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from pingpong_event_pipeline.marker_spin import read_tracks, make_pairs, sha256
from pingpong_event_pipeline.serve_spin import analyze_serves, fit_two_view
from pingpong_event_pipeline.spin import _rotate_vectors
from pingpong_event_pipeline.rotation_weights import correspondence_weights
from pingpong_event_pipeline.marker_enhancement import enhance_for_camera
from pingpong_event_pipeline.motion import write_csv,ball_motion
from pingpong_event_pipeline.radius_review import font


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def audit_dataset(root, output):
    dataset=root/'data/server_marker_two_camera_train_v1'
    profiles=load(root/'config/marker_enhancement_v1.json')['cameras']
    records=[];queue=[]
    for camera in ['F3E','F59']:
        labeled=dataset/'datasets'/camera
        images=sorted((labeled/'images/train').glob('*.png'))
        labels=[(labeled/'labels/train'/f'{p.stem}.txt').read_text().splitlines() for p in images]
        record={'camera':camera,'reviewed_frames':len(images),'reviewed_boxes':sum(map(len,labels)),
                'reviewed_serves':sorted({p.name.split('__')[0] for p in images}),
                'explicit_empty_negative_frames':sum(len(x)==0 for x in labels),
                'unlabeled_frames':0,'fallback_geometry_frames':0,'filter_pixel_retention':[],
                'filter_bit_exact_frames':0,'polarity_occupancy_exact_frames':0,
                'manifest_step_counts_us':Counter()}
        for folder in sorted((dataset/'unlabeled_inference'/camera).iterdir()):
            if not folder.is_dir():continue
            manifest=rows(folder/'manifest.csv');times=[int(r['timestamp_us']) for r in manifest]
            record['manifest_step_counts_us'].update(map(int,np.diff(times)))
            assert all(y>x for x,y in zip(times,times[1:])),folder
            raw_frames=load(root/'output/latest/tracking'/camera/folder.name/'raw_frames.json')
            byframe={r['frame_index']:r for r in raw_frames}
            proposals=[]
            for r in manifest:
                record['unlabeled_frames']+=1
                fallback=r['track_support']!='ring_supported'
                record['fallback_geometry_frames']+=int(fallback)
                raw=np.array(Image.open(folder/'binary_unfiltered'/Path(r['image']).name).convert('L'))
                filtered=np.array(Image.open(folder/r['image']).convert('L'))
                expected=enhance_for_camera(raw,camera,profiles)
                assert np.array_equal(filtered,expected),(folder,r['image'])
                assert not np.any((filtered!=0)&(raw==0))
                record['filter_bit_exact_frames']+=1
                polarity=np.array(Image.open(folder/r['polarity_reference']).convert('RGB'))
                # The stored red/blue polarity rendering has black background.
                assert np.array_equal(polarity.any(axis=2),raw!=0),(folder,r['image'])
                record['polarity_occupancy_exact_frames']+=1
                retention=float(np.count_nonzero(filtered)/max(1,np.count_nonzero(raw)))
                record['filter_pixel_retention'].append(retention)
                f=byframe[int(r['frame_index'])]
                reasons=[];score=0
                if fallback:reasons.append('球几何无圆周支持');score+=3
                if f['rejected_detections']:reasons.append('高置信框被边缘/几何规则拒绝');score+=2
                if f['high_threshold_boxes_before_edge'] and not f['tracks']:
                    reasons.append('高置信检出但未形成稳定轨迹');score+=2
                if f['baseline_boxes_025']>f['high_threshold_boxes_before_edge']:
                    reasons.append('高低阈值分歧');score+=1
                if retention<.2:reasons.append('去噪删除较多事件像素');score+=1
                if score:proposals.append((score,r,reasons,retention))
            selected=[]
            for score,r,reasons,retention in sorted(proposals,key=lambda v:(-v[0],int(v[1]['timestamp_us']))):
                if any(abs(int(r['timestamp_us'])-int(x['timestamp_us']))<10000 for x in selected):continue
                selected.append(r)
                key=f"{folder.name}_{r['frame_index']}"
                item={'id':key,'camera':camera,'group':folder.name,'frame_index':int(r['frame_index']),
                      'timestamp_us':int(r['timestamp_us']),'track_support':r['track_support'],
                      'reasons':reasons,'pixel_retention':retention,'review_status':'unreviewed',
                      'suggestion_is_not_a_label':True,'image':f'queue/{key}.jpg'}
                sheet=Image.new('RGB',(1024,330),'#142130');draw=ImageDraw.Draw(sheet)
                for j,(sub,title) in enumerate([('binary_unfiltered','原事件占用'),('binary','当前去噪'),('polarity_reference','原极性'),('alternative','较弱去噪，仅供比较')]):
                    if sub=='alternative':
                        p=dict(profiles[camera]);p.update({'min_neighbors':1} if camera=='F3E' else {'threshold':.02})
                        raw=np.array(Image.open(folder/'binary_unfiltered'/Path(r['image']).name).convert('L'))
                        image=Image.fromarray(enhance_for_camera(raw,camera,{camera:p})).convert('RGB')
                    else:image=Image.open(folder/sub/Path(r['image']).name).convert('RGB')
                    sheet.paste(image,(j*256,45));draw.text((j*256+4,8),title,font=font(17),fill='white')
                draw.text((8,305),f"{key} | {item['timestamp_us']} μs | 仅待审，不写回数据集",font=font(15),fill='white')
                sheet.save(output/item['image'],quality=94);queue.append(item)
                if len(selected)>=5:break
        ret=record.pop('filter_pixel_retention')
        record['filter_pixel_retention_quantiles']={str(q):float(np.quantile(ret,q)) for q in [0,.5,.95,1]}
        record['production_checkpoint_sha256']=sha256(root/f'experiments/two_camera_v1/{camera}/train/weights/last.pt')
        records.append(record)
    dump(output/'dataset_audit.json',records);dump(output/'review_queue.json',queue)
    return records,queue


def audit_rotation(root, output):
    folder=output/'rotation';folder.mkdir(exist_ok=True)
    baseline=load(root/'output/latest_smoothing/after_spin.json')
    tracks=root/'output/latest_smoothing/smoothed_rotation_tracks.csv'
    assert baseline['tracks_sha256']==sha256(tracks)
    dataset=root/'data/server_marker_two_camera_train_v1';config=baseline['config']
    assert 'trajectory_smoothing' not in config # explicit CSV already smoothed
    dump(folder/'uniform.json',baseline)
    fixed={s['serve']:{k:s[k] for k in ['R_F59_to_F3E','orientation_training_serves']} for s in baseline['serves']}
    candidate_path=folder/'cluster_equal.json'
    if candidate_path.exists():
        candidate=load(candidate_path)
        assert candidate['tracks_sha256']==sha256(tracks)
        assert candidate['config']=={**config,'pair_weighting':'cluster_equal'}
    else:
        candidate=analyze_serves(tracks,dataset,{**config,'pair_weighting':'cluster_equal'},fixed_relative_orientations=fixed)
        dump(candidate_path,candidate)
    segments,_,_=read_tracks(tracks,dataset)
    grouped=defaultdict(list)
    for (cam,group,segment),run in segments.items():grouped[(group.rsplit('_',1)[0],cam)].append((segment,run))
    matched=[];sensitivity=[];per_serve=[]
    for before,after in zip(baseline['serves'],candidate['serves']):
        assert before['serve']==after['serve'];serve=before['serve']
        for b,a in zip(before['windows'],after['windows']):
            assert b['window_id']==a['window_id']
            for source in ['JOINT','F3E','F59']:
                x=b if source=='JOINT' else b['mono'][source];y=a if source=='JOINT' else a['mono'][source]
                axis=None
                if x.get('candidate_axis_camera') is not None and y.get('candidate_axis_camera') is not None:
                    axis=float(np.degrees(np.arccos(np.clip(np.dot(x['candidate_axis_camera'],y['candidate_axis_camera']),-1,1))))
                matched.append({'serve':serve,'window_id':b['window_id'],'source':source,
                                'before_usable':x['numerically_usable'],'after_usable':y['numerically_usable'],
                                'before_rps':x['candidate_rps'],'after_rps':y['candidate_rps'],'axis_change_degrees':axis})
            # Leave one whole marker cluster out, refit, then predict that cluster.
            # This measures model sensitivity, not a confidence interval or accuracy.
            if not b['numerically_usable']:continue
            pairs={c:make_pairs(grouped[(serve,c)],b['start_us'],b['end_us'],config.get('maximum_pair_us',2000)) for c in ['F3E','F59']}
            support=b['common_support']
            pairs={c:[p for p in pp if p[1]['timestamp']>=support['start_us'] and p[2]['timestamp']<=support['end_us']] for c,pp in pairs.items()}
            for camera in ['F3E','F59']:
                ids=sorted({p[1]['track_id'] for p in pairs[camera]})
                if len(ids)<3:continue
                for cluster in ids:
                    held=[p for p in pairs[camera] if p[1]['track_id']==cluster]
                    training={c:[p for p in pp if c!=camera or p[1]['track_id']!=cluster] for c,pp in pairs.items()}
                    for mode,full in [('uniform',b),('cluster_equal',a)]:
                        fit=fit_two_view(training,np.array(before['R_F59_to_F3E']),{**config,'pair_weighting':mode})
                        error=None;delta=None
                        if fit['numerically_usable']:
                            omega=np.array(fit['omega_camera_rad_s'])
                            if camera=='F59':omega=np.array(before['R_F59_to_F3E']).T@omega
                            start=np.array([p[1]['sphere'] for p in held]);end=np.array([p[2]['sphere'] for p in held]);dt=np.array([p[3] for p in held])
                            px=np.linalg.norm((_rotate_vectors(start,omega,dt)-end)[:,:2],axis=1)*np.array([p[1]['radius'] for p in held])
                            error=float(np.median(px))
                            if full['numerically_usable']:delta=abs(fit['candidate_rps']-full['candidate_rps'])
                        sensitivity.append({'serve':serve,'window_id':b['window_id'],'held_camera':camera,'held_cluster':cluster,
                                            'mode':mode,'held_pairs':len(held),'refit_usable':fit['numerically_usable'],
                                            'rps_change':delta,'held_cluster_median_prediction_error_px':error})
        mode_summary={}
        for mode,report in [('uniform',before),('cluster_equal',after)]:
            joint=[w['candidate_rps'] for w in report['windows'] if w['numerically_usable']]
            mode_summary[mode]={'joint_usable_windows':len(joint),'joint_median_rps':float(np.median(joint)) if joint else None}
        per_serve.append({'serve':serve,**mode_summary})
        print('Rotation audit',serve,mode_summary,flush=True)
    write_csv(folder/'matched_windows.csv',matched)
    write_csv(folder/'leave_cluster_out.csv',sensitivity,fields=['serve','window_id','held_camera','held_cluster','mode','held_pairs','refit_usable','rps_change','held_cluster_median_prediction_error_px'])
    summary={'per_serve':per_serve,'sensitivity':{},'production_default':'uniform',
             'candidate_promoted':False,'physical_accuracy_validated':False,
             'comparison_controls':['same smoothed observations','same radius','same windows','same camera orientation'],
             'scope':'Sensitivity diagnostics; overlapping pairs/windows are correlated, not independent trials.'}
    for mode in ['uniform','cluster_equal']:
        selected=[s for s in sensitivity if s['mode']==mode]
        errors=[s['held_cluster_median_prediction_error_px'] for s in selected if s['held_cluster_median_prediction_error_px'] is not None]
        deltas=[s['rps_change'] for s in selected if s['rps_change'] is not None]
        summary['sensitivity'][mode]={'refits':len(selected),'usable_refits':sum(s['refit_usable'] for s in selected),
                                     'median_held_cluster_error_px':float(np.median(errors)) if errors else None,
                                     'median_rps_change':float(np.median(deltas)) if deltas else None}
    paired_errors=defaultdict(dict)
    for row in sensitivity:
        if row['refit_usable']:
            paired_errors[(row['window_id'],row['held_camera'],row['held_cluster'])][row['mode']]=row['held_cluster_median_prediction_error_px']
    both=[v for v in paired_errors.values() if len(v)==2]
    summary['paired_usable_refits']={'count':len(both),**{mode:float(np.median([v[mode] for v in both])) if both else None for mode in ['uniform','cluster_equal']}}
    dump(folder/'summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for source,col in [('JOINT','teal'),('F3E','tab:orange'),('F59','tab:blue')]:
        rr=[r for r in matched if r['source']==source and r['before_usable'] and r['after_usable']]
        axes[0].scatter([r['before_rps'] for r in rr],[r['after_rps'] for r in rr],label=source,color=col,s=20,alpha=.65)
    axes[0].plot([0,65],[0,65],'k--',alpha=.4);axes[0].legend()
    axes[0].set(xlabel='Uniform pair mass (RPS)',ylabel='Equal cluster mass (RPS)',title='Fixed geometry / windows; candidates, not ground truth')
    values=[[r['held_cluster_median_prediction_error_px'] for r in sensitivity if r['mode']==m and r['held_cluster_median_prediction_error_px'] is not None] for m in ['uniform','cluster_equal']]
    if all(values):axes[1].boxplot(values,labels=['Uniform','Cluster equal'],showfliers=False)
    axes[1].set(ylabel='Held cluster median prediction error (px)',title='Whole-cluster exclusion; correlated windows')
    fig.savefig(output/'plots/rotation_comparison.png',dpi=150);plt.close(fig)
    return summary


def audit_tracking(root,output):
    dataset=root/'data/server_marker_two_camera_train_v1'
    segments,_,_=read_tracks(root/'output/latest_smoothing/smoothed_rotation_tracks.csv',dataset)
    details=[]
    for (camera,group,segment),run in segments.items():
        details.append({'camera':camera,'group':group,'segment_id':segment,'frames':len(run),
                        'duration_us':run[-1]['timestamp']-run[0]['timestamp'],
                        'supported_frames':sum(r['track_support']=='ring_supported' for r in run),
                        'optical_flow_frames':sum(r['source']=='optical_flow' for r in run),
                        'maximum_step_us':int(np.max(np.diff([r['timestamp'] for r in run])))})
    write_csv(output/'tracking_segments.csv',details)
    summary={'observations':sum(r['frames'] for r in details),'segments':len(details),
             'segment_frame_counts':dict(Counter(r['frames'] for r in details)),
             'segments_shorter_than_2ms':sum(r['duration_us']<2000 for r in details),
             'source_counts':dict(Counter(r['source'] for run in segments.values() for r in run)),
             'minimum_3_consecutive_frames_checked':True,'manifest_identity_and_time_alignment_checked':True,
             'pure_prediction_observations':0,
             'note':'Three-frame retention alone does not establish an observable 3D rotation. Fit checks duration, geometry, distinct observations and conditioning separately.'}
    dump(output/'tracking_audit.json',summary)
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    ax.hist([r['duration_us']/1000 for r in details],bins=25,color='teal')
    ax.axvline(2,color='tab:orange',linestyle='--',label='Minimum spin window duration (not the only constraint)')
    ax.set(xlabel='Continuous segment duration (ms)',ylabel='Segments',title='Retained marker segments; three-frame minimum is not sufficient for spin')
    ax.legend();fig.savefig(output/'plots/track_duration.png',dpi=150);plt.close(fig)
    return summary


def audit_speed(root,output):
    records,groups=ball_motion(root/'data/server_marker_two_camera_train_v1')
    write_csv(output/'ball_motion.csv',records)
    speeds=np.array([r['candidate_transverse_speed_mps'] for r in records if r['velocity_usable']])
    summary={'groups':groups,'candidate_transverse_speed_quantiles_mps':{str(q):float(np.quantile(speeds,q)) for q in [0,.5,.95,1]},
             'samples_above_engineering_limit_30mps':int(np.sum(speeds>30)),
             'values_clipped':False,'measured_3d_speed':False,'physical_accuracy_validated':False}
    dump(output/'ball_motion_audit.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--output',type=Path,default=Path('output/system_iteration_v11'));p.add_argument('--skip-dataset',action='store_true')
    a=p.parse_args();root=a.root.resolve();output=a.output.resolve()
    for f in ['plots','queue','rotation','videos','posters']:(output/f).mkdir(parents=True,exist_ok=True)
    if not a.skip_dataset:
        audit_dataset(root,output);print('Dataset / denoising audit complete',flush=True)
    audit_tracking(root,output)
    audit_speed(root,output)
    summary=audit_rotation(root,output)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
