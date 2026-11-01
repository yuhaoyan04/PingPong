"""Replay frozen YOLO outputs and refine evidence-backed track continuity.

Keeps the original and geometry-only references. The combined result uses new
tracking, the retained RAW geometry, material anchor localization, and one
trajectory smoothing pass. No detector retraining or prediction gap filling.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/'yolo'))

import cv2
import numpy as np
from marker_tracker import BallGeometry,MarkerTracker,TrackerConfig
from track_segments import select_consecutive_segments,rotation_rows
from pingpong_event_pipeline.raw_sync import load_serves
from pingpong_event_pipeline.marker_spin import read_tracks,sha256
from pingpong_event_pipeline.motion import write_csv
from pingpong_event_pipeline.trajectory_smoothing import csv_rows
from pingpong_event_pipeline.serve_spin import analyze_serves
from pingpong_event_pipeline.spin_fusion import frame_spin
from refine_observations import load,dump,rows,transform_row,localize_group,eligible_rows


def build(root,prior,geometry,output):
    output.mkdir(parents=True,exist_ok=False)
    for sub in ['tracking','tracks','localization','spin']:(output/sub).mkdir()
    dataset=root/'data/server_marker_two_camera_train_v1';rebuilt=geometry/'dataset'
    config=TrackerConfig(**load(root/'configs/inference_tracking.json'))
    summaries=[];raw_rows=[];all_combined=[];hashes={}
    for spec in load_serves(dataset):
        for camera in ['F3E','F59']:
            name=f"{spec['serve']}_{camera}";folder=Path(spec['cameras'][camera]['folder'])
            manifest=spec['cameras'][camera]['rows'];target=rebuilt/'unlabeled_inference'/camera/name
            new_manifest=rows(target/'manifest.csv');assert len(new_manifest)==len(manifest)
            checkpoint=root/'experiments/two_camera_v1'/camera/'train/weights/last.pt'
            model_sha=sha256(checkpoint)
            assert model_sha==load(checkpoint.parents[2]/'training_completion.json')['checkpoint_sha256']
            cached=root/'experiments/two_camera_v1'/camera/'qualitative'/name/'predictions.json'
            cache_summary=load(cached.parent.parent/'summary.json')
            assert cache_summary['checkpoint_sha256']==model_sha
            assert cache_summary['prediction_floor']<=config.continuation_confidence
            detections={r['image']:r for r in load(cached)};hashes[str(cached.relative_to(root))]=sha256(cached)
            hashes[str(checkpoint.relative_to(root))]=model_sha
            tracker=MarkerTracker(config,BallGeometry());frames=[]
            for row in manifest:
                entry=detections[Path(row['image']).name];assert int(row['timestamp_us'])==entry['timestamp_us']
                image=cv2.imread(str(folder/row['image']),cv2.IMREAD_GRAYSCALE);assert image is not None
                result=tracker.update(image,entry['detections'],int(row['timestamp_us']))
                frames.append({'camera':camera,'group':name,'image':Path(row['image']).name,
                    'frame_index':int(row['frame_index']),'track_support':row['track_support'],**result})
            selected,segments,rejected,selection=select_consecutive_segments(frames,config.minimum_consecutive_frames,config.consecutive_max_gap_us)
            destination=output/'tracking'/camera/name;destination.mkdir(parents=True)
            dump(destination/'raw_frames.json',frames);dump(destination/'frames.json',selected)
            dump(destination/'rotation_segments.json',segments);dump(destination/'rejected_segments.json',rejected)
            dump(destination/'lifecycle_events.json',tracker.events)
            records=rotation_rows(selected);raw_rows.extend(records)
            write_csv(destination/'rotation_tracks.csv',records)
            transformed=[]
            for r in records:
                i=int(r['frame_index']);old=manifest[i];new=new_manifest[i]
                assert int(old['timestamp_us'])==int(new['timestamp_us'])==int(r['timestamp_us'])
                keys=['center_x_px','center_y_px','radius_px']
                mapped=transform_row(r,np.array([float(old[k]) for k in keys]),np.array([float(new[k]) for k in keys]),1.2)
                mapped.update({k:new[k] for k in ['geometry_continuity_id','geometry_window_consistent','geometry_rotation_usable']})
                transformed.append(mapped)
            combined,diagnostics=localize_group(transformed,target) if transformed else ([],[])
            all_combined.extend(combined);dump(output/'localization'/f'{name}_combined.json',diagnostics)
            previous=load(root/'output/latest/tracking'/camera/name/'rotation_filter_summary.json')
            old_frames=load(root/'output/latest/tracking'/camera/name/'frames.json')
            summary={'camera':camera,'group':name,'previous_observations':previous['retained_samples'],
                'observations':selection['retained_samples'],'previous_segments':previous['retained_segments'],
                'segments':len(segments),'previous_observed_frames':sum(bool(f['tracks']) for f in old_frames),
                'observed_frames':sum(bool(f['tracks']) for f in selected),'created_tracks':tracker.next_id-1,
                'flow_methods':dict(Counter(t.get('flow_method') for f in selected for t in f['tracks'] if t['source']=='optical_flow')),
                'selection':selection,'termination_reasons':dict(Counter(r['reason'] for r in tracker.terminated))}
            summaries.append(summary);dump(destination/'summary.json',summary)
            print(name,summary['previous_observations'],'->',summary['observations'],'frames',summary['previous_observed_frames'],'->',summary['observed_frames'],flush=True)
    write_csv(output/'tracks/retracked_raw.csv',raw_rows)
    write_csv(output/'tracks/combined_all_observations.csv',all_combined)
    eligible,dropped=eligible_rows(all_combined);path=output/'tracks/combined_eligible.csv';write_csv(path,eligible)
    runs,_,_=read_tracks(path,rebuilt,smoothing_config=load(root/'config/marker_trajectory_smoothing.json'))
    effective=output/'tracks/combined.csv';write_csv(effective,csv_rows(runs))
    variants=[]
    for mode in ['baseline','geometry_only']:
        shutil.copy2(prior/'tracks'/f'{mode}.csv',output/'tracks'/f'{mode}.csv')
        shutil.copy2(prior/'spin'/f'{mode}.json',output/'spin'/f'{mode}.json')
        v=next(v for v in load(prior/'variants.json') if v['mode']==mode)
        variants.append({**v,'tracks':str((output/'tracks'/f'{mode}.csv').resolve())})
    variants.append({'mode':'combined','dataset':str(rebuilt.resolve()),'tracks':str(effective.resolve()),
        'tracks_sha256':sha256(effective),'observations':len(eligible),'dropped':dropped})
    dump(output/'variants.json',variants);dump(output/'tracking_summary.json',summaries)
    dump(output/'continuity_provenance.json',{'tracker':asdict(config),'input_sha256':hashes,
        'dataset':str(dataset),'geometry_cache':str(geometry),'prior_review':str(prior),
        'predictions_count_as_observations':False,'production_models_changed':False,
        'timestamps_changed':False,'physical_accuracy_validated':False})
    baseline=load(output/'spin/baseline.json')
    fixed={s['serve']:{k:s[k] for k in ['R_F59_to_F3E','orientation_training_serves']} for s in baseline['serves']}
    report=analyze_serves(effective,rebuilt,baseline['config'],fixed_relative_orientations=fixed)
    dump(output/'spin/combined.json',report)
    diagnose(root,prior,output,summaries)
    return summaries


def diagnose(root,prior,output,summaries):
    """Separate observed tracking, numerical fits, and displayed coverage."""
    old={s['serve']:s for s in load(prior/'spin/combined.json')['serves']}
    new={s['serve']:s for s in load(output/'spin/combined.json')['serves']}
    previous={s['serve']:s for s in load(prior/'frame_values.json')}
    audits=[];timeline=[]
    for spec in load_serves(root/'data/server_marker_two_camera_train_v1'):
        name=spec['serve'];times=previous[name]['timestamps_us'];tally=Counter()
        for t,v in zip(times,previous[name]['values']):
            states={'previous_display':v['combined'],'overlap_fix_only':frame_spin(old[name],t),
                'tracking_and_overlap_fix':frame_spin(new[name],t)}
            for stage,state in states.items():
                for c in ['F3E','F59']:
                    tally[stage+'/'+c]+=int(state['independent'][c]['available'])
                tally[stage+'/system']+=int(state['system']['available'])
                tally[stage+'/joint']+=int(state['system'].get('source')=='JOINT')
                timeline.append({'serve':name,'timestamp_us':t,'stage':stage,
                    'F3E_rps':state['independent']['F3E']['rps'],'F59_rps':state['independent']['F59']['rps'],
                    'system_rps':state['system']['rps'],'system_source':state['system'].get('source')})
        entry={'serve':name,'displayed_frames':dict(tally),'trajectory_counts':[
            {k:r[k] for k in ['camera','previous_observations','observations','previous_segments','segments','previous_observed_frames','observed_frames']}
            for r in summaries if r['group'].startswith(name+'_')],
            'joint_usable_windows_before':old[name]['joint_usable_windows'],'joint_usable_windows_after':new[name]['joint_usable_windows'],
            'mono_usable_windows':{c:{'before':old[name]['views'][c]['aligned_usable_windows'],
                'after':new[name]['views'][c]['aligned_usable_windows']} for c in ['F3E','F59']}}
        audits.append(entry)
    dump(output/'continuity_audit.json',audits);write_csv(output/'continuity_timeline.csv',timeline)
    dump(output/'serve05_diagnostics.json',next(r for r in audits if r['serve']=='serve05'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior',type=Path,default=Path('output/latest_observations'))
    p.add_argument('--geometry',type=Path,default=Path('output/observation_refinement_v14'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(Path(__file__).resolve().parents[1],a.prior.resolve(),a.geometry.resolve(),a.output.resolve())


if __name__=='__main__':main()
