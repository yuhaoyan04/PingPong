"""Rebuild RAW crops and refine marker centers in isolated observation variants."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from pingpong_event_pipeline.raw_sync import IndexedEvents,load_serves
from pingpong_event_pipeline.events import trajectory_compensated_ball_patch
from pingpong_event_pipeline.observation_refinement import (
    CircleConfig,fit_boundary_circle,LocalizationConfig,registration_image,marker_features,register_marker)
from pingpong_event_pipeline.trajectory_smoothing import local_cubic,SmoothingConfig,csv_rows
from pingpong_event_pipeline.marker_enhancement import enhance_for_camera
from pingpong_event_pipeline.marker_spin import read_tracks,sha256
from pingpong_event_pipeline.motion import write_csv


def load(path):return json.loads(Path(path).read_text())
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def render_events(events,times,geometry,now,scale):
    surface=trajectory_compensated_ball_patch(events,times,geometry[:,:2],geometry[:,2]*scale,256,int(now),500,1000)
    off=surface[0]*255>45;on=surface[1]*255>45
    rgb=np.zeros((256,256,3),np.uint8);rgb[off,2]=255;rgb[on,:2]=255
    return ((off|on)*255).astype(np.uint8),rgb


def warp_to_reference(events,times,geometry,now):
    t=events['timestamp'].astype(np.int64)
    centers=np.column_stack([np.interp(t,times,geometry[:,k]) for k in [0,1]])
    radius=np.interp(t,times,geometry[:,2]);reference=np.array([np.interp(now,times,geometry[:,k]) for k in range(3)])
    xy=np.column_stack([events['x'],events['y']]).astype(float)
    return (xy-centers)*reference[2]/radius[:,None]+reference[:2],np.exp((t-now)/500.)


def smooth_circles(times,seed,results):
    """Smooth only consecutive accepted measurements, then recheck gate arcs."""
    proposed=np.array([r['circle'] for r in results]);active=np.flatnonzero([r['accepted'] for r in results])
    cfg=SmoothingConfig(half_window_us=1500,noise_floor_px=.15,maximum_correction_px=2.)
    for part in np.split(active,np.flatnonzero((np.diff(active)!=1)|(np.diff(times[active])>750))+1):
        if len(part)<7:continue
        correction=proposed[part]-seed[part]
        xy,_,_=local_cubic(times[part],correction[:,:2],config=cfg)
        rr,_,_=local_cubic(times[part],np.c_[correction[:,2],np.zeros(len(part))],config=cfg)
        proposed[part,:2]=seed[part,:2]+xy;proposed[part,2]=seed[part,2]+rr[:,0]
    final=seed.copy()
    for i,r in enumerate(results):
        if not r['accepted']:continue
        p=np.array(r['edge_points']);ids=np.array(r['edge_sector_ids']);valid=ids%3==1;audit=ids%3==2
        c=proposed[i];v=abs(np.linalg.norm(p[valid]-c[:2],axis=1)-c[2])
        keep=(r['validation_before_median_px']-np.median(v)>=.10 and np.mean(v<=1.5)>=.60
              and np.mean(v<=1.5)+.03>=r['validation_before_inlier_fraction'])
        if np.linalg.norm(c[:2]-seed[i,:2])>seed[i,2]*CircleConfig().maximum_center_fraction:keep=False
        r['raw_proposed_circle']=r['proposed_circle'];r['smoothed_proposed_circle']=c.tolist()
        r['final_accepted']=bool(keep)
        r['final_validation_median_px']=float(np.median(v))
        r['final_audit_median_px']=float(np.median(abs(np.linalg.norm(p[audit]-c[:2],axis=1)-c[2])))
        if keep:final[i]=c
    return final


def continuous_circles(times,seed,results,bridge=True):
    """Fit a latent correction from independent arc-qualified measurements.

    A circle already agreeing with the old seed still informs this model. It is
    not discarded just because it has little improvement over that seed. Fit
    quadratic corrections over +/-3 ms with robust weights; unsupported regime
    changes are exported as explicit spin-segment boundaries.
    """
    cfg=CircleConfig();measurements=[];quality=[]
    for i,r in enumerate(results):
        if r.get('proposed_circle') is None:continue
        c=np.array(r['proposed_circle']);margin=np.array([seed[i,2]*cfg.maximum_center_fraction]*2+[seed[i,2]*cfg.maximum_radius_fraction])
        valid=(r.get('validation_after_inlier_fraction',0)>=.60 and r.get('validation_after_median_px',99)<=1.25
               and r.get('independent_arc_disagreement_px',99)<=1.5 and r.get('condition',99)<=cfg.maximum_condition
               and np.all(abs(c-seed[i])<margin-.05))
        if valid:
            measurements.append(i);quality.append(1/(.25+r['validation_after_median_px']**2))
    ids=np.array(measurements,int);quality=np.array(quality,float);final=seed.copy()
    accepted=np.zeros(len(times),bool)
    for i,r in enumerate(results):
        r['final_accepted']=False;r['geometry_continuity_method']='robust_local_quadratic_correction'
        r['geometry_rotation_usable']=True;r['geometry_current_arcs_supported']=False
        near=ids[abs(times[ids]-times[i])<=3000]
        if len(near)<5 or times[near[-1]]-times[near[0]]<2000 or not times[near[0]]<=times[i]<=times[near[-1]]:
            r['continuous_status']='insufficient_temporal_geometry_support';continue
        if 'edge_points' not in r and not bridge:
            r['continuous_status']='missing_current_boundary_support';continue
        x=(times[near]-times[i])/3000.;design=np.c_[np.ones(len(near)),x,x*x]
        target=np.array([results[j]['proposed_circle'] for j in near])-seed[near]
        base=quality[np.searchsorted(ids,near)]*np.exp(-.5*x*x);weight=base.copy()
        for _ in range(6):
            a=design*np.sqrt(weight)[:,None];co=np.linalg.lstsq(a,target*np.sqrt(weight)[:,None],rcond=None)[0]
            error=np.linalg.norm(target-design@co,axis=1)
            robust=np.minimum(1,.6/np.maximum(error,1e-12));weight=base*robust
        c=seed[i]+co[0]
        bounded=(np.linalg.norm(c[:2]-seed[i,:2])<seed[i,2]*cfg.maximum_center_fraction
                 and abs(c[2]-seed[i,2])<seed[i,2]*cfg.maximum_radius_fraction)
        supported=False;contradicted=False
        if 'edge_points' in r:
            points=np.array(r['edge_points']);sectors=np.array(r['edge_sector_ids']);gate=sectors%3==1;audit=sectors%3==2
            residual=abs(np.linalg.norm(points[gate]-c[:2],axis=1)-c[2])
            supported=np.median(residual)<=1.25 and np.mean(residual<=cfg.inlier_px)>=.60
            contradicted=np.median(residual)>2.5 or np.mean(residual<=cfg.inlier_px)<.25
            r['final_validation_median_px']=float(np.median(residual))
            r['final_audit_median_px']=float(np.median(abs(np.linalg.norm(points[audit]-c[:2],axis=1)-c[2])))
        use=bounded and (bridge or supported)
        r['continuous_status']=('current_arcs_supported' if supported else 'short_gap_temporal_geometry') if use else 'geometry_model_rejected'
        r['geometry_current_arcs_supported']=bool(supported)
        r['geometry_rotation_usable']=not bool(contradicted and use)
        r['smoothed_proposed_circle']=c.tolist();r['final_accepted']=bool(use)
        if use:final[i]=c;accepted[i]=True
    regime=np.r_[0,np.cumsum(accepted[1:]!=accepted[:-1])]
    return final,regime


def transform_row(row,old,new,scale):
    """Original crop -> sensor at frame time -> rebuilt crop; no guessed FPS."""
    row=dict(row);factor=old[2]/new[2]
    shift=(old[:2]-new[:2])*128/(new[2]*scale)
    for x,y in [('center_x_px','center_y_px'),('x1','y1'),('x2','y2'),('raw_center_x_px','raw_center_y_px')]:
        if x not in row:continue
        p=(np.array([float(row[x]),float(row[y])])-128)*factor+128+shift
        row[x],row[y]=map(float,p)
    row['position_estimate_method']+=' + exact frame-time crop coordinate transform'
    return row


def eligible_rows(records):
    """Keep observations, but split spin exports at outside-sphere positions."""
    groups=defaultdict(list);kept=[];dropped=Counter()
    for r in records:groups[r['segment_id']].append(r)
    for segment,run in groups.items():
        sphere=np.array([np.linalg.norm([float(r['center_x_px'])-128,float(r['center_y_px'])-128])<128/1.2 for r in run])
        consistent=np.array([str(r.get('geometry_window_consistent','true')).lower()=='true' for r in run])
        supported=np.array([str(r.get('geometry_rotation_usable','true')).lower()=='true' for r in run])
        valid=sphere&consistent&supported
        idx=np.flatnonzero(valid)
        dropped['outside_sphere']+=int((~sphere).sum())
        dropped['mixed_geometry_window']+=int((sphere&~consistent).sum())
        dropped['contradicted_geometry']+=int((sphere&consistent&~supported).sum())
        regimes=np.array([str(r.get('geometry_continuity_id','original')) for r in run])
        breaks=(np.diff(idx)!=1)|(regimes[idx[1:]]!=regimes[idx[:-1]])
        for j,part in enumerate(np.split(idx,np.flatnonzero(breaks)+1)):
            if len(part)<3:dropped['short_after_geometry_split']+=len(part);continue
            name=segment if len(part)==len(run) else segment+f'/R{j:03d}'
            for offset,i in enumerate(part):kept.append({**run[i],'segment_id':name,'segment_frame_offset':offset,'segment_length':len(part)})
    return kept,dict(dropped)


def localize_group(records,folder):
    by_segment=defaultdict(list)
    for r in records:by_segment[r['segment_id']].append(r)
    images={};output=[];diagnostics=[]
    def image(row):
        name=Path(row['image']).name
        if name not in images:
            rgb=cv2.cvtColor(cv2.imread(str(folder/'polarity_reference'/name)),cv2.COLOR_BGR2RGB)
            images[name]=registration_image(rgb)
        return images[name]
    cfg=LocalizationConfig()
    for segment,run in by_segment.items():
        boxes=[np.array([float(r[k]) for k in ['x1','y1','x2','y2']]) for r in run]
        features=[len(marker_features(image(r),box)) for r,box in zip(run,boxes)]
        anchor=max(range(len(run)),key=lambda i:(features[i]>=cfg.minimum_features,
                         min(features[i],30),-abs(i-(len(run)-1)/2)))
        reference=run[anchor];center=np.array([float(reference[k]) for k in ['center_x_px','center_y_px']])
        for i,r in enumerate(run):
            point=np.array([float(r[k]) for k in ['center_x_px','center_y_px']])
            if i==anchor:
                fit={'accepted':False,'status':'fixed_reference_center','center':point.tolist(),'features':features[i]}
            else:fit=register_marker(image(reference),image(r),boxes[anchor],center,point,cfg)
            p=np.array(fit['center'])
            if np.linalg.norm(p-128)>=128/1.2:
                fit.update(accepted=False,status='outside_visible_sphere',center=point.tolist());p=point
            result={**r,'localization_baseline_center_x_px':float(point[0]),'localization_baseline_center_y_px':float(point[1]),
                    'center_x_px':float(p[0]),'center_y_px':float(p[1]),
                    'center_registration_applied':fit['accepted'],'center_registration_status':fit['status'],
                    'center_registration_correction_px':float(np.linalg.norm(p-point)),
                    'center_registration_anchor_frame':int(reference['frame_index']),
                    'position_estimate_method':r['position_estimate_method']+(' + anchor-affine material-point candidate' if fit['accepted'] else '')}
            output.append(result)
            diagnostics.append({'camera':r['camera'],'group':r['group'],'segment_id':segment,'frame_index':int(r['frame_index']),
                                'timestamp_us':int(r['timestamp_us']),'track_support':r['track_support'],
                                'anchor_frame_index':int(reference['frame_index']),**fit})
    return output,diagnostics


def finalize_exports(output,dataset):
    root=Path(__file__).resolve().parents[1];smoothing=load(root/'config/marker_trajectory_smoothing.json')
    variants=[]
    for mode in ['marker_only','geometry_only','combined']:
        if not (output/'tracks'/f'{mode}_all_observations.csv').is_file():continue
        records=rows(output/'tracks'/f'{mode}_all_observations.csv')
        data=dataset if mode=='marker_only' else output/'dataset'
        if mode!='marker_only':
            quality={}
            for path in (data/'unlabeled_inference').glob('*/*/manifest.csv'):
                manifest=rows(path);regime=np.array([r['geometry_continuity_id'] for r in manifest])
                for i,r in enumerate(manifest):
                    r['geometry_window_consistent']=bool(i<2 or regime[i]==regime[max(0,i-2)])
                    quality[(path.parent.name,int(r['frame_index']))]=(r['geometry_window_consistent'],r.get('geometry_rotation_usable','true'))
                write_csv(path,manifest)
            for r in records:r['geometry_window_consistent'],r['geometry_rotation_usable']=quality[(r['group'],int(r['frame_index']))]
            write_csv(output/'tracks'/f'{mode}_all_observations.csv',records)
        eligible,dropped=eligible_rows(records)
        if not eligible:raise ValueError(f'{mode}: no eligible track segments')
        path=output/'tracks'/f'{mode}_eligible.csv';write_csv(path,eligible)
        segments,_,count=read_tracks(path,data,smoothing_config=smoothing if mode!='geometry_only' else None)
        path=output/'tracks'/f'{mode}_effective.csv';write_csv(path,csv_rows(segments))
        variants.append({'mode':mode,'dataset':str(data),'tracks':str(path),'observations':count,'dropped':dropped,
                         'effective_tracks_sha256':sha256(path)})
    dump(output/'variants.json',variants)
    return variants


def build(dataset,cache,output,groups=None,measurements_from=None,geometry_model='continuous_bridge',include_marker_only=False):
    dataset,cache,output=map(lambda p:Path(p).resolve(),[dataset,cache,output])
    output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    read_tracks(root/'output/latest/tracking/rotation_tracks_all.csv',dataset)
    read_tracks(root/'output/latest_smoothing/smoothed_rotation_tracks.csv',dataset)
    for sub in ['dataset/unlabeled_inference','tracks','geometry','localization']:(output/sub).mkdir(parents=True)
    (output/'dataset/reference_videos').symlink_to(dataset/'reference_videos',target_is_directory=True)
    for camera,filename in [('F3E','master_DSP01F1FFF3E.raw'),('F59','slave_DSP01F1FFF59.raw')]:
        expected=[s for s in load_serves(dataset) if not groups or f"{s['serve']}_{camera}" in groups]
        if any(not (cache/f"{s['serve']}_{camera}.npy").is_file() for s in expected):
            from pingpong_event_pipeline.raw_sync import extract_camera
            print(camera,'rebuilding absent RAW range caches with independent EVT3/SDK agreement checks',flush=True)
            audit=extract_camera(root/'data'/filename,dataset,output/'raw_audit',cache,camera)
            if not audit['all_selected_windows_sdk_reference_agree']:raise ValueError('RAW independent decoder / SDK disagreement')
    profile=load(root/'config/marker_enhancement_v1.json')['cameras']
    raw=rows(root/'output/latest/tracking/rotation_tracks_all.csv')
    effective=rows(root/'output/latest_smoothing/smoothed_rotation_tracks.csv')
    variant_rows=defaultdict(list);summary=[]
    for spec in load_serves(dataset):
        for camera in ['F3E','F59']:
            name=f"{spec['serve']}_{camera}"
            if groups and name not in groups:continue
            folder=Path(spec['cameras'][camera]['folder']);manifest=spec['cameras'][camera]['rows'];meta=spec['cameras'][camera]['metadata']
            if not np.isclose(meta['crop_radius_scale'],1.2):raise ValueError('current export expects crop_radius_scale=1.2')
            times=np.array([int(r['timestamp_us']) for r in manifest],np.int64)
            if np.any(np.diff(times)!=500):raise ValueError('this experiment requires contiguous 500 us source crops')
            original=np.array([[float(r[k]) for k in ['center_x_px','center_y_px','radius_px']] for r in manifest])
            source=IndexedEvents(np.load(cache/f"{spec['serve']}_{camera}.npy",mmap_mode='r'))
            results=[];reconstruction=[]
            reused=None
            if measurements_from:
                reused=load(Path(measurements_from)/'geometry'/f'{name}.json')
                assert reused['source_manifest_sha256']==sha256(folder/'manifest.csv')
                np.testing.assert_array_equal(reused['original_circles'],original)
                for r in reused['frames']:
                    if 'config' in r:
                        assert r['config']==asdict(CircleConfig()),'circle measurement parameters changed; rerun RAW fits'
            for i,(t,seed) in enumerate(zip(times,original)):
                events=source.window(t-1000,t)
                if i<2:
                    results.append({'accepted':False,'circle':seed.tolist(),'status':'original_path_context_unavailable','sectors':0});continue
                binary,_=render_events(events,times,original,t,meta['crop_radius_scale'])
                stored=cv2.imread(str(folder/'binary_unfiltered'/Path(manifest[i]['image']).name),cv2.IMREAD_GRAYSCALE)
                union=np.count_nonzero((binary!=0)|(stored!=0));intersection=np.count_nonzero((binary!=0)&(stored!=0))
                reconstruction.append(intersection/max(1,union))
                if reused:results.append(reused['frames'][i])
                else:
                    xy,weight=warp_to_reference(events,times,original,t)
                    results.append(fit_boundary_circle(xy,seed,weight))
            if np.median(reconstruction)<.999:raise ValueError(f'{name}: source RAW crop reconstruction mismatch {np.median(reconstruction)}')
            if geometry_model=='framewise':
                geometry=smooth_circles(times,original,results);regime=np.zeros(len(times),int)
            else:geometry,regime=continuous_circles(times,original,results,bridge=geometry_model=='continuous_bridge')
            target=output/'dataset/unlabeled_inference'/camera/name
            for sub in ['binary','binary_unfiltered','polarity_reference']:(target/sub).mkdir(parents=True,exist_ok=True)
            new_manifest=[]
            for i,(t,row) in enumerate(zip(times,manifest)):
                filename=Path(row['image']).name
                if i<2:
                    for sub in ['binary','binary_unfiltered','polarity_reference']:shutil.copy2(folder/sub/filename,target/sub/filename)
                else:
                    binary,rgb=render_events(source.window(t-1000,t),times,geometry,t,meta['crop_radius_scale'])
                    enhanced=enhance_for_camera(binary,camera,profile)
                    assert cv2.imwrite(str(target/'binary_unfiltered'/filename),binary)
                    assert cv2.imwrite(str(target/'binary'/filename),enhanced)
                    assert cv2.imwrite(str(target/'polarity_reference'/filename),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                new_manifest.append({**row,**dict(zip(['center_x_px','center_y_px','radius_px'],map(float,geometry[i]))),
                    'geometry_refinement_applied':bool(results[i].get('final_accepted',False)),
                    'geometry_continuity_id':int(regime[i]),
                    'geometry_rotation_usable':bool(results[i].get('geometry_rotation_usable',True)),
                    'geometry_current_arcs_supported':bool(results[i].get('geometry_current_arcs_supported',False))})
            write_csv(target/'manifest.csv',new_manifest)
            dump(target/'metadata.json',{**meta,'geometry_refinement':'RAW angular-sector circle candidate with gated robust continuous quadratic correction',
                'source_manifest_sha256':sha256(folder/'manifest.csv'),'geometry_physically_validated':False,
                'warning':'Rebuilt unannotated candidate crops. Original frame times, source identities and support flags retained; no labels copied.'})
            dump(output/'geometry'/f'{name}.json',{'camera':camera,'group':name,'timestamps_us':times.tolist(),
                'source_manifest_sha256':sha256(folder/'manifest.csv'),'original_circles':original.tolist(),
                'effective_circles':geometry.tolist(),'frames':results,'source_reconstruction_ious':reconstruction})
            selected=[r for r in raw if r['group']==name]
            marker=[]
            if include_marker_only:
                marker,diagnostic=localize_group(selected,folder)
                variant_rows['marker_only'].extend(marker)
                dump(output/'localization'/f'{name}_marker_only.json',diagnostic)
            transformed=[transform_row(r,original[int(r['frame_index'])],geometry[int(r['frame_index'])],meta['crop_radius_scale']) for r in selected]
            for r in transformed:r['geometry_continuity_id']=int(regime[int(r['frame_index'])])
            combined,diag=localize_group(transformed,target)
            variant_rows['combined'].extend(combined)
            dump(output/'localization'/f'{name}_combined.json',diag)
            for r in effective:
                if r['group']==name:
                    i=int(r['frame_index']);variant_rows['geometry_only'].append({**transform_row(r,original[i],geometry[i],meta['crop_radius_scale']),
                                                                                 'geometry_continuity_id':int(regime[i])})
            measured=sum(bool(r.get('final_accepted',False)) for r in results)
            summary.append({'camera':camera,'group':name,'frames':len(manifest),'refined_geometry_frames':measured,
                'source_reconstruction_median_iou':float(np.median(reconstruction)),
                'median_center_change_sensor_px':float(np.median(np.linalg.norm(geometry[:,:2]-original[:,:2],axis=1))),
                'maximum_center_change_sensor_px':float(np.max(np.linalg.norm(geometry[:,:2]-original[:,:2],axis=1))),
                'median_radius_change_sensor_px':float(np.median(geometry[:,2]-original[:,2])),
                **({'marker_only_refined_points':sum(r['center_registration_applied'] for r in marker)} if include_marker_only else {}),
                'combined_refined_points':sum(r['center_registration_applied'] for r in combined)})
            dump(output/'group_summary.json',summary)
            print(name,summary[-1],flush=True)
    for mode,records in variant_rows.items():
        write_csv(output/'tracks'/f'{mode}_all_observations.csv',records)
    finalize_exports(output,dataset)
    dump(output/'provenance.json',{'source_dataset':str(dataset),'source_tracks_sha256':sha256(root/'output/latest/tracking/rotation_tracks_all.csv'),
        'source_effective_tracks_sha256':sha256(root/'output/latest_smoothing/smoothed_rotation_tracks.csv'),
        'reused_raw_circle_measurements_from':str(Path(measurements_from).resolve()) if measurements_from else None,
        'geometry_model':geometry_model,
        'circle_config':asdict(CircleConfig()),'marker_localization_config':asdict(LocalizationConfig()),
        'temporal_geometry_model':{'half_window_us':3000,'minimum_measurements':5,'minimum_measurement_span_us':2000,
                                   'degree':2,'iterations':6,'huber_residual_scale_sensor_px':.6,'extrapolation':False},
        'marker_anchor':'fixed reference bbox midpoint transported by local affine image registration; not a labeled physical dot center',
        'timestamps_changed':False,'original_dataset_modified':False,'production_models_changed':False,'physical_accuracy_validated':False})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('data/server_marker_two_camera_train_v1'))
    p.add_argument('--cache',type=Path,default=Path('/tmp/pingpong_raw_sync_cache_v9'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--groups',nargs='*')
    p.add_argument('--measurements-from',type=Path,help='Reuse RAW circle fits after source-manifest/seed checks; rechecks crop reconstruction')
    p.add_argument('--geometry-model',choices=['continuous_bridge','continuous_strict','framewise'],default='continuous_bridge')
    p.add_argument('--include-marker-only',action='store_true',help='Optional development ablation; omitted from the retained three-result review')
    a=p.parse_args();build(a.dataset,a.cache,a.output,a.groups,a.measurements_from,a.geometry_model,a.include_marker_only)
