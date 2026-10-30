"""Select complete contiguous evidence-backed track runs for later rotation analysis.

This is offline export filtering, not additional tracking or a rotation estimator.
Prediction-only holds split segments. Camera/group/track ID together define identity.
"""
from collections import Counter, defaultdict
from copy import deepcopy

from common import dump, table

OBSERVED_SOURCES=frozenset(('detection_high','detection_low','optical_flow'))

ROTATION_FIELDS=['camera','group','segment_id','track_id','segment_frame_offset','segment_length',
                 'image','frame_index','timestamp_us','center_x_px','center_y_px',
                 'x1','y1','x2','y2','source','detection_confirmed_this_frame',
                 'last_detection_confidence','position_estimate_method','track_support']


def select_consecutive_segments(frames,min_frames=3,max_gap_us=750):
    if not isinstance(min_frames,int) or isinstance(min_frames,bool) or min_frames<3:
        raise ValueError('Rotation exports require at least 3 consecutive supported frames')
    if max_gap_us<=0:
        raise ValueError('Consecutive frame time gap must be positive')
    output=deepcopy(frames)
    observations=defaultdict(list)
    counts=Counter(raw_track_samples=0,prediction_only_samples_excluded=0,unsupported_source_samples_excluded=0,
                   observed_samples=0,discarded_short_segments=0,discarded_short_samples=0,
                   retained_segments=0,retained_samples=0)
    for i,frame in enumerate(frames):
        frame_index=int(frame['frame_index']);timestamp=int(frame['timestamp_us'])
        seen=set()
        for track in frame['tracks']:
            track_id=int(track['track_id'])
            if track_id in seen:
                raise ValueError('Duplicate track ID in one frame')
            seen.add(track_id)
            counts['raw_track_samples']+=1
            source=track['source']
            if source not in OBSERVED_SOURCES:
                key='prediction_only_samples_excluded' if source=='prediction_hold' else 'unsupported_source_samples_excluded'
                counts[key]+=1
                continue
            identity=(frame.get('camera',''),frame.get('group',''),track_id)
            observations[identity].append((i,frame_index,timestamp,track))
            counts['observed_samples']+=1
        output[i]['tracks']=[]
        output[i]['raw_track_count']=len(frame['tracks'])
        output[i]['rotation_filter_applied']=True
    segments=[];rejected=[]
    for identity,items in observations.items():
        runs=[];run=[]
        for item in items:
            if run and (item[0]!=run[-1][0]+1 or item[1]!=run[-1][1]+1 or
                        not 0<item[2]-run[-1][2]<=max_gap_us):
                runs.append(run);run=[]
            run.append(item)
        if run:
            runs.append(run)
        camera,group,track_id=identity
        for ordinal,run in enumerate(runs,1):
            segment_id=f'{camera}/{group}/T{track_id:04d}/S{ordinal:03d}'
            start=run[0];end=run[-1]
            record=dict(camera=camera,group=group,track_id=track_id,segment_id=segment_id,
                        segment_length=len(run),first_frame_index=start[1],last_frame_index=end[1],
                        first_timestamp_us=start[2],last_timestamp_us=end[2],duration_us=end[2]-start[2],
                        sources=dict(Counter(item[3]['source'] for item in run)),
                        confirmation_timestamp_us=run[min_frames-1][2] if len(run)>=min_frames else None,
                        position_estimate_method='Kalman-filtered centers updated by detector or verified optical flow')
            if len(run)<min_frames:
                counts['discarded_short_segments']+=1;counts['discarded_short_samples']+=len(run)
                rejected.append(dict(**record,reason='fewer_than_minimum_consecutive_observed_frames'))
                continue
            counts['retained_segments']+=1;counts['retained_samples']+=len(run)
            segments.append(record)
            trail=[]
            for offset,(i,_,_,track) in enumerate(run):
                retained=deepcopy(track)
                center=retained.get('center_pixels')
                if center is None:
                    b=retained['xyxy_pixels'];center=[(b[0]+b[2])/2,(b[1]+b[3])/2]
                trail.append(center)
                retained.update(segment_id=segment_id,segment_length=len(run),segment_frame_offset=offset,
                                rotation_eligible=True,confirmation_timestamp_us=record['confirmation_timestamp_us'],
                                trail=deepcopy(trail[-12:]))
                output[i]['tracks'].append(retained)
    segments.sort(key=lambda s:(s['camera'],s['group'],s['first_timestamp_us'],s['track_id']))
    rejected.sort(key=lambda s:(s['camera'],s['group'],s['first_timestamp_us'],s['track_id']))
    for frame in output:
        frame['tracks'].sort(key=lambda t:t['track_id'])
    counts['discarded_samples']=counts['raw_track_samples']-counts['retained_samples']
    assert counts['discarded_samples']==counts['prediction_only_samples_excluded']+counts['unsupported_source_samples_excluded']+counts['discarded_short_samples']
    summary=dict(**counts,minimum_consecutive_frames=min_frames,max_consecutive_frame_gap_us=max_gap_us,
                 observed_sources=sorted(OBSERVED_SOURCES),prediction_holds_count_as_observations=False,
                 selection_mode='offline complete-segment selection; retained initial points preserve original timestamps',
                 tracker_identity_changes=False,rotation_speed_computed=False)
    return output,segments,rejected,summary


def rotation_rows(frames):
    records=[]
    for frame in frames:
        for t in frame['tracks']:
            b=t['xyxy_pixels'];center=t.get('center_pixels',[(b[0]+b[2])/2,(b[1]+b[3])/2])
            records.append(dict(camera=frame.get('camera',''),group=frame.get('group',''),segment_id=t['segment_id'],
                                track_id=t['track_id'],segment_frame_offset=t['segment_frame_offset'],segment_length=t['segment_length'],
                                image=frame['image'],frame_index=frame['frame_index'],timestamp_us=frame['timestamp_us'],
                                center_x_px=center[0],center_y_px=center[1],x1=b[0],y1=b[1],x2=b[2],y2=b[3],source=t['source'],
                                detection_confirmed_this_frame=t['detection_confirmed_this_frame'],
                                last_detection_confidence=t['last_detection_confidence'],
                                position_estimate_method='Kalman-filtered from detector/verified optical flow',
                                track_support=frame.get('track_support','unknown')))
    return records


def write_rotation_exports(directory,frames,segments,rejected,summary):
    directory.mkdir(parents=True,exist_ok=True)
    dump(directory/'rotation_segments.json',segments)
    dump(directory/'rejected_segments.json',rejected)
    dump(directory/'rotation_filter_summary.json',summary)
    table(directory/'rotation_tracks.csv',rotation_rows(frames),ROTATION_FIELDS)
