"""Causal marker tracking: high-confidence births, gated association, LK support, bounded coasting.

Uses the high/low score association idea from ByteTrack, not its official implementation.
All positions refer to the unchanged 256x256 crop. Prediction scores are not detection probabilities.
"""
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackerConfig:
    birth_confidence: float = 0.75
    continuation_confidence: float = 0.25
    confirmation_hits: int = 2
    max_unobserved_frames: int = 3
    max_visual_only_frames: int = 40
    max_time_gap_us: int = 5000
    nominal_step_us: int = 500
    edge_margin_pixels: float = 12.0
    minimum_box_inside_fraction: float = 0.65
    duplicate_nms_iou: float = 0.50
    minimum_event_pixels: int = 8
    max_center_distance: float = 18.0
    low_score_appearance_min: float = 0.35
    flow_appearance_min: float = 0.35
    flow_forward_backward_max: float = 1.0
    flow_min_points: int = 3
    minimum_consecutive_frames: int = 3
    consecutive_max_gap_us: int = 750

    def __post_init__(self):
        if not 0 < self.continuation_confidence <= self.birth_confidence <= 1:
            raise ValueError('Require 0 < continuation <= birth <= 1')
        if self.confirmation_hits < 1 or self.max_unobserved_frames < 0 or self.max_visual_only_frames < 1:
            raise ValueError('Invalid track lifecycle settings')
        if self.edge_margin_pixels < 0 or not 0 < self.minimum_box_inside_fraction <= 1:
            raise ValueError('Invalid ball edge settings')
        if self.minimum_consecutive_frames<3 or self.consecutive_max_gap_us<=0:
            raise ValueError('Rotation output requires >=3 consecutive evidence-backed frames and a positive time-gap limit')


@dataclass
class BallGeometry:
    center_x: float = 128.0
    center_y: float = 128.0
    radius: float = 128.0 / 1.2
    source: str = 'crop_radius_scale_geometry_unverified'

    def __post_init__(self):
        if not np.isfinite([self.center_x,self.center_y,self.radius]).all() or self.radius<=0:
            raise ValueError('Ball geometry must be finite with positive radius')

    def gate(self, box, config):
        inner = self.radius - config.edge_margin_pixels
        b = np.asarray(box, dtype=float)
        center = (b[:2] + b[2:]) / 2
        radial = float(np.linalg.norm(center - [self.center_x, self.center_y]))
        if inner <= 0 or radial > inner:
            return False, dict(reason='box_center_near_ball_edge', radial_distance=radial, safe_radius=inner)
        xs = np.linspace(b[0], b[2], 7)
        ys = np.linspace(b[1], b[3], 7)
        xx, yy = np.meshgrid(xs, ys)
        fraction = float(np.mean((xx-self.center_x)**2 + (yy-self.center_y)**2 <= inner**2))
        return fraction >= config.minimum_box_inside_fraction, dict(
            reason='box_overlaps_ball_edge' if fraction < config.minimum_box_inside_fraction else 'inside',
            radial_distance=radial, safe_radius=inner, box_inside_fraction=fraction)


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    area = np.maximum(np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]), 0).prod()
    return float(area / (np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-area+1e-9))


def bounds(box, shape):
    h, w = shape[:2]
    x1,y1 = np.floor(np.maximum(box[:2],0)).astype(int)
    x2,y2 = np.ceil(np.minimum(box[2:],[w,h])).astype(int)
    return x1,y1,x2,y2


def appearance(image, box):
    x1,y1,x2,y2 = bounds(box,image.shape)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((24,24),np.float32), 0
    patch = image[y1:y2,x1:x2]
    pixels = int(np.count_nonzero(patch))
    patch = cv2.GaussianBlur(patch.astype(np.float32)/255, (0,0), 1.0)
    return cv2.resize(patch,(24,24),interpolation=cv2.INTER_AREA), pixels


def similarity(a,b):
    x,y = a.flatten().astype(float),b.flatten().astype(float)
    x-=x.mean(); y-=y.mean()
    denominator = np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/denominator) if denominator > 1e-9 else 0.0


@dataclass
class Track:
    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    size: np.ndarray
    template: np.ndarray
    event_pixels: float
    last_detection_confidence: float
    hits: int = 1
    high_hits: int = 1
    age: int = 1
    missed: int = 0
    since_detection: int = 0
    confirmed: bool = False
    source: str = 'tentative'
    history: list = field(default_factory=list)

    def box(self):
        return np.r_[self.mean[:2]-self.size/2,self.mean[:2]+self.size/2]

    def predict(self,dt):
        f=np.eye(4); f[0,2]=f[1,3]=dt
        self.mean=f@self.mean
        self.covariance=f@self.covariance@f.T+np.diag([0.5*dt,0.5*dt,0.8*dt,0.8*dt])
        self.age+=1

    def correct(self,center,noise=2.0):
        h=np.zeros((2,4)); h[:2,:2]=np.eye(2)
        r=np.eye(2)*noise**2
        innovation=np.asarray(center)-h@self.mean
        s=h@self.covariance@h.T+r
        k=np.linalg.solve(s,(self.covariance@h.T).T).T
        self.mean+=k@innovation
        # Joseph form keeps the covariance positive semidefinite over long sequences.
        ident=np.eye(4)-k@h
        self.covariance=ident@self.covariance@ident.T+k@r@k.T


class MarkerTracker:
    def __init__(self,config=None,geometry=None):
        self.config=config or TrackerConfig()
        self.geometry=geometry or BallGeometry()
        self.tracks=[]
        self.previous_image=None
        self.previous_timestamp=None
        self.next_id=1
        self.events=[]
        self.terminated=[]

    def _end(self,track,reason,timestamp):
        record=dict(track_id=track.track_id,reason=reason,timestamp_us=timestamp,
                    age_frames=track.age,detector_hits=track.hits,confirmed=track.confirmed)
        self.events.append(dict(event='ended',**record))
        self.terminated.append(record)

    def _flow(self,track,previous_box,image):
        if self.previous_image is None:
            return None
        mask=np.zeros(image.shape,np.uint8)
        x1,y1,x2,y2=bounds(previous_box,image.shape)
        if x2-x1<4 or y2-y1<4:
            return None
        mask[y1:y2,x1:x2]=255
        previous=cv2.GaussianBlur(self.previous_image,(3,3),0.7)
        current=cv2.GaussianBlur(image,(3,3),0.7)
        p0=cv2.goodFeaturesToTrack(previous,mask=mask,maxCorners=30,qualityLevel=0.01,minDistance=2,blockSize=3)
        if p0 is None or len(p0)<self.config.flow_min_points:
            return None
        options=dict(winSize=(15,15),maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,0.01))
        p1,s1,_=cv2.calcOpticalFlowPyrLK(previous,current,p0,None,**options)
        if p1 is None:
            return None
        back,s2,_=cv2.calcOpticalFlowPyrLK(current,previous,p1,None,**options)
        if back is None:
            return None
        good=(s1.ravel()>0)&(s2.ravel()>0)&np.isfinite(p1).all(axis=(1,2))
        good &= np.linalg.norm(back-p0,axis=2).ravel()<self.config.flow_forward_backward_max
        offsets=(p1-p0).reshape(-1,2)[good]
        if len(offsets)<self.config.flow_min_points:
            return None
        median=np.median(offsets,axis=0)
        deviations=np.linalg.norm(offsets-median,axis=1)
        offsets=offsets[deviations<=max(1.0,3*float(np.median(deviations)))]
        if len(offsets)<self.config.flow_min_points:
            return None
        shift=np.median(offsets,axis=0)
        if np.linalg.norm(shift)>self.config.max_center_distance:
            return None
        flow_box=previous_box+np.r_[shift,shift]
        patch,pixels=appearance(image,flow_box)
        corr=similarity(track.template,patch)
        valid,_=self.geometry.gate(flow_box,self.config)
        if not valid or pixels<max(self.config.minimum_event_pixels,0.20*track.event_pixels):
            return None
        if corr<self.config.flow_appearance_min:
            return None
        center=(flow_box[:2]+flow_box[2:])/2
        if np.linalg.norm(center-track.mean[:2])>self.config.max_center_distance:
            return None
        return dict(center=center,correlation=corr,points=len(offsets))

    def _associate(self,tracks,detections,flow,low=False):
        if not tracks or not detections:
            return []
        cost=np.full((len(tracks),len(detections)),1e6,dtype=float)
        for i,t in enumerate(tracks):
            predicted=flow[t.track_id]['center'] if t.track_id in flow else t.mean[:2]
            gate=min(self.config.max_center_distance,max(7.0,float(np.linalg.norm(t.size)*0.65)))
            for j,d in enumerate(detections):
                b=d['box']; center=(b[:2]+b[2:])/2
                distance=float(np.linalg.norm(center-predicted))
                size=b[2:]-b[:2]
                ratio=np.maximum(size/t.size,t.size/size).max()
                corr=similarity(t.template,d['patch'])
                if distance>gate or ratio>2.0 or corr<(self.config.low_score_appearance_min if low else 0.05):
                    continue
                innovation=center-t.mean[:2]
                mahal=float(innovation@np.linalg.solve(t.covariance[:2,:2]+np.eye(2)*9,innovation))
                if mahal>16 and t.track_id not in flow:
                    continue
                cost[i,j]=0.45*distance/gate+0.25*(1-iou(t.box(),b))+0.30*(1-corr)/2
        ii,jj=linear_sum_assignment(cost)
        return [(tracks[i],detections[j]) for i,j in zip(ii,jj) if cost[i,j] < (0.72 if low else 0.80)]

    def update(self,image,detections,timestamp_us):
        image=np.asarray(image)
        if image.shape!=(256,256) or image.dtype!=np.uint8:
            raise ValueError('Expected uint8 single-channel 256x256 crop')
        timestamp_us=int(timestamp_us)
        if self.previous_timestamp is not None and timestamp_us<=self.previous_timestamp:
            raise ValueError('Timestamps must strictly increase within each sequence')
        reset=False
        if self.previous_timestamp is not None and timestamp_us-self.previous_timestamp>self.config.max_time_gap_us:
            for t in self.tracks:
                self._end(t,'timestamp_gap',timestamp_us)
            self.tracks=[]; self.previous_image=None; reset=True
        dt=1.0 if self.previous_timestamp is None or reset else (timestamp_us-self.previous_timestamp)/self.config.nominal_step_us
        rejected=[]; accepted=[]
        for index,d in enumerate(detections):
            score=float(d['confidence']); b=np.asarray(d['xyxy_pixels'],dtype=float)
            if score<self.config.continuation_confidence:
                continue
            if b.shape!=(4,) or not np.isfinite(b).all() or np.any(b[2:]<=b[:2]) or not np.isfinite(score):
                raise ValueError('Invalid detection')
            valid,edge=self.geometry.gate(b,self.config)
            if not valid:
                rejected.append(dict(index=index,confidence=score,box=b.tolist(),**edge)); continue
            patch,pixels=appearance(image,b)
            if pixels<self.config.minimum_event_pixels:
                rejected.append(dict(index=index,confidence=score,box=b.tolist(),reason='insufficient_event_support')); continue
            accepted.append(dict(index=index,score=score,box=b,patch=patch,pixels=pixels))
        accepted.sort(key=lambda d:-d['score'])
        nms=[]
        for d in accepted:
            if any(iou(d['box'],other['box'])>self.config.duplicate_nms_iou for other in nms):
                rejected.append(dict(index=d['index'],confidence=d['score'],box=d['box'].tolist(),reason='duplicate_nms'))
            else:
                nms.append(d)
        accepted=nms
        flow={}
        for t in self.tracks:
            previous_box=t.box().copy()
            t.predict(dt)
            f=self._flow(t,previous_box,image) if t.confirmed else None
            if f is not None:
                flow[t.track_id]=f
        high=[d for d in accepted if d['score']>=self.config.birth_confidence]
        low=[d for d in accepted if d['score']<self.config.birth_confidence]
        matches=self._associate(self.tracks,high,flow)
        matched_ids={t.track_id for t,d in matches}
        matches+=self._associate([t for t in self.tracks if t.track_id not in matched_ids and t.confirmed],low,flow,low=True)
        matched_ids={t.track_id for t,d in matches}
        used={d['index'] for t,d in matches}
        for t,d in matches:
            t.correct((d['box'][:2]+d['box'][2:])/2,noise=1.5 if d['score']>=self.config.birth_confidence else 3.0)
            t.size=0.65*t.size+0.35*(d['box'][2:]-d['box'][:2])
            alpha=0.10 if d['score']>=self.config.birth_confidence else 0.02
            t.template=(1-alpha)*t.template+alpha*d['patch']
            t.event_pixels=0.8*t.event_pixels+0.2*d['pixels']
            t.last_detection_confidence=d['score']; t.hits+=1
            t.high_hits+=d['score']>=self.config.birth_confidence
            t.missed=0; t.since_detection=0
            t.source='detection_high' if d['score']>=self.config.birth_confidence else 'detection_low'
            if not t.confirmed and t.high_hits>=self.config.confirmation_hits:
                t.confirmed=True
                self.events.append(dict(event='confirmed',track_id=t.track_id,timestamp_us=timestamp_us))
        survivors=[]
        for t in self.tracks:
            if t.track_id not in matched_ids:
                t.since_detection+=1
                if t.confirmed and t.track_id in flow:
                    t.correct(flow[t.track_id]['center'],noise=3.0)
                    t.missed=0; t.source='optical_flow'
                else:
                    t.missed+=1; t.source='prediction_hold'
            valid,_=self.geometry.gate(t.box(),self.config)
            reason=None
            if not valid:
                reason='ball_edge_exit'
            elif not t.confirmed and t.missed>0:
                reason='tentative_not_confirmed'
            elif t.missed>self.config.max_unobserved_frames:
                reason='cluster_evidence_lost'
            elif t.since_detection>self.config.max_visual_only_frames:
                reason='detector_absence_timeout'
            if reason:
                self._end(t,reason,timestamp_us)
            else:
                survivors.append(t)
        self.tracks=survivors
        for d in high:
            if d['index'] in used:
                continue
            center=(d['box'][:2]+d['box'][2:])/2
            if any(iou(t.box(),d['box'])>0.35 or np.linalg.norm(t.mean[:2]-center)<0.35*min(t.size) for t in self.tracks):
                continue
            t=Track(self.next_id,np.r_[center,0.,0.],np.diag([4.,4.,25.,25.]),d['box'][2:]-d['box'][:2],
                    d['patch'].copy(),float(d['pixels']),d['score'],confirmed=self.config.confirmation_hits==1)
            t.source='detection_high'
            self.next_id+=1
            self.tracks.append(t)
            self.events.append(dict(event='created',track_id=t.track_id,timestamp_us=timestamp_us,confirmed=t.confirmed))
        output=[]
        for t in self.tracks:
            if not t.confirmed:
                continue
            t.history.append(t.mean[:2].tolist())
            output.append(dict(track_id=t.track_id,xyxy_pixels=np.clip(t.box(),0,256).tolist(),
                               center_pixels=t.mean[:2].tolist(),source=t.source,
                               detection_confirmed_this_frame=t.source.startswith('detection'),
                               last_detection_confidence=t.last_detection_confidence,
                               frames_since_detection=t.since_detection,unobserved_frames=t.missed,
                               detector_hits=t.hits,age_frames=t.age,trail=t.history[-12:]))
        self.previous_image=image.copy(); self.previous_timestamp=timestamp_us
        return dict(timestamp_us=timestamp_us,tracks=output,rejected_detections=rejected,
                    high_confidence_after_filters=[dict(xyxy_pixels=d['box'].tolist(),confidence=d['score']) for d in high],
                    tentative_tracks=sum(not t.confirmed for t in self.tracks),reset_due_to_gap=reset,
                    geometry=asdict(self.geometry))
