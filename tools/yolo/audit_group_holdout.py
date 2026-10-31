"""Isolated whole-serve holdout diagnostics; production all-label models stay intact.

Only serve01/06 reviewed labels are used. Fixed final-epoch checkpoints, no
validation-based selection, no random frame split, no new labels/negatives.
These two-fold within-recording diagnostics are NOT a new independent test set.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from common import ROOT,DATA,dump,rows,table
from train_camera import TrainOnlyTrainer,params
from infer_camera import predict_batched,boxes


def iou_matrix(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    intersection=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    aa=np.maximum(0,a[:,2:]-a[:,:2]).prod(1);bb=np.maximum(0,b[:,2:]-b[:,:2]).prod(1)
    return intersection/np.maximum(aa[:,None]+bb[None,:]-intersection,1e-9)


def match(gt,predictions,threshold):
    chosen=sorted([p for p in predictions if p['confidence']>=threshold],key=lambda p:-p['confidence'])
    matrix=iou_matrix([p['xyxy_pixels'] for p in chosen],gt);used=set();matched=[]
    for i,p in enumerate(chosen):
        candidates=[j for j in range(len(gt)) if j not in used and matrix[i,j]>=.5]
        if candidates:
            j=max(candidates,key=lambda j:matrix[i,j]);used.add(j)
            center=(np.array(p['xyxy_pixels'])[:2]+np.array(p['xyxy_pixels'])[2:])/2
            reference=(gt[j,:2]+gt[j,2:])/2
            matched.append({'prediction':i,'gt':j,'iou':float(matrix[i,j]),'center_error_px':float(np.linalg.norm(center-reference))})
    return {'tp':len(matched),'fp':len(chosen)-len(matched),'fn':len(gt)-len(matched),'matches':matched}


def labels(path):
    result=[]
    for line in path.read_text().splitlines():
        cls,x,y,w,h=map(float,line.split());assert cls==0
        result.append([(x-w/2)*256,(y-h/2)*256,(x+w/2)*256,(y+h/2)*256])
    return np.array(result,float).reshape(-1,4)


def run(camera,device,output,epochs,augmentation='none'):
    import torch
    from ultralytics import YOLO,settings
    output=Path(output)/camera;output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    settings.update({k:False for k in ['sync','wandb','mlflow','comet','clearml','neptune','tensorboard'] if k in settings})
    source=DATA/'datasets'/camera
    images=sorted((source/'images/train').glob('*.png'))
    groups={p.name.split('__')[0] for p in images}
    if groups!={f'serve01_{camera}',f'serve06_{camera}'}:raise ValueError('expected exactly two reviewed source serves')
    source_hashes={str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for image in images for p in [image,source/'labels/train'/f'{image.stem}.txt']}
    dump(output/'source_hashes.json',source_hashes)
    reports=[]
    for held in sorted(groups):
        fold=output/f'holdout_{held}';fold.mkdir()
        dataset=fold/'dataset'
        for split in ['train','test']:
            for kind in ['images','labels']:(dataset/kind/split).mkdir(parents=True)
        counts={'train':0,'test':0};held_paths=[]
        for image in images:
            split='test' if image.name.split('__')[0]==held else 'train'
            shutil.copy2(image,dataset/'images'/split/image.name)
            shutil.copy2(source/'labels/train'/f'{image.stem}.txt',dataset/'labels'/split/f'{image.stem}.txt')
            counts[split]+=1
            if split=='test':held_paths.append(dataset/'images/test'/image.name)
        data_config={'path':str(dataset.resolve()),'train':'images/train','val':'images/test','test':'images/test',
                     'channels':1,'names':{0:'marker'}}
        yaml_path=dataset/'dataset.yaml';yaml_path.write_text(yaml.safe_dump(data_config,sort_keys=False))
        parameters=params(camera,device,epochs)
        if augmentation=='conservative':
            # Detector-only geometric augmentation; no altered event timestamps,
            # invented backgrounds, HSV changes, mosaic or copied marker labels.
            parameters.update(translate=.05,scale=.10,fliplr=.5)
        parameters.update(data=str(yaml_path.resolve()),project=str(fold.resolve()),name='train',workers=0,plots=False,verbose=False,save_period=-1)
        dump(fold/'parameters.json',parameters)
        dump(fold/'split.json',{'train_serve':sorted(groups-{held}),'heldout_serve':held,'counts':counts,
                              'random_frame_split':False,'production_models_changed':False,
                              'independent_final_test':False,'purpose':'within-recording whole-serve transfer diagnostic',
                              'epochs_fixed_before_holdout_inference':epochs,'confidence_thresholds_fixed':[.25,.75]})
        started=time.monotonic();model=YOLO(str(ROOT/'weights/yolo11n.pt'))
        model.train(trainer=TrainOnlyTrainer,**parameters)
        checkpoint=fold/'train/weights/last.pt'
        assert checkpoint.exists() and not (checkpoint.parent/'best.pt').exists()
        model=YOLO(str(checkpoint));assert model.model.model[0].conv.in_channels==1
        observed=[];predictions=predict_batched(model,held_paths,device,.05,observed)
        records=[];prediction_records=[]
        for image,prediction in zip(held_paths,predictions):
            gt=labels(dataset/'labels/test'/f'{image.stem}.txt');detections=boxes(prediction)
            prediction_records.append({'image':image.name,'detections':detections})
            for threshold in [.25,.75]:
                result=match(gt,detections,threshold)
                records.append({'image':image.name,'confidence':threshold,'gt_boxes':len(gt),
                                **{k:result[k] for k in ['tp','fp','fn']},
                                'matched_center_errors_px':[x['center_error_px'] for x in result['matches']]})
        metrics=[]
        for threshold in [.25,.75]:
            selected=[r for r in records if r['confidence']==threshold]
            tp,fp,fn=[sum(r[k] for r in selected) for k in ['tp','fp','fn']]
            errors=[x for r in selected for x in r['matched_center_errors_px']]
            metrics.append({'confidence':threshold,'tp':tp,'fp':fp,'fn':fn,
                            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
                            'matched_center_error_median_px':float(np.median(errors)) if errors else None,
                            'matched_center_error_p95_px':float(np.quantile(errors,.95)) if errors else None,
                            'explicit_empty_negative_frames':sum(r['gt_boxes']==0 for r in selected),
                            'iou_threshold':.5})
        report={'camera':camera,'heldout_serve':held,'train_serve':sorted(groups-{held}),**counts,
                'augmentation':augmentation,
                'epochs':epochs,'metrics':metrics,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'pretrained_sha256':hashlib.sha256((ROOT/'weights/yolo11n.pt').read_bytes()).hexdigest(),
                'elapsed_seconds':time.monotonic()-started,'channels_observed':observed,
                'scope':'two-fold within-recording transfer diagnosis, not production-model accuracy or independent final test',
                'checkpoint_selection':'fixed final epoch EMA; heldout unused during optimization and selection',
                'physical_spin_accuracy_validated':False}
        dump(fold/'diagnostic.json',report);dump(fold/'heldout_predictions.json',prediction_records)
        table(fold/'heldout_frames.csv',records)
        worst=sorted([r for r in records if r['confidence']==.75],key=lambda r:-(r['fp']+r['fn']))[:12]
        sheet=Image.new('RGB',(4*320,3*352),'#142130');draw=ImageDraw.Draw(sheet)
        pmap={r['image']:r['detections'] for r in prediction_records}
        for i,row in enumerate(worst):
            x=i%4*320;y=i//4*352
            sheet.paste(Image.open(dataset/'images/test'/row['image']).convert('RGB').resize((320,320)),(x,y+32))
            draw.text((x+3,y+3),f"{held} FP={row['fp']} FN={row['fn']}",fill='white')
            for gt in labels(dataset/'labels/test'/f"{Path(row['image']).stem}.txt"):
                draw.rectangle(tuple(np.array(gt)*1.25+np.array([x,y+32,x,y+32])),outline='lime',width=2)
            for d in pmap[row['image']]:
                if d['confidence']>=.75:draw.rectangle(tuple(np.array(d['xyxy_pixels'])*1.25+np.array([x,y+32,x,y+32])),outline='orange',width=2)
        sheet.save(fold/'worst_frames_gt_green_pred_orange.jpg',quality=93)
        reports.append(report);dump(output/'holdout_summary.json',reports)
        del model;torch.cuda.empty_cache()
        print('HOLDOUT_COMPLETE',camera,held,json.dumps(metrics),flush=True)
    for relative,digest in source_hashes.items():assert hashlib.sha256((DATA/relative).read_bytes()).hexdigest()==digest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--camera',choices=['F3E','F59'],required=True)
    p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--augmentation',choices=['none','conservative'],default='none')
    a=p.parse_args();run(a.camera,a.device,a.output,a.epochs,a.augmentation)
