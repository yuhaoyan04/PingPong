"""Portable standard-library audit for the two-camera training package."""
import csv
import hashlib
import json
import math
import struct
from pathlib import Path


def png_check(path, grayscale=False):
    with path.open('rb') as f: head=f.read(26)
    if len(head)<26 or head[:8] != b'\x89PNG\r\n\x1a\n': raise ValueError(f'not PNG: {path}')
    if struct.unpack('>II',head[16:24]) != (256,256): raise ValueError(f'wrong image size: {path}')
    if grayscale and head[25] != 0: raise ValueError(f'not grayscale PNG: {path}')


def main():
    root=Path(__file__).resolve().parent
    for cam in ('F3E','F59'):
        dataset=root/'datasets'/cam
        manifest=json.loads((dataset/'split_manifest.json').read_text(encoding='utf8'))
        if not manifest.get('train_only') or set(manifest['train']) != {f'serve01_{cam}',f'serve06_{cam}'} or manifest['val'] or manifest['test']:
            raise ValueError(f'wrong train-only split: {cam}')
        yaml=(dataset/'dataset.yaml').read_text(encoding='utf8')
        if 'train: images/train' not in yaml or 'val: images/train' not in yaml or 'channels: 1' not in yaml:
            raise ValueError(f'wrong dataset config: {cam}')
        imgs=sorted((dataset/'images'/'train').glob('*.png'))
        labels={p.stem:p for p in (dataset/'labels'/'train').glob('*.txt')}
        if not imgs or set(labels) != {p.stem for p in imgs}: raise ValueError(f'image-label mismatch: {cam}')
        boxes=negatives=fallback=0
        support={}
        for group in manifest['train']:
            with (root/'group_metadata'/cam/group/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:
                for row in csv.DictReader(f):
                    support[group+'__'+Path(row['image']).stem]=row['track_support']
        for image in imgs:
            png_check(image,True)
            if image.stem not in support: raise ValueError(f'missing source row: {image}')
            fallback+=support[image.stem]=='fallback_unverified'
            lines=labels[image.stem].read_text(encoding='utf8').splitlines()
            negatives+=not lines
            for line in lines:
                vals=line.split()
                if len(vals)!=5 or vals[0]!='0': raise ValueError(f'invalid label: {image}')
                x,y,w,h=map(float,vals[1:])
                eps=1e-7  # exporter serializes normalized values to eight decimal places
                if not all(map(math.isfinite,(x,y,w,h))) or not (0<w<=1 and 0<h<=1 and -eps<=x-w/2<=x+w/2<=1+eps and -eps<=y-h/2<=y+h/2<=1+eps):
                    raise ValueError(f'out of bounds: {image}: {line}')
                boxes+=1
        print(f'{cam}: train {len(imgs)} images, {boxes} boxes, {negatives} explicit negatives, {fallback} fallback frames; no independent validation/test')
        qroot=root/'unlabeled_inference'/cam
        groups=sorted(qroot.glob('serve*'))
        if {p.name for p in groups} != {f'serve{i:02d}_{cam}' for i in (2,3,4,5,7,8,9)}:
            raise ValueError(f'wrong qualitative groups for {cam}')
        for group in groups:
            with (group/'manifest.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
            names={Path(r['image']).name for r in rows}
            for sub in ('binary','binary_unfiltered','polarity_reference'):
                images=list((group/sub).glob('*.png'))
                if {p.name for p in images} != names: raise ValueError(f'missing frames: {group/sub}')
                for image in images: png_check(image,sub!='polarity_reference')
            if list(group.rglob('*.txt')) or list((group/'binary').glob('*.json')):
                raise ValueError(f'qualitative data must not contain labels: {group}')
            print(f'  {group.name}: {len(rows)} unlabeled frames (qualitative only)')
    for line in (root/'SHA256SUMS.txt').read_text(encoding='utf8').splitlines():
        expected,relative=line.split('  ',1); path=(root/relative).resolve()
        if root.resolve() not in path.parents: raise ValueError('invalid checksum path')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=expected: raise ValueError(f'checksum mismatch: {relative}')
    print('Checksums, PNG headers, labels, splits and qualitative file counts passed.')


if __name__=='__main__': main()
