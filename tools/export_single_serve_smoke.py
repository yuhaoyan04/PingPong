"""Make an explicitly non-independent temporal smoke-test YOLO dataset."""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pingpong_event_pipeline.annotations import rectangle_to_yolo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--train-end', type=int, default=139)
    parser.add_argument('--val-start', type=int, default=150)
    parser.add_argument('--val-end', type=int, default=189)
    parser.add_argument('--test-start', type=int, default=200)
    args = parser.parse_args()
    if not 0 <= args.train_end < args.val_start <= args.val_end < args.test_start:
        parser.error('invalid chronological split')
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    with (args.group/'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        manifest = {int(row['frame_index']): row for row in csv.DictReader(f)}
    items, excluded = [], []
    for annotation in sorted((args.group/'binary').glob('*.json')):
        data = json.loads(annotation.read_text(encoding='utf-8'))
        if data.get('flags', {}).get('reviewed') is not True:
            continue
        image = annotation.with_suffix('.png')
        index = int(image.stem.rsplit('_', 1)[1])
        if index not in manifest or Path(data['imagePath']).name != image.name:
            raise ValueError(f'bad image reference: {annotation}')
        img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (256, 256) or not np.all((img == 0) | (img == 255)):
            raise ValueError(f'invalid binary image: {image}')
        if (data.get('imageWidth'), data.get('imageHeight')) != (256, 256):
            raise ValueError(f'invalid annotation size: {annotation}')
        if not isinstance(data.get('shapes'), list):
            raise ValueError(f'bad shapes: {annotation}')
        labels = [rectangle_to_yolo(s, 256, 256) for s in data['shapes']]
        if index <= args.train_end: split = 'train'
        elif args.val_start <= index <= args.val_end: split = 'val'
        elif index >= args.test_start: split = 'test'
        else:
            excluded.append(index)
            continue
        items.append((split, image, annotation, labels, manifest[index]))
    for split in ('train', 'val', 'test'):
        if not any(s == split for s, *_ in items):
            raise ValueError(f'no confirmed images for {split}')
    report = {'purpose': 'SMOKE_TEST_ONLY_SAME_SERVE_NOT_INDEPENDENT',
              'source_group': args.group.name, 'excluded_gap_indexes': excluded, 'splits': {}}
    rows = []
    for split, image, annotation, labels, info in items:
        out_images, out_labels = args.output/'images'/split, args.output/'labels'/split
        out_images.mkdir(parents=True, exist_ok=True); out_labels.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, out_images/image.name)
        (out_labels/(image.stem+'.txt')).write_text('\n'.join(labels) + ('\n' if labels else ''), encoding='utf-8')
        rows.append({'split': split, 'frame_index': info['frame_index'], 'timestamp_us': info['timestamp_us'],
                     'image': image.name, 'boxes': len(labels), 'annotation': annotation.name})
    for split in ('train', 'val', 'test'):
        subset = [r for r in rows if r['split'] == split]
        report['splits'][split] = {'images': len(subset), 'boxes': sum(r['boxes'] for r in subset),
                                  'negatives': sum(r['boxes'] == 0 for r in subset),
                                  'min_index': min(int(r['frame_index']) for r in subset),
                                  'max_index': max(int(r['frame_index']) for r in subset)}
    (args.output/'dataset.yaml').write_text('train: images/train\nval: images/val\ntest: images/test\nchannels: 1\nnames:\n  0: marker\n', encoding='utf-8')
    (args.output/'split_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (args.output/'manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    template = Path(__file__).resolve().parents[1]/'docs'/'training'/'SERVER_SINGLE_SERVE_PROMPT.md'
    shutil.copy2(template, args.output/'SERVER_PROMPT.md')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
