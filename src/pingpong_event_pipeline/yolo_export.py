"""Convert manually reviewed Labelme rectangles to a grouped YOLO dataset."""
import argparse
import json
import re
import shutil
import struct
from pathlib import Path

from .annotations import rectangle_to_yolo


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--train', nargs='+', required=True, help='whole group names')
    p.add_argument('--val', nargs='+', default=[], help='different whole group names')
    p.add_argument('--test', nargs='*', default=[])
    p.add_argument('--train-only', action='store_true', help='all labeled groups in train; YAML val points to train for schema compatibility, not evaluation')
    p.add_argument('--accept-labelme-json', action='store_true', help='also accept external Labelme JSON without a reviewed flag')
    args = p.parse_args()
    if args.train_only and (args.val or args.test): p.error('--train-only cannot be combined with --val or --test')
    if not args.train_only and not args.val: p.error('--val is required unless --train-only is used')
    groups = args.train + args.val + args.test
    if len(set(groups)) != len(groups) or any(Path(g).name != g for g in groups):
        p.error('groups must be unique simple names; never split one group across sets')
    split_by_serve = {}
    for split, names in [('train', args.train), ('val', args.val), ('test', args.test)]:
        for group in names:
            match = re.match(r'^(serve\d+)(?:_|$)', group, flags=re.IGNORECASE)
            if match:
                key = match.group(1).lower()
                if key in split_by_serve and split_by_serve[key] != split:
                    p.error(f'{key} camera views must remain in the same split')
                split_by_serve[key] = split
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    jobs = []
    for split, names in [('train', args.train), ('val', args.val), ('test', args.test)]:
        for group in names:
            folder = args.dataset / group / 'binary'
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            for annotation in sorted(folder.glob('*.json')):
                data = json.loads(annotation.read_text(encoding='utf-8'))
                reviewed = data.get('flags', {}).get('reviewed')
                if reviewed is not True and not (reviewed is None and args.accept_labelme_json):
                    continue
                image = annotation.with_suffix('.png')
                if not image.is_file() or Path(data.get('imagePath', '')).name != image.name:
                    raise ValueError(f'annotation image mismatch: {annotation}')
                width, height = int(data['imageWidth']), int(data['imageHeight'])
                if width <= 0 or height <= 0:
                    raise ValueError(f'invalid dimensions: {annotation}')
                with image.open('rb') as f:
                    header = f.read(24)
                if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or struct.unpack('>II', header[16:24]) != (width, height):
                    raise ValueError(f'image dimensions do not match: {annotation}')
                if not isinstance(data.get('shapes'), list):
                    raise ValueError(f'missing shapes list: {annotation}')
                lines = [rectangle_to_yolo(shape, width, height) for shape in data['shapes']]
                jobs.append((split, group, image, lines))
    required_splits = ('train',) if args.train_only else ('train', 'val', 'test') if args.test else ('train', 'val')
    for split in required_splits:
        if not any(job[0] == split for job in jobs):
            raise ValueError(f'no annotated samples in {split}; unannotated images are not negatives')
    for split, group, image, lines in jobs:
        image_dir, label_dir = args.output / 'images' / split, args.output / 'labels' / split
        image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True)
        name = group + '__' + image.stem
        shutil.copy2(image, image_dir / (name + '.png'))
        (label_dir / (name + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    data_yaml = ("train: images/train\nval: " + ('images/train' if args.train_only else 'images/val') + "\nchannels: 1\n" + ("test: images/test\n" if args.test else '') + "names:\n  0: marker\n")
    (args.output / 'dataset.yaml').write_text(data_yaml, encoding='utf-8')
    (args.output / 'split_manifest.json').write_text(json.dumps({'train': args.train, 'val': args.val, 'test': args.test,
        'annotated_images': len(jobs), 'train_only': args.train_only,
        'note': ('No independent validation: YAML val points to training images for compatibility. ' if args.train_only else '') +
                'Empty Labelme shapes are explicitly reviewed negatives.'}, indent=2), encoding='utf-8')
    print(f'Exported {len(jobs)} annotated images to {args.output.resolve()}')


if __name__ == '__main__':
    main()
