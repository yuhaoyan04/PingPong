"""Package unlabelled ball crops for held-out inference, not evaluation labels."""
import argparse
import csv
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    result = []
    groups = sorted(p for p in args.source.iterdir() if p.is_dir() and p.name.startswith('serve')
                    and p.name != 'serve01_F3E')
    if len(groups) != 17:
        raise ValueError(f'expected 17 held-out camera groups, found {len(groups)}')
    args.output.mkdir(parents=True)
    for group in groups:
        with (group/'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        source = json.loads((group/'source.json').read_text(encoding='utf-8'))
        dest = args.output/'groups'/group.name
        (dest/'binary').mkdir(parents=True)
        (dest/'polarity_reference').mkdir()
        for row in rows:
            for key in ('image', 'polarity_reference'):
                rel = Path(row[key])
                if rel.is_absolute() or '..' in rel.parts or not (group/rel).is_file():
                    raise ValueError(f'invalid source: {group/rel}')
                shutil.copy2(group/rel, dest/rel)
        shutil.copy2(group/'manifest.csv', dest/'manifest.csv')
        metadata = {k: v for k, v in source.items() if k not in ('raw', 'video')}
        metadata['original_clip_name'] = Path(source['video']).name
        metadata['category'] = 'same_serve_other_camera' if group.name.startswith('serve01_') else 'different_serve'
        metadata['annotation_status'] = 'unlabeled; not ground truth and not negative labels'
        (dest/'source.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        result.append({'group': group.name, 'frames': len(rows), 'category': metadata['category']})
    (args.output/'inventory.json').write_text(json.dumps({'groups': result,
        'frames': sum(r['frames'] for r in result), 'labels_present': False}, indent=2), encoding='utf-8')
    instructions = Path(__file__).resolve().parents[1]/'docs'/'training'/'GENERALIZATION_INFERENCE_README.md'
    shutil.copy2(instructions, args.output/'README.md')
    print(json.dumps({'groups': len(groups), 'frames': sum(r['frames'] for r in result)}, indent=2))


if __name__ == '__main__': main()
