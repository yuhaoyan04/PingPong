"""Confirm saved nonempty marker drafts after their author requests it."""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('group', type=Path)
    args = parser.parse_args()
    folder = args.group / 'binary'
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    jobs, skipped_empty, already = [], 0, 0
    for path in sorted(folder.glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        shapes = data.get('shapes')
        if not isinstance(shapes, list) or not isinstance(data.get('flags'), dict):
            raise ValueError(f'invalid annotation structure: {path}')
        if not shapes:
            skipped_empty += 1
            continue
        if any(s.get('label') != 'marker' or s.get('shape_type') != 'rectangle' for s in shapes):
            raise ValueError(f'unsupported annotation: {path}')
        if data['flags'].get('reviewed') is True:
            already += 1
            continue
        data['flags']['reviewed'] = True
        jobs.append((path, data))
    backup = args.group / 'annotation_backups' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if jobs:
        backup.mkdir(parents=True, exist_ok=False)
        for path, _ in jobs:
            shutil.copy2(path, backup / path.name)
        for path, data in jobs:
            temporary = path.with_suffix('.json.confirm.tmp')
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(path)
    print(f'confirmed={len(jobs)} already_confirmed_nonempty={already} untouched_empty={skipped_empty} backup={backup if jobs else "none"}')


if __name__ == '__main__':
    main()
