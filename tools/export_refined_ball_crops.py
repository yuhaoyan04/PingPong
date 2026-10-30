"""Create a separate, unlabeled review dataset from refined candidate tracks."""
import argparse
import json
from pathlib import Path

import numpy as np

from batch_marker_crops import export
from pingpong_event_pipeline.events import RawWindowReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reviews', required=True, type=Path)
    parser.add_argument('--originals', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    args.output.mkdir(parents=True)
    readers = {}
    results = []
    for review in sorted(args.reviews.iterdir()):
        if not review.is_dir() or not (review/'candidate_tracks.npz').exists():
            continue
        old_group = args.originals/review.name
        source = json.loads((old_group/'source.json').read_text(encoding='utf-8'))
        points = np.load(review/'candidate_tracks.npz')['new']
        if len(points) < 2:
            results.append({'group': review.name, 'status': 'no_candidate', 'frames': 0})
            continue
        raw = source['raw']
        if raw not in readers:
            readers[raw] = RawWindowReader(raw)
        first, step = int(source['video_first_us']), int(source['video_step_us'])
        if step != 5000 or np.any((points[:, 0].astype(np.int64)-first) % step):
            raise ValueError(f'timestamp grid mismatch: {review}')
        members = [(int((row[0]-first)/step), tuple(map(float, row[1:4]))) for row in points]
        count = export(args.output/review.name, readers[raw], {'members': members}, first,
                       Path(source['video']), source['camera'])
        meta_path = args.output/review.name/'source.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        meta['method'] = 'motion_consistency_and_ring_score_v2'
        meta['annotations'] = 'new pixel coordinates; original annotations not copied'
        meta['review_comparison'] = str((review/'comparison.mp4').resolve())
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        results.append({'group': review.name, 'status': 'unverified_candidate', 'frames': count})
        print(f'{review.name}: {count} frames', flush=True)
    (args.output/'batch_report.json').write_text(json.dumps({'results': results,
        'warning': 'Unverified alternative crops; do not transfer old annotations automatically.'}, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
