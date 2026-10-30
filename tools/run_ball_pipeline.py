"""Batch candidate RAW-event trajectory refinement and quality diagnostics."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from pingpong_event_pipeline.ball_pipeline_quality import assess_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', required=True, type=Path)
    parser.add_argument('--coarse', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--only', nargs='*', help='optional exact group names')
    parser.add_argument('--adaptive-radius', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    groups = sorted(p for p in args.groups.iterdir() if p.is_dir() and (p/'source.json').is_file())
    if args.only is not None:
        requested = set(args.only)
        groups = [p for p in groups if p.name in requested]
        if {p.name for p in groups} != requested:
            raise ValueError('requested group not found')
    if not groups:
        raise ValueError('no source groups found')
    args.output.mkdir(parents=True)
    reviewer = Path(__file__).resolve().parent/'review_event_refinement.py'
    results = []
    for group in groups:
        coarse = args.coarse/group.name
        out = args.output/group.name
        if not (coarse/'candidate_tracks.npz').is_file():
            results.append({'group': group.name, 'status': 'no_coarse_candidate'})
            print(group.name, 'no coarse candidate', flush=True)
            continue
        command = [sys.executable, str(reviewer), '--review', str(coarse), '--group', str(group), '--output', str(out)]
        if args.adaptive_radius: command.append('--adaptive-radius')
        run = subprocess.run(command,
                             text=True, encoding='utf-8', errors='replace', capture_output=True,
                             env=os.environ.copy())
        if run.returncode:
            results.append({'group': group.name, 'status': 'failed', 'error': (run.stderr or run.stdout)[-2500:]})
            print(group.name, 'failed:', (run.stderr or run.stdout).splitlines()[-1], flush=True)
            continue
        path = np.load(out/'raw_refined_path.npz')
        quality = assess_candidate(path['records'], len(path['coarse']))
        (out/'quality.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding='utf-8')
        results.append({'group': group.name, **quality})
        print(group.name, f"supported={quality['supported_samples']}/{quality['sample_count']}",
              'flags=' + ','.join(quality['diagnostic_flags']), flush=True)
    summary = {'status': 'manual_review_required', 'groups': results,
               'warning': 'No manual ball-center ground truth; diagnostics are not accuracy certification. '
                          'No original images or annotations are modified.'}
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'reviewed_candidate_groups={len(results)} output={args.output.resolve()}', flush=True)


if __name__ == '__main__': main()
