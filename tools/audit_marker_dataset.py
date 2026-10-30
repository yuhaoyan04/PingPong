"""Check binary crops and manifest consistency without interpreting annotations."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset', type=Path)
    args = p.parse_args()
    manifests = sorted(args.dataset.glob('*/manifest.csv'))
    total = bad = 0
    for manifest in manifests:
        with manifest.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        times = [int(r['timestamp_us']) for r in rows]
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError(f'non-monotonic timestamps: {manifest}')
        for row in rows:
            image = cv2.imread(str(manifest.parent / row['image']), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (256, 256) or not np.all((image == 0) | (image == 255)):
                bad += 1
            total += 1
    print(f'groups={len(manifests)} images={total} invalid={bad}')
    if not manifests or bad:
        raise SystemExit(1)


if __name__ == '__main__': main()
