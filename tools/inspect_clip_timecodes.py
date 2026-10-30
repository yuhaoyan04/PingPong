"""Read burned-in clip timecodes using the exact OpenCV font used to write them.

This is an aid, not a substitute for an independent source timestamp audit.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def components(mask):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return sorted([tuple(map(int, s)) for s in stats[1:] if s[4] >= 4], key=lambda s: s[0])


def read_time(frame):
    mask = ((frame[:26, :500, 1] > 150) & (frame[:26, :500, 2] > 150)
            & (frame[:26, :500, 0] < 140)).astype(np.uint8)
    all_components = components(mask)
    # Locate t= in the timestamp label; exclude the following 'window' label.
    start = next((s[0] for s in all_components if 190 <= s[0] <= 205), None)
    if start is None:
        raise ValueError('timestamp prefix not found')
    items = [s for s in all_components if start + 20 < s[0] < 310]
    # Last glyph is the s suffix; characters before it are seconds with dot.
    if not 5 <= len(items) <= 8:
        raise ValueError(f'unexpected timestamp glyphs: {items}')
    glyphs = []
    for digit in '0123456789':
        tmp = np.zeros((30, 30), np.uint8)
        cv2.putText(tmp, digit, (3, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, 255, 1, cv2.LINE_AA)
        rects = components(tmp > 50)
        x, y, w, h, _ = max(rects, key=lambda s: s[4])
        glyphs.append(cv2.resize(tmp[y:y+h, x:x+w], (18, 24), interpolation=cv2.INTER_AREA))
    chars = []
    for x, y, w, h, area in items[:-1]:
        if h <= 4:
            chars.append('.')
            continue
        glyph = cv2.resize(mask[y:y+h, x:x+w] * 255, (18, 24), interpolation=cv2.INTER_AREA)
        scores = [np.mean(np.abs(glyph.astype(float) - model.astype(float))) for model in glyphs]
        chars.append(str(int(np.argmin(scores))))
    value = ''.join(chars)
    if value.count('.') != 1 or len(value.split('.')[1]) != 3:
        raise ValueError(f'bad timestamp: {value}')
    return float(value), value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    result = []
    for path in sorted(args.directory.glob('*.mp4')):
        video = cv2.VideoCapture(str(path))
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        entries = []
        for i in (0, min(50, count - 1), count - 1):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = video.read()
            try:
                sec, text = read_time(frame) if ok else (None, 'read failure')
            except ValueError as e:
                sec, text = None, str(e)
            entries.append({'index': i, 'seconds': sec, 'read': text})
        video.release()
        result.append({'file': path.name, 'frames': count, 'timecodes': entries})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
