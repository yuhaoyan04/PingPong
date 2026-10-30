"""Render short ball-centred event windows before/after burst filtering."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from audit_ball_event_filters import load_ball_events, stc_cut_trail_mask


def image(u, v, polarity, selected, radius, bins):
    positive = selected & (polarity == 1)
    negative = selected & (polarity == 0)
    hist_range = [[-radius, radius], [-radius, radius]]
    on = np.histogram2d(v[positive], u[positive], bins=bins, range=hist_range)[0]
    off = np.histogram2d(v[negative], u[negative], bins=bins, range=hist_range)[0]
    return np.arcsinh(on) - np.arcsinh(off)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--inner-radius', type=float, default=.65)
    p.add_argument('--threshold-us', type=int, default=5000)
    p.add_argument('--window-us', type=int, default=2000)
    p.add_argument('--bins', type=int, default=128)
    args = p.parse_args()
    x, y, t, pol, u, v, inside = load_ball_events(args.raw, args.path, args.inner_radius)
    filtered = stc_cut_trail_mask(x, y, t, pol, int(x.max())+1,
                                  int(y.max())+1, args.threshold_us)
    first, last = int(t[inside].min()), int(t[inside].max())
    times = np.linspace(first + args.window_us, last - args.window_us, 5).astype(np.int64)
    fig, axes = plt.subplots(len(times), 2, figsize=(8, 18), dpi=135)
    for row, start in enumerate(times):
        window = inside & (t >= start) & (t < start + args.window_us)
        for col, selected in enumerate((window, window & filtered)):
            img = image(u, v, pol, selected, args.inner_radius, args.bins)
            axes[row, col].imshow(img, cmap='coolwarm', origin='lower',
                                  vmin=-3, vmax=3, interpolation='nearest')
            axes[row, col].set_title(f'{"raw" if col == 0 else "filtered"}: '
                                     f't={((start-first)/1000):.1f} ms, n={int(selected.sum())}')
            axes[row, col].set_axis_off()
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
