"""Export DVSLume RAW ball events to Metavision's documented EventCD ABI.

The resulting .npy can be loaded on a machine with Metavision SDK and passed
to TripletMatchingFlowAlgorithm.process_events. No Prophesee camera or RAW
decoder is needed there. This exporter does not run or validate that SDK.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_ball_event_filters import load_ball_events, stc_cut_trail_mask


EVENTCD_DTYPE = np.dtype({
    'names': ['x', 'y', 'p', 't'],
    'formats': ['<u2', '<u2', '<i2', '<i8'],
    'offsets': [0, 2, 4, 8],
    'itemsize': 16,
})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--inner-radius', type=float, default=.65)
    p.add_argument('--filter-threshold-us', type=int, default=5000)
    args = p.parse_args()
    x,y,t,pol,u,v,inside = load_ball_events(args.raw,args.path,args.inner_radius)
    filtered = stc_cut_trail_mask(x,y,t,pol,int(x.max())+1,int(y.max())+1,args.filter_threshold_us)
    selected = inside & filtered
    out = np.empty(int(selected.sum()), dtype=EVENTCD_DTYPE)
    out['x'],out['y'],out['p'],out['t'] = x[selected],y[selected],pol[selected],t[selected]
    if len(out) > 1 and np.any(np.diff(out['t']) < 0):
        raise ValueError('Input events are not timestamp sorted')
    target = Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    np.save(target,out)
    info = {'raw':args.raw,'path':args.path,'events':len(out),
            'first_t_us':int(out['t'][0]),'last_t_us':int(out['t'][-1]),
            'dtype':out.dtype.descr,'inner_radius':args.inner_radius,
            'filter_threshold_us':args.filter_threshold_us,
            'note':'STC-CUT-TRAIL is an offline approximation, not hardware-bit-exact.'}
    target.with_suffix('.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in info.items() if k!='dtype'},indent=2))


if __name__ == '__main__':
    main()
