"""Manual LED-board capture. Preview snapshots are NOT synchronized measurements."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from dual_capture_isolated import BIAS, put_latest
from pingpong_event_pipeline.acquisition import (
    configure_boolean_parameter, configure_event_rate_control, configure_trail_filter,
)


def detect_grid(image, detector):
    """Find a complete 3-by-4 grid; failure is not a zero-dot count."""
    small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ok, centers = cv2.findCirclesGrid(
        small, (3, 4), flags=cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
        blobDetector=detector)
    return centers.reshape(-1, 2) * 2 if ok and centers is not None else None


def detection_thread(jobs, results, quit_event):
    from live_stereo_view import make_blob_detector
    detector = make_blob_detector()
    while not quit_event.is_set():
        try:
            frames, metadata = jobs.get(timeout=0.2)
        except queue.Empty:
            continue
        result = {}
        for role in ('master', 'slave'):
            try:
                centers = detect_grid(frames[role], detector)
                result[role] = {'found': centers is not None,
                                'host_monotonic': metadata[role]['host_monotonic'],
                                'centers': centers.tolist() if centers is not None else None}
            except Exception as exc:
                result[role] = {'found': False, 'error': str(exc),
                                'host_monotonic': metadata[role]['host_monotonic'], 'centers': None}
        put_latest(results, result)


def worker(role, serial, batch_us, window_us, directory, gate, started, messages, previews, shutdown):
    camera = None
    streaming = False
    try:
        previews.cancel_join_thread()
        from dvsense_driver.camera_manager import DvsCameraManager
        from dvsense_driver.hal.camera_tool import ToolType
        manager = DvsCameraManager()
        manager.update_cameras()
        camera = manager.open_camera(serial)
        if camera is None:
            raise RuntimeError(f'Cannot open {serial}')
        sync = camera.get_tool(getattr(ToolType, 'SYNC', getattr(ToolType, 'TOOL_SYNC', ToolType(2))))
        if not sync.set_param('mode', role.upper()):
            raise RuntimeError('Cannot set sync mode')
        ok, mode = sync.get_param('mode')
        if not ok or str(mode) != role.upper():
            raise RuntimeError(f'Sync readback failed: {mode}')
        config = {'serial': serial, 'mode': str(mode), 'batch_us': batch_us}
        bias = camera.get_tool(ToolType.BIAS)
        for name in BIAS:
            if not bias.set_param(name, 0):
                raise RuntimeError(f'Cannot set {name}')
            ok, value = bias.get_param(name)
            if not ok or int(value) != 0:
                raise RuntimeError(f'Bias readback failed: {name}={value}')
            config[name] = int(value)
        config['event_rate_control'] = configure_event_rate_control(camera.get_tool(ToolType.EVENT_RATE_CONTROL), False, 5, serial)
        config['trail_filter'] = configure_trail_filter(camera.get_tool(ToolType.EVENT_TRAIL_FILTER), False, 1, serial)
        config['anti_flicker_enabled'] = configure_boolean_parameter(camera.get_tool(ToolType.ANTI_FLICKER), 'enable', False, serial)
        roi = camera.get_tool(ToolType.ROI)
        config['roi'] = {}
        for name in ('x', 'y', 'width', 'height'):
            ok, value = roi.get_param(name)
            if not ok:
                raise RuntimeError(f'ROI readback failed: {name}')
            config['roi'][name] = int(value)
        width, height = int(camera.get_width()), int(camera.get_height())
        config.update(width=width, height=height)
        camera.set_batch_events_time(batch_us)
        messages.put(('configured', role, config))
        if not gate.wait(30) or shutdown.is_set():
            return
        rc = camera.start()
        if rc != 0:
            raise RuntimeError(f'Camera start failed: {rc}')
        streaming = True
        started.set()
        messages.put(('streaming', role, {'host_monotonic': time.monotonic()}))
        activity = np.zeros((height, width), np.uint8)
        first = last = None
        count = sampled_count = 0
        window_start_host = time.monotonic()
        while not shutdown.is_set():
            events = camera.get_next_batch()
            if events is None or len(events) == 0:
                continue
            ts = events['timestamp']
            first = int(ts[0]) if first is None else first
            last = int(ts[-1])
            count += len(events)
            # Cap transient allocations and CPU time at high event rates. This
            # is a diagnostic image, not a complete event representation.
            stride = max(1, (len(events) + 49999) // 50000)
            events = events[::stride]
            sampled_count += len(events)
            x = events['x'].astype(np.intp)
            y = events['y'].astype(np.intp)
            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            activity[y[valid], x[valid]] = 255
            if time.monotonic() - window_start_host >= window_us / 1e6:
                now = time.monotonic()
                put_latest(previews, (activity, {'first_event_us': first, 'last_event_us': last,
                                                'events': count, 'sampled_events': sampled_count,
                                                'host_window_start': window_start_host, 'host_monotonic': now}))
                activity = np.zeros((height, width), np.uint8)
                first = last = None
                count = sampled_count = 0
                window_start_host = now
    except Exception:
        messages.put(('error', role, traceback.format_exc()))
    finally:
        if streaming:
            try:
                camera.stop()
            except Exception:
                messages.put(('error', role, traceback.format_exc()))
        messages.put(('closed', role, None))


def save_pair(directory, index, frames, metadata):
    paths = {}
    for role in ('master', 'slave'):
        path = directory / f'{role}_{index:03d}.png'
        if path.exists() or not cv2.imwrite(str(path), frames[role]):
            raise RuntimeError(f'Failed to save {path}')
        paths[role] = path.name
    record = {'index': index, 'images': paths, 'preview_windows': metadata.copy(),
              'host_save_monotonic': time.monotonic(),
              'note': 'Sampled event-activity preview images, independent host-time windows, not synchronized or validated calibration data.'}
    (directory / f'pair_{index:03d}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def main():
    mp.freeze_support()
    p = argparse.ArgumentParser()
    p.add_argument('--master', default='DSP01F1FFF59')
    p.add_argument('--slave', default='DSP01F1FFF3E')
    p.add_argument('--batch-us', type=int, default=1000)
    p.add_argument('--window-ms', type=int, default=200)
    p.add_argument('--output-dir', type=Path, default=Path('calibration/manual_sessions'))
    p.add_argument('--smoke-seconds', type=float, default=0)
    p.add_argument('--smoke-save', action='store_true', help='save one pair automatically during a smoke test')
    args = p.parse_args()
    if args.master == args.slave or args.batch_us <= 0 or args.window_ms <= 0 or args.smoke_seconds < 0:
        p.error('invalid serials or timing')
    directory = (args.output_dir / datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    messages = mp.Queue()
    shutdown = mp.Event()
    roles = ('master', 'slave')
    gates = {r: mp.Event() for r in roles}
    started = {r: mp.Event() for r in roles}
    previews = {r: mp.Queue(maxsize=1) for r in roles}
    workers = {}
    info, streaming, errors = {}, {}, []
    frames, meta, saved = {}, {}, []
    detection_jobs, detection_results = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
    detection_stop = threading.Event()
    last_detection, next_detection, checks_completed = {}, 0.0, 0
    cv2.setNumThreads(1)
    try:
        for role, serial in [('slave', args.slave), ('master', args.master)]:
            proc = mp.Process(target=worker, args=(role, serial, args.batch_us, args.window_ms * 1000,
                                                   str(directory), gates[role], started[role], messages,
                                                   previews[role], shutdown))
            workers[role] = proc
            proc.start()
            while role not in info:
                kind, source, value = messages.get(timeout=30)
                if kind == 'error':
                    raise RuntimeError(value)
                if kind == 'configured':
                    info[source] = value
        fields = list(BIAS) + ['event_rate_control', 'trail_filter', 'anti_flicker_enabled', 'roi', 'width', 'height']
        mismatch = [name for name in fields if info['master'][name] != info['slave'][name]]
        if mismatch:
            raise RuntimeError(f'Camera settings differ: {mismatch}')
        for role in ('slave', 'master'):
            gates[role].set()
            if not started[role].wait(15):
                raise RuntimeError(f'{role} did not start')
        print(f'READY: {directory}', flush=True)
        print('S: save pair | Q: finish | hold the board still during each capture', flush=True)
        begin = time.monotonic()
        title = 'LED calibration | S save | Q finish'
        clicks = []
        def mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONUP and y >= 315:
                if x < 190:
                    clicks.append('save')
                elif x < 380:
                    clicks.append('quit')
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, mouse)
        threading.Thread(target=detection_thread, args=(detection_jobs, detection_results, detection_stop),
                         daemon=True).start()
        while True:
            try:
                while True:
                    kind, role, value = messages.get_nowait()
                    if kind == 'error':
                        raise RuntimeError(value)
                    if kind == 'streaming':
                        streaming[role] = value
            except queue.Empty:
                pass
            if any(not proc.is_alive() for proc in workers.values()):
                raise RuntimeError('Camera worker stopped unexpectedly')
            for role in roles:
                try:
                    frames[role], meta[role] = previews[role].get_nowait()
                except queue.Empty:
                    pass
            try:
                last_detection = detection_results.get_nowait()
                checks_completed += 1
                print('GRID CHECK', {role: ('12/12' if last_detection[role]['found'] else 'NOT FOUND')
                                     for role in roles}, flush=True)
            except queue.Empty:
                pass
            now = time.monotonic()
            if len(frames) == 2 and now >= next_detection:
                put_latest(detection_jobs, (frames.copy(), meta.copy()))
                next_detection = now + 1.5
            panels = []
            for role in roles:
                image = frames.get(role, np.zeros((info[role]['height'], info[role]['width']), np.uint8))
                panel = cv2.cvtColor(cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
                age = time.monotonic() - meta[role]['host_monotonic'] if role in meta else float('inf')
                cv2.putText(panel, f'{role.upper()} {info[role]["serial"][-3:]}  age {age:.1f}s', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255) if age < 1 else (0, 0, 255), 1)
                if role in last_detection:
                    check = last_detection[role]
                    check_age = now - check['host_monotonic']
                    result_text = '12/12 FOUND' if check['found'] else 'GRID NOT FOUND'
                    cv2.putText(panel, f'Last check: {result_text} ({check_age:.1f}s ago)', (10, 51),
                                cv2.FONT_HERSHEY_SIMPLEX, .5,
                                (0, 255, 0) if check['found'] and check_age < 3 else (0, 165, 255), 1)
                else:
                    cv2.putText(panel, 'Last check: pending', (10, 51), cv2.FONT_HERSHEY_SIMPLEX,
                                .5, (0, 165, 255), 1)
                panels.append(panel)
            view = np.hstack(panels)
            cv2.rectangle(view, (0, 315), (190, 359), (30, 90, 30), -1)
            cv2.rectangle(view, (190, 315), (380, 359), (30, 30, 130), -1)
            cv2.putText(view, '[S] SAVE', (12, 345), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            cv2.putText(view, '[Q] EXIT', (202, 345), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            cv2.putText(view, f'pairs {len(saved)} | {args.window_ms}ms | HOLD STILL',
                        (395, 345), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            cv2.imshow(title, view)
            key = cv2.waitKey(20) & 255
            action = clicks.pop(0) if clicks else None
            if args.smoke_save and not saved and len(frames) == 2:
                action = 'save'
            if key in (ord('s'), ord('S')) or action == 'save':
                now = time.monotonic()
                if len(frames) != 2 or any(now - meta[r]['host_monotonic'] > 1 for r in roles):
                    print('NOT SAVED: waiting for fresh previews from both cameras', flush=True)
                else:
                    saved.append(save_pair(directory, len(saved) + 1, frames, meta))
                    print(f'SAVED pair {len(saved)}', flush=True)
            if (key in (ord('q'), ord('Q'), 27) or action == 'quit'
                    or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
                    or (args.smoke_seconds and time.monotonic() - begin >= args.smoke_seconds)):
                break
    except Exception:
        errors.append(traceback.format_exc())
        print(errors[-1], flush=True)
    finally:
        detection_stop.set()
        shutdown.set()
        for gate in gates.values():
            gate.set()
        for role, proc in workers.items():
            proc.join(timeout=4)
            if proc.is_alive():
                errors.append(f'{role} worker did not exit cleanly; terminated')
                proc.terminate()
                proc.join(timeout=3)
        try:
            while True:
                kind, role, value = messages.get_nowait()
                if kind == 'streaming': streaming[role] = value
                if kind == 'error': errors.append(value)
        except queue.Empty:
            pass
        cv2.destroyAllWindows()
        report = {'configuration': info, 'streaming': streaming,
                  'window_ms': args.window_ms, 'saved_pairs': len(saved), 'errors': errors,
                  'detection_checks_completed': checks_completed,
                  'complete': not errors and len(streaming) == 2,
                  'note': 'No RAW recording. Images are sampled independent preview windows, not evidence of synchronized exposure or valid stereo correspondences.'}
        (directory / 'session.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('RESULT', directory, json.dumps(report), flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
