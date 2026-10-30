"""Dual DVSLume capture with one SDK process per camera and independent previews."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pingpong_event_pipeline.acquisition import (
    configure_boolean_parameter, configure_event_rate_control, configure_trail_filter,
)

BIAS = ('bias_fo', 'bias_hpf', 'bias_diff_on', 'bias_diff', 'bias_diff_off', 'bias_refr')


def put_latest(channel, value):
    try:
        channel.put_nowait(value)
    except queue.Full:
        pass


def camera_worker(role, serial, batch_us, gate, started, commands, messages, preview, shutdown):
    camera = None
    streaming = recording = False
    try:
        # Preview is disposable: an unread frame must not hold process shutdown open.
        if hasattr(preview, 'cancel_join_thread'):
            preview.cancel_join_thread()
        from dvsense_driver.camera_manager import DvsCameraManager
        from dvsense_driver.hal.camera_tool import ToolType
        manager = DvsCameraManager()
        manager.update_cameras()
        camera = manager.open_camera(serial)
        if camera is None:
            raise RuntimeError(f'无法打开 {serial}')
        mode = role.upper()
        sync = camera.get_tool(getattr(ToolType, 'SYNC', getattr(ToolType, 'TOOL_SYNC', ToolType(2))))
        if not sync.set_param('mode', mode):
            raise RuntimeError(f'{serial} 无法设置 {mode}')
        ok, actual = sync.get_param('mode')
        if not ok or str(actual) != mode:
            raise RuntimeError(f'{serial} 模式回读失败: {actual}')
        config = {'serial': serial, 'sync_mode': str(actual), 'requested_batch_us': batch_us}
        bias = camera.get_tool(ToolType.BIAS)
        for name in BIAS:
            if not bias.set_param(name, 0):
                raise RuntimeError(f'{serial} 设置 {name}=0 失败')
            ok, actual = bias.get_param(name)
            if not ok or int(actual) != 0:
                raise RuntimeError(f'{serial} {name} 回读失败: {actual}')
            config[name] = int(actual)
        config['event_rate_control'] = configure_event_rate_control(camera.get_tool(ToolType.EVENT_RATE_CONTROL), False, 5, serial)
        config['trail_filter'] = configure_trail_filter(camera.get_tool(ToolType.EVENT_TRAIL_FILTER), False, 1, serial)
        config['anti_flicker_enabled'] = configure_boolean_parameter(camera.get_tool(ToolType.ANTI_FLICKER), 'enable', False, serial)
        roi = camera.get_tool(ToolType.ROI)
        config['roi'] = {}
        for name in ('x', 'y', 'width', 'height'):
            ok, value = roi.get_param(name)
            if not ok:
                raise RuntimeError(f'{serial} ROI {name} 读取失败')
            config['roi'][name] = int(value)
        camera.set_batch_events_time(batch_us)
        width, height = camera.get_width(), camera.get_height()
        config.update(width=int(width), height=int(height))
        messages.put(('configured', role, config))
        if not gate.wait(30):
            raise TimeoutError(f'{serial} 等待启动超时')
        if shutdown.is_set():
            return
        rc = camera.start()
        if rc != 0:
            raise RuntimeError(f'{serial} 启动失败，SDK 返回码 {rc}')
        streaming = True; started.set()
        messages.put(('started', role, time.monotonic()))
        n_batches = n_events = 0
        first = last = None
        last_publish = 0.0
        while True:
            if shutdown.is_set():
                break
            try:
                cmd = commands.get_nowait()
            except queue.Empty:
                cmd = None
            if cmd:
                if cmd['action'] == 'quit':
                    break
                if cmd['action'] == 'start' and not recording:
                    target = Path(cmd['directory']) / f'{role}_{serial}.raw'
                    rc = camera.start_recording(str(target))
                    if rc != 0:
                        raise RuntimeError(f'{serial} 开始录制失败，SDK 返回码 {rc}')
                    recording = True
                    messages.put(('recording', role, {'file': str(target), 'host_monotonic': time.monotonic()}))
                if cmd['action'] == 'stop' and recording:
                    rc = camera.stop_recording()
                    if rc != 0:
                        raise RuntimeError(f'{serial} 停止录制失败，SDK 返回码 {rc}')
                    recording = False
                    messages.put(('recorded', role, {'host_monotonic': time.monotonic()}))
            events = camera.get_next_batch()
            now = time.monotonic()
            if events is not None and len(events):
                ts = events['timestamp']
                n_batches += 1; n_events += len(events)
                first = int(ts[0]) if first is None else first
                last = int(ts[-1])
            if now - last_publish >= 0.05:
                image = np.full((360, 640), 127, np.uint8)
                if events is not None and len(events):
                    sample = events[::max(1, (len(events) + 99999) // 100000)]
                    x = sample['x'].astype(np.intp); y = sample['y'].astype(np.intp)
                    valid = (x < width) & (y < height)
                    px = x[valid] * 640 // width; py = y[valid] * 360 // height
                    image[py, px] = np.where(sample['polarity'][valid] != 0, 255, 0).astype(np.uint8)
                put_latest(preview, (image, now, serial, len(events) if events is not None else 0, last))
                last_publish = now
    except Exception:
        messages.put(('error', role, traceback.format_exc()))
    finally:
        if recording:
            try: camera.stop_recording()
            except Exception: pass
        if streaming:
            try: camera.stop()
            except Exception: pass
        if 'n_batches' in locals():
            messages.put(('statistics', role, {'batches': n_batches, 'events': n_events,
                                              'first_event_us': first, 'last_event_us': last}))
        messages.put(('closed', role, None))


def main():
    mp.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('--master', default='DSP01F1FFF3E')
    parser.add_argument('--slave', default='DSP01F1FFF59')
    parser.add_argument('--batch-us', type=int, default=1000)
    parser.add_argument('--smoke-seconds', type=float, default=0,
                        help='automatically record a short test after both previews appear')
    parser.add_argument('--output-dir', type=Path, default=Path('output/isolated_capture'))
    args = parser.parse_args()
    if args.master == args.slave or args.batch_us <= 0 or args.smoke_seconds < 0:
        parser.error('camera serials must differ; batch duration must be positive and smoke seconds nonnegative')
    messages = mp.Queue()
    shutdown = mp.Event()
    gates = {r: mp.Event() for r in ('slave', 'master')}
    started = {r: mp.Event() for r in gates}
    cmds = {r: mp.Queue() for r in gates}
    previews = {r: mp.Queue(maxsize=1) for r in gates}
    workers = {}
    info = {}
    record_acks = {}
    stop_acks = {}
    statistics = {}
    display_stats = {role: {'frames': 0, 'first_monotonic': None, 'last_monotonic': None,
                            'max_transport_age_s': 0.0} for role in ('master', 'slave')}
    directory = None
    status = 'STARTING'
    error = None
    try:
        # Initialize separately; each process owns exactly one camera handle.
        for role, serial in [('slave', args.slave), ('master', args.master)]:
            proc = mp.Process(target=camera_worker, args=(role, serial, args.batch_us, gates[role],
                                                           started[role], cmds[role], messages, previews[role], shutdown))
            workers[role] = proc; proc.start()
            while role not in info:
                kind, source, value = messages.get(timeout=30)
                if kind == 'error': raise RuntimeError(value)
                if kind == 'configured': info[source] = value
        fields = list(BIAS) + ['event_rate_control', 'trail_filter', 'anti_flicker_enabled', 'roi', 'width', 'height']
        mismatched = [field for field in fields if info['master'][field] != info['slave'][field]]
        if mismatched:
            raise RuntimeError(f'两机参数不一致，停止采集: {mismatched}; {info}')
        gates['slave'].set()
        if not started['slave'].wait(10): raise RuntimeError('从机启动失败')
        gates['master'].set()
        if not started['master'].wait(10): raise RuntimeError('主机启动失败')
        print('READY: MASTER', args.master, 'SLAVE', args.slave, flush=True)
        print(json.dumps(info, ensure_ascii=False), flush=True)
        status = 'READY: SPACE to start'
        pictures = {role: np.full((360, 640), 127, np.uint8) for role in gates}
        last_frame = {role: 0.0 for role in gates}
        preview_meta = {role: ('', 0, None) for role in gates}
        while True:
            try:
                while True:
                    kind, role, value = messages.get_nowait()
                    if kind == 'error': raise RuntimeError(value)
                    if kind == 'recording': record_acks[role] = value
                    if kind == 'recorded': stop_acks[role] = value
                    if kind == 'statistics': statistics[role] = value
            except queue.Empty:
                pass
            if any(not p.is_alive() for p in workers.values()):
                raise RuntimeError('相机进程意外退出；检查终端错误信息')
            for role in ('master', 'slave'):
                try:
                    while True:
                        pictures[role], last_frame[role], serial, count, last_ts = previews[role].get_nowait()
                        preview_meta[role] = (serial, count, last_ts)
                        observed = time.monotonic()
                        stats = display_stats[role]
                        stats['frames'] += 1
                        stats['first_monotonic'] = stats['first_monotonic'] or observed
                        stats['last_monotonic'] = observed
                        stats['max_transport_age_s'] = max(stats['max_transport_age_s'], observed - last_frame[role])
                except queue.Empty:
                    pass
            panels = []
            for role in ('master', 'slave'):
                panel = cv2.cvtColor(pictures[role], cv2.COLOR_GRAY2BGR)
                serial, count, last_ts = preview_meta[role]
                cv2.putText(panel, f'{role.upper()} {serial[-3:]} events={count}', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 1)
                cv2.putText(panel, f't={last_ts if last_ts is not None else "--"} us', (10, 49),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
                age = time.monotonic() - last_frame[role] if last_frame[role] else float('inf')
                if age > 1:
                    cv2.putText(panel, 'PREVIEW STALE', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                panels.append(panel)
            view = np.hstack(panels)
            cv2.putText(view, status, (10, 345), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
            cv2.imshow('DVS independent streams | SPACE start/stop | Q quit', view)
            key = cv2.waitKey(20) & 255
            if args.smoke_seconds and directory is None and all(last_frame.values()):
                key = ord(' ')
            elif args.smoke_seconds and len(record_acks) == 2 and not stop_acks:
                since = max(item['host_monotonic'] for item in record_acks.values())
                if time.monotonic() - since >= args.smoke_seconds:
                    key = ord(' ')
            if key == ord(' ') and directory is None:
                directory = (args.output_dir / datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
                directory.mkdir(parents=True, exist_ok=False)
                for q in cmds.values(): q.put({'action': 'start', 'directory': str(directory)})
                status = 'STARTING RECORDING...'
            elif key == ord(' ') and directory is not None and len(record_acks) == 2 and not stop_acks:
                for q in cmds.values(): q.put({'action': 'stop'})
                status = 'STOPPING...'
            if len(record_acks) == 2 and not stop_acks:
                status = 'RECORDING: SPACE to stop'
            if len(stop_acks) == 2:
                break
            if key in (ord('q'), 27):
                if directory is not None and len(stop_acks) != 2:
                    error = '用户提前退出，录制可能不完整'
                break
    except Exception as exc:
        error = str(exc)
        print('ERROR:', error, flush=True)
    finally:
        shutdown.set()
        for role in workers:
            gates[role].set(); cmds[role].put({'action': 'quit'})
        clean_shutdown = True
        for role, proc in workers.items():
            proc.join(timeout=8)
            if proc.is_alive():
                clean_shutdown = False
                print(f'{role} 未能正常退出；请检查设备', flush=True)
        try:
            while True:
                kind, role, value = messages.get_nowait()
                if kind == 'statistics': statistics[role] = value
                if kind == 'recording': record_acks[role] = value
                if kind == 'recorded': stop_acks[role] = value
                if kind == 'error': print(role, value, flush=True)
        except queue.Empty:
            pass
        cv2.destroyAllWindows()
        if directory is not None:
            report = {'complete': error is None and clean_shutdown and len(record_acks) == 2 and len(stop_acks) == 2,
                      'camera_processes_stopped': clean_shutdown,
                      'error': error, 'configuration': info, 'record_start': record_acks,
                      'record_stop': stop_acks, 'preview_statistics': statistics,
                      'display_statistics': display_stats}
            (directory / 'capture.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print('RESULT', directory, json.dumps(report, ensure_ascii=False), flush=True)
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
