"""Interactive, timestamp-preserving dual DVSLume preview and parameter tuning."""
from __future__ import annotations

import json
import multiprocessing as mp
import queue
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


BIAS_RANGES = {
    'bias_diff_on': (-80, 145), 'bias_diff_off': (-30, 200),
    'bias_fo': (-20, 0), 'bias_hpf': (0, 120),
    'bias_diff': (-25, 23), 'bias_refr': (-20, 235),
}
PARAMETERS = {
    **{key: ('BIAS', key) for key in BIAS_RANGES},
    'erc_enable': ('EVENT_RATE_CONTROL', 'enable'),
    'erc_rate': ('EVENT_RATE_CONTROL', 'max_event_rate'),
    'trail_enable': ('EVENT_TRAIL_FILTER', 'enable'),
    'trail_threshold': ('EVENT_TRAIL_FILTER', 'threshold'),
    'anti_flicker': ('ANTI_FLICKER', 'enable'),
}
BOOLEAN_KEYS = {'erc_enable', 'trail_enable', 'anti_flicker'}


def read_values(camera, ToolType):
    values = {}
    for key, (tool_name, name) in PARAMETERS.items():
        try:
            ok, value = camera.get_tool(getattr(ToolType, tool_name)).get_param(name)
            values[key] = value if ok else None
        except Exception:
            values[key] = None
    for key, tool_name, name in [('trail_type', 'EVENT_TRAIL_FILTER', 'type'),
                                  ('afk_low_hz', 'ANTI_FLICKER', 'low_frequency'),
                                  ('afk_high_hz', 'ANTI_FLICKER', 'high_frequency'),
                                  ('afk_mode', 'ANTI_FLICKER', 'fliter_mode')]:
        try:
            ok, actual = camera.get_tool(getattr(ToolType, tool_name)).get_param(name)
            values[key] = actual if ok else None
        except Exception:
            values[key] = None
    return values


def checked_set(camera, ToolType, key, value):
    tool_name, param = PARAMETERS[key]
    tool = camera.get_tool(getattr(ToolType, tool_name))
    if not tool.set_param(param, value):
        raise RuntimeError(f'{key} 写入失败')
    ok, actual = tool.get_param(param)
    if not ok or actual != value:
        raise RuntimeError(f'{key} 回读不一致：设定 {value}，实际 {actual}')


def event_view(events, width, height):
    """Preview is sampled for display only; hardware RAW recording remains separate."""
    image = np.full((height, width, 3), 55, dtype=np.uint8)
    n = len(events)
    if n:
        events = events[::max(1, (n + 199999) // 200000)]
        x, y = events['x'].astype(np.intp), events['y'].astype(np.intp)
        p = events['polarity'].astype(bool)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        image[y[valid & ~p], x[valid & ~p]] = (230, 100, 30)
        image[y[valid & p], x[valid & p]] = (20, 220, 255)
    return cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)


def worker(commands, messages, frames):
    cameras = {}
    started = []
    recording = False
    run_dir = None
    try:
        from dvsense_driver.camera_manager import DvsCameraManager
        from dvsense_driver.hal.camera_tool import ToolType
        manager = DvsCameraManager()
        manager.update_cameras()
        serials = {'master': 'DSP01F1FFF59', 'slave': 'DSP01F1FFF3E'}
        found = {d.serial for d in manager.get_camera_descs()}
        if not set(serials.values()) <= found:
            raise RuntimeError(f'未发现两台相机；当前设备：{sorted(found)}。请关闭其他相机软件。')
        cameras = {key: manager.open_camera(serial) for key, serial in serials.items()}
        sync_type = getattr(ToolType, 'SYNC', getattr(ToolType, 'TOOL_SYNC', ToolType(2)))
        for key, mode in [('master', 'MASTER'), ('slave', 'SLAVE')]:
            tool = cameras[key].get_tool(sync_type)
            if not tool.set_param('mode', mode):
                raise RuntimeError(f'{key} 主从模式设置失败')
            ok, actual = tool.get_param('mode')
            if not ok or str(actual) != mode:
                raise RuntimeError(f'{key} 主从模式回读失败：{actual}')
            cameras[key].set_batch_events_time(10000)
        cameras['slave'].start(); started.append('slave')
        time.sleep(0.15)
        cameras['master'].start(); started.append('master')
        messages.put(('ready', {key: read_values(cam, ToolType) for key, cam in cameras.items()}))
        deadline = None
        preview_ms = 10
        while True:
            try:
                cmd = commands.get_nowait()
            except queue.Empty:
                cmd = None
            if cmd:
                action = cmd['action']
                if action == 'quit':
                    break
                if action == 'set':
                    if recording:
                        messages.put(('error', '录制时不能更改硬件参数；请先停止录制。'))
                    else:
                        results = []
                        for role in cmd['roles']:
                            try:
                                checked_set(cameras[role], ToolType, cmd['key'], cmd['value'])
                                results.append(f'{role}: 成功')
                            except Exception as exc:
                                results.append(f'{role}: {exc}')
                        messages.put(('values', ({key: read_values(cam, ToolType) for key, cam in cameras.items()}, '；'.join(results))))
                if action == 'refresh':
                    messages.put(('values', ({key: read_values(cam, ToolType) for key, cam in cameras.items()}, '已从相机回读')))
                if action == 'preview':
                    preview_ms = cmd['ms']
                    for camera in cameras.values():
                        camera.set_batch_events_time(preview_ms * 1000)
                    messages.put(('status', f'预览批次设为 {preview_ms} ms；只影响预览取流'))
                if action == 'record' and not recording:
                    run_dir = Path(__file__).resolve().parents[1] / 'output' / 'tuning_capture' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    run_dir.mkdir(parents=True, exist_ok=False)
                    settings = {key: read_values(cam, ToolType) for key, cam in cameras.items()}
                    metadata = {'started': datetime.now().astimezone().isoformat(), 'duration_s': cmd['duration'],
                                'preview_batch_ms': preview_ms, 'serials': serials, 'settings': settings, 'completed': False}
                    (run_dir / 'capture.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
                    cameras['slave'].start_recording(str(run_dir / f"slave_{serials['slave']}.raw"))
                    recording = True
                    cameras['master'].start_recording(str(run_dir / f"master_{serials['master']}.raw"))
                    deadline = time.monotonic() + cmd['duration']
                    messages.put(('recording', str(run_dir)))
                if action == 'stop' and recording:
                    deadline = 0
            se = cameras['slave'].get_next_batch()
            me = cameras['master'].get_next_batch()
            panels = []
            for role, events in [('master', me), ('slave', se)]:
                cam = cameras[role]
                panel = event_view(events, cam.get_width(), cam.get_height())
                cv2.putText(panel, f'{role.upper()} | {len(events)} events / batch', (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
                panels.append(panel)
            ok, encoded = cv2.imencode('.jpg', np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                try:
                    frames.put_nowait(encoded.tobytes())
                except queue.Full:
                    pass
            if recording and time.monotonic() >= deadline:
                cameras['slave'].stop_recording(); cameras['master'].stop_recording()
                recording = False
                metadata['completed'] = deadline != 0
                metadata['ended'] = datetime.now().astimezone().isoformat()
                (run_dir / 'capture.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
                messages.put(('recorded', str(run_dir)))
    except Exception:
        messages.put(('fatal', traceback.format_exc()))
    finally:
        if recording:
            for role in ('slave', 'master'):
                try: cameras[role].stop_recording()
                except Exception: pass
        for role in ('slave', 'master'):
            if role in started:
                try: cameras[role].stop()
                except Exception: pass
        messages.put(('stopped', '相机已停止'))


def main():
    import io
    import tkinter as tk
    from tkinter import ttk, messagebox
    from PIL import Image, ImageTk

    mp.freeze_support()
    commands, messages, frames = mp.Queue(), mp.Queue(), mp.Queue(maxsize=1)
    process = mp.Process(target=worker, args=(commands, messages, frames))
    process.start()
    root = tk.Tk()
    root.title('双事件相机实时调参 | 主机左 / 从机右')
    root.geometry('1450x870')
    status = tk.StringVar(value='正在连接相机，请稍候……')
    target = tk.StringVar(value='两台')
    selected = tk.StringVar(value='bias_diff_on')
    value = tk.StringVar(value='0')
    preview_ms = tk.StringVar(value='10')
    duration = tk.StringVar(value='30')
    video = ttk.Label(root, text='正在等待画面……', anchor='center')
    video.pack(fill='both', expand=True, padx=8, pady=6)
    controls = ttk.Frame(root, padding=8)
    controls.pack(fill='x')
    row = ttk.Frame(controls); row.pack(fill='x', pady=3)
    ttk.Label(row, text='应用到').pack(side='left')
    ttk.Combobox(row, textvariable=target, values=('两台', '主机', '从机'), width=8, state='readonly').pack(side='left', padx=5)
    ttk.Label(row, text='参数').pack(side='left')
    ttk.Combobox(row, textvariable=selected, values=list(PARAMETERS), width=20, state='readonly').pack(side='left', padx=5)
    ttk.Label(row, text='值（开关：0 / 1）').pack(side='left')
    ttk.Entry(row, textvariable=value, width=9).pack(side='left', padx=5)
    def apply():
        key = selected.get()
        try:
            val = int(value.get())
            if key in BOOLEAN_KEYS:
                if val not in (0, 1): raise ValueError('开关值必须为 0 或 1')
                val = bool(val)
            elif key in BIAS_RANGES:
                lo, hi = BIAS_RANGES[key]
                if not lo <= val <= hi: raise ValueError(f'官方范围：{lo} 至 {hi}')
            elif key == 'trail_threshold' and not 1 <= val <= 100:
                raise ValueError('TRAIL 阈值范围为 1 至 100 ms')
            elif key == 'erc_rate' and not 1 <= val <= 320:
                raise ValueError('ERC 上限范围为 1 至 320 MEv/s')
            roles = {'两台': ['master', 'slave'], '主机': ['master'], '从机': ['slave']}[target.get()]
            commands.put({'action': 'set', 'roles': roles, 'key': key, 'value': val})
        except ValueError as exc:
            messagebox.showerror('参数错误', str(exc))
    ttk.Button(row, text='写入并回读', command=apply).pack(side='left', padx=5)
    ttk.Button(row, text='刷新两机参数', command=lambda: commands.put({'action': 'refresh'})).pack(side='left', padx=5)
    row2 = ttk.Frame(controls); row2.pack(fill='x', pady=3)
    ttk.Label(row2, text='预览累积 / 批次 ms').pack(side='left')
    ttk.Combobox(row2, textvariable=preview_ms, values=('1', '2', '5', '10', '20', '33'), width=5, state='readonly').pack(side='left', padx=5)
    ttk.Button(row2, text='应用预览时长', command=lambda: commands.put({'action': 'preview', 'ms': int(preview_ms.get())})).pack(side='left')
    ttk.Label(row2, text='录制秒数').pack(side='left', padx=(15, 0))
    ttk.Combobox(row2, textvariable=duration, values=('5', '10', '30'), width=5, state='readonly').pack(side='left', padx=5)
    ttk.Button(row2, text='开始 RAW 录制', command=lambda: commands.put({'action': 'record', 'duration': int(duration.get())})).pack(side='left')
    ttk.Button(row2, text='停止录制', command=lambda: commands.put({'action': 'stop'})).pack(side='left', padx=5)
    ttk.Label(controls, text='Bias 为相对出厂值的偏移，0 是默认；diff_on/off 对应亮/暗变化。开关 0=关闭、1=开启。').pack(anchor='w')
    ttk.Label(controls, text='erc_enable / erc_rate：事件限流开关 / 上限；trail_enable / trail_threshold：轨迹过滤开关 / 毫秒。').pack(anchor='w')
    ttk.Label(controls, text='anti_flicker：防闪烁开关。参数范围来自官方文档，实际是否生效以双机回读为准。').pack(anchor='w')
    ttk.Label(controls, text='预览是事件累积，不是普通照片；限流和滤波可能删除纹理。录制时参数锁定。').pack(anchor='w')
    info = tk.Text(controls, height=4, width=160, wrap='word')
    info.pack(fill='x', pady=3)
    ttk.Label(root, textvariable=status, anchor='w').pack(fill='x', padx=8, pady=4)
    closing = False
    def close():
        nonlocal closing
        if not closing:
            closing = True
            commands.put({'action': 'quit'})
            status.set('正在停止相机……')
    root.protocol('WM_DELETE_WINDOW', close)
    def poll():
        nonlocal closing
        try:
            while True:
                kind, payload = messages.get_nowait()
                if kind in ('ready', 'values'):
                    vals, note = (payload, '双机已连接；参数是相机当前值') if kind == 'ready' else payload
                    info.delete('1.0', 'end')
                    for role in ('master', 'slave'):
                        info.insert('end', f'{role.upper()}: ' + '  '.join(f'{k}={v}' for k, v in vals[role].items()) + '\n')
                    status.set(note)
                elif kind in ('error', 'fatal'):
                    status.set(payload)
                    if kind == 'fatal': messagebox.showerror('相机错误', payload)
                elif kind == 'stopped':
                    closing = True
                else:
                    status.set(('录制中：' if kind == 'recording' else '录制已保存：' if kind == 'recorded' else '') + str(payload))
        except queue.Empty:
            pass
        try:
            data = frames.get_nowait()
            pic = ImageTk.PhotoImage(Image.open(io.BytesIO(data)))
            video.configure(image=pic, text='')
            video.image = pic
        except queue.Empty:
            pass
        if closing and not process.is_alive():
            process.join(timeout=0.1)
            root.destroy()
            return
        root.after(50, poll)
    root.after(50, poll)
    root.mainloop()


if __name__ == '__main__':
    main()
