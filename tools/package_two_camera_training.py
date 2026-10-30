"""Package reviewed camera-separated YOLO datasets and unlabeled serve reviews."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--zip', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.zip.exists():
        raise FileExistsError('refusing to overwrite existing package or zip')
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True)
    exporter = root/'tools'/'export_marker_yolo.py'
    for cam in ('F3E', 'F59'):
        target = args.output/'datasets'/cam
        subprocess.run([sys.executable, str(exporter), '--dataset', str(args.dataset), '--output', str(target),
                        '--train', f'serve01_{cam}', f'serve06_{cam}', '--train-only'], check=True)
    videos = args.output/'reference_videos'; videos.mkdir()
    stats = []
    for group in sorted(args.dataset.glob('serve*_F*')):
        if not (group/'source.json').exists(): continue
        source = json.loads((group/'source.json').read_text(encoding='utf8'))
        cam = source['camera']; serve = group.name.split('_')[0]
        if cam not in ('F3E','F59') or len(serve) != 7: raise ValueError(f'unexpected group {group}')
        video = Path(source['video']); destination = videos/video.name
        if not destination.exists(): shutil.copy2(video, destination)
        with (group/'manifest.csv').open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
        reviewed = []
        for row in rows:
            annotation = (group/row['image']).with_suffix('.json')
            if annotation.exists():
                obj = json.loads(annotation.read_text(encoding='utf8'))
                if obj.get('flags',{}).get('reviewed') is True: reviewed.append(row)
        is_labeled_serve = serve in ('serve01','serve06')
        dest = (args.output/'group_metadata'/cam/group.name if is_labeled_serve else
                args.output/'unlabeled_inference'/cam/group.name)
        dest.mkdir(parents=True)
        shutil.copy2(group/'manifest.csv', dest/'manifest.csv')
        meta = {k:source[k] for k in ('camera','video_first_us','video_step_us','frames','crop_radius_scale','denoise','fallback_frames') if k in source}
        meta.update(reference_video=f'reference_videos/{video.name}', group=group.name,
                    warning='Automatic candidate track and time base unverified; reference video time sampling differs.')
        (dest/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2), encoding='utf8')
        if not is_labeled_serve:
            if reviewed: raise ValueError(f'unexpected reviewed annotations in qualitative-only group {group.name}')
            for folder in ('binary','binary_unfiltered','polarity_reference'):
                (dest/folder).mkdir()
                for row in rows:
                    name = Path(row['image']).name
                    shutil.copy2(group/folder/name, dest/folder/name)
        stats.append({'group':group.name, 'frames':len(rows), 'reviewed':len(reviewed),
                      'reviewed_fallback':sum(r.get('track_support') == 'fallback_unverified' for r in reviewed),
                      'purpose':'labeled_split' if is_labeled_serve else 'unlabeled_qualitative_only'})
        print(group.name, len(rows), len(reviewed), flush=True)
    if len(stats) != 18 or len(list(videos.glob('*.mp4'))) != 9:
        raise ValueError('expected 18 groups and 9 original reference videos')
    (args.output/'package_manifest.json').write_text(json.dumps({'groups':stats,'warning':
       'Both labeled serves are train only; no independent validation or test. Unlabeled clips are qualitative only.'},indent=2),encoding='utf8')
    shutil.copy2(root/'docs'/'training'/'SERVER_TWO_CAMERA_PROMPT.md', args.output/'SERVER_PROMPT.md')
    shutil.copy2(root/'tools'/'verify_two_camera_package.py', args.output/'verify_package.py')
    (args.output/'README.md').write_text(
        '# 双相机事件标记数据包\n\n先阅读 SERVER_PROMPT.md。运行 `python verify_package.py`。\n\n'
        'F3E 与 F59 分开训练；01 和 06 的已确认帧全部训练，没有独立验证集。'
        '其余发球只有未标注候选，不是定量测试集。reference_videos 保存原复核视频，'
        '时间对应信息见各组 metadata 和 manifest。\n',encoding='utf8')
    hashes=[]
    for file in sorted(p for p in args.output.rglob('*') if p.is_file()):
        digest=hashlib.sha256()
        with file.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): digest.update(chunk)
        hashes.append(f'{digest.hexdigest()}  {file.relative_to(args.output).as_posix()}')
    (args.output/'SHA256SUMS.txt').write_text('\n'.join(hashes)+'\n',encoding='utf8')
    with zipfile.ZipFile(args.zip,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
        for file in sorted(p for p in args.output.rglob('*') if p.is_file()):
            z.write(file,Path(args.output.name)/file.relative_to(args.output))
    print(f'package={args.output.resolve()} zip={args.zip.resolve()}',flush=True)


if __name__=='__main__': main()
