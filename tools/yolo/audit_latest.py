"""Full pixel/label/provenance audit; original package remains immutable."""
import hashlib
import json
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

from common import ROOT, DATA, EXP, dump, rows, table
import numpy as np
import yaml
from PIL import Image, ImageDraw


def png(path, gray=True):
    with Image.open(path) as im:
        assert im.size == (256, 256)
        if gray:
            assert im.mode == 'L', (path, im.mode)
            a = np.asarray(im)
            assert np.isin(a, [0, 255]).all(), path
            return a.copy()
        assert im.mode == 'RGB', (path, im.mode)
        return np.asarray(im).copy()


def main():
    out = EXP / 'audit'
    out.mkdir(exist_ok=True, parents=True)
    result = subprocess.run([sys.executable, str(DATA / 'verify_package.py')], capture_output=True, text=True)
    (out / 'package_verification.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    inventory = json.loads((DATA / 'package_manifest.json').read_text())
    expected = {g['group']: g for g in inventory['groups']}
    all_stats, provenance = [], []
    camera_stats = {}
    rng = random.Random(42)
    for camera in ('F3E', 'F59'):
        source = DATA / 'datasets' / camera
        cfg = yaml.safe_load((source / 'dataset.yaml').read_text())
        assert cfg == {'train': 'images/train', 'val': 'images/train', 'channels': 1, 'names': {0: 'marker'}}
        sources = {}
        for p in sorted((DATA / 'group_metadata' / camera).glob('*/manifest.csv')):
            for r in rows(p):
                sources[r['group'] + '__' + __import__('pathlib').Path(r['image']).stem] = r
        count = Counter()
        groups = defaultdict(Counter)
        train_hashes = defaultdict(list)
        images = sorted((source / 'images/train').glob('*.png'))
        for image in images:
            assert image.stem in sources
            r = sources[image.stem]
            png(image)
            label = source / 'labels/train' / (image.stem + '.txt')
            lines = label.read_text().splitlines()
            assert len(lines) == len(set(lines))
            for line in lines:
                a = np.array([float(v) for v in line.split()])
                assert len(a) == 5 and np.isfinite(a).all()
                c, x, y, w, h = a
                assert c == 0 and 0 < w <= 1 and 0 < h <= 1
                assert min(x - w / 2, y - h / 2) >= -1e-7 and max(x + w / 2, y + h / 2) <= 1 + 1e-7
            rec = dict(camera=camera, group=r['group'], image=image.name, frame_index=int(r['frame_index']),
                       timestamp_us=int(r['timestamp_us']), track_support=r['track_support'], boxes=len(lines), explicit_negative=not lines)
            provenance.append(rec)
            for counts in (count, groups[r['group']]):
                counts['images'] += 1
                counts['boxes'] += len(lines)
                counts['explicit_negatives'] += not lines
                counts['fallback_frames'] += r['track_support'] == 'fallback_unverified'
            train_hashes[hashlib.sha256(image.read_bytes()).hexdigest()].append(image.name)
        for group, counts in groups.items():
            assert counts['images'] == expected[group]['reviewed']
            assert counts['fallback_frames'] == expected[group]['reviewed_fallback']
        # Framework cache files go into this copy; package data and YAML are untouched.
        work = EXP / 'working_data' / camera
        if not work.exists():
            shutil.copytree(source, work)
        work_cfg = dict(cfg, path=str(work))
        (work / 'dataset.yaml').write_text(yaml.safe_dump(work_cfg, sort_keys=False))
        for p in (source / 'images/train').glob('*.png'):
            assert p.read_bytes() == (work / 'images/train' / p.name).read_bytes()
        samples = rng.sample(images, min(12, len(images)))
        dump(out / f'{camera}_GT_samples.json', [p.name for p in samples])
        sheet = Image.new('RGB', (4 * 384, 3 * 414), '#161616')
        for i, p in enumerate(samples):
            im = Image.open(p).convert('RGB').resize((384, 384), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(im)
            for line in (source / 'labels/train' / (p.stem + '.txt')).read_text().splitlines():
                _, x, y, w, h = map(float, line.split())
                draw.rectangle(((x-w/2)*384, (y-h/2)*384, (x+w/2)*384, (y+h/2)*384), outline='lime', width=2)
            col, row = i % 4, i // 4
            sheet.paste(im, (col * 384, row * 414 + 30))
            ImageDraw.Draw(sheet).text((col*384+4, row*414+3), p.stem, fill='white')
        sheet.save(out / f'{camera}_training_GT.jpg', quality=92)
        camera_stats[camera] = dict(count, training_groups={g: dict(v) for g, v in groups.items()},
                                    source_candidate_manifest_frames=len(sources),
                                    unreviewed_candidate_frames_not_used=len(sources)-len(images),
                                    independent_validation_images=0, independent_test_images=0,
                                    duplicate_train_contents=[v for v in train_hashes.values() if len(v)>1])
        for directory in sorted((DATA / 'unlabeled_inference' / camera).glob('serve*')):
            records = rows(directory / 'manifest.csv')
            meta = json.loads((directory / 'metadata.json').read_text())
            assert len(records) == meta['frames'] == expected[directory.name]['frames']
            assert len({r['image'] for r in records}) == len(records)
            stamps = [int(r['timestamp_us']) for r in records]
            assert all(b > a for a, b in zip(stamps, stamps[1:]))
            fallback = 0
            overlap = []
            for r in records:
                filename = __import__('pathlib').Path(r['image']).name
                assert r['group'] == directory.name
                filtered = png(directory / 'binary' / filename)
                unfiltered = png(directory / 'binary_unfiltered' / filename)
                polarity = png(directory / 'polarity_reference' / filename, gray=False)
                # No interpolation/coordinate change: filtered occupancy is a subset of the raw map.
                assert np.all(filtered <= unfiltered), (directory.name, filename, 'filtered pixels moved')
                assert np.array_equal(np.any(polarity != 0, axis=2), unfiltered != 0), (directory.name, filename, 'polarity coordinate mismatch')
                fallback += r['track_support'] == 'fallback_unverified'
                digest = hashlib.sha256((directory / 'binary' / filename).read_bytes()).hexdigest()
                if digest in train_hashes:
                    overlap.append(filename)
            assert fallback == meta['fallback_frames']
            video = DATA / meta['reference_video']
            assert video.is_file()
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                             'stream=width,height,r_frame_rate,nb_frames:format=duration', '-of', 'json', str(video)], text=True))
            stat = dict(camera=camera, group=directory.name, unlabeled_frames=len(records), fallback_frames=fallback,
                        first_timestamp_us=stamps[0], last_timestamp_us=stamps[-1],
                        timestamp_step_counts={str(k): v for k, v in Counter(np.diff(stamps).tolist()).items()},
                        reference_video=meta['reference_video'], reference_video_probe=probe,
                        training_content_overlap=overlap, coordinate_alignment='passed: filtered subset of unfiltered; polarity occupancy equals unfiltered')
            all_stats.append(stat)
        camera_stats[camera]['unlabeled_frames'] = sum(g['unlabeled_frames'] for g in all_stats if g['camera'] == camera)
        camera_stats[camera]['unlabeled_fallback_frames'] = sum(g['fallback_frames'] for g in all_stats if g['camera'] == camera)
    archive = ROOT / 'data/server_marker_two_camera_train_v1.zip'
    original_hashes = {str(p.relative_to(DATA)): hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.rglob('*') if p.is_file()}
    dump(out / 'source_integrity.json', dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=original_hashes))
    report = dict(status='passed', cameras=camera_stats, unlabeled_groups=all_stats,
                  total_annotated_frames=sum(v['images'] for v in camera_stats.values()),
                  total_boxes=sum(v['boxes'] for v in camera_stats.values()),
                  total_unlabeled_frames=sum(g['unlabeled_frames'] for g in all_stats),
                  independent_validation_or_test=False, random_split_performed=False,
                  limitation='All confirmed serve01/serve06 frames train only; other serves unlabeled, qualitative inference only.')
    dump(out / 'dataset_audit.json', report)
    table(out / 'training_provenance.csv', provenance)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
