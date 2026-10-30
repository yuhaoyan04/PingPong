"""Export a common RAW time range using DVSense's own video exporter."""
import argparse
import json
from pathlib import Path

from dvsense_driver.raw_file_reader import RawFileReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--master-raw', type=Path, required=True)
    parser.add_argument('--slave-raw', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    readers = {}
    ranges = {}
    for role, path in [('master', args.master_raw), ('slave', args.slave_raw)]:
        reader = RawFileReader(str(path.resolve()))
        if not reader.load_file():
            raise RuntimeError(f'Cannot load {path}')
        ok_start, start = reader.get_start_timestamp()
        ok_end, end = reader.get_end_timestamp()
        if not ok_start or not ok_end or end <= start:
            raise RuntimeError(f'Invalid timestamps for {path}')
        readers[role] = reader
        ranges[role] = (int(start), int(end))
    start = max(v[0] for v in ranges.values())
    end = min(v[1] for v in ranges.values())
    if start >= end:
        raise RuntimeError('The two RAW recordings do not overlap')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {role: (args.output_dir / f'sdk_{role}.mp4').resolve() for role in readers}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError('SDK output already exists; choose a new output directory')
    report = {'start_us': start, 'end_us': end, 'exports': {}}
    for role, reader in readers.items():
        print(f'Exporting {role} to {outputs[role]}', flush=True)
        ok = bool(reader.export_event_to_video(start, end, str(outputs[role])))
        report['exports'][role] = {'ok': ok, 'file': str(outputs[role])}
        if not ok:
            break
    (args.output_dir / 'sdk_export.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if len(report['exports']) == 2 and all(x['ok'] for x in report['exports'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
