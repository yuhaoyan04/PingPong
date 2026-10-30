from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import run_offline


def main():
    parser = argparse.ArgumentParser(description="DVSLume 双目乒乓球离线追踪")
    parser.add_argument("--master-raw", required=True)
    parser.add_argument("--slave-raw", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--calibration")
    parser.add_argument("--output", default="output/states.csv")
    parser.add_argument("--preview", default="output/preview.mp4")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--start-offset-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float)
    args = parser.parse_args()
    result = run_offline(args.master_raw, args.slave_raw, load_config(args.config), args.output,
                         None if args.no_preview else args.preview, args.calibration, args.max_windows,
                         args.start_offset_s, args.duration_s)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
