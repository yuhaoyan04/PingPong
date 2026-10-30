import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pingpong_event_pipeline.spin_acceptance import build_acceptance


def main():
    parser = argparse.ArgumentParser(description="转速数量级、有向转轴稳定性与逐事件时间审核")
    parser.add_argument("--spin-report", type=Path, required=True)
    parser.add_argument("--motion-csv", type=Path, required=True)
    parser.add_argument("--timing-reports", type=Path, nargs="*", default=[])
    parser.add_argument("--orientation-profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_acceptance(args.spin_report, args.motion_csv, args.timing_reports, args.output, args.orientation_profile)
    print(report["group_status_counts"])
    print(args.output / "review.html")


if __name__ == "__main__":
    main()
