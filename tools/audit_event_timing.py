"""Export every adjacent CD-event interval and cross-check against EVT3 bytes."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pingpong_event_pipeline.event_timing import audit_raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-events", type=int, default=200000)
    parser.add_argument("--no-event-export", action="store_true")
    args = parser.parse_args()
    report = audit_raw(args.raw, args.output, args.batch_events, not args.no_event_export)
    print(json.dumps({k: report[k] for k in ["raw", "sdk", "sdk_reference_exact_agreement", "decoded_intervals_verified", "current_file_matches_historical_record"]}, ensure_ascii=False, indent=2))
    return 0 if report["decoded_intervals_verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
