"""Cross-platform source-checkout entry point: python pingpong.py --help."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pingpong_event_pipeline.workflow import main

if __name__ == "__main__":
    main()
