"""Compatibility entry point for validated YOLO dataset export."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pingpong_event_pipeline.yolo_export import main, rectangle_to_yolo

__all__ = ['main', 'rectangle_to_yolo']


if __name__ == '__main__':
    main()
