"""Compatibility entry point for the marker annotation UI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pingpong_event_pipeline.annotation_ui import AnnotationCanvas, Window, main

__all__ = ['AnnotationCanvas', 'Window', 'main']


if __name__ == '__main__':
    main()
