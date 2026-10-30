"""Transcode OpenCV FMP4 review videos to browser-compatible H.264."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def transcode(source: Path) -> Path:
    destination = source.with_name(source.stem + ".browser.mp4")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames_expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"avc1"),
                             fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("H.264 encoder could not be opened")
    frames = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        writer.write(image); frames += 1
    capture.release(); writer.release()
    if frames != frames_expected or not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"incomplete transcode {source}: {frames}/{frames_expected}")
    check = cv2.VideoCapture(str(destination))
    codec = int(check.get(cv2.CAP_PROP_FOURCC)); check_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()
    codec_name = "".join(chr((codec >> (8 * index)) & 255) for index in range(4))
    if codec_name.lower() not in {"h264", "avc1"} or check_frames != frames:
        raise RuntimeError(f"invalid browser video {destination}: {codec_name}, {check_frames}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    sources = sorted(path for path in args.root.rglob("*.mp4")
                     if not path.name.endswith(".browser.mp4"))
    for index, source in enumerate(sources, 1):
        destination = transcode(source)
        print(f"[{index}/{len(sources)}] {destination}")


if __name__ == "__main__":
    main()
