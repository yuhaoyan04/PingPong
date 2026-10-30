#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# Fresh environment setup (do not replace a working CUDA installation blindly):
# python3.12 -m venv .venv
# .venv/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# .venv/bin/python -m pip install -r requirements-train.txt --index-url https://pypi.org/simple
# Keep the latest package extracted at data/server_marker_two_camera_train_v1/.
# Official pretrained weights: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt
if [[ -e experiments/two_camera_v1/F3E/train || -e experiments/two_camera_v1/F59/train ]]; then
  echo 'Preserve the completed run. Reproduce in a fresh workspace with this dataset, weights, environment and scripts.' >&2
  exit 1
fi
mkdir -p experiments/two_camera_v1/audit experiments/two_camera_v1/F3E experiments/two_camera_v1/F59
.venv/bin/python -u tools/yolo/audit_latest.py > experiments/two_camera_v1/audit/full_audit.log 2>&1
.venv/bin/python -u tools/yolo/train_camera.py --camera F3E --device "${F3E_GPU:-5}" > experiments/two_camera_v1/F3E/training.log 2>&1
.venv/bin/python -u tools/yolo/train_camera.py --camera F59 --device "${F59_GPU:-7}" > experiments/two_camera_v1/F59/training.log 2>&1
.venv/bin/python -u tools/yolo/infer_camera.py --camera F3E --device "${F3E_GPU:-5}" > experiments/two_camera_v1/F3E/inference.log 2>&1
.venv/bin/python -u tools/yolo/infer_camera.py --camera F59 --device "${F59_GPU:-7}" > experiments/two_camera_v1/F59/inference.log 2>&1
.venv/bin/python tools/yolo/report_latest.py
