#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
camera="${1:?Usage: bash tools/yolo/predict_latest.sh F3E_or_F59 image_or_directory [gpu]}"
source_path="${2:?Provide a filtered binary image or directory for the matching camera}"
gpu="${3:-0}"
case "$camera" in F3E|F59) ;; *) echo 'Camera must be F3E or F59' >&2; exit 1 ;; esac
export YOLO_CONFIG_DIR="$PWD/experiments/two_camera_v1/ultralytics_config"
export MPLBACKEND=Agg
if [[ $# -ge 3 ]]; then shift 3; else shift 2; fi
# Tracking is meaningful on an ordered directory/sequence. A single image uses
# high-confidence + ball-ROI filtering without a temporal confirmation delay.
exec .venv/bin/python -u tools/yolo/apply_model.py \
  --camera "$camera" --source "$source_path" --device "$gpu" "$@"
