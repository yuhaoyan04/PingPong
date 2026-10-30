#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_path="${PINGPONG_PYTHON:-$PWD/.venv/bin/python}"
# Install only the SDK into this project environment; preserve working CUDA/numpy.
"$python_path" -m pip install --no-deps --only-binary=:all: --index-url https://pypi.org/simple -r requirements-camera.txt
"$python_path" tools/check_camera_sdk.py
