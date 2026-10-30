#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PINGPONG_PYTHON:-$PWD/.venv/bin/python}" -u -m pingpong_event_pipeline.workflow "$@"
