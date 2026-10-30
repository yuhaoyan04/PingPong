from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["cameras"]["master_serial"] == config["cameras"]["slave_serial"]:
        raise ValueError("主从相机序列号不能相同")
    if config["events"]["step_us"] > config["events"]["window_us"]:
        raise ValueError("step_us 不能大于 window_us")
    return config
