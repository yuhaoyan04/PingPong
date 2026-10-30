"""Checked acquisition controls shared by live DVSLume entry points."""
from __future__ import annotations


def _as_bool(value) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "on", "enabled"}:
            return True
        if lowered in {"false", "0", "off", "disabled"}:
            return False
    return bool(value)


def configure_event_rate_control(tool, enabled: bool, maximum_rate_mevs: int,
                                 context: str = "camera") -> dict:
    """Set ERC without silently discarding events and verify hardware readback."""
    if enabled:
        if maximum_rate_mevs <= 0:
            raise ValueError("maximum_rate_mevs must be positive when ERC is enabled")
        if not tool.set_param("max_event_rate", int(maximum_rate_mevs)):
            raise RuntimeError(f"不能设置 {context} ERC={maximum_rate_mevs} MEv/s")
        ok, actual_rate = tool.get_param("max_event_rate")
        if not ok or int(actual_rate) != int(maximum_rate_mevs):
            raise RuntimeError(f"{context} ERC 限值回读失败: {actual_rate}")
    if not tool.set_param("enable", bool(enabled)):
        raise RuntimeError(f"不能设置 {context} ERC enable={enabled}")
    ok, actual_enabled = tool.get_param("enable")
    if not ok or _as_bool(actual_enabled) != bool(enabled):
        raise RuntimeError(f"{context} ERC enable 回读失败: {actual_enabled}")
    return {
        "enabled": bool(enabled),
        "maximum_rate_mevs": int(maximum_rate_mevs) if enabled else None,
    }


def configure_trail_filter(tool, enabled: bool, threshold_ms: int,
                           context: str = "camera") -> dict:
    """Configure the texture-deleting trail filter and verify its enable bit."""
    if enabled:
        if threshold_ms <= 0:
            raise ValueError("threshold_ms must be positive when Trail Filter is enabled")
        if not tool.set_param("threshold", int(threshold_ms)):
            raise RuntimeError(f"不能设置 {context} Trail Filter threshold={threshold_ms}")
        if not tool.set_param("type", "TRAIL"):
            raise RuntimeError(f"不能设置 {context} Trail Filter type=TRAIL")
        ok, actual_threshold = tool.get_param("threshold")
        if not ok or int(actual_threshold) != int(threshold_ms):
            raise RuntimeError(f"{context} Trail Filter threshold 回读失败: {actual_threshold}")
    if not tool.set_param("enable", bool(enabled)):
        raise RuntimeError(f"不能设置 {context} Trail Filter enable={enabled}")
    ok, actual_enabled = tool.get_param("enable")
    if not ok or _as_bool(actual_enabled) != bool(enabled):
        raise RuntimeError(f"{context} Trail Filter enable 回读失败: {actual_enabled}")
    return {
        "enabled": bool(enabled),
        "threshold_ms": int(threshold_ms) if enabled else None,
    }


def spin_capture_config_valid(config: dict) -> bool:
    """Return true only when texture-deleting acquisition controls are known off."""
    required = ("bias_diff_on", "bias_diff_off", "trail_filter_enabled",
                "event_rate_control_enabled")
    return bool(
        all(name in config for name in required)
        and config["trail_filter_enabled"] is False
        and config["event_rate_control_enabled"] is False)


def configure_boolean_parameter(tool, name: str, enabled: bool,
                                context: str = "camera") -> bool:
    """Set a boolean hardware parameter and fail if readback disagrees."""
    if not tool.set_param(name, bool(enabled)):
        raise RuntimeError(f"不能设置 {context} {name}={enabled}")
    ok, actual = tool.get_param(name)
    if not ok or _as_bool(actual) != bool(enabled):
        raise RuntimeError(f"{context} {name} 回读失败: {actual}")
    return bool(enabled)
