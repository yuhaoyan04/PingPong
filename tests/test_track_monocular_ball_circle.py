import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "track_monocular_ball_circle",
    Path(__file__).parents[1] / "tools" / "track_monocular_ball_circle.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_start_offset_uses_shared_origin_when_provided():
    assert module.resolve_start_us(1000, .5, 2000) == 502000


def test_start_offset_defaults_to_individual_raw_origin():
    assert module.resolve_start_us(1000, .5) == 501000
