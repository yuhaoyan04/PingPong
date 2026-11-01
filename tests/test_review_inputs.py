import json
import zipfile

import pytest

from pingpong_event_pipeline.artifacts import atomic_json, file_sha256
from review_spin_direction import prepare, package


@pytest.fixture
def source(tmp_path):
    source = tmp_path / "source"
    (source / "tracks").mkdir(parents=True)
    (source / "spin").mkdir()
    variants = []
    for mode in ["baseline", "geometry_only", "combined"]:
        tracks = source / "tracks" / f"{mode}.csv"
        tracks.write_text("fixture input for " + mode, encoding="utf-8")
        atomic_json(source / "spin" / f"{mode}.json",
                    {"tracks_sha256": file_sha256(tracks), "serves": [], "config": {}})
        variants.append({"mode": mode, "dataset": "dataset", "tracks": f"tracks/{mode}.csv"})
    atomic_json(source / "variants.json", variants)
    return source


def test_source_profiles_are_portable_and_changed_valid_inputs_refresh_outputs(tmp_path, source):
    output = tmp_path / "review"
    prepared = prepare(tmp_path, source, output)
    assert prepared["baseline"]["dataset"] == str(source / "dataset")
    tracks = source / "tracks/combined.csv"
    tracks.write_text("new reviewed observations", encoding="utf-8")
    atomic_json(source / "spin/combined.json",
                {"tracks_sha256": file_sha256(tracks), "serves": [], "config": {}})
    updated = prepare(tmp_path, source, output)
    assert updated["combined"]["tracks_sha256"] == file_sha256(tracks)
    assert (output / "tracks/combined.csv").read_bytes() == tracks.read_bytes()


def test_stale_report_is_rejected_before_replacing_previous_review_inputs(tmp_path, source):
    output = tmp_path / "review"
    prepare(tmp_path, source, output)
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    (source / "tracks/combined.csv").write_text("changed without refitting", encoding="utf-8")
    with pytest.raises(ValueError, match="spin report / tracks mismatch"):
        prepare(tmp_path, source, output)
    assert before == {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}


def test_package_failure_keeps_previous_download_intact(tmp_path, monkeypatch):
    (tmp_path / "review.html").write_text('<a href="spin_review_bundle.zip">下载审核包</a>', encoding="utf-8")
    package(tmp_path)
    archive = tmp_path / "spin_review_bundle.zip"
    before = archive.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated full disk")
    (tmp_path / "new.csv").write_text("new data", encoding="utf-8")
    monkeypatch.setattr(zipfile.ZipFile, "write", fail)
    with pytest.raises(OSError, match="full disk"):
        package(tmp_path)
    assert archive.read_bytes() == before
    assert not list(tmp_path.glob(".spin_review_bundle-*"))
