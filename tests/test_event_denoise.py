import numpy as np
from pingpong_event_pipeline.event_denoise import neighbor_filter, temporal_neighbor_filter, multiscale_contrast_filter
import pytest


def test_filter_is_subset_and_keeps_nearby_cluster_at_edge():
    image = np.zeros((30, 30), np.uint8)
    image[0, 0:3] = 255
    image[15, 15] = 255
    out = neighbor_filter(image, radius=2, min_neighbors=2)
    assert np.count_nonzero(out) == 3
    assert not out[15, 15]
    assert np.all(out <= image)


def test_empty_and_no_neighbors():
    image = np.zeros((5, 5), np.uint8)
    assert not neighbor_filter(image).any()
    image[2, 2] = 255
    assert np.array_equal(neighbor_filter(image, min_neighbors=0), image)


def test_temporal_filter_only_preserves_spatially_and_temporally_supported_current_pixels():
    image = np.zeros((30, 30), np.uint8)
    image[10, 10:13] = 255
    image[20, 20:23] = 255
    old = np.zeros_like(image); old[10, 9] = 255
    out = temporal_neighbor_filter(image, old, radius=2, min_neighbors=1, motion_radius=4)
    assert out[10, 10:13].all()
    assert not out[20].any()
    assert np.all(out <= image)


def test_multiscale_prominent_cluster_and_subset():
    rng = np.random.default_rng(123)
    img = ((rng.random((128, 128)) < .08)*255).astype(np.uint8)
    img[58:70, 58:70] = 255
    out = multiscale_contrast_filter(img, threshold=.04)
    assert np.all(out <= img)
    assert out[62:66, 62:66].all()
    assert np.count_nonzero(out) < np.count_nonzero(img)
    strong = multiscale_contrast_filter(img, threshold=.10)
    assert np.all(strong <= out)


def test_multiscale_validates_parameters_and_handles_empty():
    assert not multiscale_contrast_filter(np.zeros((20, 20), np.uint8)).any()
    with pytest.raises(ValueError): multiscale_contrast_filter(np.zeros((20, 20)), small_sigma=12, large_sigma=3)
