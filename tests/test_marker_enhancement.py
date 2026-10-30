import numpy as np
import pytest
from pingpong_event_pipeline.marker_enhancement import enhance_for_camera


def test_per_camera_profiles_and_no_added_pixels():
    rng = np.random.default_rng(3)
    img = ((rng.random((64,64)) < .1)*255).astype(np.uint8)
    profiles = {'F3E': {'method':'neighbor','radius':3,'min_neighbors':2},
                'F59': {'method':'multiscale_contrast','threshold':.035,'small_sigma':3,'large_sigma':12}}
    for cam in profiles:
        out = enhance_for_camera(img, cam, profiles)
        assert out.shape == img.shape and np.all(out <= img)
    with pytest.raises(ValueError): enhance_for_camera(img, 'unknown', profiles)
