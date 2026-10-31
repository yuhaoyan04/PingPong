import copy

import numpy as np
import pytest

from pingpong_event_pipeline.marker_spin import read_tracks
from pingpong_event_pipeline.radius_review import current_fit, baseline_source, frame_value
from test_yolo_marker_spin import input_fixture


def test_pixel_radius_changes_sphere_geometry_without_mutating_pixels_or_time(tmp_path):
    tracks, dataset = input_fixture(tmp_path)
    before = {p: p.read_bytes() for p in dataset.rglob('*') if p.is_file()}
    default, _, _ = read_tracks(tracks, dataset)
    scaled, sources, _ = read_tracks(tracks, dataset, sphere_radius_multiplier=1.5)
    a, b = next(iter(default.values()))[0], next(iter(scaled.values()))[0]
    assert b['radius'] == pytest.approx(160)
    assert b['sphere'][0] == pytest.approx(a['sphere'][0] / 1.5)
    assert b['timestamp'] == a['timestamp']
    np.testing.assert_array_equal(b['point'], a['point'])
    assert all(p.read_bytes() == content for p, content in before.items())
    assert sources['F3E/serve02_F3E']['nominal_sphere_radius_px'] == pytest.approx(106.6666667)


@pytest.mark.parametrize('factor', [0., -1., float('nan'), float('inf')])
def test_invalid_radius_factor_is_rejected_before_reading_inputs(factor):
    with pytest.raises(ValueError, match='finite and positive'):
        read_tracks('missing.csv', 'missing_data', sphere_radius_multiplier=factor)


def fixture_serve(rps=2):
    fit = {'numerically_usable': True, 'candidate_rps': rps, 'candidate_spin_rpm': rps*60,
           'omega_camera_rad_s': [rps*2*np.pi, 0, 0], 'observation_timestamps_us': [100, 500],
           'inlier_fraction': 1., 'unique_observations': 8,
           'rotation_signal': 'near_zero_unresolved_direction' if rps == 0 else 'resolved_rotation_candidate'}
    window = {**fit, 'window_id': 'serve/0', 'start_us': 0, 'end_us': 1000,
              'mono': {'F3E': copy.deepcopy(fit), 'F59': copy.deepcopy(fit)}}
    return {'windows': [window], 'R_F59_to_F3E': np.diag([-1.,1.,-1.]).tolist(),
            'joint_summary': {'engineering_stable': False}, 'views': {}}


def test_overlay_never_holds_a_value_into_an_unobserved_gap():
    serve = fixture_serve()
    assert current_fit(serve, 100, 'JOINT')
    assert current_fit(serve, 300, 'JOINT') is None
    assert current_fit(serve, 1100, 'F3E') is None
    assert baseline_source(serve, 100) == 'JOINT'
    assert baseline_source(serve, 300) is None
    assert frame_value(serve, 300, 'JOINT')['rps'] is None


def test_zero_spin_is_displayed_as_zero_and_opposite_camera_directions_are_signed():
    zero = frame_value(fixture_serve(0), 100, 'JOINT')
    assert zero['available'] and zero['rps'] == zero['rpm'] == 0
    assert zero['axis'] is None
    assert zero['directions']['F3E']['front_motion_label'] == 'unresolved'
    rotating = frame_value(fixture_serve(), 100, 'JOINT')
    assert rotating['directions']['F3E']['front_motion_label'] == 'down'
    assert rotating['directions']['F59']['front_motion_label'] == 'up'
    assert rotating['rpm'] == 60 * rotating['rps']


def test_radius_variant_does_not_switch_source_when_locked_joint_fit_is_rejected():
    serve = fixture_serve()
    serve['windows'][0]['numerically_usable'] = False
    value = frame_value(serve, 100, 'JOINT')
    assert not value['available'] and value['rps'] is None
    assert value['independent']['F3E']['rps'] == 2


def test_radius_comparison_does_not_switch_to_a_different_overlapping_window():
    serve = fixture_serve()
    other = copy.deepcopy(serve['windows'][0])
    other['window_id'] = 'serve/500'
    other['start_us'] = 500
    other['end_us'] = 1500
    serve['windows'].append(other)
    serve['windows'][0]['numerically_usable'] = False
    assert current_fit(serve, 500, 'JOINT')[0]['window_id'] == 'serve/500'
    pinned = frame_value(serve, 500, 'JOINT', 'serve/0')
    assert pinned['rps'] is None and pinned['comparison_window_id'] == 'serve/0'
