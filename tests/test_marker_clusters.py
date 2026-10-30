import numpy as np

from pingpong_event_pipeline.marker_clusters import cluster_events, confirm_offline, associate_offline


def test_single_polarity_and_near_edge_are_not_rejected():
    image = np.zeros((256, 256, 3), np.uint8)
    image[125:128, 242:245, 0] = 255
    clusters, denoised, _, _ = cluster_events(image)
    assert len(clusters) == 1
    assert np.count_nonzero(denoised[:, :, 0]) == 9


def test_yellow_only_and_blue_only_clusters_are_detected():
    image = np.zeros((256, 256, 3), np.uint8)
    image[100:103, 100:103, 0] = 255
    image[140:143, 140:143, 1:3] = 255
    clusters, _, _, _ = cluster_events(image, neighbor_radius=5, min_neighbors=3,
                                        merge_radius=4, min_pixels=5)
    assert len(clusters) == 2


def test_near_mixed_polarities_merge_but_distant_ones_do_not():
    image = np.zeros((256, 256, 3), np.uint8)
    image[100:103, 100:103, 0] = 255
    image[100:103, 109:112, 1:3] = 255
    image[100:103, 140:143, 0] = 255
    clusters, _, labels, _ = cluster_events(image, neighbor_radius=5, min_neighbors=3,
                                             merge_radius=4, min_pixels=5)
    assert len(clusters) == 2
    assert labels[101, 101] == labels[101, 110]
    assert labels[101, 101] != labels[101, 141]


def test_isolated_pixel_removed_and_nearby_pieces_merged():
    image = np.zeros((256, 256, 3), np.uint8)
    image[100:102, 100:102, 1] = 255
    image[100:102, 110:112, 0] = 255
    image[160, 160, 0] = 255
    clusters, denoised, _, _ = cluster_events(image)
    assert len(clusters) == 1 and clusters[0]['pixels'] == 8
    assert not denoised[160, 160].any()


def test_shape_filter_does_not_exclude_all_near_edge_candidates():
    image = np.zeros((256, 256, 3), np.uint8)
    image[125:130, 240:245, 0] = 255
    image[100:103, 80:125, 1] = 255
    clusters, _, _, _ = cluster_events(image, merge_radius=2, max_extent=35, max_aspect=3)
    assert len(clusters) == 1 and clusters[0]['x'] > 230


def test_future_confirms_first_frame_and_isolated_noise_is_rejected():
    frames = [
        [{'x': 10, 'y': 10, 'pixels': 8}, {'x': 100, 'y': 100, 'pixels': 5}],
        [{'x': 12, 'y': 10, 'pixels': 7}],
        [],
        [{'x': 16, 'y': 10, 'pixels': 9}],
    ]
    assert confirm_offline(frames, distance=5) == [{0}, {0}, set(), set()]


def test_similar_position_with_incompatible_size_does_not_confirm():
    frames = [[{'x': 10, 'y': 10, 'pixels': 5}], [{'x': 11, 'y': 10, 'pixels': 30}]]
    assert confirm_offline(frames, size_ratio=4) == [set(), set()]


def test_track_recovers_weak_start_and_gap_without_fabricating_pixels():
    frames = [[{'x': 10, 'y': 10, 'pixels': 3, 'strong': False}],
              [{'x': 12, 'y': 10, 'pixels': 6, 'strong': True}], [],
              [{'x': 15, 'y': 10, 'pixels': 3, 'strong': False}],
              [{'x': 100, 'y': 100, 'pixels': 8, 'strong': True}]]
    assert associate_offline(frames, distance=10) == [{0}, {0}, set(), {0}, set()]


def test_weak_only_and_short_tracks_do_not_qualify():
    frames = [[{'x': 10, 'y': 10, 'pixels': 3, 'strong': False}],
              [{'x': 11, 'y': 10, 'pixels': 3, 'strong': False}],
              [{'x': 12, 'y': 10, 'pixels': 3, 'strong': False}]]
    assert associate_offline(frames) == [set(), set(), set()]
