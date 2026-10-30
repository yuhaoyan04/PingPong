"""Camera-specific, reproducible binary-event enhancement profiles."""
from .event_denoise import neighbor_filter, multiscale_contrast_filter


def enhance_for_camera(binary, camera, profiles):
    if camera not in profiles:
        raise ValueError(f'no enhancement profile for camera {camera!r}')
    config = profiles[camera]
    if config['method'] == 'neighbor':
        return neighbor_filter(binary, int(config['radius']), int(config['min_neighbors']))
    if config['method'] == 'multiscale_contrast':
        return multiscale_contrast_filter(binary, float(config['threshold']),
                                          float(config['small_sigma']), float(config['large_sigma']))
    raise ValueError(f'unknown enhancement method for {camera!r}: {config["method"]!r}')
