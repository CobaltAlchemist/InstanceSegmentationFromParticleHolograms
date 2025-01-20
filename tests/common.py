import pytest
import json

@pytest.fixture(scope="session")
def config():
    cfg = '{"contour_gen": {"kernel_size": 8, "threshold": 0.94, "min_area": 5, "max_count": 300, "type": "Blob"}, "medium": {"ior": {"real": {"vmin": 0.3, "vmax": 1.3, "dist": "Uniform"}, "imaginary": {"vmin": 0.0, "vmax": 0.0, "dist": "Uniform"}}, "intensity": {"vmin": 0.2, "vmax": 1.0, "dist": "Uniform"}}, "laser": {"wavelength": {"vmin": 0.238, "vmax": 0.938, "dist": "Uniform"}}, "property_gen": {"depth": {"vmin": 1000.0, "vmax": 1000000.0, "dist": "Exponential"}, "far_field_thresh": 1000000.0, "distribution": "exponential", "ior": {"real": {"vmin": 1.0, "vmax": 1.8, "dist": "Uniform"}, "imaginary": {"vmin": 0.0, "vmax": 1.0, "dist": "Uniform"}}}, "synthesis": {"output_size": [1024, 1024], "resolution": 4.0, "oversample": 1, "gaussian_sigma": 99, "gaussian": false, "padding": true}, "seed": 219254592, "torch": true}'
    return json.loads(cfg)