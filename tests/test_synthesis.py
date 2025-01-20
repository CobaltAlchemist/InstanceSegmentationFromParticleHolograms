import pytest
import numpy as np
import cv2
from fakeholo.synthesis import TorchSynthesizer,Medium, Laser, HologramMask

@pytest.fixture
def synthesizer():
    return TorchSynthesizer(
        resolution=20.0,
        output_size=(300, 300),
        oversample=2
    )

@pytest.fixture
def medium():
    return Medium()

@pytest.fixture
def laser():
    return Laser()

@pytest.fixture
def object():
    canvas = np.zeros((300, 300), dtype=np.uint8)
    canvas = cv2.ellipse(canvas, (150, 150), (100, 50), 0, 0, 360, 255, -1)
    return HologramMask(mask=canvas, depth=2000.0)

def test_generate_hologram_mask(synthesizer, medium, laser, object):
    result = synthesizer.generate(medium, [laser], [object])
    assert result.shape == (300, 300), f"Expected shape (300, 300), but got {result.shape}"
    assert np.isfinite(result).all(), "Result contains non-finite values"

def test_generate_with_padding(synthesizer, medium, laser, object):
    synthesizer.enable_padding = True
    result = synthesizer.generate(medium, [laser], [object])
    assert result.shape == (300, 300), f"Expected shape (300, 300), but got {result.shape}"
    assert np.isfinite(result).all(), "Result contains non-finite values"

def test_generate_with_gaussian(synthesizer, medium, laser, object):
    synthesizer.enable_gaussian = True
    result = synthesizer.generate(medium, [laser], [object])
    assert result.shape == (300, 300), f"Expected shape (300, 300), but got {result.shape}"
    assert np.isfinite(result).all(), "Result contains non-finite values"