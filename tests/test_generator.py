import pytest
from common import config
from fakeholo.generator import SampleGenerator

@pytest.fixture
def generator(config):
    return SampleGenerator.from_dict(config)

def test_generate(generator):
    holo = generator.generate((128, 128))
    assert holo.image is not None
    assert holo.bounding_boxes is not None