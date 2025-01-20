from PIL import Image
import numpy as np
from typing import Union
from enum import Enum
from typing import Union
import numpy as np

class LazyImageArray:
    def __init__(self, file, img_size, index=0):
        self.file = file
        self.index = index
        self.img_size = img_size
        self.array = None

    def _load_image(self):
        if self.array is not None:
            return
        pil_img = Image.open(self.file)
        if 0 != self.index:
            assert '.tif' in self.file, "File has index, but is not a tiff"
            pil_img.seek(self.index)
        if self.img_size != pil_img.size:
            pil_img = pil_img.resize(self.img_size)
        self.array = np.asarray(pil_img.convert('L'))[None]

    def __array__(self, dtype=None):
        self._load_image()
        return self.array if dtype is None else self.array.astype(dtype)

    def __getattr__(self, name):
        self._load_image()
        return getattr(self.array, name)

    def __getitem__(self, key):
        self._load_image()
        return self.array.__getitem__(key)

    def __setitem__(self, key, value):
        self._load_image()
        return self.array.__setitem__(key, value)

    def __repr__(self):
        self._load_image()
        return self.array.__repr__()


class Distribution(Enum):
    UNIFORM = "Uniform"
    EXPONENTIAL = "Exponential"
    CONSTANT = "Constant"


class RandomRange:
    min: float
    max: float
    dist: Distribution = Distribution.UNIFORM

    def __init__(self, vmin: float, vmax: float, dist: Union[str, Distribution] = None):
        self.min = vmin
        self.max = vmax
        if dist is None:
            dist = Distribution.UNIFORM
        if isinstance(dist, str):
            dist = Distribution[dist.upper()]
        self.dist = dist

    def __call__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if self.dist is Distribution.UNIFORM:
            return rng.uniform(self.min, self.max)
        if self.dist is Distribution.EXPONENTIAL:
            logmin = np.log10(self.min) if self.min > 0 else 0
            return 10 ** rng.uniform(logmin, np.log10(self.max))
        if self.dist is Distribution.CONSTANT:
            rng.uniform(0, 1)  # Consume a random number for consistency
            return self.max
        assert False, "Invalid distribution"

    def to_dict(self):
        return {
            "vmin": self.min,
            "vmax": self.max,
            "dist": str(self.dist).split('.')[-1]
        }


class RandomComplex:
    real: RandomRange
    imaginary: RandomRange

    def __init__(self, real: Union[RandomRange, dict], imaginary: Union[RandomRange, dict]):
        if isinstance(real, dict):
            real = RandomRange(**real)
        if isinstance(imaginary, dict):
            imaginary = RandomRange(**imaginary)
        self.real = real
        self.imaginary = imaginary

    def __call__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return complex(self.real(rng), self.imaginary(rng))

    def to_dict(self):
        return {
            "real": self.real.to_dict(),
            "imaginary": self.imaginary.to_dict()
        }


class RandomBool:
    probability: float

    def __init__(self, probability: float):
        self.probability = probability

    def __call__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if self.probability < 1e-6:
            return False
        if self.probability > 1 - 1e-6:
            return True
        return rng.uniform(0, 1) < self.probability

    def to_dict(self):
        return {
            "probability": self.probability
        }
