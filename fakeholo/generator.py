from ast import literal_eval
import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple
import cv2
import scipy

import numpy as np
from .data import BoundingBox, HoloImage, HoloSample, BoundingBoxFormat

from .synthesis import Synthesizer, Laser, Medium, HologramMask, TorchSynthesizer
from .utils import RandomComplex, RandomRange

@dataclass
class GeneratedHologram:
    seed: int
    image: np.ndarray
    contours: np.ndarray
    depths: List[float]


class DepthDistribution(str, Enum):
    UNIFORM = 'Uniform'
    EXPONENTIAL = 'Exponential'


class ContourGenerator:
    def generate(self, output_size: Tuple[int, int], rng: np.random.BitGenerator):
        raise NotImplementedError()


class BlobContourGenerator(ContourGenerator):
    def __init__(self, threshold=0.95, kernel_size=8, min_area=0, max_count=None, circularity_threshold=None, pinch_threshold=None, solidity_threshold=None):
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.min_area = min_area
        self.max_count = max_count
        self.pinch_threshold = pinch_threshold
        self.circularity_threshold = circularity_threshold
        self.solidity_threshold = solidity_threshold

    def _curvature(self, cv2_contour):
        dx = np.gradient(cv2_contour[:, 0, 0])
        dy = np.gradient(cv2_contour[:, 0, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        return (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

    def _is_pinched(self, cv2_contour):
        try:
            hull = cv2.convexHull(cv2_contour, returnPoints=False)
            defects = cv2.convexityDefects(cv2_contour, hull)
            area = cv2.contourArea(cv2_contour)
            if defects is None:
                return False
            farthest_points = [defect[0][2] for defect in defects]
            pairwise_distances = scipy.spatial.distance.pdist(
                cv2_contour[farthest_points, 0, :])
            if any(pairwise_distances < (self.pinch_threshold * np.sqrt(area))):
                return True
            return False
        except Exception as e:
            return True

    def _circularity(self, cv2_contour):
        area = cv2.contourArea(cv2_contour)
        perimeter = cv2.arcLength(cv2_contour, True)
        return 4 * np.pi * area / perimeter ** 2

    def _solidity(self, cv2_contour):
        area = cv2.contourArea(cv2_contour)
        hull = cv2.convexHull(cv2_contour)
        hull_area = cv2.contourArea(hull)
        return area / hull_area

    def generate(self,
                 output_size: Tuple[int, int],
                 rng: np.random.BitGenerator = None):
        if rng is None:
            rng = np.random.default_rng()
        img = rng.random(output_size)
        img = cv2.GaussianBlur(
            img, (0, 0), sigmaX=self.kernel_size, borderType=cv2.BORDER_ISOLATED)
        img = img - img.min()
        img = img / img.max()
        _, img = cv2.threshold(
            img, self.threshold, 1, cv2.THRESH_BINARY)
        img = img.astype(np.uint8)
        contours, _ = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if self.min_area:
            contours = tuple(
                filter(lambda c: cv2.contourArea(c) >= self.min_area, contours))
        if self.pinch_threshold:
            contours = tuple(
                filter(lambda c: not self._is_pinched(c), contours))
        if self.solidity_threshold:
            contours = tuple(
                filter(lambda c: self._solidity(c) >= self.solidity_threshold, contours))
        if self.circularity_threshold:
            contours = tuple(
                filter(lambda c: self._circularity(c) >= self.circularity_threshold, contours))
        if self.max_count:
            indices = rng.choice(len(contours), min(
                len(contours), self.max_count), replace=False)
            contours = [contours[i] for i in indices]
        return contours


CONTOUR_GENERATORS = [
    BlobContourGenerator
]


class SampleGenerator:
    def __init__(
            self,
            synthesizer: Synthesizer,
            medium: Medium,
            laser: Laser,
            contour_generator: ContourGenerator,
            depth_rng: RandomRange,
            ior_rng: RandomRange,
            default_size: Tuple[int, int] = None,
            default_seed: int = -1,
            far_field_thresh: float = None,
            random_max_depth: bool = True):
        self.synthesizer = synthesizer
        self.medium = medium
        self.laser = laser
        self.contour_generator = contour_generator
        self.depth_rng = depth_rng
        self.ior_rng = ior_rng
        self.default_size = default_size
        self.default_seed = default_seed
        self.far_field_thresh = far_field_thresh
        self.random_max_depth = random_max_depth

    @staticmethod
    def cv2toholo_contour(cv2contour):
        return list(map(tuple, cv2contour.reshape(-1, 2)))

    @staticmethod
    def holotocv2_contour(holocontour):
        if isinstance(holocontour, str):
            holocontour = literal_eval(holocontour)
        return np.array(holocontour).reshape(-1, 1, 2)

    @staticmethod
    def collapse_rng(json: Any, rng: np.random.BitGenerator):
        if isinstance(json, dict):
            if all(key in json.keys() for key in ['vmin', 'vmax', 'dist']):
                return RandomRange(**json)(rng)
            for key in json.keys():
                json[key] = SampleGenerator.collapse_rng(json[key], rng)
            if all(key in json.keys() for key in ['real', 'imaginary']):
                json = complex(json['real'], json['imaginary'])
            return json
        return json

    @classmethod
    def from_dict(cls, json: Dict[str, Any]):
        if not 'output_size' in json:
            json['output_size'] = None
        if json.get('seed', -1) == -1:
            json['seed'] = np.random.randint(0, np.iinfo(np.int32).max)
        if not 'far_field_thresh' in json['property_gen']:
            json['property_gen']['far_field_thresh'] = None
        if not 'torch' in json:
            json['torch'] = False
        rng = np.random.default_rng(1+json['seed'])
        property_generation = json.pop('property_gen')
        json = SampleGenerator.collapse_rng(json, rng)
        contour_type = json['contour_gen'].pop(
            'type', 'Blob') + 'ContourGenerator'
        contour_generator = None
        for cntgen in CONTOUR_GENERATORS:
            if contour_type == cntgen.__name__:
                contour_generator = cntgen(**json['contour_gen'])
        assert contour_generator is not None, f"Unknown contour generator type {contour_type}"
        synth_type = Synthesizer
        if json['torch']:
            synth_type = TorchSynthesizer
        return cls(
            synthesizer=synth_type(**json['synthesis']),
            medium=Medium(**json['medium']),
            laser=Laser(**json['laser']),
            contour_generator=contour_generator,
            depth_rng=RandomRange(**property_generation['depth']),
            ior_rng=RandomComplex(**property_generation['ior']),
            default_size=tuple(json['synthesis']['output_size']),
            default_seed=json['seed'],
            far_field_thresh=property_generation['far_field_thresh'],
            random_max_depth=property_generation.get('random_max', True))

    def to_dict(self):
        synth = self.synthesizer.to_dict()
        return {
            'contourgen_type': self.contour_generator.__class__.__name__,
            'contour_gen': self.contour_generator.__dict__,
            'medium': self.medium.__dict__,
            'synthesis': synth,
            'laser': self.laser.__dict__,
            'property_gen': {
                'depth': self.depth_rng.to_dict(),
                'ior': self.ior_rng.to_dict(),
                'far_field_thresh': self.far_field_thresh
            },
            'seed': self.default_seed,
            'torch': isinstance(self.synthesizer, TorchSynthesizer)
        }

    def generate_holos(self,
                       output_size: Tuple[int, int] = None,
                       seed: int = -1) -> GeneratedHologram:
        if seed == -1:
            seed = np.random.randint(0, np.iinfo(np.int32).max)
        rng = np.random.default_rng(seed=seed)
        contours = self.contour_generator.generate(output_size, rng)
        assert len(
            contours) > 0, "No contours generated, restrictions may be too high"
        masks = []
        depths = []
        keeps = []
        mask = np.zeros(output_size, dtype=np.uint8)
        random_max = self.depth_rng(rng)
        random_max_depth_rng = RandomRange(
            self.depth_rng.min, random_max, self.depth_rng.dist) if self.random_max_depth else self.depth_rng
        for contour in contours:
            mask[...] = 0
            mask = cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
            ior = self.ior_rng(rng)
            depth = random_max_depth_rng(rng)
            if None is self.far_field_thresh or depth < self.far_field_thresh:
                keeps.append(contour)
            depths.append(depth)
            masks.append(HologramMask(ior=ior, mask=mask.copy(), depth=depth))
        gen = self.synthesizer.generate(
            self.medium, [self.laser], masks, raw=False)
        gen *= 255
        gen = gen.astype(np.uint8)
        return GeneratedHologram(seed, gen, tuple(keeps), depths)

    def generate_sample(self, holo: GeneratedHologram) -> HoloSample:
        ih, iw = holo.image.shape
        img = self.generate_image(holo, ih, iw)
        bbs = self.generate_bbs(holo, ih, iw)
        return HoloSample(img, bbs)

    def generate_image(self, holo: GeneratedHologram, ih, iw) -> HoloImage:
        img = HoloImage.from_array(
            holo.image,
            file=f"Generated_{holo.seed}.png",
            index=0
        )
        return img

    def generate_bbs(self, holo: GeneratedHologram, ih, iw) -> List[BoundingBox]:
        bbs = []
        for contour, depth in zip(holo.contours, holo.depths):
            x, y, w, h = cv2.boundingRect(contour)
            x += w // 2
            y += h // 2
            cnt = self.cv2toholo_contour(contour)
            bb = BoundingBox(
                holo_class=0,
                x=x,
                y=y,
                w=w,
                h=h,
                z=depth,
                diameter=0,
                contour=np.asarray(cnt),
                format=BoundingBoxFormat.PIXEL)
            bb.__setattr__("bbnew", contour)
            bbs.append(bb)
        return bbs

    def generate(self, output_size: Tuple[int, int] = None, seed: int = None) -> HoloSample:
        assert output_size or self.default_size, "Need an output size"
        if None is output_size:
            output_size = self.default_size
        if None is seed:
            seed = self.default_seed
        return self.generate_sample(self.generate_holos(output_size, seed))
