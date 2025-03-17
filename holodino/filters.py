from abc import ABC, abstractmethod
import cv2

import numpy as np

from fakeholo.data import BoundingBox, BoundingBoxFormat


class Filter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, data) -> bool:
        pass


class ContourAvailableFilter(Filter):
    def __init__(self):
        super().__init__()

    def __call__(self, bb: BoundingBox) -> bool:
        contour = np.array(bb.contour)
        return bb.contour is not None and contour.shape[0] >= 3


class ContourInboundsFilter(ContourAvailableFilter):
    def __init__(self, margin=1e-3):
        super().__init__()
        self.margin = margin

    def __call__(self, bb: BoundingBox) -> bool:
        if not super().__call__(bb):
            return False
        contour = np.array(bb.contour)
        # Check if contour touches the edge of the image
        mins = contour.min(axis=0)
        maxs = contour.max(axis=0)
        if np.any(mins < self.margin) or np.any(maxs > 1 - self.margin):
            return False
        return True


class CircularityFilter(ContourAvailableFilter):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def __call__(self, bb: BoundingBox) -> bool:
        assert bb.format == BoundingBoxFormat.PIXEL, "Circularity filter only works on pixel coordinates"
        if not super().__call__(bb):
            return False
        cv2_contour = bb.getcv2contour()
        area = cv2.contourArea(cv2_contour)
        perimeter = cv2.arcLength(cv2_contour, True)
        circularity = 4 * np.pi * area / perimeter ** 2
        return circularity > self.threshold


class ScoreFilter(Filter):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def __call__(self, bb: BoundingBox) -> bool:
        return bb.score > self.threshold
