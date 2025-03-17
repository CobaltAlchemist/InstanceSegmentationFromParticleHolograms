from dataclasses import dataclass
from enum import Enum
from functools import cached_property, lru_cache
from typing import List, Tuple, Union
import cv2
import numpy as np
from .utils import LazyImageArray
from scipy.optimize import linear_sum_assignment
from PIL import Image


def intersection_over_union(box_a, box_b):
    """
    Calculates the intersection over union of two bounding boxes

    Args:
        box_a (tuple): (x1, y1, x2, y2)
        box_b (tuple): (x1, y1, x2, y2)

    Returns:
        float: Intersection over union
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou


def intersection_over_union_matrix(preds, labels):
    """
    Calculates the intersection over union of two bounding boxes

    Args:
        preds (np.ndarray): (N, 4) - (x1, y1, x2, y2)
        labels (np.ndarray): (M, 4) - (x1, y1, x2, y2)

    Returns:
        np.ndarray: (N, M) intersection over union matrix
    """
    n = preds.shape[0]
    m = preds.shape[1]

    # Calculate the coordinates of the intersection boxes
    x1 = np.maximum(preds[:, 0:1], labels[:, 0])  # (N, M)
    y1 = np.maximum(preds[:, 1:2], labels[:, 1])  # (N, M)
    x2 = np.minimum(preds[:, 2:3], labels[:, 2])  # (N, M)
    y2 = np.minimum(preds[:, 3:4], labels[:, 3])  # (N, M)

    # Calculate the areas of the intersection boxes
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)  # (N, M)

    # Calculate the areas of box_a and box_b
    area_a = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])  # (N,)
    area_b = (labels[:, 2] - labels[:, 0]) * \
        (labels[:, 3] - labels[:, 1])  # (N,)

    # Calculate the union areas
    union = area_a[:, None] + area_b - intersection  # (N, M)

    # Calculate the IoU matrix
    iou_matrix = intersection / union  # (N, M)

    return iou_matrix


@dataclass
class HoloImage:
    file: str
    index: int = 0
    height: int = 0
    width: int = 0
    _array = None

    def __array__(self, dtype=None):
        return self.as_arr

    def __hash__(self):
        return hash(self.file)

    def resize(self, dsize=None, max_size=None):
        assert dsize or max_size, "Specify specific dimensions or a max size"
        if max_size:
            h, w = self.as_arr.shape
            scale = max(h / max_size,
                        w / max_size, 1)
            return self.resize(dsize=tuple(map(int, (w // scale, h // scale))))
        img = cv2.resize(self.as_arr, dsize, interpolation=cv2.INTER_AREA)
        ret = HoloImage(self.file, self.index, dsize[1], dsize[0], img)
        return ret
    
    @property
    def as_arr(self):
        if not self._array is None:
            return self._array
        pil_img = Image.open(self.file)
        if 0 != self.index:
            assert '.tif' in self.file, "File has index, but is not a tiff"
            pil_img.seek(self.index)
        if (self.height, self.width) != pil_img.size and self.height > 0 and self.width > 0:
            pil_img = pil_img.resize((self.height, self.width))
        self._array = np.asarray(pil_img.convert('L'))
        return self._array

    @classmethod
    def from_file(cls, file: str, index: int = 0, height: int = 0, width: int = 0):
        return cls(
            file=file,
            index=index,
            height=height,
            width=width
        )
    
    @classmethod
    def from_array(cls, arr: np.ndarray, **kwargs):
        height, width = arr.shape[-2:]
        inst = cls(
            height=height,
            width=width,
            **kwargs
        )
        inst._array = arr
        return inst


class BoundingBoxFormat(Enum):
    RELATIVE = 1
    PIXEL = 2


@dataclass
class BoundingBox:
    holo_class: Union[int, str] = 0
    x: Union[float, int] = 0
    y: Union[float, int] = 0
    w: Union[float, int] = 0
    h: Union[float, int] = 0
    z: float = 0
    diameter: float = 0
    score: float = 0
    contour: np.ndarray = None
    format: BoundingBoxFormat = BoundingBoxFormat.RELATIVE

    def __post_init__(self):
        if not isinstance(self.holo_class, str):
            self.holo_class = int(self.holo_class)

    def __hash__(self):
        return hash((self.x, self.y, self.w, self.h))
        
    def minmax(self) -> List[float]:
        hw = self.w / 2
        hh = self.h / 2
        coords = (
            self.x - hw,
            self.y - hh,
            self.x + hw,
            self.y + hh
        )
        if self.format == BoundingBoxFormat.RELATIVE:
            return coords
        return tuple(map(int, coords))

    def to_pixelspace(self, img: Union[HoloImage, np.array]):
        if self.format == BoundingBoxFormat.PIXEL:
            return self
        img = np.array(img)
        if len(img.shape) == 3:
            h, w, _ = img.shape
        else:
            h, w = img.shape
        contour = None
        if not self.contour is None and len(self.contour) > 0:
            contour = (self.contour * np.array([w, h])).astype(np.int32)
        bb = BoundingBox(
            self.holo_class,
            int(self.x * w),
            int(self.y * h),
            int(self.w * w),
            int(self.h * h),
            self.z,
            self.diameter,
            contour,
            format=BoundingBoxFormat.PIXEL)
        bb._clip_coordinates(img.shape)
        return bb

    def to_relativespace(self, img: Union[HoloImage, np.array]):
        if self.format == BoundingBoxFormat.RELATIVE:
            return self
        img = np.array(img)
        if len(img.shape) == 3:
            h, w, _ = img.shape
        else:
            h, w = img.shape
        contour = None
        if not self.contour is None and len(self.contour) > 0:
            contour = (self.contour / np.array([w, h])).astype(np.float32)
        bb = BoundingBox(
            self.holo_class,
            self.x / w,
            self.y / h,
            self.w / w,
            self.h / h,
            self.z,
            self.diameter,
            contour,
            format=BoundingBoxFormat.RELATIVE)
        bb._clip_coordinates(img.shape)
        return bb

    def extract_image(self, img: Union[HoloImage, np.array], padding: Union[int, Tuple[int, int]] = 0):
        if not isinstance(padding, tuple):
            padding = (padding, padding)
        padding = tuple(map(int, padding))
        bb = self.to_pixelspace(img)
        xi, yi, xf, yf = bb.minmax()
        s = np.shape(img)
        if len(s) == 3:
            h, w, c = s
        else:
            h, w = s
        xi -= padding[0] // 2
        yi -= padding[1] // 2
        xf += padding[0] // 2
        yf += padding[1] // 2
        xi = max(xi, 0)
        yi = max(yi, 0)
        xf = min(xf, w)
        yf = min(yf, h)
        return img[yi:yf, xi:xf]

    def getcv2contour(self, img: Union[HoloImage, np.array] = None):
        ret = np.array(self.contour)
        if self.format is BoundingBoxFormat.RELATIVE:
            assert not img is None, "Image must be provided to convert contour to pixel space"
            # scale the contour by the image dimensions
            ret = ret * np.array([[np.shape(img)[1], np.shape(img)[0]]])
        return ret.astype(np.uint64)[:, None, :]

    def getcv2rectangle(self, img: Union[HoloImage, np.array]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        bb = self.to_pixelspace(img)
        if isinstance(img, HoloImage):
            img = img.as_arr
        xi, yi, xf, yf = list(map(int, bb.minmax()))
        return (xi, yi), (xf, yf)
    
    def get_mask(self, img: Union[HoloImage, np.array]) -> np.ndarray:
        bb = self.to_pixelspace(img)
        shape = img.as_arr.shape if len(
            img.as_arr.shape) == 2 else img.as_arr.shape[:2]
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, [bb.getcv2contour()], -1, 255, -1)
        return mask

    def has_contour(self):
        return not None is self.contour and len(self.contour) > 0

    def draw_contour(self, img: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
        if not self.has_contour():
            return img
        bb = self.to_pixelspace(img)
        return cv2.drawContours(img, [bb.getcv2contour()], -1, color, thickness)

    def draw_bounding_box(self, img: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
        bb = self.to_pixelspace(img)
        return cv2.rectangle(img, *bb.getcv2rectangle(img), color, thickness)
    def _clip_coordinates(self, img_shape):
        if self.format == BoundingBoxFormat.RELATIVE:
            half_w = self.w / 2
            half_h = self.h / 2
            x1 = np.clip(self.x - half_w, 0, 1)
            y1 = np.clip(self.y - half_h, 0, 1)
            x2 = np.clip(self.x + half_w, 0, 1)
            y2 = np.clip(self.y + half_h, 0, 1)
            self.x = (x1 + x2) / 2
            self.y = (y1 + y2) / 2
            self.w = x2 - x1
            self.h = y2 - y1
        elif self.format == BoundingBoxFormat.PIXEL:
            img_height, img_width = img_shape[:2]
            half_w = self.w / 2
            half_h = self.h / 2
            x1 = np.clip(self.x - half_w, 0, img_width)
            y1 = np.clip(self.y - half_h, 0, img_height)
            x2 = np.clip(self.x + half_w, 0, img_width)
            y2 = np.clip(self.y + half_h, 0, img_height)
            self.x = (x1 + x2) // 2
            self.y = (y1 + y2) // 2
            self.w = int(x2 - x1)
            self.h = int(y2 - y1)

    def draw(self, img: Union[np.ndarray, HoloImage], contours=True, boxes=True):
        img = np.array(img)
        if contours and self.has_contour():
            img = cv2.drawContours(img, [self.getcv2contour(img)], 0, (0, 255, 0), 2)
        if boxes:
            x1, y1, x2, y2 = self.to_pixelspace(img).minmax()
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return img
    
    def iou(self, other):
        # Calculate and return the IoU of self and other
        assert self.format == other.format, "Both bounding boxes must be in the same format"
        x1, y1, x2, y2 = self.minmax()
        x1_, y1_, x2_, y2_ = other.minmax()
        xA = max(x1, x1_)
        yA = max(y1, y1_)
        xB = min(x2, x2_)
        yB = min(y2, y2_)
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (x2 - x1 + 1) * (y2 - y1 + 1)
        boxBArea = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)
        return interArea / float(boxAArea + boxBArea - interArea)
        


def get_iou_matrix(predictions: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Returns a matrix of shape (predictions, boxes) with the iou of each
    prediction with each box

    Args:
        predictions (np.ndarray): Array of shape (N, 4) with the predictions
        boxes (np.ndarray): Array of shape (M, 4) with the boxes

    Returns:
        np.ndarray: Array of shape (N, M) with the iou of each prediction
    """
    @lru_cache(maxsize=10)
    def inner(predictions: bytes, boxes: bytes) -> np.ndarray:
        predictions = np.frombuffer(predictions, dtype=float).reshape((-1, 4))
        boxes = np.frombuffer(boxes, dtype=float).reshape((-1, 4))
        iou_matrix = np.zeros((predictions.shape[0], boxes.shape[0]))

        for i in range(predictions.shape[0]):
            for j in range(boxes.shape[0]):
                iou_matrix[i, j] = intersection_over_union(
                    predictions[i], boxes[j])

        return iou_matrix
    predictions = predictions.astype(float).tobytes()
    boxes = boxes.astype(float).tobytes()
    return inner(predictions, boxes)


@dataclass
class HoloSample:
    image: HoloImage
    bounding_boxes: List[BoundingBox]

    def __iter__(self):
        return iter((self.image, self.bounding_boxes))

    def __hash__(self) -> int:
        return hash(self.image) ^ hash(tuple(self.bounding_boxes))

    def precision_recall_from_iou_matrix(self, iou_matrix: np.ndarray, gt_depths: np.ndarray, iou_threshold: float = 0.5, depth_cutoff: float = None) -> Tuple[float, float]:
        num_predictions, num_gt_boxes = iou_matrix.shape
        pred_to_gt = np.argmax(iou_matrix, axis=1)
        pred_to_gt[iou_matrix[np.arange(
            num_predictions), pred_to_gt] < iou_threshold] = -1
        # Filter predictions out that matched to a gt box that is too far away (-2)
        if depth_cutoff is not None:
            # Check where gt_depths[pred_to_gt] is greater than the cutoff and pred_to_gt is not -1
            pred_to_gt[(gt_depths[pred_to_gt] > depth_cutoff) & (
                pred_to_gt != -1)] = -2
            num_predictions -= np.count_nonzero(pred_to_gt == -2)
            num_gt_boxes -= np.count_nonzero(gt_depths > depth_cutoff)
        if num_gt_boxes == 0:
            return 1 if num_predictions == 0 else 0, 1
        if num_predictions == 0:
            return 1, 0
        precision = np.count_nonzero(pred_to_gt >= 0) / num_predictions
        # Filter out duplicate predictions
        pred_to_gt = np.unique(pred_to_gt)
        recall = np.count_nonzero(pred_to_gt >= 0) / num_gt_boxes
        return precision, recall

    def precision_recall(self, predictions: np.ndarray, iou_threshold: float = 0.5, depth_cutoff: float = None) -> Tuple[float, float]:
        """
        Computes the Mean Average Precision (mAP) and recall for the given predictions.

        This method calculates the Intersection over Union (IoU) for each pair of ground truth 
        bounding boxes and predicted bounding boxes. If the IoU is above the given threshold, 
        it is considered a true positive; otherwise, it is a false positive.

        Args:
            predictions (np.ndarray): A NumPy array of shape (N, 4), where N is the number of 
                                      predicted bounding boxes, and each row represents a bounding 
                                      box in the format [x_min, y_min, x_max, y_max].
            iou_threshold (float, optional): The IoU threshold to consider a prediction as a true
                                             positive. Defaults to 0.5.
            depth_cutoff (float, optional): The depth cutoff to exclude bounding boxes beyond a certain depth.

        Returns:
            Tuple[float, float]: A tuple containing the Mean Average Precision (mAP) and recall, 
                                 both as floats between 0 and 1.
        """
        gt_boxes = np.array([bb.to_pixelspace(self.image).minmax()
                            for bb in self.bounding_boxes])
        gt_depths = np.array([bb.z for bb in self.bounding_boxes])

        if len(predictions) == 0:
            return 1, 0
        iou_matrix = intersection_over_union_matrix(predictions, gt_boxes)
        # Map predictions to ground truth boxes. If a prediction matches to a gt box with a lower IoU than the threshold, it is ignored (-1)
        return self.precision_recall_from_iou_matrix(iou_matrix, gt_depths, iou_threshold, depth_cutoff)

    def precision_recall_masks(self, predictions: np.ndarray, iou_threshold: float = 0.5, depth_cutoff: float = None) -> Tuple[float, float]:
        """
        Computes the precision and recall for the given predicted masks.

        This method calculates the Intersection over Union (IoU) for each pair of ground truth masks and 
        predicted masks. If the IoU is above the given threshold, it is considered a true positive; 
        otherwise, it is a false positive.

        Args:
            predictions (np.ndarray): A NumPy array of shape (# preds, height, width), where # preds is the
                                    number of predicted masks, and each mask has dimensions height x width.
            iou_threshold (float, optional): The IoU threshold to consider a prediction as a true positive.
                                            Defaults to 0.5.
            depth_cutoff (float, optional): The depth cutoff to exclude bounding boxes beyond a certain depth.

        Returns:
            Tuple[float, float]: A tuple containing the precision and recall, both as floats between 0 and 1.
        """
        def contour_to_mask(contour: List[Tuple[int, int]], shape: Tuple[int, int]) -> np.ndarray:
            mask = np.zeros(shape, dtype=np.uint8)
            formatted_contour = np.array(
                contour, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [formatted_contour], 1)
            return mask

        gt_masks = np.array([contour_to_mask(bb.to_pixelspace(self.image).contour,
                            self.image.as_arr.shape[:2]) for bb in self.bounding_boxes if bb.has_contour()], dtype=np.bool)
        gt_depths = np.array(
            [bb.z for bb in self.bounding_boxes if bb.has_contour()])
        predictions = predictions.astype(np.bool)

        if len(predictions) == 0:
            return 1, 0

        intersection = np.einsum(
            'nhw,mhw->nm', predictions, gt_masks, dtype=np.int32)
        union = (np.sum(predictions, axis=(1, 2))[
            :, None] + np.sum(gt_masks, axis=(1, 2))[None, :])
        iou_matrix = intersection / (union - intersection)
        return self.precision_recall_from_iou_matrix(iou_matrix, gt_depths, iou_threshold, depth_cutoff)
    
    def iou_masks(self, predictions: np.ndarray) -> np.ndarray:
        """
        Computes the Intersection over Union (IoU) for the given predicted masks.

        This method calculates the Intersection over Union (IoU) for each pair of ground truth masks and 
        predicted masks. It then provides the aggregation result of those IoU values.

        Args:
            predictions (np.ndarray): A NumPy array of shape (# preds, height, width), where # preds is the
                                    number of predicted masks, and each mask has dimensions height x width.
            depth_cutoff (float, optional): The depth cutoff to exclude bounding boxes beyond a certain depth.

        Returns
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A NumPy array of shape (# preds, ) containing the IoU per prediction,
                                            another array of the same shape containing the ground truth indices,
                                            and finally an array of the same shape containing the depth of each prediction.
                                            
        """
        if len(predictions) == 0:
            return np.array([]), np.array([]), np.array([])
        
        gt_masks = np.array([bb.get_mask(self.image) for bb in self.bounding_boxes if bb.has_contour()], dtype=np.bool)
        gt_depths = np.array(
            [bb.z for bb in self.bounding_boxes if bb.has_contour()])
        predictions = predictions.astype(np.bool)


        intersection = np.einsum(
            'nhw,mhw->nm', predictions, gt_masks, dtype=np.int32)
        union = (np.sum(predictions, axis=(1, 2))[
            :, None] + np.sum(gt_masks, axis=(1, 2))[None, :])
        iou_matrix = intersection / (union - intersection)
        best_matches = np.argmax(iou_matrix, axis=1)
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)
        iou_scores = np.zeros(len(predictions))
        match_indices = np.full(len(predictions), -1)
        depths = np.full(len(predictions), -1)
        iou_scores[row_indices] = iou_matrix[row_indices, col_indices]
        match_indices[row_indices] = col_indices
        depths[best_matches > 0] = gt_depths[best_matches[best_matches > 0]]
        return iou_scores, match_indices, depths
    
    def sizing_error(self, predictions: np.ndarray, depth_cutoff: float = None) -> np.ndarray:
        """
        Computes the sizing error for the given predicted masks.
        
        Args:
            predictions (np.ndarray): A NumPy array of shape (# preds, height, width), where # preds is the
                                    number of predicted masks, and each mask has dimensions height x width.

        Returns
            np.ndarray: A NumPy array of shape (# preds, ) containing the sizing error per prediction
        """
        if len(predictions) == 0:
            return np.array([]), np.array([]), np.array([])

        gt_masks = np.array([bb.get_mask(
            self.image) for bb in self.bounding_boxes if bb.has_contour()], dtype=np.bool)
        gt_depths = np.array(
            [bb.z for bb in self.bounding_boxes if bb.has_contour()])
        if depth_cutoff is not None:
            gt_masks = gt_masks[gt_depths < depth_cutoff]
        intersection = np.einsum(
            'nhw,mhw->nm', predictions, gt_masks, dtype=np.int32)
        original = np.sum(gt_masks, axis=(1, 2))[None, :]
        error = intersection / original
        row_indices, col_indices = linear_sum_assignment(error)
        return error[row_indices, col_indices].mean()
        
    
    def visualize(self):
        img = np.copy(self.image)
        for bb in self.bounding_boxes:
            img = bb.draw(img)
        return img
