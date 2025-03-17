from enum import Enum
from dataclasses import dataclass
from functools import lru_cache, partial
import os
from typing import List, Tuple
from detectron2.checkpoint import DetectionCheckpointer

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import torch

from fakeholo.data import BoundingBox, BoundingBoxFormat, HoloImage, HoloSample
from holodino.config import get_configuration
from holodino.filters import Filter
from holodino.reconstructor import Reconstructor
from holodino.trainer import Trainer

def mask_iou(masks: np.ndarray):
    masks = masks.astype(np.uint32)  # (N, H, W)
    areas = np.sum(masks, axis=(1, 2))  # (N,)
    inter = np.einsum('bhw,nhw->bn', masks, masks)  # (N, N)
    union = areas[:, np.newaxis] + areas - inter  # (N, N)
    return inter / union  # (N, N)


def box_iou(boxes: np.ndarray):
    xmin = np.maximum(boxes[:, None, 0], boxes[:, 0])  # (N, N)
    ymin = np.maximum(boxes[:, None, 1], boxes[:, 1])  # (N, N)
    xmax = np.minimum(boxes[:, None, 2], boxes[:, 2])  # (N, N)
    ymax = np.minimum(boxes[:, None, 3], boxes[:, 3])  # (N, N)
    inter_area = np.maximum(0, xmax - xmin) * \
        np.maximum(0, ymax - ymin)  # (N, N)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # (N,)
    union_area = areas[:, None] + areas - inter_area
    iou = inter_area / union_area
    return iou


@dataclass
class Prediction:
    image_viz: np.ndarray  # (H, W, 3), uint8
    bbs: List[BoundingBox]
    boxes: np.ndarray  # (N, 4), float32
    scores: np.ndarray  # (N,), float32
    masks: np.ndarray  # (N, H, W), uint8
    planes: List['Prediction'] = None
    reconstructed: float = None
    time: float = None

    def __post_init__(self):
        assert len(self.boxes) == len(self.scores) == len(
            self.masks), "boxes, scores and masks must have the same length"
        self.boxes = self.boxes.astype(np.float32)
        self.scores = self.scores.astype(np.float32)
        if self.masks.dtype in [np.float32, np.float64]:
            self.masks = self.masks > 0.5
        self.masks = self.masks.astype(np.uint8)

    def __len__(self):
        return len(self.boxes)
    
    def draw(self, image: np.ndarray, contours=True, boxes=True):
        for bb in self.bbs:
            image = bb.draw(image, contours=contours, boxes=boxes)
        return image
    
class Resolvers(Enum):
    NAIVE = "Naive"
    CONSENSUS = "Consensus"
    CONFIDENCE = "Confidence"
    EXPERT = "Expert"

def nms(dets, scores, thresh):
    if len(dets) == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0]  # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def mask_nms(masks: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5):
    assert np.all((masks == 1) | (masks == 0)), "Masks must be binary"
    assert masks.ndim == 3 and masks.shape[0] == scores.shape[0], "Masks and scores must have the same batch size"
    masks = masks.astype(np.uint32)
    order = np.argsort(scores)[::-1]
    areas = np.sum(masks, axis=(1, 2))
    keep = []
    while len(order) > 0:
        idx = order[0]
        keep.append(idx)
        if len(order) == 1:
            break
        inter = np.einsum('hw,bhw->b', masks[idx], masks[order[1:]])
        rem_areas = areas[order[1:]]
        union = areas[idx] + rem_areas - inter
        iou = inter / union
        order = order[1:][iou < iou_threshold]
    return keep

class MaskDinoModel:
    name: str
    recon_steps: int = 1
    recon_range: Tuple[float, float] = (0, 1)
    reconstructor: Reconstructor = None
    dilation: int = 0
    resolver: Resolvers = None
    filters: List[Filter] = None

    def __init__(self,
                 name: str,
                 backend: str,
                 reconstructor: Reconstructor = None,
                 recon_steps: int = 1,
                 recon_range: Tuple[float, float] = (0, 1),
                 dilation: int = 0,
                 resolver: Resolvers = None,
                 filters: List[Filter] = None):
        self.name = name
        self.backend = backend
        self.reconstructor = reconstructor
        self.recon_steps = recon_steps
        self.recon_range = recon_range
        self.threadpool = ThreadPoolExecutor(max_workers=recon_steps)
        self.dilation = dilation
        self.resolver = resolver if resolver is not None else Resolvers.NAIVE
        self.filters = filters if filters is not None else []
        self.model = self.load_model(backend)

    def load_model(self, path):
        dir_path = os.path.dirname(path)
        config = get_configuration(os.path.join(dir_path, 'config.yaml'))
        model = Trainer.build_model(config)
        DetectionCheckpointer(model).load(path)
        model.eval()
        return model

    def predict(self, images):
        inpt = [{
            'image': torch.from_numpy(im).repeat(3, 1, 1).float(),
            'height': im.shape[-2],
            'width': im.shape[-1],
            } for im in images]
        with torch.no_grad():
            return self.model(inpt)

    @staticmethod
    def dilate_contours(mask, dilation):
        return cv2.dilate(mask, np.ones((dilation, dilation), np.uint8), iterations=1)

    @property
    def config(self):
        return {
            'name': self.name,
            'recon_steps': self.recon_steps,
            'recon_min': self.recon_range[0],
            'recon_max': self.recon_range[1],
            'recon_wavelength': float(self.reconstructor.wavelength) if self.reconstructor is not None else None,
            'recon_resolution': float(self.reconstructor.resolution) if self.reconstructor is not None else None,
            'dilation': self.dilation,
            'resolver': self.resolver.value
        }

    def _parse_predictions(self, pred_boxes, pred_masks, scores):
        bbs = []
        for box, mask, score in zip(pred_boxes, pred_masks, scores):
            mask = mask.astype(np.uint8)
            contour, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contour) == 0:
                continue
            contour = contour[0]
            cbox = cv2.boundingRect(contour)
            contour = np.concatenate((contour, contour[:1]), axis=0)[:, 0, :].astype(np.float32)
            w = cbox[2]
            h = cbox[3]
            x = (cbox[0] + cbox[2] / 2)
            y = (cbox[1] + cbox[3] / 2)
            boxes = np.array([[box[0], box[1], box[2], box[3]],
                              [cbox[0], cbox[1], cbox[0] + cbox[2], cbox[1] + cbox[3]]])
            iou = box_iou(boxes)  # (2, 2)
            if iou[0, 1] == 0:
                continue

            bbs.append(BoundingBox(x=x, y=y, w=w, h=h,
                       contour=contour,
                       format=BoundingBoxFormat.PIXEL,
                       score=score))
        for f in self.filters:
            bbs = filter(f, bbs)
        return list(bbs)

    def _reconstruct(self, image: HoloImage):
        depths = np.linspace(*self.recon_range, self.recon_steps)
        imgs = self.reconstructor.intensity(image, depths).numpy()
        return imgs

    def _merge_bbs(self, bbs: List[List[BoundingBox]], scores: List[np.ndarray], masks: List[np.ndarray]):
        flat_bbs = [bb for sublist in bbs for bb in sublist]
        flat_scores = np.concatenate(scores)
        flat_masks = np.concatenate(masks)
        return flat_bbs, flat_scores, flat_masks

    def reconstructive_augmentation(self, image: HoloImage):
        if self.reconstructor is None or self.recon_steps <= 1:
            return [image.as_arr]
        planes = self._reconstruct(image)
        planes = (planes * 255).astype(np.uint8)
        planes = np.concatenate(
            (image.as_arr[np.newaxis, ...], planes), axis=0)
        return planes

    def dilation_augmentation(self, masks: np.ndarray):
        if self.dilation <= 0:
            return masks
        if len(masks) == 0:
            return masks
        print("Dilating contours", self.dilation)
        return np.stack([self.dilate_contours(mask, self.dilation) for mask in masks])

    def process_mask(self, masks: np.ndarray):
        masks = (masks > 0.5).astype(np.uint8)
        return self.dilation_augmentation(masks)

    def process_box(self, bbs: np.ndarray):
        return bbs

    def naive_nms(self, predictions: List[Prediction]) -> Prediction:
        masks = np.vstack([p.masks for p in predictions])
        scores = np.hstack([p.scores for p in predictions])
        boxes = np.vstack([p.boxes for p in predictions])
        keeps = mask_nms(masks, scores, 0.1)
        return Prediction(
            image_viz=predictions[0].image_viz,
            bbs=self._parse_predictions(
                boxes[keeps], masks[keeps], scores[keeps]),
            boxes=boxes[keeps],
            scores=scores[keeps],
            masks=masks[keeps],
            planes=predictions
        )

    def consensus_nms(self, predictions: List[Prediction]) -> Prediction:
        if len(predictions) == 1:
            return predictions[0]
        model_list = [np.array([i] * len(p))
                      for i, p in enumerate(predictions)]
        model = np.hstack(model_list)[:, np.newaxis]  # (N, 1)
        masks = np.vstack([p.masks for p in predictions])
        scores = np.hstack([p.scores for p in predictions])
        boxes = np.vstack([p.boxes for p in predictions])
        iou = box_iou(boxes)  # (N, N)
        # Get the indices of masks for which multiple models agree
        same_model_mask = (model == model.T)
        iou[same_model_mask] = 0
        agreement_counts = np.sum(iou > 0.1, axis=1) / len(predictions)
        conensus_indices = np.where(agreement_counts > 0.25)[0]
        masks = masks[conensus_indices]
        scores = scores[conensus_indices]
        boxes = boxes[conensus_indices]
        keeps = mask_nms(masks, scores, 0.1)
        return Prediction(
            image_viz=predictions[0].image_viz,
            bbs=self._parse_predictions(
                boxes[keeps], masks[keeps], scores),
            boxes=boxes[keeps],
            scores=scores[keeps],
            masks=masks[keeps],
            planes=predictions
        )

    def confidence_nms(self, predictions: List[Prediction]) -> Prediction:
        masks = np.vstack([p.masks for p in predictions])
        scores = np.hstack([p.scores for p in predictions])
        boxes = np.vstack([p.boxes for p in predictions])
        iou_matrix = mask_iou(masks)
        iou_threshold = 0.1
        N = len(scores)
        visited = set()
        resultant_masks = []
        resultant_scores = []
        resultant_boxes = []

        for i in range(N):
            if i in visited:
                continue
            overlapping_indices = [j for j in range(
                N) if iou_matrix[i, j] > iou_threshold and i != j]
            cluster_indices = [i] + overlapping_indices
            visited.update(cluster_indices)
            weighted_masks = np.array([masks[j] * scores[j]
                                      for j in cluster_indices])
            average_weighted_mask = np.mean(weighted_masks, axis=0)
            mask_threshold = 0.5
            resultant_mask = (average_weighted_mask >
                              mask_threshold).astype(np.uint8)
            if np.sum(resultant_mask) == 0:
                continue
            cluster_scores = [scores[j] for j in cluster_indices]
            average_score = np.mean(cluster_scores)
            resultant_box = cv2.boundingRect(resultant_mask)
            resultant_masks.append(resultant_mask)
            resultant_scores.append(average_score)
            resultant_boxes.append(resultant_box)

        resultant_masks = np.array(resultant_masks)
        resultant_scores = np.array(resultant_scores)
        resultant_boxes = np.array(resultant_boxes)
        return Prediction(
            image_viz=predictions[0].image_viz,
            bbs=self._parse_predictions(
                resultant_boxes, resultant_masks, scores),
            boxes=resultant_boxes,
            scores=resultant_scores,
            masks=resultant_masks,
            planes=predictions
        )

    def expert_nms(self, predictions: List[Prediction]):
        main_plane = [p for p in predictions if p.reconstructed == None][0]
        aggregated_reconstructed = self.consensus_nms(predictions)
        return self.naive_nms([main_plane, aggregated_reconstructed])

    def resolve_overlaps(self, predictions: List[Prediction]) -> Prediction:
        if len(predictions) == 1:
            return predictions[0]
        resolvers = {
            Resolvers.NAIVE: self.naive_nms,
            Resolvers.CONSENSUS: self.consensus_nms,
            Resolvers.CONFIDENCE: self.confidence_nms,
            Resolvers.EXPERT: self.expert_nms
        }
        assert self.resolver in resolvers, f"Unknown resolver {self.resolver}"
        print("Using resolver", self.resolver)
        return resolvers[self.resolver](predictions)

    def first_duplicate(self, boxes: np.ndarray) -> int:
        diff = np.diff(boxes, axis=0)
        rows = (diff == 0).all(axis=1)
        if True in rows:
            return np.where(rows)[0][0] + 1
        return -1

    def get_prediction(self, image: HoloImage):
        planes = self.reconstructive_augmentation(image)
        predictions = []
        for plane_num, response in enumerate(self.predict(planes)):
            if plane_num == 0:
                plane_num = None
            response = response['instances'].to("cpu")
            duplicate_index = self.first_duplicate(response.pred_boxes.tensor.numpy())
            masks = self.process_mask(
                response.pred_masks[:duplicate_index, ...].numpy())
            boxes = self.process_box(
                response.pred_boxes[:duplicate_index, ...].tensor.numpy())
            scores = response.scores[:duplicate_index].numpy()

            predictions.append(Prediction(
                image_viz=None,
                bbs=self._parse_predictions(boxes, masks, scores),
                boxes=boxes,
                scores=scores,
                masks=masks,
                reconstructed=plane_num,
                time=None
            ))

        final_prediction = self.resolve_overlaps(predictions)
        return final_prediction
