from itertools import chain
import numpy as np
import os
import json

from .data import  HoloImage, HoloSample
import cv2


def save_json(d, file):
    with open(file, 'w') as f:
        json.dump(d, f)


class Detectron2Exporter:
    def __init__(self, dataset_dir: str, val_size: int = 0.2, bbox_scale=4, max_img_size=None, skip_images=False):
        self.dataset_dir = dataset_dir
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        self.val_size = val_size
        self.bbox_scale = bbox_scale
        self.max_image_size = max_img_size
        self.skip_images = skip_images
        self.ann_id = 1
            
    def export_sample(self, sample: HoloSample):
        dset_image = sample.image
        if self.max_image_size:
            dset_image = sample.image.resize(max_size=self.max_image_size)
        original_format = dset_image.as_arr
        image = {
            'id': dset_image.file,
            'file_name': dset_image.file,
            'width': original_format.shape[1],
            'height': original_format.shape[0],
        }
        annos = []
        for bb in filter(lambda bb: len(bb.contour) > 0, sample.bounding_boxes):
            bb = bb.to_pixelspace(dset_image)
            anno = {
                'id': self.ann_id,
                'image_id': image['id'],
                'iscrowd': int(0),
                'category_id': int(0),
                'bbox': self.getBoundingRect(dset_image, bb.contour),
                'bbox_mode': int(1),
                'segmentation': list(map(float, chain(*bb.contour))),
                'area': cv2.contourArea(bb.contour),
            }
            self.ann_id += 1
            if len(anno['segmentation']) <= 4 or len(anno['segmentation']) % 2:
                print("Invalid segmentation detected for annotation "
                      f"{self.ann_id}, image {dset_image.file}. Skipping...")
                continue
            anno['segmentation'] = [anno['segmentation']]
            annos.append(anno)
        return image, annos, dset_image

    def getBoundingRect(self, image: HoloImage, contour: np.ndarray):
        height, width = image.as_arr.shape[:2]
        x, y, w, h = cv2.boundingRect(contour)
        x = max(0, x + w * (1 - self.bbox_scale) // 2)
        y = max(0, y + h * (1 - self.bbox_scale) // 2)
        w = min(width - x, w * self.bbox_scale)
        h = min(height - y, h * self.bbox_scale)
        return x, y, w, h
