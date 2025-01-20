import os
from glob import glob

import logging
import backoff
from detectron2.config import configurable
from detectron2.data.datasets import register_coco_instances
from detectron2.data import transforms as T
from detectron2.data import DatasetCatalog, MetadataCatalog
from torch.utils.data.dataset import Dataset
from fakeholo.generator import SampleGenerator
from fakeholo.data import HoloSample
from fakeholo.exporter import Detectron2Exporter
from .reconstructor import ReconstructionTransformer
from fvcore.transforms.transform import Transform
import numpy as np
import json
import torch
from torch.utils.data import IterableDataset
from detectron2.data import detection_utils as utils
from detectron2.structures import PolygonMasks
from pycocotools import mask as coco_mask
import requests
import pickle
from maskdino import COCOInstanceNewBaselineDatasetMapper

def register_holo_instance(root):
    for json in glob(os.path.join(root, '*', f'*.json')):
        folder = os.path.dirname(json)
        name = os.path.basename(folder)
        register_coco_instances(name, {}, json, os.path.join(folder, 'images'))
            
class Initializer:
    def __init__(self, cls, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.cls = cls
    
    def __call__(self):
        return self.cls(*self.args, **self.kwargs)
            
def register_holo_generative(root):
    files = glob(os.path.join(root, '**', 'config.json')) + \
            glob(os.path.join(root, '**', 'cfg.json'))
    for config_file in files:
        folder = os.path.dirname(config_file)
        name = os.path.basename(folder)
        with open(config_file) as f:
            data = json.load(f)
        DatasetCatalog.register(f'{name}_autogen', Initializer(HolodinoInstanceGenerativeDataset, name, data))
        MetadataCatalog.get(f'{name}_autogen').set(json_file = config_file)
        DatasetCatalog.register(f'{name}_autogen_s', Initializer(HolodinoInstanceGenerativeDatasetServer, name, data, 'http://localhost:5000'))
        MetadataCatalog.get(f'{name}_autogen_s').set(json_file = config_file)

        
def build_transform_gen(cfg, is_train):
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE
    #crop_size = cfg.HOLODINO.CROP_SIZE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        if cfg.INPUT.RANDOM_FLIP == "horizontal" or cfg.INPUT.RANDOM_FLIP == "both":
            augmentation.append(T.RandomFlip(horizontal=True))
        if cfg.INPUT.RANDOM_FLIP == "vertical" or cfg.INPUT.RANDOM_FLIP == "both":
            augmentation.append(T.RandomFlip(vertical=True))

    augmentation.extend([
        ReconstructionTransformer(
                cfg.HOLODINO.RECONSTRUCTION_WAVELENGTH,
                cfg.HOLODINO.RECONSTRUCTION_RESOLUTION,
                *cfg.HOLODINO.RECONSTRUCTION_RANGE,
                cfg.HOLODINO.RECONSTRUCTION_PROBABILITY,
            ),
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation

class HolodinoInstanceDatasetMapper(COCOInstanceNewBaselineDatasetMapper):
    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[HolodinoInstanceDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        
    @classmethod
    def from_config(cls, cfg, is_train=True):
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class HolodinoInstanceGenerativeDatasetMapper(HolodinoInstanceDatasetMapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exporter = Detectron2Exporter('tmp')
        
    def __call__(self, sample: HoloSample):
        image_dict, annos_dict, image = self.exporter.export_sample(sample)
        dataset_dict = {**image_dict, "annotations": annos_dict}
        image = image.as_arr[...,None].repeat(3, axis=2)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, image_shape)
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if not instances.has('gt_masks'):  # this is to avoid empty annotation
                instances.gt_masks = PolygonMasks([])
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks

            dataset_dict["instances"] = instances

        return dataset_dict
    
class HolodinoInstanceGenerativeDataset(IterableDataset):
    def __init__(self, name, json_data):
        super().__init__()
        self.json = json_data
        self.json['seed'] = -1
        self.json['torch'] = True
        self.output_size = self.json['synthesis']['output_size']
        self.name = name
        
    def __len__(self):
        return -1
        
    def __iter__(self):
        while True:
            sample = SampleGenerator.from_dict(self.json).generate(self.output_size)
            yield sample
        
class HolodinoInstanceGenerativeDatasetServer(HolodinoInstanceGenerativeDataset):
    def __init__(self, name, json_data, server_url):
        super().__init__(name, json_data)
        self.server_url = server_url

    @backoff.on_exception(backoff.expo, requests.exceptions.ConnectionError, max_time=10)
    def send_config(self):
        response = requests.post(f"{self.server_url}/set_config", json=self.json)
        if response.status_code != 200:
            raise Exception(f"Error setting config: {response.text}")
        
    @backoff.on_exception(backoff.expo, requests.exceptions.ConnectionError, max_time=10)
    @backoff.on_predicate(backoff.expo, lambda x: x is None, max_time=300)
    def get_sample(self):
        response = requests.get(f"{self.server_url}/get_sample")
        return pickle.loads(response.content) if response.status_code == 200 else None
        
    def __iter__(self):
        self.send_config()
        while True:
            yield self.get_sample()

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_holo_instance(_root)
register_holo_generative(_root)