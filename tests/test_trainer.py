import pytest
import torch
from holodino import Trainer
from holodino.overrides import *
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import Instances, Boxes
from detectron2.engine import launch, default_setup
from maskdino import add_maskdino_config
from holodino.config import add_holodino_config

@pytest.fixture
def default_config():
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    add_holodino_config(cfg)
    cfg.merge_from_file("config/holodino-large.yaml")
    cfg.merge_from_list(["SOLVER.MAX_ITER", "100",
                         "DATASETS.TRAIN", ("fakeholo_hard_autogen",),
                         "INPUT.DATASET_MAPPER_NAME", "holodino_autogen",
                         "SOLVER.IMS_PER_BATCH", 4])
    cfg.freeze()
    default_setup(cfg, None)
    return cfg

@pytest.fixture
def model(default_config):
    return Trainer.build_model(default_config)

@pytest.fixture
def trainer(default_config):
    return Trainer(default_config)

def test_build_model(model):
    assert model is not None

@pytest.mark.skip(reason="Trainer takes too long")
def test_training(trainer):
    def train():
        trainer.train()
    launch(train, 1)

