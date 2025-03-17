from detectron2.config import CfgNode as CN, get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config

def add_holodino_config(cfg):
    cfg.MODEL.RESNETS.KERNEL_SIZE = 3
    cfg.HOLODINO = CN()
    cfg.HOLODINO.RECONSTRUCTION_RANGE = (0, 100000)
    cfg.HOLODINO.RECONSTRUCTION_PROBABILITY = 0.0
    cfg.HOLODINO.RECONSTRUCTION_WAVELENGTH = 0.405
    cfg.HOLODINO.RECONSTRUCTION_RESOLUTION = 3.4
    cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1
    cfg.SOLVER.ANNEALING_CYCLES = 1

def get_configuration(file, opts = None):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    add_holodino_config(cfg)
    cfg.merge_from_file(file)
    if opts:
        cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg
