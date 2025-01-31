from detectron2.config import CfgNode as CN

def add_holodino_config(cfg):
    cfg.MODEL.RESNETS.KERNEL_SIZE = 3
    cfg.HOLODINO = CN()
    cfg.HOLODINO.RECONSTRUCTION_RANGE = (0, 100000)
    cfg.HOLODINO.RECONSTRUCTION_PROBABILITY = 0.0
    cfg.HOLODINO.RECONSTRUCTION_WAVELENGTH = 0.405
    cfg.HOLODINO.RECONSTRUCTION_RESOLUTION = 3.4
    cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1
    cfg.SOLVER.ANNEALING_CYCLES = 1
