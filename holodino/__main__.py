import sys
import warnings
try:
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass
import os
import subprocess

import torch
import time
import torch.profiler
from holodino.trainer import Trainer

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.structures import Instances, Boxes

from detectron2.evaluation import (
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

# MaskDINO
from maskdino import (
    add_maskdino_config
)
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch
)
from holodino.overrides import *
from holodino.config import add_holodino_config
from holodino.dataset import HolodinoInstanceGenerativeDatasetMapper

def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    add_holodino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    return cfg


def main(args):
    cfg = setup(args)
    print("Command cfg:", cfg)
    warnings.filterwarnings("ignore", category=FutureWarning)
    if args.eval_only or args.profile_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=0, warmup=2, active=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.OUTPUT_DIR),
            profile_memory=True,
            with_stack=True,
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            for _ in range(4):
                outputs = model([{'image': torch.randn(3, 1024, 1024), 'height': 1024, 'width': 1024, 'instances': Instances(image_size=(1024, 1024), gt_masks=torch.zeros(1, 1024, 1024, dtype=torch.uint8), gt_classes=torch.zeros((1,), dtype=torch.long), gt_boxes=Boxes(torch.tensor([[0., 0., 1024., 1024.]])))}])
                sum(outputs.values()).backward()
                profiler.step()
        if args.profile_only:
            return
        checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    if args.benchmark_iter:
        mapper = HolodinoInstanceGenerativeDatasetMapper(cfg)
        loader = build_detection_train_loader(cfg, mapper=mapper)
        start = time.perf_counter()
        for i, batch in enumerate(loader):
            print(len(batch))
            end = time.perf_counter()
            print("Batch took {} seconds".format(end - start))
            start = end
            if i > 10:
                return
            
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--profile_only', action='store_true')
    parser.add_argument('--benchmark_iter', action='store_true')
    parser.add_argument('--genserver', action='store_true')
    args = parser.parse_args()
    if args.genserver:
        subprocess.Popen(["python", "-m", "fakeholo", "server", "-l", "output/server.log", "-q"])
    print("Command Line Args:", args)
    print("pwd:", os.getcwd())
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )