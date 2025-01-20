import math
from fvcore.common.param_scheduler import CosineParamScheduler


class CosineAnnealingWithWarmRestartScheduler(CosineParamScheduler):
    def __init__(self, start_value: float, end_value: float, cycles):
        super().__init__(start_value, end_value)
        self.cycles = cycles
        
    def __call__(self, where: float) -> float:
        restart_cycle = (where * self.cycles) % 1
        if where == 1:
            restart_cycle = 1
        return super().__call__(where) * (1 + math.cos(math.pi * restart_cycle)) / 2
    
    