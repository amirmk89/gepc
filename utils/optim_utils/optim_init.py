"""
File for initializing optimizers and schedulers
"""
import torch.optim as optim
from functools import partial

from utils.optim_utils.schedulers.delayed_sched import *
from utils.optim_utils.schedulers.cosine_annealing_with_warmup import *


def init_optimizer(type_str, **kwargs):
    if type_str.lower() == 'adam':
        opt_f = optim.Adam
    else:
        return None

    return partial(opt_f, **kwargs)


def init_scheduler(type_str, lr, epochs, warmup=3):
    sched_f = None
    if type_str.lower() == 'exp_decay':
        sched_f = None
    elif type_str.lower() == 'cosine':
        sched_f = partial(optim.lr_scheduler.CosineAnnealingLR, T_max=epochs)
    elif type_str.lower() == 'cosine_warmup':
        sched_f = partial(CosineAnnealingWarmUpRestarts, T_0=epochs, T_up=warmup)
    elif type_str.lower() == 'cosine_delayed':
        sched_f = partial(DelayedCosineAnnealingLR, delay_epochs=warmup,
                          cosine_annealing_epochs=epochs)
    elif (type_str.lower() == 'tri') and (epochs >= 8):
        sched_f = partial(optim.lr_scheduler.CyclicLR,
                          base_lr=lr/10, max_lr=lr*10,
                          step_size_up=epochs//8,
                          mode='triangular2',
                          cycle_momentum=False)
    else:
        print("Unable to initialize scheduler, defaulting to exp_decay")

    return sched_f

