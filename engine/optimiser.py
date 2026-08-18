import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def create_optimizer(model: nn.Module, lr: float = 2.5e-4, weight_decay: float = 0.1, betas=(0.9, 0.95)):
    decay_params = []
    nodecay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return AdamW(optim_groups, lr=lr, betas=betas)

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)