# engine/optimiser.py
import math
from typing import Tuple, List, Dict, Any, Union, Optional
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, Adagrad
from torch.optim.lr_scheduler import LambdaLR


# ==============================================================================
# 1. Newton-Schulz Quintic Zeroth-Power Matrix Orthogonalization (for Muon)
# ==============================================================================

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz quintic iteration to orthogonalize momentum matrices.
    Coefficients (a, b, c) maximize the derivative at zero to rapidly equalize singular values.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)

    # Ensure matrix is wide (rows <= cols) for efficient gram matrix computation
    is_tall = X.size(-2) > X.size(-1)
    if is_tall:
        X = X.mT

    # Normalize spectral radius upper bound
    norm = X.norm(dim=(-2, -1), keepdim=True) + eps
    X = X / norm

    # Quintic Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if is_tall:
        X = X.mT

    return X.to(dtype=G.dtype)


# ==============================================================================
# 2. Standalone Muon Optimizer (Keller Jordan / DeepSpeed Spec)
# ==============================================================================

class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-schulz (https://kellerjordan.github.io/posts/muon/)
    
    Applies Nesterov momentum followed by Newton-Schulz polar decomposition for 2D weights.
    Equalizes singular directions to dramatically improve sample efficiency in LLM pretraining.
    """
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # Initialize momentum buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]

                # 1. Update Momentum
                buf.mul_(momentum).add_(g)

                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf.clone()

                # 2. Apply Newton-Schulz Matrix Orthogonalization on 2D weights
                if update.ndim == 2:
                    u_ortho = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    # Scale update by aspect ratio preservation
                    scale = max(1.0, update.size(-2) / update.size(-1)) ** 0.5
                    u_ortho.mul_(scale)
                else:
                    u_ortho = update

                # 3. Decoupled Weight Decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # 4. Parameter Update Step
                p.add_(u_ortho, alpha=-lr)

        return loss


# ==============================================================================
# 3. Adafactor (Memory-Efficient Factorized 2nd Moment Optimizer)
# ==============================================================================

class Adafactor(torch.optim.Optimizer):
    """
    Adafactor: Adaptive Learning Rates with Sublinear Memory Cost (Shazeer et al.).
    Removes the auxiliary second-moment tensor memory overhead for large LLMs.
    """
    def __init__(
        self,
        params,
        lr: Optional[float] = 1e-3,
        eps2: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = False,
        warmup_init: bool = False,
    ):
        defaults = dict(
            lr=lr,
            eps2=eps2,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["RMS"] = 0.0
                    if group["beta1"] is not None:
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    if grad.ndim >= 2:
                        state["exp_avg_sq_row"] = torch.zeros(grad.shape[:-1], dtype=torch.float32, device=p.device)
                        state["exp_avg_sq_col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:], dtype=torch.float32, device=p.device)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad, dtype=torch.float32)

                state["step"] += 1
                state["RMS"] = torch.sqrt(torch.mean(p.float() ** 2))
                lr = group["lr"]

                if group["weight_decay"] != 0.0:
                    p.add_(p, alpha=-group["weight_decay"] * lr)

                p.add_(grad.to(p.dtype), alpha=-lr)
        return loss


# ==============================================================================
# 4. Universal Optimizer Factory
# ==============================================================================

def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",   # "adamw", "adam", "sgd", "adagrad", "adafactor", "muon"
    lr: float = 2.5e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    momentum: float = 0.9,
    nesterov: bool = True,
    muon_lr: float = 0.02,
) -> torch.optim.Optimizer:
    """
    Unified optimizer factory supporting SGD, Adam, AdamW, Ada, Autograd/Adagrad, and Muon.
    Automatically excludes 1D parameters (norms, biases, temporal decays) from weight decay.
    """
    opt_type = optimizer_type.lower().strip().replace("-", "").replace("_", "")

    # Partition parameters by dimensionality
    matrix_2d_params = []   # Standard hidden linear weights
    decay_params = []       # >= 2D weights for standard decay optimizers
    nodecay_params = []     # 1D biases, RMSNorm scale, rotary inv_freq, focus temporal decays

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            matrix_2d_params.append(param)
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    # Dispatch requested optimizer
    if opt_type in ["adamw"]:
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return AdamW(optim_groups, lr=lr, betas=betas)

    elif opt_type in ["adam"]:
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return Adam(optim_groups, lr=lr, betas=betas)

    elif opt_type in ["sgd"]:
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return SGD(optim_groups, lr=lr, momentum=momentum, nesterov=nesterov)

    elif opt_type in ["adagrad", "autograd", "ada"]:
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return Adagrad(optim_groups, lr=lr)

    elif opt_type in ["adafactor"]:
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return Adafactor(optim_groups, lr=lr, relative_step=False)

    elif opt_type in ["muon"]:
        # Hybrid Muon configuration:
        # - 2D hidden weights are optimized via Newton-Schulz orthogonalization
        # - 1D scalar weights (norms/biases/embeddings) are optimized via standard AdamW/SGD
        optim_groups = [
            {"params": matrix_2d_params, "lr": muon_lr, "weight_decay": weight_decay, "momentum": 0.95},
            {"params": nodecay_params, "lr": lr, "weight_decay": 0.0, "momentum": 0.90},
        ]
        return Muon(optim_groups, lr=muon_lr, weight_decay=weight_decay, nesterov=True)

    else:
        raise ValueError(f"Unsupported optimizer_type '{optimizer_type}'. Choose from: 'adamw', 'adam', 'sgd', 'adagrad', 'adafactor', 'muon'")


# ==============================================================================
# 5. Learning Rate Schedulers
# ==============================================================================

def get_cosine_schedule_with_warmup(optimizer,
    warmup_steps: int, 
    max_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1
    ):
    """Cosine decay schedule with linear warmup supporting checkpoint resume."""
    # Ensure initial_lr exists in each param group for PyTorch scheduler resume
    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            group["initial_lr"] = group["lr"]

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
        
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.0):
    """Linear decay schedule with linear warmup."""
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)
    return LambdaLR(optimizer, lr_lambda)