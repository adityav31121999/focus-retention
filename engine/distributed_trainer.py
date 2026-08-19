import os
import math
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from tqdm import tqdm

from mock_d1.block import MockD1Block
from mock_d1.model_mock import MockD1ForCausalLM
from mock_d1.configure_mockd1 import MockD1Config
from .checkpoint import CheckpointManager


class DistributedTrainer:
    """
    Unified training engine for Mock-D1:
    Supports Single-GPU, DistributedDataParallel (DDP), and FullyShardedDataParallel (FSDP).
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        dataloader,
        checkpoint_manager: CheckpointManager,
        mode: str = "single",                 # 'single', 'ddp', or 'fsdp'
        mixed_precision: str = "bfloat16",    # 'bfloat16', 'float16', 'no'
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_every: int = 2500,
        device_id: Optional[int] = None,
    ):
        self.mode = mode.lower()
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.save_every = save_every
        self.dataloader = dataloader
        self.scheduler = scheduler
        self.ckpt_manager = checkpoint_manager

        # Set up distributed environment details
        self.is_distributed = self.mode in ["ddp", "fsdp"]
        if self.is_distributed:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", device_id or 0))
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set computation precision dtype
        if mixed_precision == "bfloat16" and torch.cuda.is_bf16_supported():
            self.compute_dtype = torch.bfloat16
        elif mixed_precision == "float16":
            self.compute_dtype = torch.float16
        else:
            self.compute_dtype = torch.float32

        # Wrap model according to selected parallelization mode
        self.model = self._setup_model(model)
        self.optimizer = optimizer

    def _setup_model(self, base_model: nn.Module) -> nn.Module:
        base_model = base_model.to(self.device)

        if self.mode == "fsdp":
            # Auto-wrap each MockD1Block as an independent sharding unit
            auto_wrap_policy = transformer_auto_wrap_policy(
                transformer_layer_cls={MockD1Block}
            )
            mp_policy = MixedPrecision(
                param_dtype=self.compute_dtype,
                reduce_dtype=torch.float32,
                buffer_dtype=self.compute_dtype,
            )
            fsdp_model = FSDP(
                base_model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mp_policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 Full Sharding
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                device_id=self.device,
                limit_all_gathers=True,
            )
            return fsdp_model

        elif self.mode == "ddp":
            return DDP(
                base_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )

        return base_model

    def _clip_gradients(self):
        """Dispatches gradient clipping to FSDP or standard PyTorch norm."""
        if self.mode == "fsdp":
            self.model.clip_grad_norm_(self.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

    def _all_reduce_loss(self, loss_val: float) -> float:
        """Averages scalar loss across all GPUs for accurate logging."""
        if not self.is_distributed:
            return loss_val
        loss_tensor = torch.tensor(loss_val, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        return loss_tensor.item()

    def save_checkpoint(self, step: int, loss: float):
        """Unified checkpoint saving with rank-0 consolidation."""
        if self.mode == "fsdp":
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            optim_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
            
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy, optim_policy):
                model_state = self.model.state_dict()
                optim_state = FSDP.full_optim_state_dict(self.model, self.optimizer)

                if self.rank == 0:
                    self.ckpt_manager.save_raw(
                        step=step,
                        model_state=model_state,
                        optim_state=optim_state,
                        scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
                        loss=loss
                    )
        else:
            if self.rank == 0:
                raw_model = self.model.module if hasattr(self.model, "module") else self.model
                self.ckpt_manager.save(
                    step=step,
                    model=raw_model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    loss=loss
                )

        if self.is_distributed:
            dist.barrier()

    def train(self, max_steps: int, start_step: int = 0):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        step = start_step
        running_loss = 0.0
        pbar = tqdm(total=max_steps, initial=start_step, desc=f"Training Mock-D1 [{self.mode.upper()}]") if self.rank == 0 else None

        # Determine autocast device type
        autocast_device = "cuda" if "cuda" in self.device.type else "cpu"

        for batch in self.dataloader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            # 1. Forward Pass with Mixed Precision
            with torch.autocast(device_type=autocast_device, dtype=self.compute_dtype):
                _, loss, _ = self.model(input_ids=input_ids, labels=labels)
                scaled_loss = loss / self.grad_accum_steps

            # 2. Backward Pass
            scaled_loss.backward()
            running_loss += loss.item() / self.grad_accum_steps

            # 3. Optimizer Step
            if (step + 1) % self.grad_accum_steps == 0:
                self._clip_gradients()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                step += 1
                avg_loss = self._all_reduce_loss(running_loss)

                if self.rank == 0 and pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                running_loss = 0.0

                # 4. Checkpoint Save
                if step % self.save_every == 0:
                    self.save_checkpoint(step, avg_loss)

                if step >= max_steps:
                    break

        if self.rank == 0 and pbar is not None:
            pbar.close()

        if self.is_distributed:
            dist.destroy_process_group()