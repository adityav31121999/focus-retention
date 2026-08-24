# engine/curriculum_trainer.py
import time
import math
import os
from typing import Tuple, List, Dict, Any, Optional
import torch
import torch.nn as nn
from tqdm import tqdm
from .checkpoint import CheckpointManager
from .optimiser import create_optimizer, get_cosine_schedule_with_warmup
from data.curriculum_streamer import CurriculumStreamer, create_curriculum_dataloader

# Optional PyTorch-XLA (TPU) Import
try:
    import torch_xla.core.xla_model as xm
    import torch_xla
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

# Distributed PyTorch Import
import torch.distributed as dist


class CurriculumTrainer:
    """
    Unified Cross-Hardware Curriculum Trainer for Focus-Retention (Mock-D1):
    - Accelerators : NVIDIA CUDA, AMD ROCm/HIP, Google Cloud TPU (XLA), CPU.
    - Precision    : Hardware-native BF16 (TPU, Ampere+, ROCm), FP16 AMP (Turing/Volta), pure FP16.
    - Topology     : Single Accelerator, Multi-GPU DDP (torchrun), Multi-Core TPU Pods.
    - Optimizers   : Dynamic Stage Switching (Adafactor Stages 1-3 -> Muon Stages 4-5).
    - Session      : Timed execution with dataset streaming state tracking.
    """
    def __init__(
        self,
        model: nn.Module,
        streamer: CurriculumStreamer,
        curriculum_stages: List[Dict[str, Any]],
        checkpoint_manager: CheckpointManager,
        session_duration_minutes: Optional[float] = 660.0,
        device: str = "auto",                  # 'auto', 'cuda', 'rocm', 'hip', 'xla', 'tpu', 'cpu'
        mixed_precision: str = "auto",         # 'auto', 'bfloat16', 'float16', 'no'
        max_grad_norm: float = 1.0,
        save_every: int = 500,
    ):
        self.streamer = streamer
        self.stages = curriculum_stages
        self.ckpt_manager = checkpoint_manager
        self.session_duration_sec = (session_duration_minutes * 60.0) if session_duration_minutes else None
        self.max_grad_norm = max_grad_norm
        self.save_every = save_every

        # ----------------------------------------------------------------------
        # 1. Hardware & Accelerator Detection
        # ----------------------------------------------------------------------
        dev_req = device.lower().strip()
        self.backend = "cpu"
        self.is_xla = False
        self.is_rocm = False
        self.is_cuda = False
        self.is_master = True
        self.world_size = 1

        # A. Google TPU (XLA) Check
        if dev_req in ["xla", "tpu"] or (dev_req == "auto" and HAS_XLA and "TPU" in xm.xla_device_hw(xm.xla_device())):
            self.device = xm.xla_device()
            self.backend = "tpu"
            self.is_xla = True
            self.is_master = xm.is_master_ordinal()
            self.world_size = xm.xrt_world_size()

        # B. AMD GPU (ROCm / HIP) Check
        elif torch.cuda.is_available() and (dev_req in ["rocm", "hip"] or getattr(torch.version, "hip", None) is not None):
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device(f"cuda:{local_rank}")
            self.backend = "rocm"
            self.is_rocm = True
            if dist.is_available() and dist.is_initialized():
                self.is_master = dist.get_rank() == 0
                self.world_size = dist.get_world_size()

        # C. NVIDIA GPU (CUDA) Check
        elif torch.cuda.is_available() and dev_req in ["cuda", "auto"]:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device(f"cuda:{local_rank}")
            self.backend = "cuda"
            self.is_cuda = True
            if dist.is_available() and dist.is_initialized():
                self.is_master = dist.get_rank() == 0
                self.world_size = dist.get_world_size()

        # D. CPU / Fallback
        else:
            self.device = torch.device("cpu")
            self.backend = "cpu"
            self.is_master = True
            self.world_size = 1

        # Move model to resolved device
        self.model = model.to(self.device)

        # ----------------------------------------------------------------------
        # 2. Precision & Mixed Precision (AMP / Scaler) Setup
        # ----------------------------------------------------------------------
        is_half_model = any(p.dtype in {torch.float16, torch.bfloat16} for p in self.model.parameters())
        self.use_amp = False
        self.amp_dtype = torch.float32
        self.scaler = None
        self.autocast_device = "cuda" if (self.is_cuda or self.is_rocm) else "cpu"

        if self.is_xla:
            # TPUs compute in native hardware bfloat16 without GradScaler
            self.use_amp = False
            self.amp_dtype = torch.bfloat16
            self.scaler = None
        elif is_half_model:
            # Pure FP16/BF16 model: GradScaler is disabled to prevent unscale exceptions
            self.use_amp = False
            self.amp_dtype = next(self.model.parameters()).dtype
            self.scaler = None
        elif self.is_cuda or self.is_rocm:
            if mixed_precision == "auto":
                if torch.cuda.is_bf16_supported():
                    self.use_amp = True
                    self.amp_dtype = torch.bfloat16
                    self.scaler = None
                else:
                    self.use_amp = True
                    self.amp_dtype = torch.float16
                    self.scaler = torch.amp.GradScaler("cuda")
            elif mixed_precision in ["bfloat16", "bf16"]:
                self.use_amp = True
                self.amp_dtype = torch.bfloat16
                self.scaler = None
            elif mixed_precision in ["float16", "fp16"]:
                self.use_amp = True
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler("cuda")

        self.current_optimizer_type: Optional[str] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None

    def _get_current_stage(self, step: int) -> Tuple[int, Dict[str, Any]]:
        for idx, stage in enumerate(self.stages):
            if step < stage["max_steps"]:
                return idx, stage
        return len(self.stages) - 1, self.stages[-1]

    def _configure_stage_optimizer(self, stage_idx: int, stage: Dict[str, Any]):
        """
        Policy Switch:
        - Stages 1 to 3 -> Adafactor (factorized 2nd moments, low memory)
        - Stages 4 to 5 -> Muon (Newton-Schulz polar decomposition)
        """
        target_opt = stage.get("optimizer", "adafactor" if stage_idx < 3 else "muon").lower()
        stage_lr = float(stage["lr"])
        total_steps = self.stages[-1]["max_steps"]

        if self.current_optimizer_type != target_opt or self.optimizer is None:
            if self.is_master:
                print(f"\n⚙️  [Optimizer Setup] Activating '{target_opt.upper()}' for Stage {stage_idx + 1}")
            self.optimizer = create_optimizer(
                self.model,
                optimizer_type=target_opt,
                lr=stage_lr,
                muon_lr=stage_lr if target_opt == "muon" else 0.02,
                weight_decay=0.01 if target_opt == "adafactor" else 0.1
            )
            warmup = stage.get("warmup_steps", 300)
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                warmup_steps=warmup,
                max_steps=total_steps
            )
            self.current_optimizer_type = target_opt
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = stage_lr

    def _apply_stage_config(self, stage_idx: int, stage: Dict[str, Any]):
        """Synchronizes model stage tracking, context length, and optimizer parameters."""
        seq_len = stage["seq_len"]
        batch_size = stage.get("batch_size", 2)
        grad_accum = stage.get("gradient_accumulation_steps", 8)
        stage_lr = float(stage["lr"])
        effective_tokens = seq_len * batch_size * grad_accum * self.world_size

        # 1. Update model stage tag
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(raw_model, "config"):
            raw_model.config.curriculum_stage = stage_idx + 1

        # 2. Update data streamer context length
        self.streamer.set_seq_len(seq_len)

        # 3. Configure stage optimizer
        self._configure_stage_optimizer(stage_idx, stage)

        if self.is_master:
            if self.is_xla:
                hw_desc = f"Google TPU ({self.world_size} Cores)"
            elif self.is_rocm:
                hw_desc = f"AMD ROCm/HIP ({self.world_size} GPUs)"
            elif self.is_cuda:
                hw_desc = f"NVIDIA CUDA ({self.world_size} GPUs)"
            else:
                hw_desc = "CPU"

            print("\n" + "=" * 75)
            print(f"🚀 [Curriculum Transition] Stage {stage_idx + 1}: '{stage['name']}' on {hw_desc}")
            print(f"   • Context Window (Seq Len) : {seq_len:,} tokens")
            print(f"   • Micro Batch / Device     : {batch_size}")
            print(f"   • Grad Accumulation Steps  : {grad_accum}")
            print(f"   • Global Effective Batch   : {effective_tokens:,} tokens / step")
            print(f"   • Active Optimizer         : {self.current_optimizer_type.upper()}")
            print(f"   • Base Learning Rate       : {stage_lr:.2e}")
            print(f"   • Target Step Boundary     : {stage['max_steps']:,} steps")
            print("=" * 75 + "\n")

    def train(self, start_step: int = 0):
        self.model.train()
        start_time = time.time()

        current_stage_idx, current_stage = self._get_current_stage(start_step)
        self._apply_stage_config(current_stage_idx, current_stage)

        dataloader = create_curriculum_dataloader(self.streamer, current_stage.get("batch_size", 2))
        data_iter = iter(dataloader)

        total_steps = self.stages[-1]["max_steps"]
        pbar = tqdm(total=total_steps, initial=start_step, desc="Curriculum Training") if self.is_master else None
        step = start_step
        latest_loss = 0.0

        while step < total_steps:
            # 1. Wall-Clock Session Timer Check
            if self.session_duration_sec is not None:
                elapsed_sec = time.time() - start_time
                if elapsed_sec >= self.session_duration_sec:
                    if self.is_master:
                        print(f"\n⏰ [Session Limit] Session duration reached ({elapsed_sec / 60:.1f}m). Saving state...")
                        streamer_state = self.streamer.state_dict() if hasattr(self.streamer, "state_dict") else None
                        self.ckpt_manager.save(step=step, model=self.model, streamer_state=streamer_state, loss=latest_loss)
                    break

            # 2. Stage Transition Check
            active_stage_idx, active_stage = self._get_current_stage(step)
            if active_stage["name"] != current_stage["name"]:
                current_stage_idx, current_stage = active_stage_idx, active_stage
                self._apply_stage_config(current_stage_idx, current_stage)
                dataloader = create_curriculum_dataloader(self.streamer, current_stage.get("batch_size", 2))
                data_iter = iter(dataloader)

            grad_accum = current_stage.get("gradient_accumulation_steps", 8)
            accum_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)

            # 3. Micro-Batch Accumulation Loop
            for _ in range(grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    dataloader = create_curriculum_dataloader(self.streamer, current_stage.get("batch_size", 2))
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.autocast(device_type=self.autocast_device, dtype=self.amp_dtype):
                        _, loss, _ = self.model(input_ids=input_ids, labels=labels)
                        scaled_loss = loss / grad_accum

                    if self.scaler is not None:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                else:
                    _, loss, _ = self.model(input_ids=input_ids, labels=labels)
                    scaled_loss = loss / grad_accum
                    scaled_loss.backward()

                accum_loss += scaled_loss.item()

            # 4. Gradient Step & Parameter Update (Dispatched per Backend)
            if self.is_xla:
                # TPU (XLA) Step Dispatch
                xm.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                xm.optimizer_step(self.optimizer)
                xm.mark_step()
            else:
                # GPU (CUDA / ROCm) & CPU Step Dispatch
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            latest_loss = accum_loss
            step += 1

            # 5. Distributed Loss Synchronization for Telemetry
            if self.world_size > 1:
                if self.is_xla:
                    avg_loss = xm.mesh_reduce("loss_reduce", accum_loss, lambda x: sum(x) / len(x))
                elif dist.is_available() and dist.is_initialized():
                    loss_tensor = torch.tensor(accum_loss, device=self.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                    avg_loss = loss_tensor.item()
                else:
                    avg_loss = accum_loss
            else:
                avg_loss = accum_loss

            # 6. Progress Bar & Periodic Checkpoints (Master Ordinal Only)
            if self.is_master and pbar is not None:
                current_lr = self.optimizer.param_groups[0]["lr"]
                postfix = {
                    "loss": f"{avg_loss:.4f}",
                    "opt": self.current_optimizer_type,
                    "seq": current_stage["seq_len"],
                    "lr": f"{current_lr:.2e}",
                }
                if self.session_duration_sec is not None:
                    rem_m = max(0.0, (self.session_duration_sec - (time.time() - start_time)) / 60.0)
                    postfix["rem_time"] = f"{rem_m:.1f}m"
                if hasattr(self.streamer, "samples_seen_in_current_ds"):
                    postfix["seen"] = f"{self.streamer.samples_seen_in_current_ds:,}"

                pbar.update(1)
                pbar.set_postfix(postfix)

                save_every = current_stage.get("save_every", self.save_every)
                if step % save_every == 0 or step == total_steps:
                    streamer_state = self.streamer.state_dict() if hasattr(self.streamer, "state_dict") else None
                    self.ckpt_manager.save(
                        step=step,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        loss=avg_loss,
                        streamer_state=streamer_state,
                        optimizer_type=self.current_optimizer_type
                    )

        if self.is_master and pbar is not None:
            pbar.close()

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()