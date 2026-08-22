# engine/curriculum_trainer.py
import time
import math
from typing import Tuple, List, Dict, Any, Optional
import torch
import torch.nn as nn
from tqdm import tqdm
from .checkpoint import CheckpointManager
from .optimiser import create_optimizer, get_cosine_schedule_with_warmup
from data.curriculum_streamer import CurriculumStreamer, create_curriculum_dataloader


class CurriculumTrainer:
    """
    Manages progressive context growth (128 -> 512 -> 2048 -> 8192 -> 65536 / 262144) with:
    - 1-Hour Wall-Clock Session Timer with graceful shutdown.
    - Dynamic Optimizer Switching: Adafactor (Stages 1-3) -> Muon (Stages 4-5).
    - Dataset State Tracking (Samples seen & token buffers) for seamless resume.
    - Sentence-by-sentence / micro-batch weight updates and gradient accumulation.
    """
    def __init__(
        self,
        model: nn.Module,
        streamer: CurriculumStreamer,
        curriculum_stages: List[Dict[str, Any]],
        checkpoint_manager: CheckpointManager,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        session_duration_minutes: Optional[float] = 55.0,  # 55 mins + 5 mins buffer for Colab/Kaggle
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision: str = "auto",
        max_grad_norm: float = 1.0,
        save_every: int = 500,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.streamer = streamer
        self.stages = curriculum_stages
        self.ckpt_manager = checkpoint_manager
        self.session_duration_sec = (session_duration_minutes * 60.0) if session_duration_minutes else None
        self.max_grad_norm = max_grad_norm
        self.save_every = save_every

        self.current_optimizer_type: Optional[str] = None
        self.optimizer: Optional[torch.optim.Optimizer] = optimizer
        self.scheduler: Optional[Any] = scheduler

        # Configure Mixed Precision (AMP)
        self.use_amp = False
        self.amp_dtype = torch.float32
        self.scaler = None

        if self.device.type == "cuda":
            if mixed_precision == "auto":
                if torch.cuda.is_bf16_supported():
                    self.use_amp = True
                    self.amp_dtype = torch.bfloat16
                else:
                    self.use_amp = True
                    self.amp_dtype = torch.float16
                    self.scaler = torch.amp.GradScaler("cuda")
            elif mixed_precision in ["bfloat16", "bf16"]:
                self.use_amp = True
                self.amp_dtype = torch.bfloat16
            elif mixed_precision in ["float16", "fp16"]:
                self.use_amp = True
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler("cuda")

    def _get_current_stage(self, step: int) -> Tuple[int, Dict[str, Any]]:
        for idx, stage in enumerate(self.stages):
            if step < stage["max_steps"]:
                return idx, stage
        return len(self.stages) - 1, self.stages[-1]

    def _configure_stage_optimizer(self, stage_idx: int, stage: Dict[str, Any]):
        """
        Policy Switch:
        - Stages 1 to 3 (Index 0, 1, 2) -> Adafactor (low memory, factorized 2nd moments)
        - Stages 4 to 5 (Index >= 3)    -> Muon (Newton-Schulz polar matrix orthogonalization)
        """
        # Determine target optimizer from stage name/dict or fallback to stage index rule
        target_opt = stage.get("optimizer", "adafactor" if stage_idx < 3 else "muon").lower()
        stage_lr = float(stage["lr"])
        total_steps = self.stages[-1]["max_steps"]

        if self.current_optimizer_type != target_opt or self.optimizer is None:
            print(f"\n⚙️  [Optimizer Setup] Activating '{target_opt.upper()}' for Stage {stage_idx + 1}")
            self.optimizer = create_optimizer(
                self.model,
                optimizer_type=target_opt,
                lr=stage_lr,
                muon_lr=stage_lr if target_opt == "muon" else 0.02,
                weight_decay=0.01 if target_opt == "adafactor" else 0.1
            )
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                warmup_steps=500,
                max_steps=total_steps
            )
            self.current_optimizer_type = target_opt
        else:
            # Adjust learning rate for the existing optimizer
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = stage_lr

    def _apply_stage_config(self, stage_idx: int, stage: Dict[str, Any]):
        """Synchronizes model stage tracker, sequence length, and optimizer."""
        seq_len = stage["seq_len"]
        batch_size = stage.get("batch_size", 1)
        grad_accum = stage.get("gradient_accumulation_steps", 1)
        stage_lr = float(stage["lr"])
        effective_tokens = seq_len * batch_size * grad_accum

        # 1. Update model config stage tracker
        if hasattr(self.model, "config"):
            self.model.config.curriculum_stage = stage_idx + 1
        elif hasattr(self.model, "module") and hasattr(self.model.module, "config"):
            self.model.module.config.curriculum_stage = stage_idx + 1

        # 2. Update streamer sequence length
        self.streamer.set_seq_len(seq_len)

        # 3. Configure stage optimizer & learning rate
        self._configure_stage_optimizer(stage_idx, stage)

        print("\n" + "=" * 70)
        print(f"🚀 [Curriculum Transition] Entering Stage {stage_idx + 1}: '{stage['name']}'")
        print(f"   • Context Window (Seq Len) : {seq_len:,} tokens")
        print(f"   • Micro Batch Size         : {batch_size}")
        print(f"   • Gradient Accumulation    : {grad_accum} steps")
        print(f"   • Effective Batch Size     : {effective_tokens:,} tokens / step")
        print(f"   • Active Optimizer         : {self.current_optimizer_type.upper()}")
        print(f"   • Base Learning Rate       : {stage_lr:.2e}")
        print(f"   • Target Step Boundary     : {stage['max_steps']:,} steps")
        print("=" * 70 + "\n")

    def train(self, start_step: int = 0):
        self.model.train()
        start_time = time.time()

        current_stage_idx, current_stage = self._get_current_stage(start_step)
        self._apply_stage_config(current_stage_idx, current_stage)

        dataloader = create_curriculum_dataloader(self.streamer, current_stage.get("batch_size", 1))
        data_iter = iter(dataloader)

        total_steps = self.stages[-1]["max_steps"]
        pbar = tqdm(total=total_steps, initial=start_step, desc="Curriculum Training")
        step = start_step
        latest_loss = 0.0

        while step < total_steps:
            # 1. Wall-Clock Session Limit Check
            if self.session_duration_sec is not None:
                elapsed_sec = time.time() - start_time
                if elapsed_sec >= self.session_duration_sec:
                    print(f"\n⏰ [Time Limit] Session elapsed ({elapsed_sec / 60:.1f}m). Saving state...")
                    streamer_state = self.streamer.state_dict() if hasattr(self.streamer, "state_dict") else None
                    self.ckpt_manager.save(
                        step=step,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        loss=latest_loss,
                        streamer_state=streamer_state,
                        optimizer_type=self.current_optimizer_type
                    )
                    print("✅ Session checkpoint safely saved. Ready for next session!")
                    break

            # 2. Stage Transition Check
            active_stage_idx, active_stage = self._get_current_stage(step)
            if active_stage["name"] != current_stage["name"]:
                current_stage_idx, current_stage = active_stage_idx, active_stage
                self._apply_stage_config(current_stage_idx, current_stage)
                dataloader = create_curriculum_dataloader(self.streamer, current_stage.get("batch_size", 1))
                data_iter = iter(dataloader)

            grad_accum = current_stage.get("gradient_accumulation_steps", 1)
            accum_loss = 0.0

            # 3. Micro-Batch Accumulation (or 1 update / sentence when batch=1, accum=1)
            self.optimizer.zero_grad(set_to_none=True)
            for _ in range(grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    dataloader = create_curriculum_dataloader(self.streamer, current_stage.get("batch_size", 1))
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
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

            # 4. Gradient Step & Optimizer Update
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

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]["lr"]
            postfix = {
                "loss": f"{accum_loss:.4f}",
                "opt": self.current_optimizer_type,
                "seq": current_stage["seq_len"],
                "lr": f"{current_lr:.2e}",
            }
            if self.session_duration_sec is not None:
                remaining_min = max(0.0, (self.session_duration_sec - (time.time() - start_time)) / 60.0)
                postfix["rem_time"] = f"{remaining_min:.1f}m"
            if hasattr(self.streamer, "samples_seen"):
                postfix["seen"] = f"{self.streamer.samples_seen:,}"

            pbar.update(1)
            pbar.set_postfix(postfix)

            # 5. Periodic Checkpoint Save
            if step % self.save_every == 0 or step == total_steps:
                streamer_state = self.streamer.state_dict() if hasattr(self.streamer, "state_dict") else None
                self.ckpt_manager.save(
                    step=step,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    loss=accum_loss,
                    streamer_state=streamer_state,
                    optimizer_type=self.current_optimizer_type
                )

        pbar.close()