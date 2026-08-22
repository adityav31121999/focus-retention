import torch
import torch.nn as nn
from tqdm import tqdm
from .checkpoint import CheckpointManager
from data.curriculum_streamer import CurriculumStreamer, create_curriculum_dataloader


class CurriculumTrainer:
    """
    Manages progressive context growth: 128 -> 512 -> 2048 -> 8192 -> 65536 / 262144.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        streamer: CurriculumStreamer,
        curriculum_stages: list,
        checkpoint_manager: CheckpointManager,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision: str = "auto",
        max_grad_norm: float = 1.0,
        save_every: int = 2500,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.streamer = streamer
        self.stages = curriculum_stages
        self.ckpt_manager = checkpoint_manager
        self.max_grad_norm = max_grad_norm
        self.save_every = save_every

        # Configure mixed precision
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
                    self.scaler = torch.cuda.amp.GradScaler()
            elif mixed_precision in ["bfloat16", "bf16"]:
                self.use_amp = True
                self.amp_dtype = torch.bfloat16
            elif mixed_precision in ["float16", "fp16"]:
                self.use_amp = True
                self.amp_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler()

    def _get_current_stage(self, step: int):
        for stage in self.stages:
            if step < stage["max_steps"]:
                return stage
        return self.stages[-1]

    def train(self, start_step: int = 0):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        current_stage = self._get_current_stage(start_step)
        self.streamer.set_seq_len(current_stage["seq_len"])
        dataloader = create_curriculum_dataloader(self.streamer, current_stage["batch_size"])
        data_iter = iter(dataloader)

        total_steps = self.stages[-1]["max_steps"]
        pbar = tqdm(total=total_steps, initial=start_step, desc="Curriculum Training")
        step = start_step

        while step < total_steps:
            # Check for stage transition
            active_stage = self._get_current_stage(step)
            if active_stage["name"] != current_stage["name"]:
                print(f"\n🚀 [Transition] Entering {active_stage['name']} | Seq Len: {active_stage['seq_len']}")
                current_stage = active_stage
                self.streamer.set_seq_len(current_stage["seq_len"])
                dataloader = create_curriculum_dataloader(self.streamer, current_stage["batch_size"])
                data_iter = iter(dataloader)

                # Adjust base LR for the new stage
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = current_stage["lr"]

            grad_accum = current_stage["gradient_accumulation_steps"]
            accum_loss = 0.0

            for _ in range(grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward & Backward with AMP
                if self.use_amp:
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        _, loss, _ = self.model(input_ids=input_ids, labels=labels)
                        loss = loss / grad_accum
                    
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    _, loss, _ = self.model(input_ids=input_ids, labels=labels)
                    loss = loss / grad_accum
                    loss.backward()

                accum_loss += loss.item() * grad_accum

            # Gradient step
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
            self.optimizer.zero_grad(set_to_none=True)

            step += 1
            pbar.update(1)
            pbar.set_postfix({
                "stage": current_stage["name"],
                "seq_len": current_stage["seq_len"],
                "loss": f"{accum_loss:.4f}"
            })

            # Checkpoint save
            if step % self.save_every == 0:
                self.ckpt_manager.save(step, self.model, self.optimizer, self.scheduler, accum_loss)

        pbar.close()