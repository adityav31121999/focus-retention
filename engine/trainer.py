import torch
import torch.nn as nn
from tqdm import tqdm
from .checkpoint import CheckpointManager


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        dataloader,
        checkpoint_manager: CheckpointManager,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision: str = "auto",  # 'auto', 'bfloat16', 'float16', 'no'
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_every: int = 2500,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.ckpt_manager = checkpoint_manager
        self.grad_accum_steps = grad_accum_steps
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

    def train(self, max_steps: int, start_step: int = 0):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        step = start_step
        running_loss = 0.0
        pbar = tqdm(total=max_steps, initial=start_step, desc=f"Training {getattr(self.model, 'config', {}).get('model_name', 'Mock-D1') if hasattr(self.model, 'config') else 'Mock-D1'}")

        for batch in self.dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            if self.use_amp:
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    _, loss, _ = self.model(input_ids=input_ids, labels=labels)
                    loss = loss / self.grad_accum_steps
                
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                _, loss, _ = self.model(input_ids=input_ids, labels=labels)
                loss = loss / self.grad_accum_steps
                loss.backward()

            running_loss += loss.item() * self.grad_accum_steps

            if (step + 1) % self.grad_accum_steps == 0:
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

                pbar.update(1)
                pbar.set_postfix({"loss": f"{running_loss / self.grad_accum_steps:.4f}"})
                running_loss = 0.0
                step += 1

                if step % self.save_every == 0:
                    self.ckpt_manager.save(step, self.model, self.optimizer, self.scheduler, loss.item())

                if step >= max_steps:
                    break

        pbar.close()