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
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_every: int = 2500,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.ckpt_manager = checkpoint_manager
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.save_every = save_every

    def train(self, max_steps: int, start_step: int = 0):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        step = start_step
        running_loss = 0.0
        pbar = tqdm(total=max_steps, initial=start_step, desc="Training Mock-D1 7B")

        for batch in self.dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            _, loss, _ = self.model(input_ids=input_ids, labels=labels)
            loss = loss / self.grad_accum_steps
            loss.backward()
            running_loss += loss.item() * self.grad_accum_steps

            if (step + 1) % self.grad_accum_steps == 0:
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