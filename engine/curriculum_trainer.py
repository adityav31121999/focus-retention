import torch
import torch.nn as nn
from tqdm import tqdm
from .checkpoint import CheckpointManager
from data.curriculum_streamer import CurriculumStreamer, create_curriculum_dataloader


class CurriculumTrainer:
    """
    Manages progressive context growth: 128 -> 256 -> 1024 -> 8192 -> 262144.
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
        max_grad_norm: float = 1.0,
        save_every: int = 2500,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.streamer = streamer
        self.stages = curriculum_stages
        self.ckpt_manager = checkpoint_manager
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.save_every = save_every

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

                # Forward & Backward
                _, loss, _ = self.model(input_ids=input_ids, labels=labels)
                loss = loss / grad_accum
                loss.backward()
                accum_loss += loss.item() * grad_accum

            # Gradient step
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