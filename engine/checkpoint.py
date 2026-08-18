import os
import torch
import torch.nn as nn

class CheckpointManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save(self, step: int, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, loss: float):
        path = os.path.join(self.output_dir, f"checkpoint_step_{step}.pt")
        state = {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "loss": loss,
        }
        torch.save(state, path)
        print(f"[Checkpoint] Saved checkpoint to {path}")

    def load_latest(self, model: nn.Module, optimizer=None, scheduler=None):
        checkpoints = [f for f in os.listdir(self.output_dir) if f.startswith("checkpoint_step_")]
        if not checkpoints:
            return 0
        latest = max(checkpoints, key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
        path = os.path.join(self.output_dir, latest)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model_state"])
        if optimizer and state.get("optimizer_state"):
            optimizer.load_state_dict(state["optimizer_state"])
        if scheduler and state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
        print(f"[Checkpoint] Restored from {path} at step {state['step']}")
        return state["step"]