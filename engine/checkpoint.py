# checkpoint.py
import os
import glob
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Union


class CheckpointManager:
    """
    Manages model, optimizer, scheduler, and dataset streaming state checkpoints
    with automated disk quota pruning (keeping the N latest checkpoints).
    """
    def __init__(self, output_dir: str, keep_last_n: int = 2):
        self.output_dir = output_dir
        self.keep_last_n = keep_last_n
        os.makedirs(output_dir, exist_ok=True)

    def prune_old_checkpoints(self):
        """Keeps only the latest N checkpoints to prevent storage exhaustion."""
        ckpts = glob.glob(os.path.join(self.output_dir, "checkpoint_step_*.pt"))
        if len(ckpts) <= self.keep_last_n:
            return
        
        # Sort checkpoints numerically by step count
        sorted_ckpts = sorted(
            ckpts,
            key=lambda x: int(os.path.basename(x).split("_")[-1].replace(".pt", ""))
        )
        for ckpt in sorted_ckpts[:-self.keep_last_n]:
            try:
                os.remove(ckpt)
                print(f"[Checkpoint Pruner] Removed old checkpoint: {os.path.basename(ckpt)}")
            except Exception as e:
                print(f"[Checkpoint Pruner] Failed to remove {ckpt}: {e}")

    def save_raw(
        self,
        step: int,
        model_state: dict,
        optim_state: Optional[dict],
        scheduler_state: Optional[dict],
        loss: float,
        streamer_state: Optional[dict] = None,
        optimizer_type: Optional[str] = None,
    ):
        """Saves pre-consolidated state dictionaries (used by FSDP)."""
        path = os.path.join(self.output_dir, f"checkpoint_step_{step}.pt")
        state = {
            "step": step,
            "model_state": model_state,
            "optimizer_state": optim_state,
            "scheduler_state": scheduler_state,
            "streamer_state": streamer_state,
            "optimizer_type": optimizer_type,
            "loss": loss,
        }
        torch.save(state, path)
        print(f"[Checkpoint] Saved consolidated FSDP checkpoint to {path}")
        self.prune_old_checkpoints()

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        loss: float = 0.0,
        streamer_state: Optional[dict] = None,
        optimizer_type: Optional[str] = None,
    ):
        """Standard save for Single-GPU and DDP."""
        path = os.path.join(self.output_dir, f"checkpoint_step_{step}.pt")
        
        # Unwrap module if model is wrapped in DDP or custom container
        raw_model = model.module if hasattr(model, "module") else model

        state = {
            "step": step,
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "streamer_state": streamer_state,
            "optimizer_type": optimizer_type,
            "loss": loss,
        }
        torch.save(state, path)
        print(f"[Checkpoint] Saved checkpoint to {path}")
        self.prune_old_checkpoints()

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        streamer: Optional[Any] = None,
    ) -> Tuple[int, Optional[str], Optional[Dict[str, Any]]]:
        """
        Loads the most recent checkpoint and restores model, optimizer,
        scheduler, and dataset streaming offsets.
        
        Returns:
            Tuple of (step, optimizer_type, streamer_state)
        """
        checkpoints = [
            f for f in os.listdir(self.output_dir)
            if f.startswith("checkpoint_step_") and f.endswith(".pt")
        ]
        if not checkpoints:
            print("[Checkpoint] No previous checkpoints found. Starting from Step 0.")
            return 0, None, None

        latest_file = max(
            checkpoints,
            key=lambda x: int(x.split("_")[-1].replace(".pt", ""))
        )
        path = os.path.join(self.output_dir, latest_file)
        
        print(f"[Checkpoint] Restoring training state from {path}...")
        state = torch.load(path, map_location="cpu")

        # 1. Restore Model Weights
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(state["model_state"])

        # 2. Restore Optimizer State
        if optimizer and state.get("optimizer_state"):
            try:
                optimizer.load_state_dict(state["optimizer_state"])
            except Exception as e:
                print(f"[Checkpoint] Optimizer state skipped / reset (e.g. optimizer type changed): {e}")

        # 3. Restore Scheduler State
        if scheduler and state.get("scheduler_state"):
            try:
                scheduler.load_state_dict(state["scheduler_state"])
            except Exception as e:
                print(f"[Checkpoint] Scheduler state skipped: {e}")

        # 4. Restore Dataset Streamer State (if streamer object is provided)
        streamer_state = state.get("streamer_state", None)
        if streamer is not None and streamer_state is not None:
            if hasattr(streamer, "load_state_dict"):
                streamer.load_state_dict(streamer_state)

        step = state.get("step", 0)
        opt_type = state.get("optimizer_type", None)

        print(f"[Checkpoint] Restored from {path} at step {step} (Optimizer: {opt_type})")
        return step, opt_type, streamer_state