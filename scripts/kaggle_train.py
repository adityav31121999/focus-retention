"""
scripts/kaggle_train.py
=======================
Unified, high-throughput training runner tailored for Kaggle GPU Notebooks (T4 x2, P100, A100).
Supports single-GPU execution or high-throughput Dual-GPU Distributed Data Parallel (DDP).

Usage:
------
# 1. Dual-GPU High Throughput (Recommended on Kaggle GPU T4 x2):
!torchrun --nproc_per_node=2 scripts/kaggle_train.py --model_size 1.7B --mode curriculum

# 2. Single-GPU Execution:
!python scripts/kaggle_train.py --model_size 1.7B --mode curriculum
"""

import os
import sys
import glob
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# ------------------------------------------------------------------------------
# 1. Environment & Secrets Setup
# ------------------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Ingest Kaggle Secret HF_TOKEN if available
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
except Exception:
    pass

from mock_d1 import MockD1Config, MockD1ForCausalLM
from data.tokeniser import TokenizerManager
from data.curriculum_streamer import CurriculumStreamer, create_curriculum_dataloader
from data.hf_streamer import HuggingFaceStreamer, get_hf_dataloader
from engine.optimiser import create_optimizer, get_cosine_schedule_with_warmup


# ------------------------------------------------------------------------------
# 2. Lightweight Checkpoint Manager (Kaggle Quota Guard)
# ------------------------------------------------------------------------------
class KaggleCheckpointManager:
    """
    Saves strictly FP16 model weights and streamer progress to prevent filling
    Kaggle's 57.6 GB disk quota. Keeps only the latest N checkpoints.
    """
    def __init__(self, output_dir: str = "/kaggle/working/mock_d1_weights", keep_last_n: int = 2, save_fp16: bool = True):
        self.output_dir = output_dir
        self.keep_last_n = keep_last_n
        self.save_fp16 = save_fp16
        os.makedirs(output_dir, exist_ok=True)

    def prune_old_checkpoints(self):
        ckpts = glob.glob(os.path.join(self.output_dir, "weights_step_*.pt"))
        if len(ckpts) <= self.keep_last_n:
            return
        sorted_ckpts = sorted(ckpts, key=lambda x: int(os.path.basename(x).split("_")[-1].replace(".pt", "")))
        for ckpt in sorted_ckpts[:-self.keep_last_n]:
            try:
                os.remove(ckpt)
                print(f"✂️ [Disk Pruner] Removed old checkpoint: {os.path.basename(ckpt)}")
            except Exception:
                pass

    def save(self, step: int, model: nn.Module, streamer_state: dict = None, loss: float = 0.0):
        path = os.path.join(self.output_dir, f"weights_step_{step}.pt")
        raw_model = model.module if hasattr(model, "module") else model
        state_dict = raw_model.state_dict()
        if self.save_fp16:
            state_dict = {k: v.half() if torch.is_floating_point(v) else v for k, v in state_dict.items()}

        torch.save({
            "step": step,
            "model_state": state_dict,
            "streamer_state": streamer_state,
            "loss": loss
        }, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"💾 [Checkpoint] Saved model weights to {path} ({size_mb:.1f} MB)")
        self.prune_old_checkpoints()

    def load_latest(self, model: nn.Module, streamer=None):
        search_dirs = [self.output_dir]
        input_ckpts = glob.glob("/kaggle/input/**/weights_step_*.pt", recursive=True)
        checkpoints = []
        for d in search_dirs:
            if os.path.exists(d):
                checkpoints.extend([os.path.join(d, f) for f in os.listdir(d) if f.startswith("weights_step_") and f.endswith(".pt")])
        checkpoints.extend(input_ckpts)

        if not checkpoints:
            print("🚀 [Checkpoint] No previous weights found. Starting fresh from Step 0.")
            return 0, None

        latest_file = max(checkpoints, key=lambda x: int(os.path.basename(x).split("_")[-1].replace(".pt", "")))
        print(f"🔄 [Checkpoint] Restoring model weights from {latest_file}...")
        state = torch.load(latest_file, map_location="cpu")

        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(state["model_state"])

        streamer_state = state.get("streamer_state", None)
        if streamer and streamer_state and hasattr(streamer, "load_state_dict"):
            streamer.load_state_dict(streamer_state)

        step = state.get("step", 0)
        print(f"✅ Restored weights at Step {step} (Loss: {state.get('loss', 0.0):.4f})")
        return step, streamer_state


# ------------------------------------------------------------------------------
# 3. Main Execution Function
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Kaggle Dual-GPU Training Runner (Mock-D1)")
    parser.add_argument("--model_size", type=str, default="1.7B", choices=["1.7B", "1.7b", "7B", "7b"],
                        help="Model size: '1.7B' (default) or '7B'")
    parser.add_argument("--mode", type=str, default="curriculum", choices=["curriculum", "standard"],
                        help="Training mode: 'curriculum' or 'standard'")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (defaults to configs/curriculum_kaggle.yaml)")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/mock_d1_weights",
                        help="Directory to save weights")
    parser.add_argument("--session_minutes", type=float, default=660.0,
                        help="Max runtime timer in minutes (default: 660 mins / 11 hours)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable activation checkpointing")
    args = parser.parse_args()

    # DDP Distributed Environment Setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Resolve Configuration Path
    if args.config is not None:
        config_path = args.config
    elif args.mode == "curriculum":
        config_path = "configs/curriculum_kaggle.yaml" if os.path.exists("configs/curriculum_kaggle.yaml") else "configs/curriculum_config.yaml"
    else:
        config_path = "configs/default_config.yaml"

    if local_rank == 0:
        print(f"📖 Loading Kaggle configuration from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Initialize Tokenizer
    tokenizer = TokenizerManager(cfg["data"]["tokenizer_name"])

    # 2. Instantiate Model Directly in FP16 on this GPU (Avoids 6.8 GB FP32 spike)
    model_cfg = MockD1Config.from_dict(cfg["model"])

    torch.set_default_dtype(torch.float16)
    with torch.device(device):
        model = MockD1ForCausalLM(model_cfg)
    torch.set_default_dtype(torch.float32)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if local_rank == 0:
        total_params = model.module.get_num_params() if is_distributed else model.get_num_params()
        vram_used = torch.cuda.memory_allocated(local_rank) / 1e9
        print(f"📊 Model: {model_cfg.model_name} | Parameters: {total_params:,} ({total_params / 1e9:.3f}B)")
        print(f"💾 Initial Model VRAM: {vram_used:.2f} GB / 15.0 GB")
        print(f"🎮 Active GPUs: {world_size} (Distributed DDP: {is_distributed})")

    # 3. Setup Checkpoint Manager
    output_dir = args.output_dir or cfg["training"].get("output_dir", "/kaggle/working/mock_d1_weights")
    keep_n = cfg["training"].get("keep_last_n_checkpoints", 2)
    ckpt_manager = KaggleCheckpointManager(output_dir=output_dir, keep_last_n=keep_n, save_fp16=True)

    # 4. Setup Multi-Dataset Streamer
    streamer = CurriculumStreamer(
        tokenizer=tokenizer,
        datasets_list=cfg["data"].get("datasets", None),
        dataset_name=cfg["data"].get("dataset_name", "roneneldan/TinyStories"),
        dataset_config=cfg["data"].get("dataset_config", None),
        initial_seq_len=cfg["curriculum"]["stages"][0]["seq_len"] if args.mode == "curriculum" else cfg["data"]["seq_len"],
        split=cfg["data"].get("split", "train"),
        buffer_size=cfg["data"].get("buffer_size", 25000),
        seed=cfg["data"].get("seed", 42) + local_rank  # Diversify data stream across GPUs
    )

    # Restore previous weights if resuming
    start_step, _ = ckpt_manager.load_latest(model=model, streamer=streamer)

    # 5. Execute Training Loop
    if args.mode == "curriculum":
        stages = cfg["curriculum"]["stages"]
        total_steps = stages[-1]["max_steps"]
        session_sec = float(args.session_minutes) * 60.0

        current_stage_idx = 0
        for idx, st in enumerate(stages):
            if start_step < st["max_steps"]:
                current_stage_idx = idx
                break
        current_stage = stages[current_stage_idx]

        stage_opt_type = current_stage.get("optimizer", "adafactor" if current_stage_idx < 3 else "muon").lower()
        stage_lr = float(current_stage["lr"])
        optimizer = create_optimizer(
            model,
            optimizer_type=stage_opt_type,
            lr=stage_lr,
            muon_lr=stage_lr if stage_opt_type == "muon" else 0.02,
            weight_decay=0.01 if stage_opt_type == "adafactor" else 0.1
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=500, max_steps=total_steps)

        streamer.set_seq_len(current_stage["seq_len"])
        dataloader = create_curriculum_dataloader(streamer, current_stage.get("batch_size", 2))
        data_iter = iter(dataloader)

        if local_rank == 0:
            eff_tokens = current_stage.get("batch_size", 2) * world_size * current_stage["seq_len"] * current_stage.get("gradient_accumulation_steps", 8)
            print("\n" + "=" * 70)
            print(f"🚀 [Training Active] Stage {current_stage_idx + 1}: {current_stage['name']}")
            print(f"   • Micro Batch/GPU : {current_stage.get('batch_size', 2)}")
            print(f"   • Grad Accum Steps: {current_stage.get('gradient_accumulation_steps', 8)}")
            print(f"   • Global Effective: {eff_tokens:,} tokens / step")
            print(f"   • Active Optimizer: {stage_opt_type.upper()} | Base LR: {stage_lr:.2e}")
            print("=" * 70 + "\n")

        pbar = tqdm(total=total_steps, initial=start_step, desc="Kaggle Pretraining") if local_rank == 0 else None
        step = start_step
        start_time = time.time()

        model.train()
        while step < total_steps:
            # 11-hour session limit check
            if (time.time() - start_time) >= session_sec:
                if local_rank == 0:
                    print("\n⏰ [Session Limit] 11-Hour limit reached. Saving final checkpoint...")
                    ckpt_manager.save(step, model, streamer_state=streamer.state_dict(), loss=accum_loss)
                break

            grad_accum = current_stage.get("gradient_accumulation_steps", 8)
            accum_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            for _ in range(grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    dataloader = create_curriculum_dataloader(streamer, current_stage.get("batch_size", 2))
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                _, loss, _ = model(input_ids=input_ids, labels=labels)
                scaled_loss = loss / grad_accum
                scaled_loss.backward()
                accum_loss += scaled_loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

            step += 1

            if is_distributed:
                loss_tensor = torch.tensor(accum_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                avg_loss = loss_tensor.item()
            else:
                avg_loss = accum_loss

            if local_rank == 0 and pbar is not None:
                current_lr = optimizer.param_groups[0]["lr"]
                remaining_min = max(0.0, (session_sec - (time.time() - start_time)) / 60.0)
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "seq": current_stage["seq_len"],
                    "rem_time": f"{remaining_min:.1f}m"
                })

                if step % cfg["training"].get("save_every", 500) == 0:
                    ckpt_manager.save(step, model, streamer_state=streamer.state_dict(), loss=avg_loss)

        if local_rank == 0 and pbar is not None:
            pbar.close()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()