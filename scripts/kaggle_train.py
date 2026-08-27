"""
scripts/train_accelerator.py
============================
Universal 10,000-Step Pretraining Runner for both TPU (v5e / v3-8) and GPU (Dual T4 DDP, P100, A100).
Checkpoints saved every 500 steps to /kaggle/working/mock_d1_weights.

Usage:
------
# 1. On GPU (Dual T4 DDP):
!torchrun --nproc_per_node=2 scripts/train_accelerator.py --save_every 500

# 2. On TPU (v5e-8 VM) or Single GPU / CPU:
!python scripts/train_accelerator.py --save_every 500
"""

import os
import sys
import glob
import time
import argparse
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# ------------------------------------------------------------------------------
# 1. Hardware Detection & Device Setup
# ------------------------------------------------------------------------------
is_tpu = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    # Register xla device module for autograd / checkpointing
    if hasattr(torch, "_register_device_module"):
        try:
            torch._register_device_module("xla", torch_xla)
        except Exception:
            pass
    torch.xla = torch_xla
    device = torch_xla.device()
    is_tpu = True
except Exception:
    pass

if not is_tpu and torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

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
from engine.optimiser import create_optimizer, get_cosine_schedule_with_warmup


# ------------------------------------------------------------------------------
# 2. Checkpoint Manager
# ------------------------------------------------------------------------------
class UniversalCheckpointManager:
    def __init__(self, output_dir: str = "/kaggle/working/mock_d1_weights", keep_last_n: int = 3, save_half: bool = True):
        self.output_dir = output_dir
        self.keep_last_n = keep_last_n
        self.save_half = save_half
        os.makedirs(output_dir, exist_ok=True)

    def prune_old_checkpoints(self):
        ckpts = glob.glob(os.path.join(self.output_dir, "weights_step_*.pt"))
        if len(ckpts) <= self.keep_last_n:
            return
        sorted_ckpts = sorted(
            ckpts,
            key=lambda x: int(os.path.basename(x).split("_")[-1].replace(".pt", ""))
        )
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
        if self.save_half:
            state_dict = {
                k: (v.to(torch.bfloat16) if v.dtype == torch.bfloat16 else v.half()) if torch.is_floating_point(v) else v
                for k, v in state_dict.items()
            }

        torch.save({
            "step": step,
            "model_state": state_dict,
            "streamer_state": streamer_state,
            "loss": loss
        }, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"💾 [Checkpoint] Saved model weights at step {step} to {path} ({size_mb:.1f} MB)")
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
            return 0, None, None

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
        return step, None, streamer_state


# ------------------------------------------------------------------------------
# 3. Main Universal Training Loop
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Universal TPU/GPU Pretraining Runner (Mock-D1)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML curriculum config")
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/mock_d1_weights")
    parser.add_argument("--session_hours", type=float, default=11.5, help="Safety cutoff hours")
    args = parser.parse_args()

    # Hardware Configuration
    if is_tpu:
        import torch_xla
        import torch_xla.core.xla_model as xm
        dev = torch_xla.device()
        target_dtype = torch.bfloat16
        world_size = 1
        local_rank = 0
        is_distributed = False
        accel_desc = f"Google Cloud TPU ({dev})"
    elif torch.cuda.is_available():
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        is_distributed = world_size > 1
        if is_distributed:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
            dev = torch.device(f"cuda:{local_rank}")
        else:
            dev = torch.device("cuda:0")
        target_dtype = torch.float16
        accel_desc = f"NVIDIA CUDA GPUs: {world_size}"
    else:
        dev = torch.device("cpu")
        target_dtype = torch.float32
        world_size = 1
        local_rank = 0
        is_distributed = False
        accel_desc = "CPU"

    # Select config file
    if args.config:
        config_path = args.config
    elif is_tpu and os.path.exists("configs/curriculum_tpu.yaml"):
        config_path = "configs/curriculum_tpu.yaml"
    elif os.path.exists("configs/curriculum_kaggle.yaml"):
        config_path = "configs/curriculum_kaggle.yaml"
    else:
        config_path = "configs/curriculum_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tokenizer = TokenizerManager(cfg["data"]["tokenizer_name"])
    model_cfg = MockD1Config.from_dict(cfg["model"])

    # Instantiate Model directly in target precision on device
    torch.set_default_dtype(target_dtype)
    with torch.device(dev):
        model = MockD1ForCausalLM(model_cfg)
    torch.set_default_dtype(torch.float32)

    model.gradient_checkpointing_enable()

    if not is_tpu and is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    ckpt_manager = UniversalCheckpointManager(output_dir=args.output_dir, keep_last_n=3, save_half=True)

    streamer = CurriculumStreamer(
        tokenizer=tokenizer,
        datasets_list=cfg["data"].get("datasets", None),
        dataset_name=cfg["data"].get("dataset_name", "roneneldan/TinyStories"),
        initial_seq_len=cfg["curriculum"]["stages"][0]["seq_len"],
        split="train",
        buffer_size=cfg["data"].get("buffer_size", 25000),
        seed=cfg["data"].get("seed", 42) + local_rank
    )

    start_step, _, _ = ckpt_manager.load_latest(model=model, streamer=streamer)

    # 10,000 Total Steps
    stages = cfg["curriculum"]["stages"]
    total_steps = stages[-1]["max_steps"]
    session_sec = float(args.session_hours) * 3600.0

    def get_stage(step):
        for idx, st in enumerate(stages):
            if step < st["max_steps"]:
                return idx, st
        return len(stages) - 1, stages[-1]

    current_stage_idx, current_stage = get_stage(start_step)

    stage_opt_type = current_stage.get("optimizer", "adafactor" if current_stage_idx < 3 else "muon").lower()
    stage_lr = float(current_stage["lr"])
    optimizer = create_optimizer(
        model,
        optimizer_type=stage_opt_type,
        lr=stage_lr,
        muon_lr=stage_lr if stage_opt_type == "muon" else 0.02,
        weight_decay=0.01 if stage_opt_type == "adafactor" else 0.1
    )
    warmup = cfg["training"].get("warmup_steps", 300)
    scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
        warmup_steps=warmup, 
        max_steps=total_steps,
        last_epoch=start_step - 1 if start_step > 0 else -1   # <--- Fixes scheduler resume!
    )

    streamer.set_seq_len(current_stage["seq_len"])
    batch_size = current_stage.get("batch_size", 4)
    grad_accum = current_stage.get("gradient_accumulation_steps", 4)
    dataloader = create_curriculum_dataloader(streamer, batch_size)
    data_iter = iter(dataloader)

    if local_rank == 0:
        eff_tokens = batch_size * world_size * current_stage["seq_len"] * grad_accum
        print("\n" + "=" * 75)
        print(f"🚀 [Full 10k Training Active] {accel_desc} | Precision: {target_dtype}")
        print(f"   • Starting Step      : {start_step:,}")
        print(f"   • Target Total Steps : {total_steps:,}")
        print(f"   • Checkpoint Save    : Every {args.save_every} steps")
        print(f"   • Global Effective   : {eff_tokens:,} tokens / step")
        print("=" * 75 + "\n")

    pbar = tqdm(total=total_steps, initial=start_step, desc="10k Curriculum Training") if local_rank == 0 else None
    step = start_step
    start_time = time.time()
    avg_loss = 0.0

    model.train()
    while step < total_steps:
        if (time.time() - start_time) >= session_sec:
            if local_rank == 0:
                print("\n⏰ [Session Limit] Runtime cutoff approaching. Saving state...")
            break

        active_idx, active_stage = get_stage(step)
        if active_stage["name"] != current_stage["name"]:
            current_stage_idx, current_stage = active_idx, active_stage
            streamer.set_seq_len(current_stage["seq_len"])
            batch_size = current_stage.get("batch_size", 4)
            grad_accum = current_stage.get("gradient_accumulation_steps", 4)
            dataloader = create_curriculum_dataloader(streamer, batch_size)
            data_iter = iter(dataloader)
            
            stage_opt_type = current_stage.get("optimizer", "adafactor" if current_stage_idx < 3 else "muon").lower()
            stage_lr = float(current_stage["lr"])
            optimizer = create_optimizer(
                model,
                optimizer_type=stage_opt_type,
                lr=stage_lr,
                muon_lr=stage_lr if stage_opt_type == "muon" else 0.02,
                weight_decay=0.01 if stage_opt_type == "adafactor" else 0.1
            )
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=warmup, max_steps=total_steps)

            if local_rank == 0:
                print(f"\n🚀 Transitioned to Stage {current_stage_idx + 1}: {current_stage['name']}")

        optimizer.zero_grad(set_to_none=True)
        accum_loss_tensor = None

        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                dataloader = create_curriculum_dataloader(streamer, batch_size)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(dev, non_blocking=True)
            labels = batch["labels"].to(dev, non_blocking=True)

            if not is_tpu and torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, loss, _ = model(input_ids=input_ids, labels=labels)
                    scaled_loss = loss / grad_accum
            else:
                _, loss, _ = model(input_ids=input_ids, labels=labels)
                scaled_loss = loss / grad_accum

            scaled_loss.backward()

            if accum_loss_tensor is None:
                accum_loss_tensor = scaled_loss.detach()
            else:
                accum_loss_tensor = accum_loss_tensor + scaled_loss.detach()

            if is_tpu:
                import torch_xla.core.xla_model as xm
                xm.mark_step()

        if is_tpu:
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer)
            xm.mark_step()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler:
            scheduler.step()

        step += 1
        avg_loss = accum_loss_tensor.item() if accum_loss_tensor is not None else 0.0

        if local_rank == 0 and pbar is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}",
                "seq": current_stage["seq_len"]
            })

            # Checkpoint every 500 steps
            if step % args.save_every == 0:
                ckpt_manager.save(step, model, streamer_state=streamer.state_dict(), loss=avg_loss)

    if local_rank == 0:
        if pbar is not None:
            pbar.close()
        if step % args.save_every != 0 or step == total_steps:
            ckpt_manager.save(step, model, streamer_state=streamer.state_dict(), loss=avg_loss)
        print(f"\n🏁 Finished Training at Step {step}/{total_steps}.")

    if not is_tpu and is_distributed:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()