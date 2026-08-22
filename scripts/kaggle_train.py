"""
kaggle_train.py
==============
All-in-one runner script tailored for Kaggle GPU Notebooks (T4 x2, P100, A100, L4).
Supports single-command execution for Mock-D1:1.7B mini model training or 7B model experiments.

Usage in Kaggle Notebook Cell:
-----------------------------
!python scripts/kaggle_train.py --model_size 1.7B --mode curriculum
"""

import os
import sys
import argparse
import yaml
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Optimize PyTorch memory allocator for Kaggle GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from mock_d1 import MockD1Config, MockD1ForCausalLM
from data.tokeniser import TokenizerManager
from data.curriculum_streamer import CurriculumStreamer
from data.hf_streamer import HuggingFaceStreamer, get_hf_dataloader
from engine.optimiser import create_optimizer, get_cosine_schedule_with_warmup
from engine.checkpoint import CheckpointManager
from engine.trainer import Trainer
from engine.curriculum_trainer import CurriculumTrainer


def main():
    parser = argparse.ArgumentParser(description="Kaggle GPU Training Runner for Focus-Retention (Mock-D1)")
    parser.add_argument("--model_size", type=str, default="1.7B", choices=["1.7B", "1.7b", "7B", "7b"],
                        help="Model size: '1.7B' (default) or '7B'")
    parser.add_argument("--mode", type=str, default="curriculum", choices=["curriculum", "standard"],
                        help="Training mode: 'curriculum' (progressive context) or 'standard'")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Checkpoint directory (defaults to /kaggle/working/checkpoints/...)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable activation checkpointing (default: True for Kaggle VRAM safety)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Override gradient accumulation steps")
    args = parser.parse_args()

    # Determine default config
    is_1_7b = "1.7B" in args.model_size.upper()
    if args.mode == "curriculum":
        config_path = "configs/curriculum_config.yaml" if is_1_7b else "configs/7b_curriculum.yaml"
    else:
        config_path = "configs/default_config.yaml" if is_1_7b else "configs/7b_config.yaml"

    print(f"📖 Loading Kaggle configuration from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve output directory for Kaggle
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir
    elif os.path.exists("/kaggle/working"):
        cfg["training"]["output_dir"] = f"/kaggle/working/checkpoints/mock_d1_{args.model_size.lower().replace('.', '_')}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Execution Device: {device.upper()}")
    if device == "cuda":
        print(f"   GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Build model
    model_cfg = MockD1Config.from_dict(cfg["model"])
    model = MockD1ForCausalLM(model_cfg)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("⚡ Activation Checkpointing: ENABLED")

    total_params = model.get_num_params()
    print("\n" + "=" * 65)
    print(f"🚀 Mock-D1 Training Setup: {model_cfg.model_name}")
    print(f"   - Layers: {model_cfg.num_layers} (9 Blocks x [3 Focus + 1 Retention])")
    print(f"   - Hidden Dim: {model_cfg.hidden_dim} | Focus Heads: {model_cfg.focus_heads}")
    print(f"   - Parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"   - Checkpoints: {cfg['training']['output_dir']}")
    print("=" * 65 + "\n")

    tokenizer = TokenizerManager(cfg["data"]["tokenizer_name"])
    ckpt_manager = CheckpointManager(cfg["training"]["output_dir"])

    if args.mode == "curriculum":
        optimizer = create_optimizer(
            model,
            lr=float(cfg["curriculum"]["stages"][0]["lr"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
            betas=(cfg["training"]["beta1"], cfg["training"]["beta2"])
        )
        total_max_steps = cfg["curriculum"]["stages"][-1]["max_steps"]
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=cfg["training"]["warmup_steps"],
            max_steps=total_max_steps
        )
        streamer = CurriculumStreamer(
            dataset_name=cfg["data"]["dataset_name"],
            tokenizer=tokenizer,
            initial_seq_len=cfg["curriculum"]["stages"][0]["seq_len"],
            buffer_size=cfg["data"]["buffer_size"]
        )
        trainer = CurriculumTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            streamer=streamer,
            curriculum_stages=cfg["curriculum"]["stages"],
            checkpoint_manager=ckpt_manager,
            device=device,
            mixed_precision="auto",
            max_grad_norm=cfg["training"]["max_grad_norm"],
            save_every=cfg["training"]["save_every"],
        )
    else:
        optimizer = create_optimizer(
            model,
            lr=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
            betas=(cfg["training"]["beta1"], cfg["training"]["beta2"])
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=cfg["training"]["warmup_steps"],
            max_steps=cfg["training"]["max_steps"],
            min_lr_ratio=cfg["training"]["min_lr_ratio"]
        )
        streamer = HuggingFaceStreamer(
            dataset_name=cfg["data"]["dataset_name"],
            tokenizer=tokenizer,
            seq_len=cfg["data"]["seq_len"],
            buffer_size=cfg["data"]["buffer_size"]
        )
        dataloader = get_hf_dataloader(streamer, batch_size=cfg["training"]["batch_size"])
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=dataloader,
            checkpoint_manager=ckpt_manager,
            device=device,
            mixed_precision="auto",
            grad_accum_steps=cfg["training"]["gradient_accumulation_steps"],
            max_grad_norm=cfg["training"]["max_grad_norm"],
            save_every=cfg["training"]["save_every"],
        )

    start_step = ckpt_manager.load_latest(model, optimizer, scheduler)
    print(f"🎬 Starting training at step {start_step}...")
    trainer.train(start_step=start_step)


if __name__ == "__main__":
    main()
