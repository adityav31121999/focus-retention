import os
import sys
import argparse
import yaml
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mock_d1 import MockD1Config, MockD1ForCausalLM
from data import TokenizerManager, HuggingFaceStreamer, LocalMemmapDataset, get_hf_dataloader, get_local_dataloader
from engine import Trainer, create_optimizer, get_cosine_schedule_with_warmup, CheckpointManager


def main():
    parser = argparse.ArgumentParser(description="Train Focus-Retention Architecture (Mock-D1)")
    parser.add_argument("--model_size", type=str, default="1.7B", choices=["1.7B", "1.7b", "7B", "7b", "custom"],
                        help="Target model size: '1.7B' (default for Kaggle/Single-GPU) or '7B'")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file. If omitted, picks default for model_size.")
    parser.add_argument("--use_local_data", action="store_true",
                        help="Train on local memmap binary dataset instead of streaming from HuggingFace.")
    parser.add_argument("--data_path", type=str, default="data/train.bin",
                        help="Path to local binary dataset if --use_local_data is set.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Execution device: auto (detects CUDA), cuda, or cpu.")
    parser.add_argument("--mixed_precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "no"],
                        help="Mixed precision mode for GPU acceleration.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable activation checkpointing to save VRAM.")
    args = parser.parse_args()

    # Determine configuration file
    if args.config is None:
        norm_size = args.model_size.upper()
        if "1.7B" in norm_size:
            args.config = "configs/default_config.yaml"
        elif "7B" in norm_size:
            args.config = "configs/7b_config.yaml"
        else:
            args.config = "configs/default_config.yaml"

    print(f"📖 Loading configuration from: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"🖥️ Target Compute Device: {device.upper()}")

    # Initialize model config
    model_cfg = MockD1Config.from_dict(cfg["model"])
    model = MockD1ForCausalLM(model_cfg)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("⚡ Gradient checkpointing: ENABLED")

    # Display Architecture & Parameter Breakdown
    total_params = model.get_num_params()
    print("\n" + "=" * 60)
    print(f"🧠 Model Architecture: {model_cfg.model_name}")
    print(f"   - Layers: {model_cfg.num_layers} (9 Blocks x [3 Focus + 1 Retention])")
    print(f"   - Hidden Dim: {model_cfg.hidden_dim} | KQV Dim: {model_cfg.kqv_dim}")
    print(f"   - Focus Heads: {model_cfg.focus_heads} (Head Dim: {model_cfg.focus_head_dim})")
    print(f"   - Vocab Size: {model_cfg.vocab_size:,} | Max Context: {model_cfg.max_seq_len:,}")
    print(f"   - LoRA De-Embedding Rank: {model_cfg.lora_deembed_rank}")
    print(f"   - Total Parameter Count: {total_params:,} ({total_params / 1e9:.2f}B)")
    print("=" * 60 + "\n")

    tokenizer = TokenizerManager(cfg["data"]["tokenizer_name"])

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
    ckpt_manager = CheckpointManager(cfg["training"]["output_dir"])

    if args.use_local_data:
        dataset = LocalMemmapDataset(args.data_path, seq_len=cfg["data"]["seq_len"])
        dataloader = get_local_dataloader(dataset, batch_size=cfg["training"]["batch_size"])
    else:
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
        mixed_precision=args.mixed_precision,
        grad_accum_steps=cfg["training"]["gradient_accumulation_steps"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        save_every=cfg["training"]["save_every"],
    )

    start_step = ckpt_manager.load_latest(model, optimizer, scheduler)
    trainer.train(max_steps=cfg["training"]["max_steps"], start_step=start_step)


if __name__ == "__main__":
    main()