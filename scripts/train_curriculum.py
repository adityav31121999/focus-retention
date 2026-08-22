import os
import sys
import argparse
import yaml
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mock_d1 import MockD1Config, MockD1ForCausalLM
from data.tokeniser import TokenizerManager
from data.curriculum_streamer import CurriculumStreamer
from engine.optimiser import create_optimizer, get_cosine_schedule_with_warmup
from engine.checkpoint import CheckpointManager
from engine.curriculum_trainer import CurriculumTrainer


def main():
    parser = argparse.ArgumentParser(description="Progressive Curriculum Training for Focus-Retention (Mock-D1)")
    parser.add_argument("--model_size", type=str, default="1.7B", choices=["1.7B", "1.7b", "7B", "7b", "custom"],
                        help="Target model size: '1.7B' (default for Kaggle/Single-GPU) or '7B'")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file. If omitted, picks curriculum config for model_size.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Execution device: auto, cuda, or cpu.")
    parser.add_argument("--mixed_precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "no"],
                        help="Mixed precision mode.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable activation checkpointing.")
    args = parser.parse_args()

    # Determine configuration file
    if args.config is None:
        norm_size = args.model_size.upper()
        if "1.7B" in norm_size:
            args.config = "configs/curriculum_config.yaml"
        elif "7B" in norm_size:
            args.config = "configs/7b_curriculum.yaml"
        else:
            args.config = "configs/curriculum_config.yaml"

    print(f"📖 Loading curriculum configuration from: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"🖥️ Target Compute Device: {device.upper()}")

    # Initialize model
    model_cfg = MockD1Config.from_dict(cfg["model"])
    model = MockD1ForCausalLM(model_cfg)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("⚡ Gradient checkpointing: ENABLED")

    total_params = model.get_num_params()
    print("\n" + "=" * 60)
    print(f"🧠 Model Architecture: {model_cfg.model_name}")
    print(f"   - Layers: {model_cfg.num_layers} | Hidden Dim: {model_cfg.hidden_dim}")
    print(f"   - Focus Heads: {model_cfg.focus_heads} | LoRA Rank: {model_cfg.lora_deembed_rank}")
    print(f"   - Total Parameter Count: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"   - Curriculum Stages: {len(cfg['curriculum']['stages'])} stages")
    for s_idx, st in enumerate(cfg['curriculum']['stages']):
        print(f"     * Stage {s_idx+1}: {st['name']} -> SeqLen: {st['seq_len']}, Batch: {st['batch_size']}, Steps: {st['max_steps']}")
    print("=" * 60 + "\n")

    tokenizer = TokenizerManager(cfg["data"]["tokenizer_name"])

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
    ckpt_manager = CheckpointManager(cfg["training"]["output_dir"])

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
        mixed_precision=args.mixed_precision,
        max_grad_norm=cfg["training"]["max_grad_norm"],
        save_every=cfg["training"]["save_every"],
    )

    start_step = ckpt_manager.load_latest(model, optimizer, scheduler)
    trainer.train(start_step=start_step)


if __name__ == "__main__":
    main()