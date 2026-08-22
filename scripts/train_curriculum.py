# scripts/train_curriculum.py
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
from engine.checkpoint import CheckpointManager
from engine.curriculum_trainer import CurriculumTrainer


def main():
    parser = argparse.ArgumentParser(description="Progressive Curriculum Training for Focus-Retention (Mock-D1)")
    parser.add_argument("--model_size", type=str, default="1.7B", choices=["1.7B", "1.7b", "7B", "7b", "custom"],
                        help="Target model size: '1.7B' (default) or '7B'")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file. If omitted, picks curriculum config for model_size.")
    parser.add_argument("--session_minutes", type=float, default=None,
                        help="Wall-clock training session limit in minutes (e.g. 55.0 for 1-hour Colab runs).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override checkpoint output directory.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Execution device: auto, cuda, or cpu.")
    parser.add_argument("--mixed_precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "no"],
                        help="Mixed precision mode.")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable activation checkpointing (default: True).")
    args = parser.parse_args()

    # 1. Resolve Configuration File
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

    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir

    # 2. Resolve Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"🖥️ Target Compute Device: {device.upper()}")
    if device == "cuda":
        print(f"   GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 3. Initialize Tokenizer & Vocab Safety Guard
    tokenizer_name = cfg["data"]["tokenizer_name"]
    print(f"📖 Initializing Tokenizer: {tokenizer_name}")
    tokenizer = TokenizerManager(tokenizer_name)
    
    config_vocab = cfg["model"]["vocab_size"]
    actual_vocab = len(tokenizer.tokenizer)
    print(f"   • Tokenizer Vocab Size (len) : {actual_vocab:,}")
    print(f"   • Configured Embedding Rows : {config_vocab:,}")

    assert actual_vocab <= config_vocab, (
        f"🚨 CRITICAL MISMATCH: Tokenizer vocabulary ({actual_vocab}) exceeds configured model "
        f"vocab_size ({config_vocab}). Fix 'vocab_size' in YAML before proceeding!"
    )

    # 4. Initialize Model & Activation Checkpointing
    model_cfg = MockD1Config.from_dict(cfg["model"])
    model = MockD1ForCausalLM(model_cfg)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("⚡ Gradient Checkpointing: ENABLED")

    total_params = model.get_num_params()
    print("\n" + "=" * 65)
    print(f"🧠 Model Architecture: {model_cfg.model_name}")
    print(f"   - Layers: {model_cfg.num_layers} (9 Blocks x [3 Focus + 1 Retention])")
    print(f"   - Hidden Dim: {model_cfg.hidden_dim} | Focus Heads: {model_cfg.focus_heads}")
    print(f"   - Total Parameter Count: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"   - Checkpoints: {cfg['training']['output_dir']}")
    print(f"   - Curriculum Schedule: {len(cfg['curriculum']['stages'])} Stages")
    for s_idx, st in enumerate(cfg["curriculum"]["stages"]):
        opt_name = st.get("optimizer", "adafactor" if s_idx < 3 else "muon").upper()
        print(f"     * Stage {s_idx+1}: {st['name']} -> SeqLen: {st['seq_len']}, Batch: {st.get('batch_size', 1)}, "
              f"Accum: {st.get('gradient_accumulation_steps', 1)}, Opt: {opt_name}, Max Steps: {st['max_steps']:,}")
    print("=" * 65 + "\n")

    # 5. Initialize Multi-Dataset Sequential Streamer
    datasets_list = cfg["data"].get("datasets", None)
    if datasets_list:
        print(f"📚 Dataset Pipeline: {len(datasets_list)} chained datasets configured.")
        for idx, ds in enumerate(datasets_list):
            print(f"   [{idx + 1}] {ds['name']} (subset: {ds.get('config', 'default')})")
    else:
        datasets_list = [{"name": cfg["data"]["dataset_name"], "config": cfg["data"].get("dataset_config", None)}]

    streamer = CurriculumStreamer(
        tokenizer=tokenizer,
        datasets_list=datasets_list,
        dataset_name=cfg["data"].get("dataset_name", "roneneldan/TinyStories"),
        dataset_config=cfg["data"].get("dataset_config", None),
        initial_seq_len=cfg["curriculum"]["stages"][0]["seq_len"],
        split=cfg["data"].get("split", "train"),
        buffer_size=cfg["data"].get("buffer_size", 10000),
        seed=cfg["data"].get("seed", 42)
    )

    # 6. Initialize Checkpoint Manager with Auto-Pruning
    keep_last_n = cfg["training"].get("keep_last_n_checkpoints", 2)
    ckpt_manager = CheckpointManager(cfg["training"]["output_dir"], keep_last_n=keep_last_n)

    # 7. Restore Model Weights, Optimizer Type, Dataset Offsets, and Epoch
    start_step, opt_type, streamer_state = ckpt_manager.load_latest(model=model, streamer=streamer)

    # 8. Configure Session Duration
    session_minutes = args.session_minutes
    if session_minutes is None:
        session_minutes = cfg["training"].get("session_duration_minutes", 55.0)

    # 9. Initialize Timed Curriculum Trainer
    trainer = CurriculumTrainer(
        model=model,
        streamer=streamer,
        curriculum_stages=cfg["curriculum"]["stages"],
        checkpoint_manager=ckpt_manager,
        session_duration_minutes=session_minutes,
        device=device,
        mixed_precision=args.mixed_precision,
        max_grad_norm=cfg["training"]["max_grad_norm"],
        save_every=cfg["training"]["save_every"],
    )

    # 10. Execute Training Session
    print(f"🎬 Commencing training session at Step {start_step} (Timer: {session_minutes} mins)...")
    trainer.train(start_step=start_step)


if __name__ == "__main__":
    main()