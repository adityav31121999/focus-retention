import os
import sys
import argparse
import yaml
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mock_d1 import MockD1Config, MockD1ForCausalLM
from data import TokenizerManager
from inference import MockD1Generator
from engine import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description="Generate text with Focus-Retention (Mock-D1)")
    parser.add_argument("--model_size", type=str, default="1.7B", choices=["1.7B", "1.7b", "7B", "7b", "custom"],
                        help="Target model size: '1.7B' (default) or '7B'")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file. If omitted, picks default for model_size.")
    parser.add_argument("--prompt", type=str, default="The future of focus attention architectures is",
                        help="Prompt text for generation.")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0.0 for greedy).")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k filtering parameter.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Execution device.")
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

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Loading tokenizer: {cfg['data']['tokenizer_name']}")
    tokenizer = TokenizerManager(cfg["data"]["tokenizer_name"])
    
    model_cfg = MockD1Config.from_dict(cfg["model"])
    model = MockD1ForCausalLM(model_cfg)

    ckpt_manager = CheckpointManager(cfg["training"]["output_dir"])
    ckpt_manager.load_latest(model)

    generator = MockD1Generator(model, tokenizer, device=device)
    print("\n" + "=" * 50)
    print(f"📝 Prompt: {args.prompt}")
    print("=" * 50)
    output = generator.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )
    print("\n--- Generated Output ---")
    print(output)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()