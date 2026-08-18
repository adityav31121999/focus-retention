import argparse
import yaml
import torch
from mock_d1 import MockD1Config, MockD1ForCausalLM
from data import TokenizerManager
from inference import MockD1Generator
from engine import CheckpointManager

def main():
    parser = argparse.ArgumentParser(description="Generate text with Mock-D1 7B")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--prompt", type=str, default="The future of focus attention architectures is")
    parser.add_argument("--max_tokens", type=int, default=100)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = TokenizerManager(cfg["data"]["tokenizer_name"])
    model_cfg = MockD1Config(
        vocab_size=cfg["model"]["vocab_size"],
        hidden_dim=cfg["model"]["hidden_dim"],
        kqv_dim=cfg["model"]["kqv_dim"],
        intermediate_dim=cfg["model"]["intermediate_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_blocks=cfg["model"]["num_blocks"],
    )

    model = MockD1ForCausalLM(model_cfg)
    ckpt_manager = CheckpointManager(cfg["training"]["output_dir"])
    ckpt_manager.load_latest(model)

    generator = MockD1Generator(model, tokenizer, device=device)
    print("\n--- Model Output ---")
    print(generator.generate(args.prompt, max_new_tokens=args.max_tokens))

if __name__ == "__main__":
    main()