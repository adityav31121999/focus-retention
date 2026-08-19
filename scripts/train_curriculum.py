import argparse
import yaml
import torch
from mock_d1 import MockD1Config, MockD1ForCausalLM
from data.tokeniser import TokenizerManager
from data.curriculum_streamer import CurriculumStreamer
from engine.optimiser import create_optimizer, get_cosine_schedule_with_warmup
from engine.checkpoint import CheckpointManager
from engine.curriculum_trainer import CurriculumTrainer


def main():
    parser = argparse.ArgumentParser(description="Progressive Curriculum Training for Mock-D1")
    parser.add_argument("--config", type=str, default="configs/curriculum_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    tokenizer = TokenizerManager(cfg["data"]["tokenizer_name"])
    model_cfg = MockD1Config(
        vocab_size=cfg["model"]["vocab_size"],
        max_seq_len=cfg["model"]["max_seq_len"],
        hidden_dim=cfg["model"]["hidden_dim"],
        kqv_dim=cfg["model"]["kqv_dim"],
        intermediate_dim=cfg["model"]["intermediate_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_blocks=cfg["model"]["num_blocks"],
        focus_layers_per_block=cfg["model"]["focus_layers_per_block"],
        retention_layers_per_block=cfg["model"]["retention_layers_per_block"],
        focus_heads=cfg["model"]["focus_heads"],
        retention_heads=cfg["model"]["retention_heads"],
        retention_latent_dim=cfg["model"]["retention_latent_dim"],
        phi_act=cfg["model"]["phi_act"],
        tie_word_embeddings=cfg["model"]["tie_word_embeddings"],
        use_lora_deembed=cfg["model"]["use_lora_deembed"],
        lora_deembed_rank=cfg["model"]["lora_deembed_rank"],
    )

    model = MockD1ForCausalLM(model_cfg)
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
        max_grad_norm=cfg["training"]["max_grad_norm"],
        save_every=cfg["training"]["save_every"],
    )

    start_step = ckpt_manager.load_latest(model, optimizer, scheduler)
    trainer.train(start_step=start_step)


if __name__ == "__main__":
    main()