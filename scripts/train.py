import argparse
import yaml
import torch
from mock_d1 import MockD1Config, MockD1ForCausalLM
from data import TokenizerManager, HuggingFaceStreamer, LocalMemmapDataset, get_hf_dataloader, get_local_dataloader
from engine import Trainer, create_optimizer, get_cosine_schedule_with_warmup, CheckpointManager

def main():
    parser = argparse.ArgumentParser(description="Train Mock-D1 7B Architecture")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--use_local_data", action="store_true")
    parser.add_argument("--data_path", type=str, default="data/train.bin")
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
        grad_accum_steps=cfg["training"]["gradient_accumulation_steps"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        save_every=cfg["training"]["save_every"],
    )

    start_step = ckpt_manager.load_latest(model, optimizer, scheduler)
    trainer.train(max_steps=cfg["training"]["max_steps"], start_step=start_step)

if __name__ == "__main__":
    main()