"""Shared, notebook-independent training utilities. No device initialization on import."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import yaml


def arguments(accelerator):
    parser = argparse.ArgumentParser(description=f"Focus-Retention fixed-shape {accelerator} training")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", default=str(root / "configs/curriculum_kaggle.yaml"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="Packed uint32 token file produced by prepare_tokens.py")
    source.add_argument("--synthetic", action="store_true", help="Compute benchmark only; random tokens")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1, help="Per device microbatch")
    parser.add_argument("--grad-accum", type=int, default=1 if accelerator == "tpu" else 4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--loss-chunk-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000, help="0 disables checkpoints")
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument("--output-dir", default=f"checkpoints/{accelerator}")
    parser.add_argument("--resume", help="Complete checkpoint directory from this trainer")
    parser.add_argument("--init-weights", help="Legacy weights .pt file (fresh optimizer/data position)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--tiny", action="store_true", help="Small model for launch/correctness smoke tests")
    if accelerator == "gpu":
        parser.add_argument("--strategy", choices=("fsdp", "ddp"), default="fsdp")
        parser.add_argument("--cpu", action="store_true", help="Single-process smoke test only")
    else:
        parser.add_argument("--xla-graph-blocks", type=int, default=1,
                            help="Execute after N blocks in forward/backward; 0 builds one large graph")
        parser.add_argument("--xla-optimizer-params", type=int, default=32)
        parser.add_argument("--cache-dir", default=".xla_cache")
        parser.add_argument("--xla-metrics", action="store_true")
        parser.add_argument("--expected-devices", type=int, default=8)
    args = parser.parse_args()
    for name in ("seq_len", "batch_size", "grad_accum", "chunk_size", "loss_chunk_size", "max_steps", "log_every", "keep_last"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seq_len < 2 or args.workers < 0 or args.save_every < 0 or args.warmup_steps < 0:
        parser.error("seq-len must be >=2; workers, save-every and warmup-steps must be >=0")
    if args.lr <= 0 or args.max_grad_norm <= 0:
        parser.error("lr and max-grad-norm must be positive")
    if args.resume and args.init_weights:
        parser.error("choose resume or init-weights")
    if accelerator == "tpu" and (args.xla_graph_blocks < 0 or args.xla_optimizer_params < 0):
        parser.error("XLA graph intervals must be nonnegative")
    return args


def model_config(args):
    from mock_d1 import MockD1Config
    with open(args.config, encoding="utf-8") as f:
        config = MockD1Config.from_dict(yaml.safe_load(f)["model"])
    if args.tiny:
        config = MockD1Config(vocab_size=256, hidden_dim=32, kqv_dim=32,
                              intermediate_dim=64, num_blocks=1, num_layers=4,
                              focus_heads=2, retention_latent_dim=8, lora_deembed_rank=4)
    config.chunk_size = args.chunk_size
    if args.seq_len > config.max_seq_len:
        raise ValueError("seq-len exceeds model.max_seq_len")
    return config


class PackedTokens(Dataset):
    def __init__(self, path, seq_len, vocab_size, seed=42):
        self.path = str(Path(path).resolve()) if path else None
        self.seq_len, self.vocab_size, self.seed = seq_len, vocab_size, seed
        self.tokens = None
        if self.path:
            size = Path(self.path).stat().st_size
            if size % 4:
                raise ValueError("Token file must contain little-endian uint32 values")
            self.count = size // (4 * seq_len)
            if self.count < 1:
                raise ValueError("Token file is shorter than one sequence")
        else:
            self.count = 1_000_000

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.path:
            if self.tokens is None:
                self.tokens = np.memmap(self.path, dtype="<u4", mode="r")
            start = index * self.seq_len
            ids = torch.from_numpy(self.tokens[start:start + self.seq_len].astype(np.int64))
            if int(ids.max()) >= self.vocab_size:
                raise ValueError("Token ID exceeds model vocabulary; check tokenizer/config")
            return ids
        generator = torch.Generator().manual_seed(self.seed + index)
        return torch.randint(self.vocab_size, (self.seq_len,), generator=generator)


class RankSampler(Sampler):
    """Disjoint cyclic rank partitions; offset tracks consumed, not prefetched data."""
    def __init__(self, size, rank, world_size, batch_size, consumed=0):
        self.usable = size - size % (world_size * batch_size)
        if not self.usable:
            raise ValueError("Need at least batch_size * world_size sequences in the token file")
        self.rank, self.world_size, self.consumed = rank, world_size, consumed

    def __iter__(self):
        index = self.rank + self.consumed * self.world_size
        while True:
            yield index % self.usable
            index += self.world_size


def loader(args, config, rank, world_size, consumed):
    dataset = PackedTokens(args.data, args.seq_len, config.vocab_size, args.seed)
    sampler = RankSampler(len(dataset), rank, world_size, args.batch_size, consumed)
    return DataLoader(dataset, sampler=sampler, batch_size=args.batch_size,
                      num_workers=args.workers, drop_last=True,
                      pin_memory=not hasattr(args, "xla_graph_blocks") and not getattr(args, "cpu", False),
                      persistent_workers=args.workers > 0)


def learning_rate(args, step):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(v) for v in value)
    return value


def run_signature(args, config, world_size, strategy):
    path = Path(args.data) if args.data else None
    return dict(model=config.to_dict(), seq_len=args.seq_len, batch_size=args.batch_size,
                grad_accum=args.grad_accum, world_size=world_size, strategy=strategy,
                data_size=path.stat().st_size if path else None, data_name=path.name if path else None,
                seed=args.seed, lr=args.lr, warmup_steps=args.warmup_steps,
                max_steps=args.max_steps, weight_decay=args.weight_decay)


def fsdp_state_context(model, strategy):
    if strategy != "fsdp":
        return contextlib.nullcontext()
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, LocalStateDictConfig
    return FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT, LocalStateDictConfig(offload_to_cpu=True))


def restore(args, model, optimizer, scaler, signature, rank, strategy):
    if not args.resume:
        return 0, 0
    folder = Path(args.resume)
    metadata = json.loads((folder / "complete.json").read_text())
    if metadata["signature"] != signature:
        raise ValueError("Resume configuration/data/world size differs from checkpoint; use --init-weights for a new run")
    # These are trusted local trainer checkpoints, including optimizer/RNG objects.
    saved_rank = 0 if strategy == "tpu" else rank
    state = torch.load(folder / f"rank-{saved_rank:03d}.pt", map_location="cpu", weights_only=False)
    with fsdp_state_context(model, strategy):
        model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    if strategy != "tpu" and torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state(state["cuda_rng"])
    return state["step"], state["consumed"]


def save(args, model, optimizer, scaler, signature, rank, strategy, step, consumed, barrier):
    root = Path(args.output_dir).resolve()
    folder = root / f"step-{step:08d}"
    folder.mkdir(parents=True, exist_ok=True)
    if strategy == "tpu" and rank != 0:
        # No model collectives are involved for replicated TPU parameters.
        # Wait for rank zero without making seven unnecessary host copies.
        barrier()
        barrier()
        return
    with fsdp_state_context(model, strategy):
        weights = to_cpu(model.state_dict())
    payload = dict(model=weights, optimizer=to_cpu(optimizer.state_dict()),
                   scaler=scaler.state_dict() if scaler is not None else None,
                   step=step, consumed=consumed, torch_rng=torch.get_rng_state(),
                   python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                   cuda_rng=torch.cuda.get_rng_state() if strategy != "tpu" and torch.cuda.is_available() else None)
    # Replicated TPU checkpoints need only one model/optimizer copy. Rank-local
    # consumed positions are identical because batches/steps are synchronized.
    if strategy != "tpu" or rank == 0:
        temporary = folder / f"rank-{rank:03d}.tmp"
        torch.save(payload, temporary)
        temporary.replace(folder / f"rank-{rank:03d}.pt")
    barrier()
    if rank == 0:
        (folder / "complete.json").write_text(json.dumps(dict(signature=signature, step=step)), encoding="utf-8")
        candidates = sorted(p for p in root.glob("step-*")
                            if p.name[5:].isdigit() and (p / "complete.json").is_file())
        for old in candidates[:-args.keep_last]:
            if old.resolve().parent != root or old.is_symlink():
                raise ValueError("Refusing to prune checkpoint outside output directory")
            shutil.rmtree(old)
        print(f"Saved {folder}", flush=True)
    barrier()


class Reporter:
    def __init__(self, args, rank, world_size):
        self.args, self.rank, self.world_size = args, rank, world_size
        self.started = time.perf_counter()
        self.last_step = None

    def report(self, step, loss, lr):
        now = time.perf_counter()
        if self.rank == 0:
            if self.last_step is None:
                print(f"First update completed in {now - self.started:.2f}s (includes startup/compilation); "
                      f"step={step} loss={loss:.4f} lr={lr:.3g}", flush=True)
            else:
                seconds = (now - self.started) / (step - self.last_step)
                tokens = self.args.batch_size * self.world_size * self.args.grad_accum * (self.args.seq_len - 1)
                print(f"step={step} loss={loss:.4f} lr={lr:.3g} seconds/update={seconds:.3f} target_tokens/s={tokens / seconds:.1f}", flush=True)
        self.started, self.last_step = now, step
