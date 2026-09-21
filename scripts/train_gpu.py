"""Run with torchrun --standalone --nproc_per_node=2 scripts/train_gpu.py --data tokens.bin."""
import contextlib
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist

from scripts.training_common import (
    arguments, model_config, loader, learning_rate, seed_everything,
    run_signature, restore, save, Reporter,
)
from mock_d1 import MockD1ForCausalLM
from engine.optimiser import Adafactor


def main():
    args = arguments("gpu")
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.cpu:
        if world_size != 1:
            raise ValueError("--cpu supports only one process")
        device, strategy = torch.device("cpu"), "single"
        torch.set_num_threads(1)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --cpu --tiny for a smoke test")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        strategy = args.strategy if world_size > 1 else "single"
        if world_size > 1:
            dist.init_process_group("nccl")
    barrier = dist.barrier if world_size > 1 else lambda: None
    seed_everything(args.seed)
    config = model_config(args)
    # CPU initialization avoids a full unsharded FP32 allocation on a T4.
    model = MockD1ForCausalLM(config)
    count = model.get_num_params()
    if args.init_weights:
        state = torch.load(args.init_weights, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model_state", state.get("model", state)))
    if not args.no_checkpointing:
        model.gradient_checkpointing_enable()
    if strategy == "fsdp":
        from functools import partial
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        from mock_d1.block import MockD1Block
        model = FSDP(model, device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD,
                     auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={MockD1Block}),
                     mixed_precision=MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float32,
                                                    buffer_dtype=torch.float32),
                     limit_all_gathers=True, backward_prefetch=None, use_orig_params=False)
        scaler = ShardedGradScaler()
    else:
        model = model.to(device)
        if strategy == "ddp":
            from torch.nn.parallel import DistributedDataParallel
            model = DistributedDataParallel(model, device_ids=[local_rank], gradient_as_bucket_view=True)
        scaler = torch.amp.GradScaler("cuda", enabled=not args.cpu)
    # Construct after FSDP has replaced/sharded parameters. Flat FSDP shards
    # use the unfactored second moment; there is no full Adam first moment.
    optimizer = Adafactor(model.parameters(), lr=args.lr, scale_parameter=False,
                          relative_step=False, weight_decay=args.weight_decay)
    signature = run_signature(args, config, world_size, strategy)
    step, consumed = restore(args, model, optimizer, scaler, signature, rank, strategy)
    batches = iter(loader(args, config, rank, world_size, consumed))
    if rank == 0:
        print(f"GPU trainer: strategy={strategy} devices={world_size} parameters={count:,} "
              f"sequence={args.seq_len} microbatch={args.batch_size} accumulation={args.grad_accum}", flush=True)
        print(f"PyTorch={torch.__version__}; optimizer=Adafactor; starting update={step}", flush=True)
    model.train()
    reporter = Reporter(args, rank, world_size)
    running_loss = torch.zeros((), device=device)
    last_saved = None
    try:
        while step < args.max_steps:
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(args, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss_sum = torch.zeros((), device=device)
            for microstep in range(args.grad_accum):
                ids = next(batches).to(device, non_blocking=True)
                consumed += args.batch_size
                # FSDP deliberately synchronizes each microstep: no_sync keeps
                # full gradients resident and defeats the T4 memory budget.
                context = model.no_sync() if strategy == "ddp" and microstep + 1 < args.grad_accum else contextlib.nullcontext()
                with context:
                    with torch.autocast("cuda", dtype=torch.float16, enabled=not args.cpu, cache_enabled=False):
                        _, loss, _ = model(input_ids=ids, labels=ids, return_logits=False,
                                            loss_chunk_size=args.loss_chunk_size)
                    scaler.scale(loss / args.grad_accum).backward()
                loss_sum += loss.detach() / args.grad_accum
            scaler.unscale_(optimizer)
            if strategy == "fsdp":
                model.clip_grad_norm_(args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            running_loss = loss_sum
            if step == 1 or step % args.log_every == 0 or step == args.max_steps:
                if world_size > 1:
                    dist.all_reduce(running_loss)
                    running_loss /= world_size
                if not args.cpu:
                    torch.cuda.synchronize()
                value = running_loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError("Non-finite loss; inspect precision/gradients before continuing")
                reporter.report(step, value, lr)
            if args.save_every and step % args.save_every == 0:
                save(args, model, optimizer, scaler, signature, rank, strategy, step, consumed, barrier)
                last_saved = step
        if args.save_every and last_saved != step:
            save(args, model, optimizer, scaler, signature, rank, strategy, step, consumed, barrier)
        if rank == 0 and not args.cpu:
            print(f"Peak allocated on rank 0: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
