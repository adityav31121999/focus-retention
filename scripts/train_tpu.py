"""Eight-chip PJRT trainer; launch as a normal Python script, outside a notebook kernel."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("ALLREDUCE_GRADIENTS_BUCKET_SIZE_MB", "128")


def worker(index, args):
    # Do not create any XLA tensor or query devices in the parent process.
    import math
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from torch_xla.distributed.parallel_loader import MpDeviceLoader
    from scripts.training_common import (
        model_config, loader, learning_rate, seed_everything,
        run_signature, restore, save, Reporter,
    )
    from mock_d1 import MockD1ForCausalLM
    from mock_d1.focus import FocusRoPE
    from engine.optimiser import Adafactor

    if torch.__version__.split("+")[0].split(".")[:2] != torch_xla.__version__.split("+")[0].split(".")[:2]:
        raise RuntimeError("Install matching PyTorch and torch-xla major/minor versions")
    xr.initialize_cache(str(Path(args.cache_dir).resolve() / f"worker-{index}"))
    device = torch_xla.device()
    rank, world_size = xr.global_ordinal(), xr.world_size()
    if world_size != args.expected_devices:
        raise RuntimeError(f"Expected {args.expected_devices} devices but launched {world_size}; check TPU runtime")
    seed_everything(args.seed)
    config = model_config(args)
    # Host initialization keeps random initialization out of the first XLA
    # training graph and gives every replica identical FP32 master parameters.
    model = MockD1ForCausalLM(config)
    if args.init_weights:
        state = torch.load(args.init_weights, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model_state", state.get("model", state)))
    for module in model.modules():
        if isinstance(module, FocusRoPE):
            module._build_cache(max(2048, args.seq_len))
    model = model.to(device)
    model.model.xla_graph_blocks = args.xla_graph_blocks
    if not args.no_checkpointing:
        model.gradient_checkpointing_enable()
    optimizer = Adafactor(model.parameters(), lr=args.lr, scale_parameter=False,
                          relative_step=False, weight_decay=args.weight_decay)
    optimizer.xla_parameters_per_graph = args.xla_optimizer_params
    signature = run_signature(args, config, world_size, "tpu")
    step, consumed = restore(args, model, optimizer, None, signature, rank, "tpu")
    xm.mark_step()
    xm.wait_device_ops()
    data = loader(args, config, rank, world_size, consumed)
    device_loader = MpDeviceLoader(data, device, batches_per_execution=args.grad_accum)
    batches = iter(device_loader)
    if rank == 0:
        print(f"TPU trainer: devices={world_size} parameters={model.get_num_params():,} "
              f"sequence={args.seq_len} microbatch/chip={args.batch_size} accumulation={args.grad_accum}", flush=True)
        print(f"PyTorch={torch.__version__} XLA={torch_xla.__version__}; BF16 compute/FP32 parameters; "
              f"graph boundary every {args.xla_graph_blocks} block(s); starting update={step}", flush=True)
        print("Starting first forward/backward. First update includes compilation; later timings are reported separately.", flush=True)
    reporter = Reporter(args, rank, world_size)
    barrier = lambda: xm.rendezvous("training-checkpoint")
    model.train()
    last_saved = None
    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(args, step)
        # Transfer as data, instead of embedding a changing Python LR literal
        # in every optimizer graph and recompiling on every update.
        lr_tensor = torch.tensor(lr, dtype=torch.float32).to(device)
        for group in optimizer.param_groups:
            group["lr"] = lr_tensor
        loss_sum = torch.zeros((), device=device)
        for _ in range(args.grad_accum):
            ids = next(batches)
            consumed += args.batch_size
            with torch.autocast("xla", dtype=torch.bfloat16, cache_enabled=False):
                _, loss, _ = model(input_ids=ids, labels=ids, return_logits=False,
                                    loss_chunk_size=args.loss_chunk_size)
            (loss / args.grad_accum).backward()
            loss_sum += loss.detach() / args.grad_accum
            # Finish the backward fragment; never accumulate multiple full
            # microbatch graphs before compiling/executing them.
            xm.mark_step()
        # Clip the averaged global gradient, not different local gradients.
        xm.reduce_gradients(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, foreach=False)
        xm.mark_step()
        optimizer.step()
        xm.mark_step()
        step += 1
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            mean_loss = xm.all_reduce(xm.REDUCE_SUM, loss_sum, scale=1.0 / world_size)
            xm.mark_step()
            xm.wait_device_ops()
            value = mean_loss.item()
            if not math.isfinite(value):
                raise FloatingPointError("Non-finite loss; inspect precision/gradients before continuing")
            reporter.report(step, value, lr)
            if args.xla_metrics and rank == 0:
                import torch_xla.debug.metrics as metrics
                print(metrics.short_metrics_report(), flush=True)
        if args.save_every and step % args.save_every == 0:
            save(args, model, optimizer, None, signature, rank, "tpu", step, consumed, barrier)
            last_saved = step
    if args.save_every and last_saved != step:
        save(args, model, optimizer, None, signature, rank, "tpu", step, consumed, barrier)


def main():
    from scripts.training_common import arguments
    args = arguments("tpu")
    import torch_xla
    torch_xla.launch(worker, args=(args,))


if __name__ == "__main__":
    main()
