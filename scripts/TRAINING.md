Run training directly from the repository root. The notebook and its generated `train_kaggle_gpu.py` are no longer needed for these entry points.

For automatic hardware detection, use a single command:

```bash
python scripts/train_accelerator.py --data /kaggle/working/tokens.bin
```

The launcher probes hardware in a disposable process, selects the matching model config, and runs `train_gpu.py` with all visible GPUs through torchrun (or directly for one GPU), or starts `train_tpu.py` for TPU workers. It uses each trainer's device-specific defaults. Detection prefers CUDA when CUDA GPUs are visible; otherwise it checks the TPU runtime. No available accelerator produces an error rather than silently starting CPU training. Run the launcher with Python, not torchrun. `--detect-only` prints hardware JSON, `--dry-run` shows the selected command, and `--accelerator gpu|tpu` overrides selection. Other options are forwarded to the selected trainer. The notebook uses this same launcher with `ACCELERATOR = 'auto'` by default.

Install `numpy pyyaml tqdm transformers datasets` in the Kaggle environment. Keep Kaggle's working accelerator-specific PyTorch installation; for TPU, PyTorch and `torch-xla` must have matching major/minor versions. The TPU launcher checks this. PyTorch 2.4+ APIs are used; validate the installed XLA build with the small smoke run below before a full run.

First prepare reusable packed tokens on CPU:

```bash
python scripts/prepare_tokens.py --dataset roneneldan/TinyStories --output /kaggle/working/tokens.bin --max-tokens 100000000
```

The default tokenizer matches the mini model's 128,256 vocabulary. You can use `--text-file documents.txt` instead of a Hugging Face dataset, or supply `--dataset-config`, `--split`, and `--text-column`. Output is little-endian uint32 with a JSON sidecar. Existing token files are not overwritten. Attach the prepared file as a Kaggle input in later sessions to avoid repeating tokenization. The preparation step preserves source order; shuffle/mix your corpus before packing if needed for your training objective.

For the dual T4 GPUs:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_gpu.py --data /kaggle/working/tokens.bin --seq-len 128 --batch-size 1 --grad-accum 4 --output-dir /kaggle/working/checkpoints-gpu
```

This defaults to FSDP full sharding, FP16 compute, FP32 master parameters, gradient scaling, and Adafactor. FSDP flat parameter shards use unfactored second moments, so the optimizer loses factorization savings but still avoids an Adam first-moment tensor. Each four-layer block is a sharding/checkpointing unit; the tied embedding and output head stay together at the root. The optimizer is created after wrapping. Peak memory and throughput must be measured on the actual devices.

`--strategy ddp` is available when the model and stable precision configuration fit each GPU. DDP uses `no_sync()` for intermediate accumulation microsteps. FSDP intentionally synchronizes each microstep to avoid keeping full unsharded gradients in T4 memory. A single GPU is supported by running with Python, but the full mini model may not fit; use the two-process command for the two T4s.

For the TPU v5e-8:

```bash
PJRT_DEVICE=TPU python scripts/train_tpu.py --data /kaggle/working/tokens.bin --seq-len 128 --batch-size 1 --grad-accum 1 --xla-metrics --output-dir /kaggle/working/checkpoints-tpu --cache-dir /kaggle/working/xla-cache
```

Run this in a fresh process. `torch_xla.launch` starts the workers; no device is initialized in the parent. The trainer checks that eight workers/devices are participating. Use `--expected-devices N` only when deliberately running on a different topology. Each replica has its own disjoint token partition. Each chip holds FP32 parameters and gradients with BF16 matrix compute; batch size is per chip, not global. This is replicated data parallel training, not SPMD or parameter sharding. Start at microbatch 1: v5e has 16 GB HBM per chip and FP32 parameters plus gradients alone need about 12.6 GiB for this model.

The TPU path addresses graph construction and compilation in several ways:

- Focus forward and backward use batched chunk operations, including sequences shorter than the configured chunk; no training path enters the Python per-token reference recurrence.
- Default `--xla-graph-blocks 1` puts an execution boundary after each four-layer block in forward and at the corresponding boundary in backward, without detaching gradients. This bounds graph size; it trades some fusion opportunities for shorter compilation units. Increase to 2 or 3 after measuring, or use 0 to benchmark the full graph.
- XLA's checkpoint utility preserves XLA autocast during recomputation. Autocast caching is disabled to avoid retaining a low-precision copy of every parameter alongside the FP32 master weights and gradients.
- Initialization and RoPE cache construction happen on CPU. Model initialization operations do not become part of the first training graph.
- `--xla-optimizer-params 32` splits the optimizer graph into bounded groups. The optimizer step counter and learning rate are tensors, avoiding a changing Python step/LR constant in each compiled update.
- Each worker has a persistent compilation cache. Reuse the cache directory between runs on the same compatible software/hardware environment. The cache is not a replacement for warmup and does not eliminate all compilation.
- Sequence length, microbatch and loss chunks stay fixed for the run. The existing YAML is read for model dimensions only; its automatic curriculum, large batches, Muon stages, and 65K jump are deliberately not used by these entry points. Changing shape requires a separate run and fresh compilation.
- Data loading uses `MpDeviceLoader`; gradients are averaged once per optimizer update, then globally clipped. Loss values are transferred to the host only at logging intervals.

The first completed update is timed separately. Later output reports seconds per optimizer update and global target tokens/second (accounting for the shifted label). Throughput intervals include any checkpoint overhead between reports. `--xla-metrics` prints compilation/execution counters; additional graphs during startup can be expected, but continuing compilation on every identical step needs investigation. Official background: [XLA execution and caching](https://docs.pytorch.org/xla/master/learn/pytorch-on-xla-devices.html) and [XLA performance diagnostics](https://docs.pytorch.org/xla/master/learn/troubleshoot.html).

Use synthetic input to isolate compute from token-file I/O:

```bash
PJRT_DEVICE=TPU python scripts/train_tpu.py --tiny --synthetic --seq-len 128 --max-steps 5 --save-every 0 --log-every 1 --xla-metrics
torchrun --standalone --nproc_per_node=2 scripts/train_gpu.py --tiny --synthetic --seq-len 128 --max-steps 5 --save-every 0 --log-every 1
```

After the tiny smoke run, remove `--tiny` to benchmark the full 1.693B model, first with sequence 128 and microbatch 1. Random synthetic tokens do not train a useful language model. Sweep microbatch sizes only after checking memory, and increase sequence length to 512/2048 only after establishing a stable baseline. No accelerator compile-time or throughput improvement is claimed until measured on Kaggle. Retention still has quadratic pairwise work, and the current Focus chunk scan uses a dense intra-chunk decay matrix. The changes remove Python token unrolling; they are not a fused custom CUDA/Pallas kernel or a constant-cost long-context implementation.

The new trainer fixes the previous Adafactor implementation, which allocated moments without applying them. Defaults use an explicit learning rate of 3e-4, warmup 200, cosine decay, no first moment and no parameter-relative scaling. `--lr`, `--warmup-steps`, `--weight-decay`, `--max-grad-norm`, and `--max-steps` are configurable. Monitor held-out quality and retune: old settings and loss curves came from a different update rule. The chunked Focus causal direction is also corrected, so old chunked-training checkpoints were trained with different behavior.

Checkpoints are written every 1000 updates and at completion, retaining two complete checkpoints. Set `--save-every 0` for benchmarks. A directory contains per-rank state files and a `complete.json` marker written after every rank has saved. TPU replicas share rank zero's model/optimizer checkpoint; data consumption is tracked per rank by a common count, with the sampler reconstructing each rank's distinct partition. GPU FSDP checkpoints are local shards and require the same world size and strategy to resume. Optimizer state, AMP scaler where applicable, CPU RNG state, and consumed data position are saved. These models use zero dropout and the data sampler is deterministic; stochastic XLA operations added later would additionally require per-rank XLA RNG persistence.

Resume with the same model, sequence, batch, accumulation, seed, schedule, world size, and token file:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_gpu.py --data /kaggle/working/tokens.bin --seq-len 128 --batch-size 1 --grad-accum 4 --resume /kaggle/working/checkpoints-gpu/step-00001000 --output-dir /kaggle/working/checkpoints-gpu
PJRT_DEVICE=TPU python scripts/train_tpu.py --data /kaggle/working/tokens.bin --resume /kaggle/working/checkpoints-tpu/step-00001000 --output-dir /kaggle/working/checkpoints-tpu
```

`--init-weights old_weights.pt` loads a legacy full `model_state` checkpoint before device wrapping with a fresh optimizer and fresh data position; do not combine it with `--resume`. Local FSDP shards are not portable full model weights and cannot be passed to `--init-weights`. Only load trusted checkpoint files.

For local CPU validation:

```bash
python -m pytest tests -q
python scripts/train_gpu.py --cpu --tiny --synthetic --seq-len 8 --chunk-size 4 --max-steps 3 --save-every 0 --log-every 1
```

The regression tests cover independent-reference kernel gradients, causal invariance, partial chunks, latent-projection equivalence, chunked projection/loss gradients, actual Adafactor moment updates, rank partitioning and fixed operation count with increasing tokens within a chunk. GPU/FSDP and TPU runtime behavior still require their respective hardware smoke tests.
