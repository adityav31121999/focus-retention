# Focus-Retention 1.7B: training performance analysis and implementation plan

Reviewed 2026-09-20: the current repository and `notebook/mock-d1-mini.ipynb`, including its saved GPU output. This is a source audit, mathematical analysis, and proposed implementation plan. No accelerator benchmark was run locally: this workspace has neither PyTorch installed nor an accessible Kaggle accelerator. The notebook cloned an unpinned repository revision, so its historical run may differ from the current checkout.

The parameter count is 1,693,005,798. The main problems are the implementation of the scans, unnecessary large intermediates, training correctness, and device/data orchestration. Preserve the intended architecture while fixing these before considering a smaller model.

## Establish the actual baseline and execution path.

The saved notebook run uses two T4s, sequence length 128, batch size 4 per GPU, and 4 accumulation microsteps. Its 500 updates take 1:57:16, approximately 14.07 seconds/update and 291 nominal input tokens/second across both GPUs. Loss stays around 12.6 and ends at 12.727; this warrants a learning-correctness check but does not by itself identify a cause. The single-GPU smoke test reports 9.28 GB peak allocated for batch 2, sequence 128; that does not establish the memory requirements of DDP or later stages.

Notebook cell titled “CELL 6” writes `scripts/train_kaggle_gpu.py` and launches it with torchrun. Editing `scripts/kaggle_train.py` alone therefore does not change this GPU run. Replace the embedded trainer with a call to one maintained repository entry point. The notebook launcher is CUDA-only despite the multi-core TPU title; no TPU training timing is saved here.

The standalone Kaggle runner initializes one XLA device, hardcodes world size 1, and implements neither a multi-process launch nor explicit SPMD sharding. Detecting eight available devices does not distribute work over them.

## Fix correctness before interpreting speed or loss improvements.

In `mock_d1/focus.py:56` and `:121`, `arange.unsqueeze(0) - arange.unsqueeze(1)` produces column minus row. Masking this with `>= 0` gives an upper-triangular matrix. Multiplying it by token contributions includes future contributions in earlier states. The intended causal recurrence requires row minus column and a lower-triangular matrix. The custom backward implements the causal recurrence and is inconsistent with the current chunked forward.

A scalar check with gamma=0.5 and contributions [1,2,4] gives current chunk states [3,4,4], whereas the intended recurrence gives [1,2.5,5.25]. This checks the scan algebra without requiring PyTorch. The existing gradchecks cover the reference functions, not these chunked implementations. The notebook smoke test uses default chunk size 512 and sequence 128, while GPU training uses chunk size 128: the smoke test exercises a different Focus path.

Add small, meaningful tests for chunked/reference output and gradient equivalence, multiple chunks, partial final chunks, decay gradients, and causal invariance under changes to future tokens. Compare full-sequence and token-by-token results. Use an ordinary autograd reference independent of the custom backward.

In `engine/optimiser.py:189`, “Adafactor” computes RMS and then applies a plain gradient update. Its allocated factored second moments are never updated or used. Replace it with a validated implementation and retune learning rate; existing learning-rate assumptions do not transfer automatically. Measure optimizer cost and validation loss together.

The trainers initialize trainable weights directly in FP16/BF16. CUDA autocast alone does not supply gradient scaling or FP32 master updates. Establish a stable precision policy: FP16 compute on T4, BF16 compute on TPU, FP32 sensitive reductions and appropriate optimizer/master state. Use compatible loss scaling for FP16. Do not just add standard GradScaler to the existing pure-half parameter setup. Standard AMP keeps trainable parameters in FP32, which changes the memory budget. [PyTorch AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html).

RMSNorm currently squares in the input dtype; compute its variance in FP32. Validate scan accumulations and learned decay numerics separately. Save optimizer, scheduler, scaler where applicable, token position, and RNG state for genuine training resumes. Current weights-only resumes and stage transitions discard optimizer history.

## Replace Focus token loops with correct scans and bounded intermediates.

Focus holds a d_head x d_head matrix per head and token. With 18 heads and head dimension 128, one FP16 state tensor at batch 1, chunk 128 is 72 MiB; batch 4 makes it 288 MiB. Forward and backward have several such tensors. At TPU stage-one batch 32 and sequence 128, one full state is 2.25 GiB; the reference path allocates both M and S, before other intermediates and model memory.

Chunked backward still loops over every token (`focus.py:169`). At sequence 8192 that means 221,184 Python token-loop iterations across 27 Focus layers per microstep, each containing multiple tensor operations. The reference forward also has token loops. On CUDA this produces many small launches; on XLA it builds large unrolled graphs.

Implement the affine scan M_t = gamma*M_(t-1) + P_t and its reverse gradient scan. Compose affine transforms associatively, preserve gradients through chunk-boundary carries, checkpoint only needed boundaries, and reconstruct local states during backward. The current dense chunk-decay multiplication costs O(batch*heads*sequence*chunk*head_dim^2); an efficient scan can avoid the additional dense chunk factor. Avoid inverse powers of gamma, which can be numerically unstable.

For CUDA, first benchmark a vectorized tiled implementation; then consider a fused CUDA kernel targeting T4's SM75 if launch or memory traffic still dominates. Verify compiler/kernel support on T4 before choosing a backend. For TPU, use fixed shapes and a supported compiled scan/control-flow implementation. PyTorch/XLA provides experimental scan facilities, but compatibility and performance must be checked against the installed version. A compiled sequential scan reduces graph size; parallel scan and tiling address execution throughput. [PyTorch/XLA scan documentation](https://docs.pytorch.org/xla/master/features/scan.html).

Use separate Focus and Retention chunk sizes. Sweep Focus chunks 32,64,128 first; benchmark partial chunks too. Increasing a shared chunk size to 512 is not automatically an optimization. Standard scaled-dot-product attention is not mathematically equivalent to either custom operator, so FlashAttention is not a direct substitution.

## Remove Retention's repeated prefix work and exploit latent projections exactly.

In `mock_d1/retention.py:110`, each backward chunk reconstructs a full square score matrix for the preceding prefix. Summing prefix squares costs O(sequence^3/chunk) score work at fixed feature width, instead of the O(sequence^2) pairwise work needed by the current equations. At sequence 8192 and chunk 128, these extra prefix reconstructions contain approximately 41 times as many score entries as one tiled forward score pass. This is an operation-count ratio, not a predicted training speedup.

At sequence 65536 and chunk 512, the largest backward prefix score tensor alone is about 7.88 GiB in BF16. Multiple prefix intermediates coexist. Chunked forward therefore does not make this backward memory-bounded by chunk size.

Save cumulative score boundary vectors during forward, or regenerate them once in a forward sweep before backward. Each reverse chunk then starts from its saved boundary instead of recomputing the entire square prefix. All saved boundaries together cost O(batch*sequence^2/chunk); use sparse boundary checkpointing/recomputation if that is too large. Keep score and reverse-scan arithmetic tiled. The exact Retention equations still require quadratic pairwise work after this fix.

There is also an exact algebraic improvement. Let L be latent KV, and let Wk and Wv denote the PyTorch expansion weights, each shaped [2304,256]:

    K = L @ Wk.T
    V = L @ Wv.T
    Q @ K.T = (Q @ Wk) @ L.T
    alpha @ V = (alpha @ L) @ Wv.T

Compute pairwise scores using Q_lat = Q @ Wk and L, and aggregate latent values before expanding. Keep the original score scale 1/sqrt(2304), cumulative SiLU scores, causal masks, softmax, and parameterization. This reduces the pairwise matmul feature width from 2304 to 256, a factor of 9 for those matmuls, while adding a query transformation. It is not a factor-of-9 whole-model speed claim and remains quadratic in sequence length. Compare forward and all parameter gradients to the original in FP64/FP32, then validate mixed precision.

## Chunk the vocabulary projection and loss together.

`mock_d1/model_mock.py:180` constructs the complete [batch,sequence,128256] logits before chunking cross-entropy. The LoRA output path produces another vocabulary-sized tensor. Chunking only CE does not remove those allocations, and autograd can retain CE intermediates across chunks.

One FP16/BF16 logits tensor at batch 1 is approximately 0.49 GiB for sequence 2048, 1.96 GiB for 8192, and 15.66 GiB for 65536. The final case already exceeds one chip's practical capacity before weights and gradients.

Add a training loss-only path that chunks hidden states before both tied projection and LoRA projection, using checkpointed projection-plus-CE chunks or a validated custom backward that recomputes logits. Merely summing ordinary chunk losses can still retain every chunk's intermediates. Preserve gradients to hidden states, tied embeddings, and both LoRA matrices. Preserve token shifting and normalize by the number of valid labels. Keep full logits optional for inference. Benchmark chunk sizes 64,128,256.

## Repair data loading and multi-device execution.

`data/curriculum_streamer.py` stores seed and buffer_size but does not use them to shuffle, shard, or prefetch. Both GPU ranks iterate the same ordered stream. Different seed arguments do not fix this. With this implementation, nominal global token throughput overstates distinct training examples; identical rank streams give roughly half the unique throughput on two GPUs.

Pre-tokenize into local packed uint32 shards or memory-mapped files. Explicitly partition by global rank and worker, preserve independent positions on resume, and verify batch hashes differ across ranks. Then tune workers/prefetch and pinned memory for CUDA; simply increasing workers now would duplicate iterable data again. Streaming resume through itertools.islice still consumes preceding examples; indexed local shards make resumes cheaper.

For DDP, wrap both forward and backward in no_sync for all but the last accumulation microstep. The current code all-reduces gradients 4 times per update in stage one and 8 times in stage two. This reduces synchronization rounds to one, not whole-step time by the same factor. Evaluate gradient_as_bucket_view to reduce gradient/bucket duplication. Aggregate losses on-device and log every 20-50 updates; the notebook currently calls .item() every microstep. [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide).

For TPU, create devices/models inside a version-supported torch_xla.launch worker, initialize identical replicas, shard inputs, and synchronize gradients once per optimizer update. Alternatively use explicitly configured SPMD/FSDP sharding. Avoid parent-process XLA device initialization before launch. Use MpDeviceLoader with intentional execution boundaries; do not stack redundant mark_step calls on top of loader synchronization. Use fixed per-stage shapes, prebuild RoPE caches, and inspect CompileTime, ExecuteTime, transfers, and fallback counters. Changing optimizer Python scalar values can also affect graph reuse, so check compilation counts rather than assuming shapes alone solve it. [PJRT launch documentation](https://docs.pytorch.org/xla/master/learn/pjrt.html), [XLA troubleshooting](https://docs.pytorch.org/xla/master/learn/troubleshoot.html).

## Fit a numerically sound training setup to each device.

| Decision | Dual T4 | TPU v5e-8 |
|---|---|---|
| Matrix compute | FP16, with compatible loss scaling | BF16 |
| Initial benchmark shape after fixes | Sequence 128/512, batch 1 per GPU | Sequence 128/512, batch 1 per chip |
| Increase batch | Sweep 1,2,4 if memory permits | Sweep 1,2,4,8 if memory permits |
| Distributed mode | DDP if stable precision fits; otherwise FSDP with optimizer chosen for sharded parameters | Eight replicas if each fits; otherwise SPMD/FSDP sharding |
| Initial context ceiling | 2048 until profiling establishes headroom | 2048, then 4096 after profiling |
| Long-context promotion | Require corrected tiled backward and loss path | Same; 65536 requires a separate feasibility study |

These are benchmark candidates, not claims that all settings fit. Two GPU memories do not combine under DDP. Eight TPU chips have 16 GB HBM each, not a single freely usable 128 GB allocation. [Google v5e specifications](https://docs.cloud.google.com/tpu/docs/v5e).

For this parameter count, FP16 weights alone occupy 3.15 GiB; FP32 parameters plus FP32 gradients occupy 12.61 GiB; FP32 Adam parameters, gradients and two moments occupy 25.23 GiB before activations, buckets and temporary allocations. Standard AMP cannot simply replace pure FP16 without memory planning. On T4, prefer sharding with a validated memory-efficient optimizer when stable DDP cannot fit. Ordinary Adam fully sharded over only two GPUs still averages 12.61 GiB of persistent state per GPU before transient gathers and activations. Factorized optimizer behavior must be validated on the chosen sharded parameter layout; flattening can remove its matrix-factorization advantage. Construct the optimizer after wrapping when the wrapper changes parameter objects.

Checkpointing is currently unconditional across four-layer blocks while custom kernels also recompute intermediates. After fixing kernel memory, compare whole-block, smaller-unit, and selective checkpointing to avoid unnecessary duplicate work. Keep the fastest variant that fits and remains correct.

Keep Muon out of the first performance baseline. Its current implementation applies Newton-Schulz to the large embedding table and uses CUDA BF16 capability to choose dtype even on TPU. On T4 that selects FP32 matrix iterations, and on TPU it is the wrong device check. If retained, benchmark it separately, choose dtype by actual device, and restrict orthogonalization to intended hidden matrices with a separate validated optimizer for embeddings and other parameters. Do not assume optimizer switching improves wall-clock convergence.

## Measure improvements at equal work and equal learning quality.

First run small correctness tests and a repeatable short learning experiment. Then benchmark one Focus layer, one Retention layer, the loss head, and the complete update separately. Use synthetic batches already on device to isolate compute; compare with local real data to expose input stalls. Report compilation/startup separately from steady-state timing, and separately account for checkpoints.

After warmup and stable graph compilation, measure at least 50 completed updates. Synchronize the accelerator at timing boundaries. Record global non-padding target tokens/second, unique data processed, update latency, peak memory per device, forward/backward/optimizer/communication time, finite-gradient checks, and held-out loss. Hold the global token batch constant when comparing batch/accumulation choices. Use the lowest accumulation count that provides the desired token batch and fits memory; reducing accumulation alone does less work per update.

The TPU YAML schedules about 1.769 billion nominal tokens across eight chips, not roughly 600 million. The GPU YAML schedules 253.952 million nominal tokens across two GPUs, before correcting duplicate data. For a rough dense-training work proxy, 6*N*tokens divided by advertised peak throughput gives about 3.17 hours for that TPU schedule and 5.51 hours for the GPU schedule. These are idealized compute reference numbers, not rigorous lower bounds or runtime predictions for this custom architecture: embedding use, recomputation, custom operations, communication and actual utilization alter the result. The TPU config's 2.5-3.5 hour claim leaves essentially no allowance for those costs under this proxy. Peak specifications: [v5e](https://docs.cloud.google.com/tpu/docs/v5e), [T4](https://www.nvidia.com/en-us/data-center/tesla-t4/).

Implement in this order: causal/gradient and optimizer correctness; one maintained launcher and correct data partitioning; stable precision/memory policy; Focus scans; Retention prefix and latent algebra; projection-plus-loss chunking; measured device and checkpoint tuning. Add small regression benchmarks after the kernel fixes. Do not promise an aggregate speedup before measuring the corrected run.

Only if exact-kernel optimization remains insufficient should architecture changes be evaluated: smaller Focus head dimension at fixed total projection width, fewer Retention layers, bounded-window Retention, or a smaller vocabulary. These alter capacity, semantics, or checkpoint compatibility and require new quality comparisons. The MLPs alone contain about 1.019 billion parameters, so even fully optimized attention leaves substantial dense compute.
