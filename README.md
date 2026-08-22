# Focus Retention Architecture

This repository is for experimental **focus-retention** architecture. Using this, we will train a 7B parameter model.

## FOCUS ATTENTION

Let `S` be the scaled dot product of `Q` and `K` transpose.
$$S=softmax(\frac{Q \times K^T}{\sqrt{d}})$$
Here `d` is the dimension embeddings. `Q` is the query and `K` is the key matrix.

Now modifying the the process as following:
$$S=softmax(\frac{Q^T \times K}{\sqrt{d}})$$

- This means that attention has fixed dimensions compared to original one.
- No KV-caching, no load on oboard memory and fast inference.

Following is the breakdown:
- `S` has the size of $d_i \times d_i$
- `T` is $C \times d_{head}$
- `A` will be obtained as $A = V \times S$
- This means that:
    - `Q, K, V` will have size of $C \times d_i$
    - $W_{(Q, K, V)} = d_{head} \times d_i$
    - $W_O = d_{head} \times d_{head}$
- Treating $S = softmax(M)$ with $M = \frac{Q^T \times K}{\sqrt{d}}$, we get:
    - Each row is processed by softmax
    - $M_i = M_{i-1} + \frac{Q_i^T \times K_i}{\sqrt{d}}$
    - This forms a recursive relationship with rank-update for each token.
    - This means, only M is cached rather than `K` and `V`. Reducing the need of memory.
- `C` will be concated `A` from each head as $C = concat(A)$. Further: $L = C \times W_O$
- FFN is used to process `L` as $X = f_{FFN}(L)$

This is similar to linear attention, but instead of $QK^TV$, we take $VQ^TK$. If we look closely, this almost looks similar, instead of doing feature extractions, we apply softmax. The content is also in `K` and `Q`, if we have to go by standard definition, this will blur the past, but if we consider this as new mechanism, and ignore the standard vanilla transformer, this attention is doing same thing as linear attention but more on vanilla standard manner.

> Added Decay Factor ($\gamma$) for $M_{i-1}$ so that we do not accumulate all past and corrupt the future predictions. If we let the accumulation for longer context, we meight end up with values too large or too small.

This changes the formula:
$$M_i = \gamma \cdot M_{i-1} + \frac{Q_i^T \times K_i}{\sqrt{d}}$$
$$\gamma \in (0.1, 0.9)$$

---

## RETENTION MECHANISM

The retention mechanism is just reverse engineered attention mechanism, starting from
$$A_i = \alpha_i \times V$$
This $\alpha_i$ is the softmax of $S_i$ which is defined as:
$$\alpha_i = softmax(S_i)$$
$$S_i = S_{i-1} + \phi(Q_i \times K^T)$$

The recursive relation with previous `S` helps in keeping the past history alive in current step and the $\phi(Q_i \times K^T)$ is current steps history. $\phi$ is the feature extraction function, unlike linear attention, this is activates the Q and K. Softmax is applied on `S` to form $\alpha$ and then retention happens with this probability distribution i.e., whichever token has more contribution in selecting new tokens has more probability than others and only those value vectors contribute with it.

To overcome the blurriness problem of focus attention, we take global retention.

For memory footprint reduction, we will use concept similar to Deepseek MHLA.
- Key and Value weights will be product of 2 matrices each with a common KV latent Weight $W_{KV}^L$ and key specific expansion $W_K^L$ and Value specific expansion $W_V^L$.
- $L_{KV}$: Latent KV cache = $X \times W_{KV}^L$
- For Key and values:
    - $K = L_{KV} \times W_K^L$
    - $V = L_{KV} \times W_V^L$
- $S_i = S_{i-1} + \phi(Q \times K^T) = S_{i-1} + \phi(Q \times (X \times W_{KV}^L)^T) = S_{i-1} + \phi((Q \times W_{KV}^{LT}) \times L_{KV}^T)$
- $\alpha = softmax(S)$
- $A = \alpha \times V = (\alpha_i \times L_{KV}) \times W_V^L$

> We only take latent representatin of Keys and Values, not Queries. This latent representation reduces the memory requirement of KV-cache.

## Un-Embedding

De-embedding or un-embedding will be used to produce output of transformer, which will be softmaxed to get probability distribution.

$$D = E + 

---

## Backpropagation

### A. Focus Attention: Closed-Form Softmax VJP & Gradients

Given:
$$M = \frac{Q^T K}{\sqrt{d_h}} \in \mathbb{R}^{d_h \times d_h}, \quad S = \text{softmax}(M) \in \mathbb{R}^{d_h \times d_h}, \quad A = V S \in \mathbb{R}^{C \times d_h}$$

Incoming upstream gradient: $G_A = \nabla_A \mathcal{L} = \frac{\partial \mathcal{L}}{\partial A}$

1. **Gradient w.r.t $V$ and $S$**:
   $$\nabla_V \mathcal{L} = G_A S^T$$
   $$G_S = \nabla_S \mathcal{L} = V^T G_A$$

2. **Softmax VJP (Eliminates building explicit Jacobian matrices)**:
   For row-wise softmax $S_{r, :}$, the exact VJP for row $r$ is:
   $$\nabla_M \mathcal{L} = S \odot \left( G_S - \sum_k (G_S \odot S) \right) = S \odot \Big( G_S - \text{rowsum}(G_S \odot S) \Big)$$

3. **Gradient w.r.t $Q$ and $K$**:
   $$\nabla_Q \mathcal{L} = \frac{1}{\sqrt{d_h}} K (\nabla_M \mathcal{L})^T$$
   $$\nabla_K \mathcal{L} = \frac{1}{\sqrt{d_h}} Q (\nabla_M \mathcal{L})$$

---

### B. Retention: Reverse Cumulative Scan & Latent-KV Backprop

In forward pass:
$$P_i = \phi\left(\frac{Q_i K^T}{\sqrt{D_{KQV}}}\right), \quad S_i = S_{i-1} + P_i = \sum_{t=1}^i P_t, \quad \alpha_i = \text{softmax}(S_i), \quad A_i = \alpha_i V$$

$P_t$ is simply an **alias (shorthand notation)** for the new token contribution at step $t$:

$$P_t = \phi\left( \frac{Q_t \times K^T}{\sqrt{d}} \right)$$

---

### 1. Why is it written as $\sum_{t=1}^i P_t$?

When we unroll your recursive formula step by step:

- **Step 1 ($i=1$):**  
  $$S_1 = P_1$$
- **Step 2 ($i=2$):**  
  $$S_2 = S_1 + P_2 = P_1 + P_2$$
- **Step 3 ($i=3$):**  
  $$S_3 = S_2 + P_3 = P_1 + P_2 + P_3$$
- **Step $i$:**  
  $$S_i = S_{i-1} + P_i = P_1 + P_2 + \dots + P_i = \sum_{t=1}^i P_t$$

Here, $t$ is just the **dummy summation index running from time $t = 1$ to $t = i$**. It represents all past history accumulated up to step $i$.

---

### 2. Why is this unrolling crucial for backpropagation?

This unrolling directly explains the **triangular gradient accumulation**:

1. **In the Forward Pass**: Step $i$'s feature $P_i$ is added to $S_i$, which stays alive in all subsequent states:
   $$S_i, \; S_{i+1}, \; S_{i+2}, \; \dots, \; S_C$$

2. **In the Backward Pass (Chain Rule)**:
   Because $\frac{\partial S_k}{\partial P_i} = 1$ for all $k \ge i$, the total gradient for token $i$'s feature $P_i$ is:
   $$\frac{\partial \mathcal{L}}{\partial P_i} = \sum_{k=i}^{C} \frac{\partial \mathcal{L}}{\partial S_k} \underbrace{\frac{\partial S_k}{\partial P_i}}_{= 1} = \sum_{k=i}^{C} \frac{\partial \mathcal{L}}{\partial S_k}$$

This is why token $i$ receives the sum of gradients from tokens $i$ to $i+j$. Using $P_t$ allows this sum to be computed across the whole sequence simultaneously as a **reverse prefix/suffix sum** (or cumulative scan) without slow loops.

In backward pass, because token $i$ feeds into all downstream states $S_i, S_{i+1}, \dots, S_{i+j}$, the gradient with respect to step $i$'s contribution $P_i$ is the **reverse cumulative sum (suffix sum)** across future tokens:

$$\nabla_{P_i} \mathcal{L} = \sum_{k=i}^{C} \nabla_{S_k} \mathcal{L}$$

1. **Softmax VJP on $\alpha_i$**:
   $$G_{\alpha_i} = (\nabla_{A_i} \mathcal{L}) V^T$$
   $$\nabla_{S_i} \mathcal{L} = \alpha_i \odot \Big( G_{\alpha_i} - \text{sum}(G_{\alpha_i} \odot \alpha_i) \Big)$$

2. **Vectorized Reverse Suffix Scan (No slow for-loops)**:
   $$\nabla_P \mathcal{L} = \text{flip}\Big(\text{cumsum}(\text{flip}(\nabla_S \mathcal{L}, \text{dim}=C), \text{dim}=C)\Big)$$

3. **Backprop through $\phi$ (e.g. SiLU $\phi(x) = x \cdot \sigma(x)$)**:
   $$\phi'(x) = \sigma(x) \cdot \big(1 + x \cdot (1 - \sigma(x))\big)$$
   $$G_{\text{score}} = \nabla_P \mathcal{L} \odot \phi'(\text{score})$$

4. **Gradients w.r.t $Q, K, V$ and Latent Weights**:
   $$\nabla_Q \mathcal{L} = \frac{1}{\sqrt{D_{KQV}}} G_{\text{score}} K, \quad \nabla_K \mathcal{L} = \frac{1}{\sqrt{D_{KQV}}} G_{\text{score}}^T Q, \quad \nabla_V \mathcal{L} = \alpha^T (\nabla_A \mathcal{L})$$
   $$\nabla_{L_{KV}} \mathcal{L} = \nabla_K \mathcal{L} (W_K^L)^T + \nabla_V \mathcal{L} (W_V^L)^T$$
   $$\nabla_{W_{KV}^L} = X^T (\nabla_{L_{KV}} \mathcal{L}), \quad \nabla_{W_K^L} = L_{KV}^T (\nabla_K \mathcal{L}), \quad \nabla_{W_V^L} = L_{KV}^T (\nabla_V \mathcal{L})$$

---

## Parallel Execution

The recursive formulation of `M_i` (Focus) and `S_i` (Retention) looks inherently sequential, but it splits cleanly into a **parallel phase** and a **much shorter sequential phase**. This section describes that split and how it bounds the memory needed to hold state for backpropagation over very long contexts.

### 1. Two-Phase Execution

**Phase A — Per-token contribution (fully parallel).**
For every token `i` in the context (n tokens total), the token's own raw contribution does not depend on any other token, so it can be computed for the whole sequence in one batched op:

- Focus: $P_i = Q_i^{T} \times K_i \in \mathbb{R}^{d_h \times d_h}$
- Retention: $P_i = \phi(Q_i \times K^{T})$

**Phase B — Recursive accumulation (token-wise, decayed).**
Only the *carry* between tokens is sequential:

$$M_i = \gamma \cdot M_{i-1} + Q_i^{T} \times K_i, \qquad M_0 = \mathbf{0}^{d \times d}, \qquad 0.1 < \gamma < 0.9$$

with $\gamma$ a learnable per-head decay applied throughout every layer so that history doesn't accumulate unboundedly and corrupt future predictions. The retention path uses the same two-phase structure, just without the decay term on the pre-softmax score:

$$S_i = S_{i-1} + \phi(Q_i \times K^{T})$$

Because this recurrence is linear in the previous state, it does **not** need to run as a literal Python `for t in range(C)` loop (as `FocusAttentionFunction` and `RetentionFunction` currently do). It should instead run as a **chunked/blockwise scan**:

1. Split the sequence into blocks of size `B`.
2. Compute all `P_i` for every token in every block in parallel (Phase A, batched matmul).
3. Compute the intra-block cumulative sums in parallel.
4. Propagate only the block-boundary carry (`M`/`S` at the end of each block) sequentially across `C / B` blocks, instead of across all `C` tokens.

This turns an `O(C)` sequential dependency into `O(C / B)` sequential steps, with all other work batched — the same trick used by chunked linear-attention/RetNet-style recurrences, and it removes the need to ever materialize the full `[B, H, C, d_h, d_h]` state tensor at once.

### 2. Why the Naive Loop Doesn't Scale

At `max_seq_len = 262,144`, a per-token Python loop over `M` (shape `[B, H, C, d_h, d_h]`) is both slow (no batched parallelism across the sequence axis) and memory-heavy, since the current `forward`/`backward` in `focus.py` explicitly allocate and store `M` for every timestep for use by the backward pass. The chunked scan above avoids storing every intermediate `M_t`/`S_t` by only checkpointing block boundaries and recomputing the intra-block values on demand during backward (the same idea as activation checkpointing, applied at the block level instead of the layer level).


| | Original (per-token loop) | New (chunked parallel scan) |
|---|---|---|
| Sequential steps | `O(C)` | `O(C/B)` |
| Peak state memory | `O(C · d_h²)` — scales with full context | `O(B · d_h²)` — bounded by block size |
| GPU utilization | Poor (tiny per-step ops) | High (large batched matmuls) |
| Correctness | Exact, easy to reason about | Must match reference numerically; more failure modes (chunk-boundary decay, FP8 error) |
| Implementation effort | Trivial (already written) | Nontrivial custom kernel + checkpointing logic |
| Feasible at `C = 262,144`? | **No** | Yes (by design) |
| Best use | Curriculum Stages 1–4, correctness testing, small-context debugging | Curriculum Stage 5, real training/inference at target context |


---

## Mock-D1
This model is based on focus attention and recursion based retention mechanism called **Focus-Retention Architecture**. This model is structured as follows:
- 3:1 ratio for Focus and Retention Layer per block. Total 9 blocks, i.e., 27 Focus and 9 Retention heads.
- Focus Layer will have 32 heads, and Retention layer will always have only `one` head.
- All retention heads use global attention with latent KV-cache.


### Structure

Let `F` denote focus attention layer, `R` denote retention layer. Following is how this model assembles:

```
    ____________________________________________________________________________________
    |                                |                                                 |
X0 ----> F --(+)--> F --(+)--> F ---(+)---> R --(+)--> F --(+)--> F --(+)-->.......---(+)--> R ---> Yn
     |________| |________|  |____________________| |________| 
```

Each `R` get token embedding injection as residual connect. For All `F`, the connections are made between their input and output. For `F-R-F`, the connection between `F-R` and `R-F` is skipped and connection is directly made between input of `F-`, skip `R` and connect ouput of `R` to form input of `-F`.

Token injection maintains the continuity of original information, while residual connects from `F-F` maintain the process.


### General Dimensions

- $D_E$: Embedding dimension = $3072$
- $D_{KQV}$: Key, Query, and Value dimension = $4096$
- $D_{FFN}$: Up-projection dimension in FFN = $12288$
- $N_H$: Number of Heads in Focus Attention = $32$
- $N_L$: Total Number of Layers = $36$
- $N_F$: Focus Attention Layers per block = $3$
- $N_R$: Retention Layers per block = $1$
- $N_B$: Number of F-R blocks = $36 \div (3 + 1) = 9$
- $C_W$: Context Window = $2^{18} = 262,144$
- $VocabSize$: Vocabulary size = $262,144$ (Gemma vocabulary)


#### Focus Attention Weights

- $d_h$: Head dimension = $D_{KQV} \div N_H = 4096 \div 32 = 128$
- $W_{Q,K,V} = D_E \times D_{KQV} = 3072 \times 4096$
- $M_i, S = d_h \times d_h = 128 \times 128$
- $A_i = \mathbb{R}^{C \times d_h}$ per head
- $C = \text{concat}(A_i) \in \mathbb{R}^{C \times D_{KQV}}$
- $W_O = D_{KQV} \times D_E = 4096 \times 3072$


#### Retention Weights

- $W_Q = D_E \times D_{KQV} = 3072 \times 4096$
- $d_L$: Latent KV-cache dimension = $512$
- Latent Projections:
    - $W_{KV}^L = D_E \times d_L = 3072 \times 512$
    - $L_{KV} = C \times d_L$
    - $W_K^L, W_V^L = d_L \times D_{KQV} = 512 \times 4096$
- $W_O = D_{KQV} \times D_E = 4096 \times 3072$

### Embeddings, De-Embedding Head, and FFN

#### Embeddings & LoRA De-Embedding Head
- **Shared Input Embedding**: $W_{emb} = VocabSize \times D_E = 262,144 \times 3,072 = 3 \times 2^{28}$
- **De-Embedding Head**: Uses shared embedding with a LoRA residual matrix:
  $$D_{\text{de-embed}} = W_{emb}^T + (A \times B)$$
  - $A = \mathbb{R}^{D_E \times r} = 3,072 \times 256$
  - $B = \mathbb{R}^{r \times VocabSize} = 262,144 \times 256$
  - LoRA rank $r = 256$

#### FFN (SwiGLU)
- **Layer 1**:
    - $W_1 = D_E \times D_{FFN} = 3,072 \times 12,288$
    - $\text{Gate} = D_E \times D_{FFN} = 3,072 \times 12,288$
- **Layer 2**:
    - $W_2 = D_{FFN} \times D_E = 12,288 \times 3,072$

#### Parameter Count Summary
- **Backbone (36 Layers)**: ~5.714 Billion Parameters
- **Shared Embeddings ($W_{emb}$)**: ~0.805 Billion Parameters
- **LoRA De-Embedding Head ($r=256$)**: ~0.068 Billion Parameters
- **Total Model Parameters**: **~6.588 Billion Parameters** (~6.6B Class / **Mock-D1:7B**)

---

## 25B and 100B

- These will be reasoning and MoE models. Experts are routed gates, not complete projection weights.
- In these variants, the FFN has gates for up-projection along with experts. We will use these gates as experts, not complete FFNs.
- Each expert is fetched via router through output vectors of each layer.
- Number of shared experts are less than experts. Shared expert is pre-fetched by using system prompt from multiple shared experts. This will be refered to as system expert.

---

## For Kaggle Notebook based training

### Architectural & Parameter Summary (~1.69B Class)

| Dimension / Specification | Value | Note |
| :--- | :--- | :--- |
| **Embedding Dim ($D_E$)** | **$1,536$** | Halved from 3072 |
| **KQV Dim ($D_{KQV}$)** | **$2,304$** | $18 \text{ heads} \times 128 \text{ head dim}$ |
| **Focus Heads per Layer ($N_H$)** | **$18$** | Reduced from 36/32 |
| **Head Dimension ($d_h$)** | **$128$** | Standard power-of-2 head dim |
| **SwiGLU Intermediate Dim ($D_{FFN}$)** | **$6,144$** | $4 \times D_E$ |
| **Retention Latent Dim ($d_L$)** | **$256$** | 9x compression over $D_{KQV}$ |
| **Focus Layers ($N_F$)** | **$27$ layers** | 3 per block $\times$ 9 blocks |
| **Retention Layers ($N_R$)** | **$9$ layers** | 1 per block $\times$ 9 blocks |
| **Total Blocks ($N_B$) / Layers ($N_L$)** | **$9$ blocks / $36$ layers** | Exact original 3:1 topology |
| **Context Window ($C_W$)** | **$65,536$ (64K)** | Reduced from 262k |
| **Vocabulary Size ($VocabSize$)** | **$128,256$** | SmolLM3 Tokenizer |
| **LoRA De-Embedding Rank ($r$)** | **$128$** | Halved from 256 |


### Parameter Count Breakdown

1. **Shared Input Embedding ($W_{emb}$)**: $128,256 \times 1,536 \approx \mathbf{0.197\text{ B}}$
2. **27 Focus Layers**:
   - Projections: $4 \times (1536 \times 2304) \approx 14.16\text{M}$
   - SwiGLU MLP: $3 \times (1536 \times 6144) \approx 28.31\text{M}$
   - Per Layer: $\approx 42.47\text{M} \implies 27 \times 42.47\text{M} \approx \mathbf{1.147\text{ B}}$
3. **9 Retention Layers**:
   - Projections ($W_Q, W_{KV}^L, W_K^L, W_V^L, W_O$): $\approx 8.65\text{M}$
   - SwiGLU MLP: $\approx 28.31\text{M}$
   - Per Layer: $\approx 36.96\text{M} \implies 9 \times 36.96\text{M} \approx \mathbf{0.333\text{ B}}$
4. **LoRA De-Embedding Head ($r=128$)**: $(1536 \times 128) + (128 \times 128256) \approx \mathbf{0.017\text{ B}}$
5. **Total Model Parameters**: $\mathbf{\approx 1.694\text{ Billion Parameters}}$ (~1.7B Model)
