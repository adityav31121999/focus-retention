# Focus Retention Architecture

This repository is for experimental **focus-retention** architecture. Using this, we will train and 7B parameter model.

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

---

### Parameter Count Summary
- **Backbone (36 Layers)**: ~5.714 Billion Parameters
- **Shared Embeddings ($W_{emb}$)**: ~0.805 Billion Parameters
- **LoRA De-Embedding Head ($r=256$)**: ~0.068 Billion Parameters
- **Total Model Parameters**: **~6.588 Billion Parameters** (~6.6B Class / **Mock-D1:7B**)
