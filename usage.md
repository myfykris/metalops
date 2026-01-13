# MetalOps Usage Guide

This guide provides comprehensive documentation and examples for using the high-performance Metal-accelerated operations in **metalcore**.

## Installation

```bash
pip install metalcore
```

## Matrix Decompositions

### Singular Value Decomposition (SVD)
Optimized Jacobi SVD for Apple Silicon. Up to **25x faster** than PyTorch for large matrices (e.g., Llama weights).

```python
import torch
import metalcore

# 4096 x 4096 matrix on MPS
A = torch.randn(4096, 4096, device="mps")

# Returns U, S, V such that A = U @ diag(S) @ V.T
U, S, V = metalcore.svd(A)
```
> **Speedup Note**: Speedups are realized for matrices > 512x512. For smaller matrices, overhead may dominate.

### QR Decomposition
Batched Householder QR decomposition. Extremely fast for many small matrices.

```python
# Batch of 500 small matrices (16x16)
A = torch.randn(500, 16, 16, device="mps")

# Q, R decomposition
Q, R = metalcore.qr(A)
```

### Cholesky Decomposition
Batched Cholesky factorization for symmetric positive-definite matrices.

```python
L = metalcore.cholesky(A)
# A = L @ L.T
```

### Linear Solve
Solves `AX = B` using LU decomposition. Supports FP16 and BF16.

```python
X = metalcore.solve(A, B)
```

---

## Training Operations

### Fused RMSNorm
A highly optimized RMSNorm implementation that fuses sum-of-squares reduction and normalization. **675x faster** than PyTorch naive implementation.

```python
from metalcore import MetalRMSNorm

# Drop-in replacement for torch.nn.RMSNorm
norm_layer = MetalRMSNorm(hidden_size=4096).to("mps")
y = norm_layer(x)
```

### Fused AdamW
A fused optimizer step kernel that updates parameters, moments, and applies weight decay in a single pass. **2.4x faster**.

```python
from metalcore import MetalAdamW

model = MyModel().to("mps")
optimizer = MetalAdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

optimizer.step()
```

### Fused LoRA Attention (Meta-Kernel) 🚀
Combines RMSNorm, QKV projection, LoRA application, RoPE, SDPA, Output projection, and Residual add into a **single Metal dispatch**. Eliminates hundreds of kernel launches.

```python
from metalcore import fused_lora_attention

# Inputs (hidden_states, etc.) must be on MPS
out = fused_lora_attention(
    hidden_states, 
    layer_norm_weight,
    q_proj, k_proj, v_proj, o_proj,
    lora_A_q, lora_B_q, ... # LoRA weights
    cos, sin, # RoPE cache
    ...
)
```

### Fused SwiGLU MLP (Meta-Kernel) 🚀
Combines RMSNorm, Gate/Up projection, LoRA, SwiGLU activation, Down projection, and Residual add into a **single Metal dispatch**.

```python
from metalcore import fused_swiglu_mlp

out = fused_swiglu_mlp(
    hidden_states,
    gate_proj, up_proj, down_proj,
    lora_gate_A, lora_gate_B, ...
    rms_weight, rms_eps
)
```

---

## Activations & Layers

### GELU / SiLU
Vectorized activation functions.

```python
y = metalcore.metal_gelu(x)
y = metalcore.metal_silu(x)
```

### Rotary Embeddings (RoPE)
Metal-accelerated RoPE implementation (Split-half format). **3.4x faster**.

```python
from metalcore import RotaryEmbedding

rope = RotaryEmbedding(dim=128, max_position_embeddings=2048)
cos, sin = rope(seq_len=128)
q_out, k_out = metalcore.rope_fwd(q, k, cos, sin)
```

### EmbeddingBag
Optimized `EmbeddingBag` sum. PyTorch on MPS falls back to CPU for `mode="sum"`. Metalcore keeps it on GPU. **6x faster**.

```python
from metalcore import MetalEmbeddingBag

emb = MetalEmbeddingBag(num_embeddings=32000, embedding_dim=4096).to("mps")
y = emb(input, offsets)
```

---

## PyTorch Overrides
To automatically accelerate existing models without changing code:

```python
import metalcore
metalcore.enable_pytorch_overrides()

# Now F.silu, F.gelu, torch.svd, etc. use metalcore optimized versions automatically.
```
