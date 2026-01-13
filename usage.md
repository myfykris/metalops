# MetalOps Usage Guide

This guide provides comprehensive documentation for **metalcore**, a library of high-performance Metal-accelerated linear algebra and neural network primitives for PyTorch on macOS.

## Installation

```bash
pip install metalcore
```

---

## 🚀 Easy Mode: Automatic Acceleration

The easiest way to use metalcore is to enable PyTorch overrides. This automatically routes supported operations to optimized Metal kernels without changing your model code.

```python
import metalcore
# Enable all safe overrides
metalcore.enable_pytorch_overrides()
```

### What gets overridden?
| Operation | Implementation | Condition | Speedup |
|-----------|----------------|-----------|---------|
| `torch.nn.RMSNorm` | `MetalRMSNorm` | Always (Class Replacement) | **~1.5x** |
| `torch.optim.AdamW` | `MetalAdamW` | Always (Class Replacement) | **2.4x** |
| `torch.linalg.svd` | Optimized Jacobi | Matrix dim >= 512 | **Up to 12x** |
| `torch.linalg.qr` | Batched Householder | Batched inputs (3+ dims) | **Up to 20x** |
| `F.silu` | `metal_silu` | All inputs | **1.1x** |
| `F.softmax` | `fused_softmax` | All inputs on MPS | **Parity/Fast** |
| `F.embedding_bag` | Coalesced Kernel | `mode="sum"` | **6x** (avoids CPU fallback) |

**Note**: `enable_pytorch_overrides` now replaces classes like `RMSNorm` and `AdamW` directly. New model instantiations will automatically use the optimized versions. Existing instances created *before* calling this function will need manual patching.

---

## ⚡ Transformer Acceleration

Optimized primitives for LLM training and inference.

### Fused RMSNorm
Fused sum-of-squares reduction and normalization.
- **When to use**: Always. Drop-in replacement for `LlamaRMSNorm`, `Qwen2RMSNorm`, etc.
- **Speedup**: **~1.5x** over PyTorch native.

```python
from metalcore import MetalRMSNorm

# 1. Direct Usage
layer = MetalRMSNorm(4096).to("mps")
y = layer(x)

# 2. Patch Existing Model
from metalcore import patch_transformers_rmsnorm
patch_transformers_rmsnorm(my_huggingface_model)
```

### Fused AdamW
Fused optimizer step (params + moments + decay + unscale).
- **When to use**: Training deep learning models on Mac.
- **Speedup**: **2.4x**.

```python
from metalcore import MetalAdamW
optimizer = MetalAdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
optimizer.step()
```

### Rotary Embeddings (RoPE)
Metal-accelerated RoPE (Split-half format).
- **When to use**: Llama/Mistral/Qwen models.
- **Speedup**: **3.4x**.

```python
from metalcore import patch_transformers_rope
patch_transformers_rope(my_huggingface_model)
```

### Fused SwiGLU MLP
Merges Gate/Up proj, SwiGLU, Down proj, and Residual into a single kernel dispatch.
- **When to use**: Custom model implementations requiring max throughput.
- **Speedup**: Reduces kernel launch overhead significantly.

```python
from metalcore import fused_swiglu_mlp

out = fused_swiglu_mlp(
    hidden_states,
    gate_proj, up_proj, down_proj,
    lora_gate_A, lora_gate_B, ... # Optional LoRA
    rms_weight, rms_eps
)
```

---

## 📐 Matrix Decompositions

Metal-accelerated linear algebra operations.

### Singular Value Decomposition (SVD)
Jacobi-based SVD.
- **When to use**: Large matrices (N >= 512). Small matrices are faster on CPU/Apple AMX.
- **Speedup**: **Up to 12x** on M3 Max for large inputs.

```python
U, S, V = metalcore.svd(A) # A = U @ diag(S) @ V.T
```

### QR Decomposition
Householder QR.
- **When to use**: **Batched** small matrices (e.g., thousands of 16x16 or 64x64 matrices).
- **Speedup**: **Up to 20x** over CPU-bound torch.linalg.qr loops.

```python
# A: [Batch, M, N]
Q, R = metalcore.qr(A)
```

### Eigendecomposition (Eigh)
Symmetric eigendecomposition.
- **When to use**: Symmetric matrices where you need eigenvectors.
- **Speedup**: **Up to 3.5x** for batched inputs.

```python
L, Q = metalcore.eigh(A) # Eigenvalues, Eigenvectors
```

### Cholesky Decomposition
- **When to use**: Batched symmetric positive-definite matrices.
- **Speedup**: **33x** for large batches.

```python
L = metalcore.cholesky(A) # Returns lower triangular L
```

---

## 🔢 Linear Solvers

### Solve (LU-based)
Solves `AX = B`.
- **When to use**: General square linear systems. Supports FP16/BF16 (unlike PyTorch MPS).
- **Speedup**: **10x** for batched systems.

```python
X = metalcore.solve(A, B)
```

### Least Squares (Lstsq)
Solves `min ||AX - B||`.
- **When to use**: Overdetermined or underdetermined systems via SVD.

```python
X, residuals, rank, s = metalcore.lstsq(A, B)
```

### Pseudo-Inverse (Pinv)
Moore-Penrose inverse.
- **When to use**: Ill-conditioned matrices or non-square inversion.

```python
A_inv = metalcore.pinv(A)
```

### Triangular Solve (TRSM)
Solves `AX=B` where A is triangular.
- **When to use**: Back-substitution after QR/Cholesky.

```python
X = metalcore.trsm(A, B, lower=True, transpose=False)
```

---

## 🧩 Activations & Utilities

### Activations
- `metal_silu(x)`: **1.1x faster** than PyTorch.
- `metal_gelu(x)`: Included for completeness, but PyTorch native is currently faster (0.55x speedup only).

### EmbeddingBag
Optimized `mode="sum"`.
- **When to use**: Sparse feature lookups (DLRM-style). PyTorch falls back to CPU for `sum` on MPS, causing massive sync overhead.
- **Speedup**: **6x** (GPU vs CPU fallback).

```python
from metalcore import MetalEmbeddingBag
emb = MetalEmbeddingBag(num, dim).to("mps")
y = emb(input, offsets)
```

### Quantization (INT4/INT8)
Primitives for quantized inference.
- `quantize_int4` / `dequantize_int4`: Group-wise quantization.
- `matmul_int4`: Accelerated mixed-precision matmul (W4A16).

---

## ⚠️ Known Limitations / Anti-Patterns

1.  **Small Matrix SVD/QR**: Do not use `metalcore.svd` for single small matrices (e.g., 64x64). The CPU overhead of dispatching kernels dominates. Use PyTorch native (which uses Apple Accelerate on CPU) for these.
2.  **GELU**: Do not override GELU globally; PyTorch's native MPS implementation is highly optimized.
3.  **SDPA**: `metalcore` includes a Metal SDPA implementation, but it is currently **slower** (~6x) than PyTorch's Flash Attention on MPS. It is provided for research/fallback only.
