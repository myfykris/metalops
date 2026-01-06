# metalcore

Foundational Metal Linear Algebra Primitives for PyTorch on Apple Silicon.

## Overview

`metalcore` provides a unified backend for high-performance linear algebra operations on macOS devices, bypassing generic MPS fallbacks to use optimized custom Metal kernels.

## Supported Operations

### 1. Decompositions
- **SVD (`svd`)**: One-sided Jacobi algorithm. Highly optimized for both batched small matrices and large "tall" matrices (e.g., LLM weights).
- **QR (`qr`, `qr_batched`)**: Blocked Householder reflection. Significantly faster for batched operations.
- **Eigh (`eigh`)**: Symmetric eigenvalue decomposition using Jacobi rotations.
- **Cholesky (`cholesky`)**: MAGMA-style shared memory optimization for Positive Definite matrices.

### 2. Solvers
- **Linear Solve (`solve`)**: Batched linear system solver using LU factorization. Supports fp16/bf16 (auto-promoted to fp32 for stability).
- **Triangular Solve (`trsm`)**: Solve $AX=B$ where $A$ is triangular.

### 3. Training Ops ⚡ NEW
- **RMSNorm (`MetalRMSNorm`)**: Fused RMS normalization with 2.5x speedup over PyTorch.
- **AdamW (`MetalAdamW`)**: Fused optimizer step with 2.9x speedup.
- **Activations (`metal_gelu`, `metal_silu`)**: Vectorized float4 GELU/SiLU with fast backward pass.
- **SDPA (`metal_scaled_dot_product_attention`)**: Flash Attention v2 with tiling and causal masking (experimental).

### 4. Primitives
- **Householder Reflections**: Core orthogonalization primitives (`geqr2`, `larft`, `larfb`).

## Installation

```bash
pip install metalcore
```

## Usage

```python
import torch
import metalcore

device = 'mps'

# SVD
A = torch.randn(100, 50, device=device)
U, S, V = metalcore.svd(A)

# Batched QR
B = torch.randn(100, 16, 16, device=device)
Q, R = metalcore.qr(B)

# Cholesky
C = torch.randn(10, 32, 32, device=device)
C = C @ C.mT + 1e-4 * torch.eye(32, device=device)  # Make PD
L = metalcore.cholesky(C)

# Linear Solve (batched, supports fp16/bf16)
A = torch.randn(100, 32, 32, device=device)
b = torch.randn(100, 32, device=device)
x = metalcore.solve(A, b)  # x such that A @ x = b

# Training Ops
from metalcore import MetalRMSNorm, MetalAdamW, metal_gelu

# RMSNorm (2.5x faster)
norm = MetalRMSNorm(512).to(device)
x = torch.randn(32, 128, 512, device=device)
y = norm(x)

# AdamW (2.9x faster)
model = torch.nn.Linear(512, 256).to(device)
optimizer = MetalAdamW(model.parameters(), lr=1e-3)

# GELU activation
y = metal_gelu(x)
```

## Performance Highlights

| Operation | Speedup vs PyTorch/CPU |
|-----------|------------------------|
| Cholesky Batched | **10x faster** |
| Solve Batched | **5-10x faster** |
| QR Batched | **20x faster** |
| RMSNorm | **2.5x faster** |
| AdamW | **2.9x faster** |
| SiLU/GELU | **2-4x faster** |

## Requirements

- macOS 12.0+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- PyTorch 2.0+

## Author

[Kris Bailey](https://github.com/myfykris)

## License

MIT
