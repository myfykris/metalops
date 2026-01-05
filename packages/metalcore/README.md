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
- **Linear Solve (`solve`)**: Batched linear system solver using QR factorization and triangular solve.
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

| Operation | Speedup vs PyTorch |
|-----------|-------------------|
| RMSNorm (4096x4096) | **2.5x** |
| AdamW (16M params) | **2.9x** |
| SiLU (256x1024) | **4x** |
| QR Batched (500x16x16) | **20x** |

## Requirements

- macOS 12.0+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- PyTorch 2.0+

## License

MIT
