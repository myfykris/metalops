# metalcore

Foundational Metal Linear Algebra Primitives for PyTorch on Apple Silicon.

## Overview

`metalcore` provides a unified backend for high-performance linear algebra operations on macOS devices, bypassing generic MPS fallbacks to use optimized custom Metal kernels.

## Supported Operations

### 1. Decompositions
- **SVD (`svd`)**: One-sided Jacobi algorithm. Highly optimized for both batched small matrices and large "tall" matrices (e.g., LLM weights).
- **QR (`qr`, `qr_batched`)**: Blocked Householder reflection. significantly faster for batched operations.
- **Eigh (`eigh`)**: Symmetric eigenvalue decomposition using Jacobi rotations.
- **Cholesky (`cholesky`)**: MAGMA-style shared memory optimization for Positive Definite matrices.

### 2. Solvers
- **Linear Solve (`solve`)**: Batched linear system solver using QR factorization and triangular solve.
- **Triangular Solve (`trsm`)**: Solve $AX=B$ where $A$ is triangular.

### 3. Primitives
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
Q, R = metalcore.qr_batched(B)

# Cholesky
C = torch.randn(10, 32, 32, device=device)
C = C @ C.mT + 1e-4 * torch.eye(32, device=device) # Make PD
L = metalcore.cholesky(C)
```

## Requirements

- macOS 12.0+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- PyTorch 2.0+

## License

MIT
