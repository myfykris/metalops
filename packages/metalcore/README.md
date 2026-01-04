# metalcore

Foundational Metal Linear Algebra Primitives for PyTorch on Apple Silicon.

## Overview

`metalcore` provides GPU-accelerated building blocks for linear algebra operations:

- **Triangular Solve (trsm)** - Solve Ax=b where A is triangular
- **Householder Reflections** - Core orthogonalization primitive
- **QR Decomposition** - Blocked Householder algorithm

Other packages (`metalsvd`, `metaleig`, etc.) build on these primitives.

## Installation

```bash
pip install metalcore
```

## Usage

```python
import torch
import metalcore

device = 'mps'

# Triangular solve
L = torch.tril(torch.randn(100, 100, device=device))
b = torch.randn(100, device=device)
x = metalcore.trsm(L, b, lower=True)

# QR decomposition
A = torch.randn(100, 50, device=device)
Q, R = metalcore.qr(A)

# Householder reflection
v, tau = metalcore.householder_vector(A[:, 0])
```

## Requirements

- macOS 12.0+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- PyTorch 2.0+

## License

MIT
