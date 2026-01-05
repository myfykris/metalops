# metalcore

Foundational Metal Linear Algebra Primitives for PyTorch on Apple Silicon.

## Overview

`metalcore` provides building blocks for linear algebra operations:

- **Triangular Solve (trsm)** - Solve Ax=b where A is triangular
- **Householder Reflections** - Core orthogonalization primitive  
- **QR Decomposition** - Blocked Householder algorithm (WY representation)

## Current Status

✅ **Correctness**: All algorithms numerically correct
- QR decomposition: ~2e-6 reconstruction error (matches PyTorch reference)
- Orthogonality: ~4e-7 (near machine epsilon)
- Triangular solve: ~2e-7 residual

### Performance

See **[benchmarks.md](../../benchmarks.md)** for full results.

| Use Case | Best Speedup | Example |
|----------|--------------|---------|
| **QR Batched (small)** | up to 20× | 500×8×8 |
| **QR Batched (medium)** | up to 14× | 1000×16×16 |
| **QR Single** | CPU faster | Sequential dependencies |

**Why single QR is CPU-bound**: Unlike SVD (where Jacobi rotations are embarrassingly parallel), QR has inherent sequential dependencies within each panel. CPU LAPACK is extremely optimized with hand-tuned BLAS3 kernels.

For **batched operations**, `qr_batched()` provides significant speedups by processing many small matrices in parallel.

## Path to GPU Speedup (Single Matrix)

To achieve GPU speedups for QR/eigensolvers on large matrices, the approach would be:

1. **Fused panel factorization kernel** - Process entire panel in one kernel launch
2. **Use MPS GEMM for trailing update** - This is already done
3. **Reduce synchronization** - Use indirect command buffers
4. **Consider different algorithms** - e.g., Communication-avoiding QR (CAQR)

## Installation

```bash
pip install metalcore
```

## Usage

```python
import torch
import metalcore_backend as mc

device = 'mps'

# QR decomposition (correct but not GPU-optimized)
A = torch.randn(100, 50, device=device)
Q, R = mc.qr(A)  # Q orthogonal, R upper triangular
assert torch.allclose(Q @ R, A, atol=1e-5)

# Triangular solve  
L = torch.tril(torch.randn(50, 50, device=device) + 10*torch.eye(50, device=device))
b = torch.randn(50, device=device)
x = mc.trsm(L, b, lower=True)
assert torch.allclose(L @ x, b, atol=1e-5)

# Panel QR (returns V stored below diagonal + tau)
R_V, tau = mc.geqr2(A)  

# Build T for WY representation
T = mc.larft(R_V, tau, 0)

# Apply block Householder reflector
C = torch.randn(100, 30, device=device)
C_updated = mc.larfb(C, R_V, T, trans=True, panel_start=0)
```

## Algorithm Details

### Blocked QR (based on rocSOLVER pattern)

```
for each block of columns:
    1. Factor panel with geqr2 (unblocked Householder)
    2. Build T matrix with larft (WY representation)  
    3. Apply block reflector with larfb (trailing update - GEMM)
```

The WY representation allows converting nb Householder updates into matrix multiplications:
- H = I - V @ T @ V^T
- H @ C = C - V @ (T @ (V^T @ C))

## Requirements

- macOS 12.0+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- PyTorch 2.0+

## License

MIT
