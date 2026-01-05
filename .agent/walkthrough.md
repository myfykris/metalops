# Metalcore - Consolidated LA for Metal

## Current Status
metalcore consolidates all Metal-accelerated linear algebra operations:
- SVD, Eigh, QR, Cholesky, Solve, TRSM
- Plus 6 high-impact ops: LU, SYRK, FrobNorm, Softmax, Trace

## Key Files
- `packages/metalcore/native/core_kernels.metal` - 3,200+ lines of Metal kernels
- `packages/metalcore/native/core_mps.mm` - C++ dispatch + PYBIND11
- `packages/metalcore/src/metalcore/` - Python wrappers

## Usage
```python
import metalcore as mc

# SVD
U, S, V = mc.svd(A)

# QR
Q, R = mc.qr(A)

# Cholesky
L = mc.cholesky(A)

# High-impact ops
C = mc.syrk_batched(A)  # A.T @ A
norms = mc.frobenius_norm_batched(A)
y = mc.softmax_batched(x)
```

## Benchmark
```bash
python benchmark.py --quick --compare
```
