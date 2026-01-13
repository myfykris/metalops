# Cholesky Decomposition

## Overview
Compute lower triangular L such that A = L @ L.T for symmetric positive definite matrices.

## Why We Built It
- **Linear solvers**: Cholesky is fastest for SPD systems
- **Batched support**: Process many small matrices in parallel
- **GPU native**: Keep data on GPU for subsequent operations

## Formula
```
A = L @ L.T
```

## Performance

| Config | Metal | CPU | Speedup |
|--------|-------|-----|---------|
| Batch 100×32×32 | ~0.5ms | ~1ms | **2x** ✓ |
| Batch 500×16×16 | ~0.3ms | ~0.5ms | **1.7x** ✓ |

## Usage
```python
import metalcore

L = metalcore.cholesky(A)  # A must be SPD
```

## Notes
- Input must be symmetric positive definite
- Batched version highly efficient for many small matrices
- Used internally by linear solve for SPD systems
