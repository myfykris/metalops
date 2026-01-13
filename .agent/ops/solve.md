# Linear Solve

## Overview
Solve linear systems Ax = b efficiently on GPU.

## Why We Built It
- **Batched systems**: Solve many small systems in parallel
- **Least squares**: Support for overdetermined systems
- **GPU native**: Avoid CPU round-trip for solver

## Functions
1. `solve(A, b)` - General linear solve
2. `solve_triangular(A, b, upper=False)` - Triangular solve
3. `lstsq(A, b)` - Least squares solution

## Performance

| Config | Metal | CPU | Speedup |
|--------|-------|-----|---------|
| Batch 100×64×64 | ~1ms | ~2ms | **2x** ✓ |

## Usage
```python
import metalcore

x = metalcore.solve(A, b)
```

## Notes
- Uses LU decomposition for general matrices
- Cholesky for symmetric positive definite
- Batched triangular solve highly optimized
