# Eigendecomposition (EIGH)

## Overview
Symmetric eigenvalue decomposition for real symmetric/Hermitian matrices.

## Why We Built It
- **PCA/dimensionality reduction**: Eigendecomposition for feature analysis
- **Batched support**: Many small matrices in parallel
- **GPU native**: Keep spectral analysis on GPU

## Formula
```
A = V @ diag(eigenvalues) @ V.T
```

## Performance

| Config | Metal | CPU | Status |
|--------|-------|-----|--------|
| 256×256 | ~5ms | ~3ms | CPU faster |
| Batch 100×32×32 | ~2ms | ~3ms | **GPU** ✓ |

## Usage
```python
import metalcore

eigenvalues, eigenvectors = metalcore.eigh(A)
```

## Notes
- GPU wins on batched small matrices
- CPU wins on single large matrices (LAPACK is highly optimized)
- Uses Jacobi eigenvalue algorithm
