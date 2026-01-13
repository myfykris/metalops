# SVD (Singular Value Decomposition)

## Overview
Full and truncated SVD using one-sided Jacobi algorithm on GPU.

## Why We Built It
- **LoRA training**: SVD used for weight initialization and analysis
- **GPU native**: Keep large matrices on GPU for decomposition
- **Batched support**: Process many small matrices in parallel

## Formula
```
A = U @ diag(S) @ V.T
```

## Performance

| Size | Metal | CPU | Status |
|------|-------|-----|--------|
| 512×512 | 15ms | 21ms | **GPU 0.7x** ✓ |
| 1024×1024 | 70ms | 91ms | **GPU 0.76x** ✓ |
| 2048×2048 | 386ms | 571ms | **GPU 0.68x** ✓ |
| 4096×11008 | 6s | 12s | **GPU 2x** ✓ |

## Usage
```python
import metalcore

U, S, V = metalcore.svd(A)
```

## Notes
- GPU wins big on LLM-sized matrices (4096+ dimensions)
- Small matrices (<256) favor CPU due to dispatch overhead
- Uses one-sided Jacobi with ICB for numerical stability
