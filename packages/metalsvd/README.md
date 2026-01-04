# MetalSVD: GPU-Accelerated SVD for PyTorch MPS (Apple Silicon)

**Problem:** PyTorch does **not** support `torch.linalg.svd` / `torch.svd` on the MPS backend. On Apple Silicon, these operations either error or fall back to the CPU.

**Solution:** **MetalSVD** is a GPU-accelerated Metal implementation of Singular Value Decomposition for PyTorch **MPS** tensors. It provides a fast, drop-in replacement optimized for Apple GPU.

---

## Quick Start

```python
import torch
import metalsvd

metalsvd.patch_torch()  # Overrides torch.linalg.svd for MPS tensors

device = torch.device("mps")
A = torch.randn(64, 128, 128, device=device)

# Now accelerated on Apple GPU
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
```

Or use directly:

```python
U, S, V = metalsvd.svd(A)  # A = U @ diag(S) @ V.T
```

---

## Performance

MetalSVD provides significant speedups over CPU for batched and large matrices:

### Batched Square Matrices (Best Use Case)

| Size | Batch | Speedup vs CPU |
|------|-------|----------------|
| 64×64 | 256 | **3.5x** |
| 128×128 | 256 | **6.0x** |
| 256×256 | 256 | **1.9x** |
| 512×512 | 64 | **1.5x** |

### Large Single Matrices (Gram Strategy)

| Shape | Metal | CPU | Speedup |
|-------|-------|-----|---------|
| 4096×14336 | 3.1s | 11.9s | **3.9x** |
| 14336×4096 | 3.0s | 7.8s | **2.6x** |
| 4096×4096 | 2.9s | 5.0s | **1.7x** |
| 1024×4096 | 0.08s | 0.29s | **3.8x** |

---

## Features

- **Drop-in Replacement**: Works with `torch.linalg.svd` via `metalsvd.patch_torch()`
- **Universal Shape Support**: Tall, wide, and square matrices; batched and unbatched
- **Smart Execution**: Automatically chooses optimal strategy based on matrix size
- **Python 3.9-3.14**: Tested on all recent Python versions

### Execution Strategies

- **Small matrices (N < 1024)**: Uses optimized CPU path to avoid GPU overhead
- **Large matrices (N ≥ 1024)**: Uses Gram strategy (A^T @ A + eigh) for maximum speed
- **Batched operations**: Uses parallel One-Sided Jacobi on Metal GPU

---

## Installation

```bash
pip install metalsvd
```

### Requirements

- macOS 12.0+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- PyTorch 2.0+

### Supported Data Types

- `torch.float32` — Recommended
- `torch.float16` — Supported
- `torch.bfloat16` — macOS 14+ (Metal 3.1+)

---

## Configuration

```python
import metalsvd

# Adjust Gram strategy threshold (default: 512)
metalsvd.config.GRAM_THRESHOLD = 512   # Use Gram for N >= 512
metalsvd.config.GRAM_THRESHOLD = 1024  # More conservative (higher precision)

# Disable CPU fallback for small matrices
metalsvd.config.ENABLE_CPU_FALLBACK = False
```

---

## API

### `metalsvd.svd(A, strategy='auto')`

Computes thin SVD: `A = U @ diag(S) @ V.T`

**Args:**
- `A`: Input tensor `(..., M, N)` on MPS device
- `strategy`: `'auto'`, `'gram'`, or `'standard'`

**Returns:**
- `U`: Left singular vectors `(..., M, K)` where K = min(M, N)
- `S`: Singular values `(..., K)`
- `V`: Right singular vectors `(..., N, K)`

### `metalsvd.patch_torch()`

Patches `torch.linalg.svd` to use MetalSVD for MPS tensors.

---

## How It Works

### One-Sided Jacobi (Batched Small Matrices)

Custom Metal kernels implement parallel column rotations with:
- Specialized fused kernels for N=64, 128, 256, 512, 1024
- Float4 vectorized memory operations
- De Rijk column pre-sorting for faster convergence

### Gram Strategy (Large Matrices)

For large matrices where N ≥ threshold:
1. Form Gram matrix: K = A^T @ A (uses Metal matmul)
2. Eigendecomposition: K = V @ diag(λ) @ V^T (CPU eigh)
3. Recover singular values: S = sqrt(λ)
4. Recover U: U = A @ V @ diag(1/S)

This hybrid approach leverages fast GPU matmul while using efficient CPU eigensolvers.

---

## License

MIT
