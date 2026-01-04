# metaleig

GPU-Accelerated Eigendecomposition on Apple Metal for PyTorch.

## Installation

```bash
pip install metaleig
```

**Requirements:**
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- PyTorch 2.0+ with MPS backend

## Usage

```python
import torch
import metaleig

# Create a symmetric matrix on MPS
A = torch.randn(1000, 1000, device='mps')
A = A + A.T  # Make symmetric

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = metaleig.eigh(A)

# Or just eigenvalues
eigenvalues = metaleig.eigvalsh(A)
```

## Performance

metaleig uses an optimized One-Sided Jacobi algorithm on Apple Metal:

| Size | Batch | Speedup vs CPU |
|------|-------|----------------|
| 64×64 | 64 | ~3.2x |
| 128×128 | 64 | ~2.4x |

For batched operations, metaleig provides significant speedups over CPU.

## API

### `metaleig.eigh(A)`
Computes eigenvalues and eigenvectors of a symmetric matrix.

**Args:**
- `A`: Symmetric tensor of shape `(..., N, N)`

**Returns:**
- `eigenvalues`: Tensor of shape `(..., N)` in ascending order
- `eigenvectors`: Tensor of shape `(..., N, N)`

### `metaleig.eigvalsh(A)`
Computes only eigenvalues (faster than full eigh).

## License

MIT License
