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
A = torch.randn(64, 128, 128, device='mps')
A = A + A.transpose(-2, -1)  # Make symmetric

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = metaleig.eigh(A)

# Or just eigenvalues
eigenvalues = metaleig.eigvalsh(A)
```

## Performance

metaleig provides significant speedups for **batched small matrices**:

| Size | Batch | Speedup vs CPU |
|------|-------|----------------|
| 64×64 | 64 | **2.6x** |
| 128×128 | 64 | **2.3x** |

For single matrices and larger sizes, metaleig automatically uses CPU (which is faster due to optimized LAPACK Divide & Conquer).

### Strategy Selection

```python
# Auto-select best strategy (default)
L, V = metaleig.eigh(A, strategy='auto')

# Force specific strategy
L, V = metaleig.eigh(A, strategy='jacobi')   # Metal Jacobi kernels
L, V = metaleig.eigh(A, strategy='cpu')      # CPU fallback
L, V = metaleig.eigh(A, strategy='tridiag')  # Tridiagonalization (experimental)
```

**Auto-selection logic:**
- Batched, N ≤ 128: Uses Jacobi (2-3x speedup)
- Otherwise: Uses CPU (faster for single/large matrices)

## API

### `metaleig.eigh(A, strategy='auto')`
Computes eigenvalues and eigenvectors of a symmetric matrix.

**Args:**
- `A`: Symmetric tensor of shape `(..., N, N)`
- `strategy`: Algorithm selection ('auto', 'jacobi', 'cpu', 'tridiag')

**Returns:**
- `eigenvalues`: Tensor of shape `(..., N)` in ascending order
- `eigenvectors`: Tensor of shape `(..., N, N)`

### `metaleig.eigvalsh(A, strategy='auto')`
Computes only eigenvalues (may be faster for some strategies).

## License

MIT License
