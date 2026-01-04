# MetalOps

**MetalOps** is a collection of high-performance Metal-accelerated operations for PyTorch on macOS (Apple Silicon).

## Packages

| Package | Description | Install |
|---------|-------------|---------|
| **[metalsvd](packages/metalsvd)** | GPU-Accelerated SVD | `pip install metalsvd` |
| **[metaleig](packages/metaleig)** | GPU-Accelerated Eigendecomposition | `pip install metaleig` |

## Quick Start

```python
import torch
import metalsvd
import metaleig

device = torch.device("mps")

# SVD (up to 6x faster for batched, 3.9x for large matrices)
A = torch.randn(64, 128, 128, device=device)
U, S, V = metalsvd.svd(A)

# Eigendecomposition (up to 3x faster for batched)
B = torch.randn(64, 64, 64, device=device)
B = B + B.transpose(-2, -1)  # Make symmetric
eigenvalues, eigenvectors = metaleig.eigh(B)
```

## Requirements

- macOS 12.0+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- PyTorch 2.0+

## Development

```bash
# Install metalsvd in editable mode
cd packages/metalsvd
pip install -e .

# Install metaleig in editable mode
cd packages/metaleig
pip install -e .
```

## License

MIT
