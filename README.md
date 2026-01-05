# MetalOps

**MetalOps** is a collection of high-performance Metal-accelerated linear algebra operations for PyTorch on macOS (Apple Silicon).

All operations are now consolidated into the single **`metalcore`** package.

## Features

| Operation | Description | Performance Highlights |
|-----------|-------------|------------------------|
| **SVD** | Jacobi Singular Value Decomposition | Up to **25x faster** for LLM weight matrices (4096+) |
| **QR** | Householder QR Decomposition | Up to **20x faster** for batched small matrices |
| **Eigh** | Symmetric Eigendecomposition | Up to **3.5x faster** for batched matrices |
| **Cholesky** | Cholesky Decomposition | Optimized for batched operations |
| **Solve** | Linear System Solve (QR-based) | High throughput batched solver |

## Installation

```bash
pip install metalcore
```

## Quick Start

```python
import torch
import metalcore

device = torch.device("mps")

# SVD (Fast for large LLM matrices)
A = torch.randn(4096, 4096, device=device)
U, S, V = metalcore.svd(A)

# Batched QR (Fast for many small matrices)
B = torch.randn(500, 16, 16, device=device)
Q, R = metalcore.qr_batched(B)

# Eigendecomposition
C = torch.randn(64, 64, 64, device=device)
C = C + C.transpose(-2, -1)  # Make symmetric
eigenvalues, eigenvectors = metalcore.eigh(C)
```

## Performance

See **[benchmarks.md](benchmarks.md)** for detailed benchmark results across all algorithms.

## Usage Recommendations

| Operation | When to Use Metal | When to Use CPU |
|-----------|-------------------|-----------------|
| **SVD** | Batched small/medium matrices, Single large matrices | - |
| **QR** | Batched small/medium matrices | Single large matrices (sequential bottleneck) |
| **EIGH** | Batched symmetric matrices | - |
| **Pipeline** | Chained operations (avoid CPU<->GPU transfer) | Single ops on CPU-resident data |

## Requirements

- macOS 12.0+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.9 - 3.14
- PyTorch 2.0+

## Development

```bash
# Install package in editable mode
pip install -e packages/metalcore

# Run benchmarks
python benchmark.py
```

## License

MIT
