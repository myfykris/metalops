# Metal SVD: Accelerated SVD for PyTorch on Apple Silicon

A high-performance implementation of Singular Value Decomposition (SVD) for PyTorch on macOS Metal (Apple Silicon M1/M2/M3). Supports **any matrix shape** (Wide, Tall, Square) and **any size** (from small batched tensors to large-scale dense matrices).

## Features

- **Universal Shape Support**: Handles Tall ($M \ge N$), Wide ($M < N$), and Square matrices automatically.
- **Batched One-Sided Jacobi SVD**: Optimized for batches of small matrices (e.g., $64 \times 128$).
- **Randomized SVD (rSVD)**: Efficiently handles large matrices (e.g., $10,000 \times 10,000$) by running entirely on the GPU.
- **Pure Metal Backend**: No CPU fallbacks for orthogonalization loop, unlike standard PyTorch MPS which falls back to CPU for `linalg.svd` and `linalg.qr`.
- **Performance**:
  - **Fused Block-Jacobi Kernel**: **3.5x - 40x speedup** compared to CPU fallback.
  - **Threadgroup Reduction**: ~20x speedup on tall matrices.
  
  | Scenario | Matrix Size | Speedup | Notes |
  | :--- | :--- | :--- | :--- |
  | **Batched SVD** | 64 x 128 x 128 | **4.23x** | Fused Kernel + Threadgroup reduction |
  | **Square SVD** | 1024 x 1024 | **3.53x** | Fused Block-Jacobi Kernel |
  | **Square SVD** | 2048 x 2048 | **16.28x** | Massive parallelism win |
  | **Tall SVD** | 4096 x 2048 | **40.52x** | GPU compute density dominates |
  | **Large rSVD** | 4096 x 4096 | **8.60x** | Randomized alg + Metal kernels |
  | **Huge rSVD** | 8192 x 8192 | **>10x** | FP16 compute limits |

## Installation

### Requirements
- macOS 12.0+ (Apple Silicon)
- PyTorch 2.0+

### Supported Data Types
- `torch.float32`: Full precision (Recommended).
- `torch.float16`: Half precision (Supported on all Apple Silicon M1/M2/M3).
- `torch.bfloat16`: Supported on Metal 3.1+ (macOS 14+). Falls back gracefully on older versions.


### Build from Source
```bash
git clone https://github.com/myfykris/metalsvd.git
cd metalsvd
pip install .
```

## Usage

### 1. Standard Batched SVD
Best for batches of small matrices (e.g., Transformers heads, LoRA adapters).

```python
import torch
import metalsvd

device = torch.device('mps')
# Shape: (Batch, Rows, Cols)
A = torch.randn(64, 128, 128, device=device)

# U, S, V such that A = U @ diag(S) @ V.T
U, S, V = metalsvd.svd(A)

print(U.shape) # (64, 128, 128)
```

### 2. Seamless Integration (Monkeypatching)
Drop-in replacement for existing codebases. Automatically routes MPS tensors to `metalsvd` while keeping CPU/CUDA behavior unchanged.

```python
import metalsvd
metalsvd.patch_torch()

# Now torch.linalg.svd uses our Metal backend for MPS tensors!
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
```

### 3. Randomized SVD (Large Matrices)
Best for large matrices where you only need the top-$k$ singular values/vectors.

```python
import torch
import metalsvd

M, N = 10000, 10000
K = 100
A = torch.randn(M, N, device=device)

# Decompose
U, S, V = metalsvd.randomized_svd(A, k=K, n_iter=2)

# Verify
A_approx = U @ torch.diag(S) @ V.T
print(f"Approximation Error: {(A - A_approx).norm() / A.norm():.2e}")
```

### 4. Lanczos SVD ("Harder but Better")
Best for precision on large matrices with slowly decaying spectra where Randomized SVD might struggle.

```python
import metalsvd

# Finds top-k singular values using Golub-Kahan Bidiagonalization with Full Re-orthogonalization
# Slower than rSVD but deterministic and robust.
U, S, V = metalsvd.lanczos_svd(A, k=10, n_iter=30)
```

## How It Works

### The One-Sided Jacobi Algorithm
We implement the "One-Sided Jacobi" algorithm, which is inherently parallelizable. It iteratively rotates pairs of columns to orthogonalize them.
- **Metal Kernel**: Our custom kernel (`svd.metal`) uses Threadgroup Reductions to parallelize the dot products and column updates, saturating the GPU bandwidth.

### Randomized SVD Pipeline
For large matrices, we use **Randomized SVD** (Probabilistic):
1. Sketch inputs ($Y = A \Omega$).
2. Orthogonalize sketch using our custom fast SVD (acting as a QR substitute).
3. Project and solve small SVD.

### Lanczos Bidiagonalization
For high-precision needs, we provide **Golub-Kahan-Lanczos**:
1. Iteratively builds a Krylov subspace.
2. Uses **Full Re-orthogonalization** (Gram-Schmidt against all previous vectors) to maintain stability in `float32`.
3. Solves the SVD of the resulting bidiagonal matrix.


## Roadmap
- [ ] **Complex Number Support**: Implementation of $2 \times 2$ unitary rotations for complex SVD.
- [ ] **Binary Wheels**: CI/CD pipeline for pre-built PyPI packages.

## Development & AI Context
This repository includes a `.agent/` directory containing detailed context for AI agents and developers:
- **[Architecture](.agent/architecture.md)**: System design explanation (C++/Metal/Python bridge).
- **[Benchmarks](.agent/benchmarks.md)**: Detailed verification and performance reports.
- **[Roadmap](.agent/release_roadmap.md)**: Future goals.
- **[Task History](.agent/task.md)**: Checklist of completed work.

Agents starting work on this repo should read `.agent/architecture.md` first.

## License
MIT
