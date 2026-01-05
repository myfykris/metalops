# System Architecture

## Overview
`metalcore` provides high-performance linear algebra operations for PyTorch on macOS by bypassing generic MPS fallbacks and executing custom Metal kernels. It consolidates SVD, Eigh, QR, Cholesky, and other LA operations into a single native backend.

## Package Structure
```
packages/metalcore/
├── native/
│   ├── core_kernels.metal   # All Metal kernels (3,200+ lines)
│   └── core_mps.mm          # C++ dispatch + PYBIND11
└── src/metalcore/
    ├── __init__.py          # Public API
    ├── svd.py               # SVD with De Rijk optimization
    ├── eigh.py              # Eigendecomposition
    ├── qr.py                # QR decomposition
    ├── cholesky.py          # Cholesky factorization
    ├── solve.py             # Linear system solve
    ├── pinv.py              # Pseudo-inverse
    └── lstsq.py             # Least squares
```

## Components

### 1. Metal Kernels (`native/core_kernels.metal`)
- **Jacobi SVD**: One-sided Jacobi with SIMD reduction
- **Jacobi Eigh**: Symmetric eigenvalue decomposition
- **Householder QR**: Batched QR with apply kernels
- **MAGMA-style Cholesky**: Shared memory optimized
- **High-impact ops**: LU, SYRK, FrobNorm, Softmax, Trace

### 2. Host Orchestrator (`native/core_mps.mm`)
- **Objective-C++**: Bridges PyTorch (C++) and Metal (Obj-C)
- **PSO Management**: Pipeline state objects for each kernel
- **PYBIND11**: Python bindings for all operations

### 3. Python Wrappers (`src/metalcore/*.py`)
- **Autograd**: Custom backward passes where needed
- **Shape handling**: Wide/tall matrix transpositions
- **Config**: Strategy selection, optimization flags

## Design Decisions
- **External Metal Source**: Loaded at runtime for hot-reload development
- **Unified Package**: All LA ops in one native extension (vs separate packages)
- **Batched First**: Most kernels optimized for batch dimension parallelism
