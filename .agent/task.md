# Tasks

## Completed

### Package Consolidation ✅
- [x] Merged metalsvd Metal kernels into metalcore
- [x] Merged metaleig Metal kernels into metalcore  
- [x] Removed packages/metalsvd and packages/metaleig directories
- [x] Updated PYBIND11 exports

### Core Operations ✅
- [x] SVD (Jacobi, De Rijk, sign canonicalization)
- [x] Eigh (Jacobi eigendecomposition)
- [x] QR (Householder, batched)
- [x] Cholesky (MAGMA-style shared memory)
- [x] TRSM (triangular solve)
- [x] Solve (QR-based batched)

### High-Impact Kernels ✅
- [x] LU decomposition (with pivoting)
- [x] SYRK (A.T @ A)
- [x] Frobenius norm
- [x] Softmax (numerically stable)
- [x] Trace

### Benchmarking ✅
- [x] Unified benchmark.py with all operations
- [x] --compare flag for historical comparison
- [x] Runtime tracking saved to benchmark_history.jsonl

## Future Work
- [ ] Publish metalcore to PyPI
- [ ] Deprecate metalsvd on PyPI (yank 0.0.3)
- [ ] Complex number support
- [ ] Additional BLAS Level 3 operations
