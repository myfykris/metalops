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
- [x] Solve (LU-based batched, fp16/bf16 supported)

### Training Ops ✅ (v0.1.7)
- [x] RMSNorm (2.5x faster, vectorized SIMD)
- [x] AdamW (2.9x faster, fused single-kernel)
- [x] GELU/SiLU (4x faster, float4 vectorized)
- [x] SDPA (Flash Attention v2, tiled, causal masking)
- [x] SDPA Backward (Metal kernel + autograd)

### Benchmarking ✅
- [x] Unified benchmark.py with all operations
- [x] --activations, --sdpa, --training flags
- [x] Runtime tracking saved to benchmark_history.jsonl

## Current Version
- metalcore v0.1.7
- Tested on Python 3.9, 3.10, 3.11, 3.12, 3.13, 3.14

## Future Work
- [ ] Complex number support
- [ ] Additional BLAS Level 3 operations
- [ ] Optimize SDPA to match PyTorch's native performance
