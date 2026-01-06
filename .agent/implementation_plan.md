# Implementation Notes

## Completed Optimizations

### LA Ops
- De Rijk column sorting for SVD (Metal kernel)
- Sign canonicalization (Metal kernel)
- Fused Q.T @ b for solve (single command buffer)
- MAGMA-style shared memory Cholesky

### Training Ops (v0.1.7)
- Vectorized float4 GELU/SiLU (2-4x bandwidth improvement)
- SIMD reductions for RMSNorm (simd_sum, simd_max)
- Fused single-kernel AdamW step
- Flash Attention v2 tiling with online softmax
- Causal masking via -INFINITY initialization

## Future Optimizations
- Optimize SDPA to match PyTorch native MPS
- FP16/BFloat16 kernels for memory efficiency
- Blocked LU for N > 64 (per ROCm research)
- Parallel panel QR factorization
