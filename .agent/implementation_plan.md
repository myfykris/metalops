# Implementation Notes

## Completed Optimizations
- De Rijk column sorting for SVD (Metal kernel)
- Sign canonicalization (Metal kernel)
- Fused Q.T @ b for solve (single command buffer)
- MAGMA-style shared memory Cholesky

## Future Optimizations
- Blocked LU for N > 64 (per ROCm research)
- Vectorized kernels (float4) for memory bandwidth
- Parallel panel QR factorization
