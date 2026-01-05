# Release Roadmap

## Current: metalcore 0.1.5
All LA + training ops consolidated into single package.

### Included in 0.1.5
- ✅ SVD, QR, Cholesky, Eigh, TRSM, Solve
- ✅ RMSNorm (2.5x faster)
- ✅ AdamW (2.9x faster)
- ✅ GELU/SiLU (4x faster)
- ✅ SDPA (Flash Attention v2)
- ✅ Python 3.9-3.13 support
- ✅ Precompiled .metallib

## Future Enhancements
- [ ] Complex number support
- [ ] Optimize SDPA to match native MPS
- [ ] More BLAS Level 3 ops
- [ ] FP16/BFloat16 kernels
