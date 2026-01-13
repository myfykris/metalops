# QR Decomposition Kernels

## Performance Summary

| Size | GPU μs/mat | CPU μs/mat | Winner |
|------|------------|------------|--------|
| 8x8 | 0.58 | 0.89 | **GPU 1.53x** ✓ |
| 16x16 | 2.35 | 2.27 | Tie |
| 32x32 | 9.42 | 6.56 | CPU 1.44x |

## Key Findings

### Why GPU Wins at 8x8
- **No barriers**: 1 thread per matrix, all work in registers
- **Simple algorithm**: 8 Householder reflections fit in thread-local state

### Why GPU Ties at 16x16
- **32 threads per matrix**: SIMD group operations
- **16 barriers**: Acceptable overhead for 16 columns

### Why CPU Wins at 32x32
- **CPU GFLOPS**: 6.7G vs GPU 4.6G (1.44x gap)
- **Root cause**: Householder requires 32 sequential reflections
- **Barrier-free kernel tested**: Single-thread shared-mem achieves SAME 9.4μs
- **Not barrier overhead**: GPU just can't match CPU's per-thread efficiency

## Analysis Deep Dive

### Dispatch Overhead
- Cold (GPU idle 0.1s): 15ms 
- Warm: 2.2ms
- At batch=10K+, amortized to negligible

### GFLOPS Scaling
| Size | GPU GFLOPS | CPU GFLOPS |
|------|------------|------------|
| 8x8 | 1.2 | 0.8 |
| 16x16 | 2.3 | 2.4 |
| 32x32 | 4.6 | 6.7 |

GPU scales 4x from 8→32, CPU scales 8x. CPU wins at larger sizes.

### What Was Tried for 32x32
1. **32-thread SIMD with barriers**: 9.4μs
2. **1-thread shared memory (no barriers)**: 9.4μs (same!)
3. **Fully unrolled loops**: No improvement
4. **Cholesky-QR alternative algorithm**: Failed (numerical issues)
5. **CGS algorithm in Python**: 4.8x slower (dispatch overhead)

## Recommendation
- Use GPU for **8x8** (1.5x win)
- Either GPU or CPU for **16x16**
- Accept 1.44x CPU advantage at **32x32** or fall back to CPU
