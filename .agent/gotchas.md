# Metal & PyTorch MPS Gotchas

## PyTorch MPS Backend
- **QR Fallback**: `torch.linalg.qr` does not currently support MPS and falls back to CPU. This incurs data transfer overhead and CPU processing time.
- **BFloat16 Support**: Requires explicit Metal 3.1+ targeting (`options.languageVersion = MTLLanguageVersion3_1` in Obj-C++ wrapper).

## Metal Kernel Development
- **Strict Type Checking**:
    - Casting literals is mandatory. `T val = 0.0;` fails for `bfloat`/`half` if `0.0` is double/float. Use `T val = (T)0.0;`.
    - Function returns must be cast explicitly. `return 1e-3;` in a `bfloat` function fails. Use `return (bfloat)1e-3;`.
- **Missing SIMD Intrinsics for BFloat16**:
    - `simd_shuffle_down` is not defined for `bfloat`. Workaround: Cast to `ushort`, shuffle, then cast back.
    - `dot` product for `vec<bfloat, 4>` (`bfloat4`) is missing. Must be implemented manually (`a.x*b.x + ...`).
- **Dependencies**: Functions used in macros (like `simd_shuffle_down` helper) must be defined *before* the macro or function that calls them. Forward declarations or careful ordering is required.

## Numerical Stability & Convergence
- **Epsilon Tuning**: `1e-6` is too strict for `half` and `bfloat16`. It causes Jacobi SVD to loop excessively or fail to converge. Use `1e-3` for these lower-precision types.

## Performance Tuning
- **Vectorization vs Occupancy**: High threadgroup occupancy (e.g., 1024 threads) can sometimes outperform theoretically "better" ILP configurations (e.g., 256 threads with more work per thread). In our "Huge Fat" SVD case, TPG=1024 was ~15% faster than TPG=256 even for the vectorized kernel, likely due to better latency hiding on the GPU.

## Matrix Shape & Occupancy
- **Huge Matrices**: For matrices larger than ~4096 on a side, the standard iterative Jacobi kernel (even with ICB) can become numerically unstable on Metal, producing reconstruction errors ~1.41 (uncorrelated output).
- **Recommendation**: Use the **Hybrid Gram Strategy** (`strategy='gram'`) for "Tall/Fat" matrices ($M \gg N$). It computes $A^T A$ on Metal (stable and fast) and solves the small square SVD on CPU (robust). This consistently achieves strict accuracy (< 1e-5) where the native kernel fails.

