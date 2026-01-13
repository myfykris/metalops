# Metal & PyTorch MPS Gotchas

## CRITICAL: Before Doing ANYTHING
**ALWAYS check for existing scripts first!** Before building wheels, running tests, or any repetitive task:
```bash
ls *.sh && cat .agent/workflows/*.md
```
Key scripts in this repo:
- `build_all_wheels.sh` - Build and patch wheels for all Python versions
- `/release` workflow - Full release process including PyPI

## PyTorch MPS Backend
- **QR Fallback**: `torch.linalg.qr` falls back to CPU on MPS. Use `metalcore.qr()` for GPU.
- **BFloat16**: Requires Metal 3.1+ (macOS 14+). Check with `__METAL_VERSION__ >= 310`.
- **SDPA**: PyTorch's native `F.scaled_dot_product_attention` is highly optimized on MPS. Our custom kernel is slower.

## Metal Kernel Development
- **Strict Type Casting**: Literals must be cast. `T val = 0.0;` fails for half/bfloat.
- **Missing SIMD for BFloat16**: `simd_shuffle_down` not defined. Cast to `ushort`, shuffle, cast back.
- **Function Dependencies**: Helpers used in macros must be defined before the macro.

## Linear Algebra Precision
- **fp16/bf16 NaN Issues**: LU factorization (used in `solve`) produces NaNs with half-precision. Always promote to fp32 for computation, convert back after.
- **Pattern**: `bool need_conversion = (dtype == kHalf || dtype == kBFloat16); A_in = need_conversion ? A.to(kFloat) : A; ... return need_conversion ? result.to(input_dtype) : result;`

## Common Pitfalls
- **Stale Encoder**: Allocate tensors BEFORE `stream->commandEncoder()`. PyTorch ops invalidate encoders.
- **Forward-Pass Sync**: Use `COMMIT` (not `COMMIT_AND_WAIT`) for ops that run mid-forward-pass to avoid "command encoder already encoding" errors.
- **Encoder Conflicts**: Call `stream->synchronize(SyncType::COMMIT)` before creating encoder in backward pass.
- **threadgroup_barrier**: Use `mem_flags::mem_device` for device memory, `mem_threadgroup` for shared.
- **Shared Memory Size**: Must be set with `setThreadgroupMemoryLength` before dispatch.

## Command Buffer Integration (HuggingFace/Transformers)
- **Problem**: metalcore ops in HuggingFace model forward passes crash with "command encoder already encoding"
- **Root Cause**: Creating encoder via `[cmdBuffer computeCommandEncoder]` conflicts with PyTorch's active encoder
- **Correct Pattern**:
  ```cpp
  auto stream = at::mps::getCurrentMPSStream();
  auto encoder = stream->commandEncoder();  // Returns existing or creates new
  // ... encode your kernel ...
  // DON'T call [encoder endEncoding] - PyTorch manages lifecycle
  stream->synchronize(SyncType::NONE);  // Let PyTorch batch commands
  ```
- **Wrong Pattern** (causes crash):
  ```cpp
  auto encoder = [stream->commandBuffer() computeCommandEncoder];  // WRONG!
  // ... encode ...
  [encoder endEncoding];  // WRONG - ends PyTorch's encoder
  stream->synchronize(SyncType::COMMIT_AND_WAIT);  // WRONG - blocks
  ```
  **CRITICAL**: If you mix custom Metal encoders with MPS kernels (like `MPSMatrixMultiplication`), you **MUST** use `[cmdBuf computeCommandEncoder]` and manually `endEncoding` before calling MPS kernels. Do **NOT** use `stream->commandEncoder()` in mixed contexts, as it causes encoder coalescing crashes.
- **Fixed ops**: rmsnorm_fwd, fused_add_rmsnorm, gelu_fwd, silu_fwd, fused_softmax, layernorm_fwd, embedding_bag
- **Unchanged ops**: SVD, QR, Eigh, Cholesky, Solve, AdamW (standalone ops that need immediate results)

## Training Ops Gotchas
- **Atomics in Backward**: SDPA backward uses `atomic_fetch_add`, which is slower but thread-safe.
- **Causal Masking**: Initialize p_reg to -INFINITY for proper mask handling.
- **AdamW Tail Handling**: Use scalar kernel for (numel % 4) tail elements to avoid OOB.
- **AdamW Mixed-Precision**: Params can be bf16/fp16 for bandwidth, but **optimizer states (exp_avg, exp_avg_sq) MUST be float32** for numerical stability. The Python wrapper should use `dtype=torch.float32` explicitly.

## Performance Tuning
- **Batched > Single**: Always prefer batched ops for GPU utilization
- **Shared Memory**: Use for N ≤ 64, reduces VRAM round-trips
- **256-1024 threads**: Sweet spot for most reductions
- **float4 Vectorization**: 2-3x bandwidth improvement for elementwise ops

## CI Testing Limitations
- **GitHub Actions has NO MPS**: `macos-latest` runners lack Metal GPU access
- **Tests MUST be import-only**: Don't add tests that call Metal kernels
- **Local testing only**: Run `pytest` and `adamw_stress_test.py` on Apple Silicon
- **See**: `tests/test_imports.py` header comments for detailed guidance

## PyTorch Override System (`overrides.py`)
- **Purpose**: `enable_pytorch_overrides()` monkey-patches F.silu, F.gelu, F.embedding_bag to use metalcore
- **Default Enables**: activations (silu, gelu) + embedding_bag (50-100x win over CPU fallback)
- **Default Disables**: normalization, softmax (near parity with PyTorch)
- **Usage**: Call once at startup: `import metalcore; metalcore.enable_pytorch_overrides()`
- **Limitation**: Cannot fully restore originals - restart interpreter for clean state

## INT4 Quantization (`quantization.py`)

### Recommended Approaches (in order):
1. **Hybrid (fastest)**: Store INT4, dequant to FP16 at load → `Int4Linear.from_float(linear, dequant_on_load=True)`
2. **GGML block_q4_0 (llama.cpp compatible)**: `matmul_ggml_q4_0()` → 4-15x overhead vs FP16

### Legacy Kernels (still in codebase, not recommended):
- `matmul_int4_*` variants (scalar, vec4, tiled, simd) - slow due to per-K dequant
- `matmul_int4_tensor` - requires M4/M5 chips (Metal 4 GPU family)
- `matmul_int4_llama` - superseded by GGML kernel

### Details
- **Dequant Formula**: `W = (q - zero) * scale` where q is [0,15]
- **GGML Formula**: `W = d * (q - 8)` where d is scale, zero is implicit at 8
- **Functions**: `quantize_int4()`, `dequantize_int4()`, `matmul_int4()`, `Int4Linear`
- **GGML Functions**: `quantize_ggml_q4_0()`, `dequantize_ggml_q4_0()`, `matmul_ggml_q4_0()`

### Future: Compressed-Space GEMM (Optional Feature)
**Goal**: Operate directly on INT4 values without dequant, apply scales post-hoc.

**Key insight**: `Y = X @ W_float = X @ ((W_q - zero) * scale)` can be decomposed:

**Option 1: Per-tensor quantization** (simplest)
```
Y = scale * (X @ W_q - zero * X.sum(dim=1, keepdim=True))
```
One GEMM on int8/FP16-small-ints + cheap correction. Loses per-group accuracy.

**Option 2: Per-group with batched GEMM**
```
Y = Σ_g scale[g] * (X_g @ W_g - zero[g] * X_g.sum())
```
`num_groups` smaller GEMMs (e.g., 32 for K=4096, group_size=128). May batch with einsum.

**Option 3: Precompute scaled weights**
Store `W_q * scale_expanded` instead of `W_q, scale` separately. Encode scale into weights.
Tradeoff: loses some compression, but enables single GEMM.

**Option 4: SmoothQuant-style** 
Balance scale between activations and weights: `Y = (X * s_inv) @ (W_q * s)`
Requires calibration data, but enables int8 GEMM.

### GGML block_q4_0 Kernel (llama.cpp port)
**Status**: Implemented and working!
- Ported llama.cpp's `kernel_mul_mm` using `simdgroup_multiply_accumulate`
- Uses GGML block_q4_0 format: 32 values per block, scale embedded
- MIT licensed, attribution included

**Performance** (M3 Max, K=N=4096):
| Batch Size | GGML Kernel | FP16 GEMM | Ratio |
|------------|-------------|-----------|-------|
| M=256 | 4.8ms | 1.1ms | 4.3x slower |
| M=128 | 3.9ms | 0.5ms | 7.3x slower |
| M=64 | 3.6ms | 0.3ms | 13.4x slower |
| M=1 | 3.5ms | 0.1ms | 27.5x slower |

**Improvement**: 36x faster than initial naive kernel (was 553x slower, now 4-15x)

**Functions**: `quantize_ggml_q4_0()`, `dequantize_ggml_q4_0()`, `matmul_ggml_q4_0()`

**Open questions**:
- Is per-tensor quantization accuracy acceptable? (test on actual models)
- Can batched group GEMMs beat single FP16 GEMM? (benchmark)
- Does Apple MPS have efficient int8 GEMM? (explore MPSMatrixMultiplication)



## Unused Variable Warnings (Known Cruft)
These are NOT half-implemented features, just cruft:
- `MAX_PANEL_M`, `MAX_PANEL_N`, `CHOL_MAX_N`, `WARP_SIZE` in .metal files - tuning constants
- `int64_t N` at line ~954 in core_mps.mm - declared but unused in apply_householder_panel
- `threads_per_pair` at line ~2282 - duplicate declaration (used elsewhere)


## Fused Kernels & Memory Layouts (SwiGLU Lessons)
- **The "Transposed Data" Trap**: `at::mm` (and `MPSMatrixMultiplication`) often returns non-contiguous or transposed outputs (e.g. column-major) to optimize for the GPU hierarchy.
    - **Symptom**: Linear kernels (`idx = id`) read garbage/NaNs/Zeros because they assume row-major contiguous layout (`idx = row * W + col`), but the actual data is strided differently.
    - **Fix**: **ALWAYS use Strided Kernels** for elementwise ops following a MatMul. Dispatch a 2D grid (`rows`, `cols`) and pass explicit strides (`stride_row`, `stride_col`).
      ```cpp
      // Valid mapping for ANY layout:
      uint idx = row * stride_row + col * stride_col;
      ```
- **Const Aliasing**: Never declare a buffer as `device const T*` if you are writing to it in-place (aliasing input as output). The compiler may cache reads assuming immutability, leading to race conditions.

## Troubleshooting
### Crash: "failed assertion 'A command encoder is already encoding to this command buffer'"
- **Cause**: A custom Metal kernel created a raw encoder (`[cmdBuf computeCommandEncoder]`) on a command buffer that PyTorch was already using.
- **Fix**: Use `stream->commandEncoder()` instead. This method is aware of PyTorch's state and will either return the active encoder or safely create a new one. **NEVER** call `[encoder endEncoding]` when using this method.
- **Affected Kernels**: Historically `adamw`, `rmsnorm`, `sdpa`, `cholesky`, `solve`. Ensure these are patched to use `stream->commandEncoder()`.
