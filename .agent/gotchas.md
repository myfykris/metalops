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

## Unused Variable Warnings (Known Cruft)
These are NOT half-implemented features, just cruft:
- `MAX_PANEL_M`, `MAX_PANEL_N`, `CHOL_MAX_N`, `WARP_SIZE` in .metal files - tuning constants
- `int64_t N` at line ~954 in core_mps.mm - declared but unused in apply_householder_panel
- `threads_per_pair` at line ~2282 - duplicate declaration (used elsewhere)
