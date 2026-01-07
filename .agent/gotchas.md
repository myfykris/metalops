# Metal & PyTorch MPS Gotchas

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
- **Encoder Conflicts**: Call `stream->synchronize(SyncType::COMMIT)` before creating encoder in backward pass.
- **threadgroup_barrier**: Use `mem_flags::mem_device` for device memory, `mem_threadgroup` for shared.
- **Shared Memory Size**: Must be set with `setThreadgroupMemoryLength` before dispatch.

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
