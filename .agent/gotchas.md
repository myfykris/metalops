# Metal & PyTorch MPS Gotchas

## PyTorch MPS Backend
- **QR Fallback**: `torch.linalg.qr` falls back to CPU on MPS. Use `metalcore.qr()` for GPU.
- **BFloat16**: Requires Metal 3.1+ (macOS 14+). Check with `__METAL_VERSION__ >= 310`.

## Metal Kernel Development
- **Strict Type Casting**: Literals must be cast. `T val = 0.0;` fails for half/bfloat.
- **Missing SIMD for BFloat16**: `simd_shuffle_down` not defined. Cast to `ushort`, shuffle, cast back.
- **Function Dependencies**: Helpers used in macros must be defined before the macro.

## Common Pitfalls
- **Stale Encoder**: Allocate tensors BEFORE `stream->commandEncoder()`. PyTorch ops invalidate encoders.
- **threadgroup_barrier**: Use `mem_flags::mem_device` for device memory, `mem_threadgroup` for shared.
- **Shared Memory Size**: Must be set with `setThreadgroupMemoryLength` before dispatch.

## Performance Tuning
- **Batched > Single**: Always prefer batched ops for GPU utilization
- **Shared Memory**: Use for N ≤ 64, reduces VRAM round-trips
- **256-1024 threads**: Sweet spot for most reductions
