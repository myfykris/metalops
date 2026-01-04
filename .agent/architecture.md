# System Architecture

## Overview
`metalsvd` provides a high-performance SVD implementation for PyTorch on macOS by bypassing the generic MPS fallback and executing custom Metal kernels.

## Components

### 1. The Host Orchestrator (`src/svd_mps.mm`)
- **Objective-C++**: Bridges PyTorch (C++) and Metal (Obj-C).
- **Dynamic Dispatch**: Checks `tensor.scalar_type()` to select the correct kernel variant (`float`, `half`, `bfloat`).
- **Durability**: Uses runtime checks and preprocessor macros (`#if __METAL_VERSION__ >= 310`) to safely degrade on older macOS versions lacking BFloat16 support.

### 2. The Metal Kernels (`src/svd.metal`)
- **One-Sided Jacobi Algorithm**: Operates on a batch of matrices $A$.
- **Parallelism**:
  - `jacobi_rotate_kernel`: Assigns a threadgroup to each matrix column-pair $(i, j)$.
  - **Threadgroup Reduction**: Uses SIMD shuffle instructions (`simd_sum`, `simd_shuffle_down`) to compute dot products $(A_i \cdot A_j)$ in effectively $O(1)$ time relative to column height (for $M \le 1024$).
  - **Shared Memory**: Caches reduction results to minimize VRAM round-trips.
- **Templating**: Uses C++ templates in MSL to instantiate `half` and `bfloat` variants from a single source of truth.

### 3. Python Wrapper (`metalsvd/func.py`)
- **Autograd**: Implements a custom `torch.autograd.Function`. The backward pass is computed analytically using the standard SVD gradient formula in pure PyTorch (efficient since $U, S, V$ are already on GPU).
- **Wide Matrices ($M < N$)**: Automatically handles wide inputs by transposing, calling SVD, and swapping $U/V$.

## Design Decisions
- **External Metal Source**: We load `svd_kernels.metal` from the file system at runtime. This allows for rapid iteration and hot-reloading during development without recompilation of the C++ extension.
- **Monkeypatching**: We deliberately provide a mechanism to overwrite `torch.linalg.svd` to allow users to upgrade existing codebases with zero refactoring.

## Optimization Strategy
- **Specialized Kernels**: 
  - **N <= 64**: Warp-synchronous kernels (TPP=32).
  - **N = 128**: Half-warp kernels (TPP=16).
  - **N = 256**: Quarter-warp kernels (TPP=8).
  - **N > 256**: Generic Block Jacobi or rSVD.
- **Batched Throughput**: Focus on minimizing kernel launch overhead by fusing the entire Jacobi sweep into a single persistent kernel.

## Developer Notes & Gotchas

### 1. The "Stale Encoder" Segfault
**Symptom**: `segmentation fault` at subclass `[encoder setComputePipelineState:]`.
**Cause**: Calling complex PyTorch operations (like `tensor.to(device)` or `torch.empty(..., device='mps')`) *after* fetching `stream->commandEncoder()` causes the underlying Metal Command Buffer to be committed or modified by PyTorch's internal pool. This invalidates the retrieved `id<MTLComputeCommandEncoder>` pointer.
**Solution**: ALWAYS perform all tensor allocations, copies (`.to()`), and shape calculations *before* calling `stream->commandEncoder()`. Once you fetch the encoder, strictly perform dispatch operations only.

### 2. Metal Versioning on MPS
**Symptom**: Crash or `nil` PSO (Pipeline State Object) when `options.languageVersion = MTLLanguageVersion3_0` is set.
**Cause**: Not all Apple Silicon devices/OS combinations report strict Metal 3.0 compliance in the way `MTLCompileOptions` expects.
**Solution**: Rely on the default compiler version. For `bfloat16`, check `__METAL_VERSION__ >= 310`.

### 3. Fused Kernel Safety
**Symptom**: Race conditions or incorrect reductions in block-Jacobi.
**Constraint**: The Fused Block-Jacobi kernel uses `threadgroup_barrier` for synchronization. For small N (<= 256), we use specialized kernels with hardcoded `ThreadsPerPair` to maximize occupancy and minimize divergence.

## Failed Approaches & Dead Ends
- **Early Termination via Atomics**: We attempted to use an atomic counter `sweep_rotations` to break the Jacobi loop early if no rotations occurred. determining convergence.
  - **Result**: Failed. The cost of atomic operations in the hot loop roughly equaled the savings from skipping the last few sweeps. For small matrices, kernel launch latency dominates anyway.
- **Generic Fused Kernel for N < 128**: We initially used the generic `svd_fused_block_kernel` with `ThreadsPerPair=1` for all sizes.
  - **Result**: Failed. Performance was 5-10x slower than CPU due to massive thread underutilization (1 thread per pair = 64 items for N=128). Specialized kernels with TPP=16/32 were required.
- **Dynamic ThreadsPerPair in Shader**: Calculating loop strides based on a uniform `ThreadsPerPair` passed at runtime.
  - **Result**: Suboptimal. The compiler could not unroll loops or optimize register usage as effectively as with `constant` template parameters or macros. Hardcoded variations (N=64, 128) proved significantly faster.

