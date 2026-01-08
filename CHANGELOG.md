# Changelog

All notable changes to `metalcore` will be documented in this file.

## [0.1.13] - 2026-01-07

### Fixed
- **Command Buffer Integration**: Fixed "command encoder already encoding" crash when using metalcore ops with PyTorch SDPA in HuggingFace models
  - Changed from manual `[cmdBuffer computeCommandEncoder]` to PyTorch's managed `stream->commandEncoder()`
  - Removed `endEncoding` calls - PyTorch now manages encoder lifecycle
  - Changed `SyncType::COMMIT_AND_WAIT` to `SyncType::NONE` for forward-pass ops

### Changed
- Cleaned up unused variables in Metal shaders and C++ code for cleaner builds

## [0.1.12] - 2026-01-07

### Added
- **`enable_pytorch_overrides()`**: Transparent acceleration for `F.silu`, `F.gelu`, and `F.embedding_bag`
- **`patch_transformers_rmsnorm(model)`**: Patches all RMSNorm modules in HuggingFace Transformers models with `MetalRMSNorm`
- `disable_pytorch_overrides()` and `get_active_overrides()` helper functions
- `test_forward_pass.py` and `test_command_buffer.py` integration test suites

### Fixed
- Forward-pass synchronization: Changed 7 ops from blocking to non-blocking sync

## [0.1.11] - 2026-01-06

### Fixed
- Verified `embedding_bag` numerical accuracy matches PyTorch

## [0.1.10] - 2026-01-06

### Added
- **Native bf16 kernels** for GELU, SiLU, and RMSNorm forward passes
  - 2x faster bf16 training by avoiding fp32 conversion overhead
  - In-register bf16→fp32→bf16 conversion for numerical stability

### Changed
- Renamed SDPA functions to `enable_slow_metal_sdpa()` with warnings about performance

## [0.1.9] - 2026-01-05

### Added
- **Native bf16/fp16 AdamW kernels** with ILP=4 optimization
- `adamw_step_ilp4` kernel for large tensors (>256KB)

## [0.1.8] - 2026-01-05

### Added
- **`fused_add_rmsnorm`**: Fused residual add + RMSNorm in single kernel
- **`fused_softmax`**: Online softmax with numerical stability
- **`layer_norm`** / `MetalLayerNorm`: Full LayerNorm implementation
- **`embedding_bag`**: GPU-native embedding bag (avoids CPU fallback, 50-100x faster)
- **`scatter_add`** and **`gather`**: Index-based tensor operations
- ILP4 AdamW kernel, half4 vectorized kernels

## [0.1.7] - 2026-01-04

### Fixed
- PyPI metadata issues
- Added `License-File` removal to build script

## [0.1.6] - 2026-01-04

### Added
- **`solve` fp16/bf16 support**: Half-precision linear system solving with automatic fp32 promotion

### Changed
- Dynamic benchmark sorting in results

## [0.1.5] - 2026-01-03

### Added
- **Training Operations**:
  - `MetalRMSNorm`: Fused RMS normalization (2.5x faster)
  - `MetalAdamW`: Fused optimizer step (2.9x faster)
  - `metal_gelu` / `metal_silu`: Vectorized activations (up to 4x faster)
  - `metal_scaled_dot_product_attention`: Flash Attention v2 implementation

## [0.1.0] - 2026-01-01

### Added
- Initial release of unified `metalcore` package
- **Linear Algebra Operations**:
  - `svd`: Jacobi SVD with De Rijk optimization (up to 25x faster for LLM matrices)
  - `qr`: Householder QR decomposition (20x faster batched)
  - `eigh`: Symmetric eigendecomposition (3.5x faster batched)
  - `cholesky`: Cholesky factorization (33x faster batched)
  - `solve`: LU-based linear system solver (10x faster batched)
  - `trsm` / `trsm_batched`: Triangular solve
- Precompiled `.metallib` for fast kernel loading
- Support for Python 3.9 - 3.14
