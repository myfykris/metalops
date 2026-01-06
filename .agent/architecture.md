# System Architecture

## Overview
`metalcore` provides high-performance linear algebra and training operations for PyTorch on macOS by bypassing generic MPS fallbacks and executing custom Metal kernels.

## Package Structure
```
packages/metalcore/
├── native/
│   ├── core_kernels.metal       # Core LA kernels
│   ├── training_kernels.metal   # RMSNorm, AdamW
│   ├── activation_kernels.metal # GELU, SiLU
│   ├── sdpa_kernels.metal       # Flash Attention v2
│   └── core_mps.mm              # C++ dispatch + PYBIND11
└── src/metalcore/
    ├── __init__.py      # Public API + enable_metal_sdpa()
    ├── svd.py           # SVD with De Rijk optimization
    ├── qr.py            # QR decomposition
    ├── cholesky.py      # Cholesky factorization
    ├── solve.py         # Linear solve (LU-based, fp16/bf16 supported)
    ├── rmsnorm.py       # MetalRMSNorm module
    ├── optim.py         # MetalAdamW optimizer
    ├── activations.py   # metal_gelu, metal_silu
    └── sdpa.py          # Flash Attention v2
```

## Components

### 1. Metal Kernels
| File | Kernels |
|------|---------|
| `core_kernels.metal` | SVD, QR, Cholesky, Eigh, TRSM |
| `training_kernels.metal` | rmsnorm_fwd/bwd, adamw_step |
| `activation_kernels.metal` | gelu_fwd/bwd, silu_fwd/bwd (float4 vectorized) |
| `sdpa_kernels.metal` | flash_attention_fwd_v2/bwd_v2, attention_naive |

### 2. Host Orchestrator (`core_mps.mm`)
- **Objective-C++**: Bridges PyTorch (C++) and Metal (Obj-C)
- **PSO Management**: Pipeline state objects (19 kernels)
- **PYBIND11**: Python bindings for all operations

### 3. Python Wrappers
- **Autograd**: Custom backward passes (GELU, SiLU, SDPA)
- **MetalRMSNorm**: `nn.Module` replacement for `torch.nn.RMSNorm`
- **MetalAdamW**: Drop-in replacement for `torch.optim.AdamW`

## Design Decisions
- **Compiled .metallib**: Precompiled for faster load times
- **Unified Package**: All ops in one native extension
- **Batched First**: Kernels optimized for batch parallelism
- **SDPA Opt-in**: Requires explicit `enable_metal_sdpa()` call
