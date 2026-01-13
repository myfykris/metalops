# System Architecture

## Overview
`metalcore` provides high-performance linear algebra and training operations for PyTorch on macOS by bypassing generic MPS fallbacks and executing custom Metal kernels.

## Package Structure
```
packages/metalcore/
├── native/
│   ├── core_kernels.metal       # Core LA kernels (SVD, QR, Cholesky, Eigh, TRSM)
│   ├── training_kernels.metal   # RMSNorm, AdamW, RoPE
│   ├── activation_kernels.metal # GELU, SiLU
│   ├── sdpa_kernels.metal       # Flash Attention v2
│   ├── quant_kernels.metal      # INT4/INT8 quantized matmul
│   └── core_mps.mm              # C++ dispatch + PYBIND11
└── src/metalcore/
    ├── __init__.py      # Public API
    ├── overrides.py     # PyTorch F.silu/F.gelu/embedding_bag + SVD/QR overrides
    ├── svd.py           # SVD with De Rijk optimization
    ├── qr.py            # QR decomposition
    ├── cholesky.py      # Cholesky factorization
    ├── solve.py         # Linear solve (LU-based, fp16/bf16 supported)
    ├── rmsnorm.py       # MetalRMSNorm module
    ├── optim.py         # MetalAdamW optimizer
    ├── activations.py   # metal_gelu, metal_silu
    ├── rope.py          # RoPE kernels + RotaryEmbedding module
    ├── quantization.py  # INT4 quantization + Int4Linear
    └── sdpa.py          # Flash Attention v2 (SLOWER than PyTorch native!)
```

## Components

### 1. Metal Kernels
| File | Kernels |
|------|---------|
| `core_kernels.metal` | SVD, QR, Cholesky, Eigh, TRSM |
| `training_kernels.metal` | rmsnorm_fwd/bwd, adamw_step, rope_fwd/bwd, fused_swiglu_mlp, fused_lora_attention |
| `activation_kernels.metal` | gelu_fwd/bwd, silu_fwd/bwd (float4 vectorized) |
| `sdpa_kernels.metal` | flash_attention_fwd_v2/bwd_v2, attention_naive |
| `quant_kernels.metal` | matmul_int4_*, matmul_ggml_q4_0 (llama.cpp), quantize_to_int4/int8 |

### 2. Host Orchestrator (`core_mps.mm`)
- **Objective-C++**: Bridges PyTorch (C++) and Metal (Obj-C)
- **PSO Management**: Pipeline state objects (25+ kernels)
- **PYBIND11**: Python bindings for all operations

### 3. Python Wrappers
- **Autograd**: Custom backward passes (GELU, SiLU, SDPA, RoPE)
- **MetalRMSNorm**: `nn.Module` replacement for `torch.nn.RMSNorm`
- **MetalAdamW**: Drop-in replacement for `torch.optim.AdamW`
- **Int4Linear**: 4x memory compression for Linear layers
- **RotaryEmbedding**: Drop-in HuggingFace RoPE replacement

## Design Decisions
- **Compiled .metallib**: Precompiled for faster load times
- **Unified Package**: All ops in one native extension
- **Batched First**: Kernels optimized for batch parallelism
- **SDPA Opt-in**: Requires explicit `enable_metal_sdpa()` call
- **Size-aware Overrides**: SVD/QR only override at beneficial sizes

