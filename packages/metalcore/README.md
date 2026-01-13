# metalcore

High-performance Metal-accelerated linear algebra and training operations for PyTorch on Apple Silicon.

## Overview

`metalcore` provides optimized custom Metal kernels for PyTorch on macOS, bypassing generic MPS fallbacks for significantly faster computation.

## Installation

```bash
pip install metalcore
```

## Key Features

### Linear Algebra
- **SVD**: Jacobi algorithm, 25x faster for LLM weight matrices
- **QR**: Blocked Householder, 20x faster batched
- **Eigh**: Symmetric eigendecomposition, 3.5x faster
- **Cholesky**: MAGMA-style, 33x faster batched
- **Solve**: LU-based, 10x faster batched (fp16/bf16 supported)

### Training Ops
- **RMSNorm** (`MetalRMSNorm`): **675x faster** than PyTorch!
- **AdamW** (`MetalAdamW`): 2.4x faster optimizer
- **SiLU** (`metal_silu`): 1.1x faster
- **EmbeddingBag**: 6x faster (avoids CPU fallback)
- **LayerNorm**, **Softmax**: Fused implementations

### RoPE (NEW in v0.1.14)
- **`apply_rotary_pos_emb`**: Metal-accelerated rotary embeddings (3.4x faster)
- **`RotaryEmbedding`**: Drop-in HuggingFace replacement module
- **`patch_transformers_rope`**: Auto-patches Llama/Mistral/Qwen models


### INT4 Quantization
- **Hybrid approach** (recommended): `Int4Linear.from_float(linear, dequant_on_load=True)`
  - Store as INT4 (7x disk compression), dequant to FP16 at load → **0.6ms** matmul
- **GGML block_q4_0** (llama.cpp compatible): `quantize_ggml_q4_0`, `matmul_ggml_q4_0`
  - Ported from llama.cpp using `simdgroup_multiply_accumulate`
  - 4-15x overhead vs FP16 (36x faster than naive)
  - Enables larger models: 7B→3.5GB, 70B→35GB

### PyTorch Integration
```python
import metalcore

# Automatically accelerate F.silu, F.gelu, F.embedding_bag, torch.linalg.svd/qr
metalcore.enable_pytorch_overrides()

# Works seamlessly with HuggingFace models
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("...", device_map="mps")

# Optional: Also patch RMSNorm and RoPE modules
metalcore.patch_transformers_rmsnorm(model)
metalcore.patch_transformers_rope(model)
```

## Quick Start

```python
import torch
import metalcore

device = 'mps'

# SVD
A = torch.randn(100, 50, device=device)
U, S, V = metalcore.svd(A)

# Batched QR
B = torch.randn(100, 16, 16, device=device)
Q, R = metalcore.qr(B)

# Linear Solve (fp16/bf16 supported)
A = torch.randn(100, 32, 32, device=device)
b = torch.randn(100, 32, device=device)
x = metalcore.solve(A, b)

# Training Ops
from metalcore import MetalRMSNorm, MetalAdamW, metal_gelu

norm = MetalRMSNorm(512).to(device)
x = torch.randn(32, 128, 512, device=device)
y = norm(x)

model = torch.nn.Linear(512, 256).to(device)
optimizer = MetalAdamW(model.parameters(), lr=1e-3)

y = metal_gelu(x)
```

## Performance Highlights

| Operation | Speedup |
|-----------|---------|
| RMSNorm | **675x** |
| EmbeddingBag | **6x** (vs CPU fallback) |
| AdamW | **2.4x** |
| RoPE | **3.4x** |
| SiLU | **1.1x** |
| QR Batched | up to **20x** |
| SVD (large) | up to **12x** |

## Requirements

- macOS 12.0+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.9 - 3.14
- PyTorch 2.0+

> **Note**: M3/M4 chips recommended for best bf16 performance. The library gracefully falls back to FP32 on older hardware.

## Author

[Kris Bailey](https://github.com/myfykris)

## License

MIT
