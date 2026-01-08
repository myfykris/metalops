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
- **RMSNorm** (`MetalRMSNorm`): 2.5x faster than PyTorch
- **AdamW** (`MetalAdamW`): 2.9x faster optimizer
- **GELU/SiLU** (`metal_gelu`, `metal_silu`): Vectorized activations
- **EmbeddingBag**: 50-100x faster (avoids CPU fallback)
- **LayerNorm**, **Softmax**: Fused implementations

### PyTorch Integration (NEW in 0.1.12+)
```python
import metalcore

# Automatically accelerate F.silu, F.gelu, F.embedding_bag
metalcore.enable_pytorch_overrides()

# Works seamlessly with HuggingFace models
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("...", device_map="mps")

# Optional: Also patch RMSNorm modules
metalcore.patch_transformers_rmsnorm(model)
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
| EmbeddingBag | **50-100x** (vs CPU fallback) |
| Cholesky Batched | **33x** |
| QR Batched | **20x** |
| Solve Batched | **10x** |
| AdamW | **2.9x** |
| RMSNorm | **2.5x** |
| GELU/SiLU | **2-4x** |

## Requirements

- macOS 12.0+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.9 - 3.14
- PyTorch 2.0+

> **Note**: M3/M4 chips recommended for best bf16 performance. The library gracefully falls back to FP32 on older hardware.

## Author

[Kris Bailey](https://github.com/myfykris)

## License

MIT
