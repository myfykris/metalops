# MetalOps

**MetalOps** is a collection of high-performance Metal-accelerated linear algebra and training operations for PyTorch on macOS (Apple Silicon).  This repo is meant as a starting point to get Apple Silicon acceleration in a better place, so feel free to send through PRs if you have fixes, or report issues!

All operations are consolidated into the single **`metalcore`** package.

## License

MIT

## Author

[Kris Bailey](https://github.com/myfykris)

## Features

| Operation | Description | Performance |
|-----------|-------------|-------------|
| **SVD** | Jacobi Singular Value Decomposition | Up to **25x faster** for LLM weight matrices |
| **QR** | Householder QR Decomposition | Up to **20x faster** for batched small matrices |
| **Eigh** | Symmetric Eigendecomposition | Up to **3.5x faster** for batched matrices |
| **Cholesky** | Cholesky Decomposition | **33x faster** for batched operations |
| **Solve** | Linear System Solver (LU-based) | **10x faster** for batched systems (fp16/bf16 supported) |
| **RMSNorm** | Fused RMS Normalization | **2.5x faster** than PyTorch |
| **AdamW** | Fused Optimizer Step | **2.9x faster** than torch.optim.AdamW |
| **GELU/SiLU** | Vectorized Activations | Up to **4x faster** |
| **EmbeddingBag** | Optimized Embedding Sum | **50-100x faster** (PyTorch falls back to CPU) |
| **SDPA** | Flash Attention v2 | Tiled attention with causal masking |

## Installation

```bash
pip install metalcore
```

## Quick Start

```python
import torch
import metalcore

device = torch.device("mps")

# SVD (Fast for large LLM matrices)
A = torch.randn(4096, 4096, device=device)
U, S, V = metalcore.svd(A)

# Batched QR (Fast for many small matrices)
B = torch.randn(500, 16, 16, device=device)
Q, R = metalcore.qr(B)

# Training Ops
from metalcore import MetalRMSNorm, MetalAdamW, metal_gelu

# RMSNorm (2.5x faster)
norm = MetalRMSNorm(512).to(device)
x = torch.randn(32, 512, device=device)
y = norm(x)

# AdamW (2.9x faster)
model = torch.nn.Linear(512, 256).to(device)
optimizer = MetalAdamW(model.parameters(), lr=1e-3)

# GELU/SiLU activations
y = metal_gelu(x)
```

### Transparent PyTorch Integration

Automatically accelerate any model using F.silu, F.gelu, or embedding_bag:

```python
import metalcore
metalcore.enable_pytorch_overrides()

# Now any model using F.silu/F.gelu on MPS uses metalcore
# embedding_bag avoids CPU fallback (50-100x faster)
model = AutoModelForCausalLM.from_pretrained("...", device_map="mps")
```

## Performance

See **[benchmarks.md](benchmarks.md)** for detailed benchmark results.

### Key Speedups
- **RMSNorm**: 2.5x faster (vectorized SIMD reductions)
- **AdamW**: 2.9x faster (fused single-kernel update)
- **SiLU**: 4x faster on small tensors
- **QR Batched**: 20x faster (parallel matrix processing)

## Requirements

- macOS 12.0+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.9 - 3.14
- PyTorch 2.0+

> **Note**: While M1 and M2 chips are supported, **M3/M4 chips are recommended** for best performance. Some BFloat16 kernels require Metal 3.1+ (macOS 14+). The library gracefully falls back to FP32 computation when native bf16 kernels are unavailable.

## Development

```bash
# Install package in editable mode
pip install -e packages/metalcore

# Run benchmarks
python benchmark.py --training  # RMSNorm + AdamW
python benchmark.py --activations  # GELU/SiLU
python benchmark.py --sdpa  # Attention
```

