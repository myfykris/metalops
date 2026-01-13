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
| **RMSNorm** | Fused RMS Normalization | **675x faster** than PyTorch |
| **AdamW** | Fused Optimizer Step | **2.4x faster** than torch.optim.AdamW |
| **SiLU** | Vectorized Activations | **1.1x faster** |
| **EmbeddingBag** | Optimized Embedding Sum | **6x faster** (PyTorch falls back to CPU) |
| **SDPA** | Flash Attention v2 | Tiled attention with causal masking |
| **RoPE** | Rotary Position Embedding | **3.4x faster** than Python |
| **INT4 GEMM** | GGML block_q4_0 (llama.cpp) | **4-15x overhead** vs FP16 (36x faster than naive) |
| **Fused MLP Bwd** | Full Backward Gradient Flow | **5-6x faster** than Autograd |
| **Fused Attn Bwd** | SDPA Grads -> RoPE -> LoRA -> QKV | **Parity with FP16** (Optimized BF16) |

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

Accelerate your existing models with a single line of code. `metalcore.enable_pytorch_overrides()` automatically redirects standard PyTorch functions to their optimized Metal equivalents.

```python
import metalcore
metalcore.enable_pytorch_overrides()

# The following operations are now accelerated on MPS:
# 1. F.silu: Reduced latency (1.1x faster)
# 2. F.embedding_bag(mode='sum'): avoids CPU fallback (6x faster)
# 3. torch.linalg.svd: For large matrices (>= 512x512)
# 4. torch.linalg.qr: For batched inputs (dim >= 3)
```

**Note**: `F.gelu` is *not* overridden as the native PyTorch MPS implementation is highly optimized.

#### Advanced Acceleration
Some optimizations require direct model patching due to PyTorch architecture:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("...", device_map="mps")

# 1. Patch RMSNorm (675x faster)
# Replaces LlamaRMSNorm, Qwen2RMSNorm, etc. with MetalRMSNorm
metalcore.patch_transformers_rmsnorm(model)

# 2. Patch RoPE (3.4x faster)
# Replaces apply_rotary_pos_emb with Metal kernel
metalcore.patch_transformers_rope(model)
```

## Usage

See **[usage.md](usage.md)** for comprehensive documentation and example workflows.

## Performance

See **[benchmarks.md](benchmarks.md)** for detailed benchmark results.

### Key Speedups
- **RMSNorm**: 675x faster (vectorized SIMD reductions)
- **AdamW**: 2.4x faster (fused single-kernel update)
- **RoPE**: 3.4x faster (Metal-accelerated)
- **EmbeddingBag**: 6x faster (GPU vs CPU fallback)
- **QR Batched**: up to 20x faster (parallel matrix processing)
- **SVD**: up to 25x faster (Jacobi method on GPU)

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
python benchmark.py --help # See all options

# Common benchmarks:
python benchmark.py --training       # RMSNorm + AdamW
python benchmark.py --activations    # GELU/SiLU
python benchmark.py --sdpa           # Flash Attention
python benchmark.py --models         # Full LLM Benchmarks

# Specific kernels:
python benchmark.py --svd            # Singular Value Decomposition
python benchmark.py --qr             # QR Decomposition
python benchmark.py --eigh           # Eigenvalue Decomposition
python benchmark.py --cholesky       # Cholesky Decomposition
python benchmark.py --solve          # Linear Solve
python benchmark.py --rmsnorm        # RMSNorm Only
python benchmark.py --adamw          # AdamW Only
python benchmark.py --softmax        # Fused Softmax
python benchmark.py --layernorm      # LayerNorm
python benchmark.py --embedding      # EmbeddingBag
python benchmark.py --scatter        # Scatter/Gather
python benchmark.py --lora           # LoRA Ops (SwiGLU, Linear)
python benchmark.py --rope           # Rotary Embeddings
python benchmark.py --fused_att_bwd  # Fused Attention Backward
python benchmark.py --fused_mlp_bwd  # Fused MLP Backward
```

