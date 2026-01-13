# Performance & Benchmarks

## Benchmark Commands
```bash
python benchmark.py --quick          # All benchmarks
python benchmark.py --training       # RMSNorm + AdamW
python benchmark.py --activations    # GELU + SiLU
python benchmark.py --sdpa           # Scaled Dot Product Attention
```

## Training Ops (v0.1.8)

| Operation | Size | Metal vs Torch | Status |
|-----------|------|----------------|--------|
| RMSNorm | 4096×4096 | **2.5x faster** | 💚 |
| AdamW | 16M params | **2.9x faster** | 💚 |
| SiLU | 256×1024 | **4x faster** | 💚 |
| GELU | 1024×4096 | ~1x (parity) | ⚪ |
| SDPA | N=256 | 14x slower | 🔴 |
| CrossEntropy | Lite | **18x faster** (fp32) | 💚 |
| KL Div | Lite | **5x faster** (fp32) | 💚 |
| SwiGLU MLP | Lite | ~1.03x (FP32, crash fix verif) | 🔵 |

**Note**: SDPA is slower than PyTorch's native implementation (which uses Apple's MPS optimizations). Use `enable_metal_sdpa()` only if you need custom backward pass behavior.

## Linear Algebra Ops

| Operation | Size | GPU vs CPU |
|-----------|------|------------|
| **Gemma-7B MLP SVD** | 3072×24576 | **25x faster** |
| **Cholesky batched** | 500×16×16 | **33x faster** |
| **QR batched** | 1000×16×16 | **14x faster** |
| **Llama-3-8B SVD** | 4096×14336 | **5.9x faster** |

## When to Use Metal

| ✅ Use Metal | ❌ Use CPU/Native |
|--------------|-------------------|
| Batched QR/Cholesky/SVD | Single small matrices |
| Large LLM weight matrices | SDPA (use native F.sdpa) |
| RMSNorm in training | Sequential operations |
| AdamW optimizer step | Small batch operations |
