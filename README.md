# MetalOps

**MetalOps** is a collection of high-performance Metal-accelerated linear algebra operations for PyTorch on macOS (Apple Silicon).

## Packages

| Package | Description | Install |
|---------|-------------|---------|
| **[metalsvd](packages/metalsvd)** | GPU-Accelerated SVD | `pip install metalsvd` |
| **[metaleig](packages/metaleig)** | GPU-Accelerated Eigendecomposition | `pip install metaleig` |
| **[metalcore](packages/metalcore)** | GPU-Accelerated QR (batched) | `pip install metalcore` |

## Performance Highlights

See **[benchmarks.md](benchmarks.md)** for full results.

| Operation | Best Speedup | When GPU Wins |
|-----------|--------------|---------------|
| **SVD** | up to 25× faster | Tall matrices (M >> N), batched, LLM weights |
| **QR Batched** | up to 20× faster | Many small matrices (batch > 100) |
| **EIGH Batched** | up to 3.5× faster | Batched medium matrices (64×64+) |
| **LLM Weights** | up to 9× faster | Gemma, Llama, Mistral, Qwen, Phi weight SVD |

## Quick Start

```python
import torch
import metalsvd
import metaleig

device = torch.device("mps")

# SVD (up to 25x faster for LLM weight matrices)
A = torch.randn(64, 128, 128, device=device)
U, S, V = metalsvd.svd(A)

# Eigendecomposition (up to 3.5x faster for batched)
B = torch.randn(64, 64, 64, device=device)
B = B + B.transpose(-2, -1)  # Make symmetric
eigenvalues, eigenvectors = metaleig.eigh(B)
```

## Requirements

- macOS 12.0+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- PyTorch 2.0+

## Development

```bash
# Install all packages in editable mode
pip install -e packages/metalsvd -e packages/metaleig -e packages/metalcore

# Run benchmarks
python benchmark.py
```

## License

MIT
