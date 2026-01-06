# Metalcore - Consolidated Metal Ops for PyTorch

## Current Status
metalcore v0.1.7 consolidates all Metal-accelerated operations:
- **LA Ops**: SVD, Eigh, QR, Cholesky, Solve, TRSM
- **Training Ops**: RMSNorm, AdamW, GELU, SiLU, SDPA

## Key Files
- `packages/metalcore/native/core_kernels.metal` - LA kernels
- `packages/metalcore/native/training_kernels.metal` - RMSNorm, AdamW
- `packages/metalcore/native/activation_kernels.metal` - GELU, SiLU
- `packages/metalcore/native/sdpa_kernels.metal` - Flash Attention v2
- `packages/metalcore/native/core_mps.mm` - C++ dispatch + PYBIND11

## Usage
```python
import metalcore as mc

# LA Ops
U, S, V = mc.svd(A)
Q, R = mc.qr(A)
L = mc.cholesky(A)

# Training Ops
from metalcore import MetalRMSNorm, MetalAdamW, metal_gelu

norm = MetalRMSNorm(512).to('mps')
y = norm(x)

optimizer = MetalAdamW(model.parameters(), lr=1e-3)
optimizer.step()

y = metal_gelu(x)
```

## Benchmark
```bash
python benchmark.py --training      # RMSNorm + AdamW
python benchmark.py --activations   # GELU + SiLU
python benchmark.py --sdpa          # Flash Attention
```
