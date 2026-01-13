# RMSNorm

## Overview
Root Mean Square Layer Normalization used in Llama, Mistral, and most modern LLMs.

## Why We Built It
- **LLM standard**: RMSNorm replaced LayerNorm in Llama and successors
- **Simpler than LayerNorm**: No mean subtraction, just scale by RMS
- **Fused variants**: Combined with residual add for memory efficiency

## Formula
```
y = x * (1 / sqrt(mean(x^2) + eps)) * gamma
```

## Functions
1. `rmsnorm_fwd(x, gamma, eps)` - Standard RMSNorm
2. `fused_add_rmsnorm(x, residual, gamma, eps)` - Fused residual + RMSNorm

## Performance

| Op | Config | Metal | CPU | Speedup |
|----|--------|-------|-----|---------|
| RMSNorm | 1024x4096 | ~0.2ms | ~0.5ms | **2.7x** ✓ |
| Fused Add+RMSNorm | 1024x4096 | ~0.3ms | ~0.8ms | **2.7x** ✓ |

## Usage
```python
from metalcore import RMSNorm

norm = RMSNorm(hidden_size, eps=1e-6)
y = norm(x)

# Or with fused residual:
from metalcore import fused_add_rmsnorm
y = fused_add_rmsnorm(x, residual, gamma, eps)
```

## Notes
- Fused add+RMSNorm saves one memory roundtrip
- Uses Welford's algorithm for numerical stability
- Half precision version available for FP16 training
