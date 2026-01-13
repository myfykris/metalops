# GELU / SiLU Activations

## Overview
Hardware-optimized activation functions for transformer MLP layers.

## Functions
1. `gelu_fwd(x)` - GELU forward (tanh approximation)
2. `gelu_bwd(dY, x)` - GELU backward
3. `silu_fwd(x)` - SiLU/Swish forward: x * sigmoid(x)
4. `silu_bwd(dY, x)` - SiLU backward

## Performance

| Op | Config | Metal | PyTorch | Status |
|----|--------|-------|---------|--------|
| GELU | 1024×4096 | ~0.2ms | ~0.2ms | 🔵 Close |
| SiLU | 1024×4096 | ~0.2ms | ~0.2ms | 🔵 Close |

## Notes
- PyTorch MPS already has optimized activation kernels
- Metal kernels provided for custom dispatch patterns
- Vectorized versions (vec4) for aligned inputs
- Half precision (FP16) and BFloat16 supported
