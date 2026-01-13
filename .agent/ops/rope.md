# RoPE (Rotary Position Embeddings)

## Overview
Position encoding used in Llama, Mistral, and most modern LLMs.

## Why We Built It
- **Standard position encoding**: RoPE is the default for Llama-family models
- **Fused variants**: Combined with attention for memory efficiency
- **Q/K specific**: Rotation applied to queries and keys before attention

## Formula
```
# For each position i and dimension pair (2j, 2j+1):
x_rot[2j] = x[2j] * cos(i*theta_j) - x[2j+1] * sin(i*theta_j)
x_rot[2j+1] = x[2j] * sin(i*theta_j) + x[2j+1] * cos(i*theta_j)
```

## Functions
1. `rope_fwd(x, cos, sin)` - Apply RoPE to tensor
2. `rope_fwd_qk(q, k, cos, sin)` - Fused RoPE for both Q and K
3. `rope_bwd(grad, cos, sin)` - Backward pass
4. `rope_sdpa_fwd(q, k, v, cos, sin)` - Fused RoPE + attention

## Performance

| Op | Config | Metal | PyTorch | Speedup |
|----|--------|-------|---------|---------|
| RoPE Q+K | seq=1024, d=128 | ~0.1ms | ~0.2ms | **2x** ✓ |

## Usage
```python
from metalcore import apply_rope, apply_rope_qk

# Single tensor
q_rot = apply_rope(q, cos, sin)

# Fused Q and K
q_rot, k_rot = apply_rope_qk(q, k, cos, sin)
```

## Notes
- Fused rope_sdpa_fwd is optimal for inference (no intermediate tensors)
- Supports Llama-style interleaved dimensions
- Half precision supported for FP16 training
