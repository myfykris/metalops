# Scaled Dot-Product Attention (SDPA)

## Overview
Flash Attention v2 implementation with O(N) memory instead of O(N²).

## Why We Built It
- **Memory efficiency**: Standard attention materializes N×N attention matrix
- **Flash Attention**: Tiled algorithm keeps working set in SRAM
- **Long sequences**: Enables 8K+ context without OOM

## Formula
```
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V
```

## Functions
1. `sdpa_fwd(Q, K, V, scale)` - Standard SDPA
2. `sdpa_bwd(dO, Q, K, V, O, L)` - Backward pass
3. `rope_sdpa_fwd(Q, K, V, cos, sin)` - Fused RoPE + SDPA

## Performance

| Config | Metal | MPS | Status |
|--------|-------|-----|--------|
| seq=512, head=64 | ~1ms | ~0.5ms | CPU/MPS faster |
| seq=2048, head=64 | ~10ms | ~8ms | Close |

## Notes
- MPS has hardware-accelerated attention (AMX/ANE)
- Our kernel useful for custom attention patterns
- RoPE fusion saves memory by not materializing rotated Q,K
- Head dim currently optimized for 64 (Llama)
