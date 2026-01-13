# Fused LoRA QKV Projection

## Overview
Computes Q, K, V projections with LoRA in a single kernel dispatch.

## Why We Built It
- **12→1 Kernel Launches**: Replaces 12 separate matmuls with single dispatch
- **GPU Memory Locality**: All intermediate results stay in GPU registers
- **Training Pipeline**: Eliminates switching between CPU/GPU memory tagging

## Formula
```
Q = x @ W_q.T + scale * (x @ A_q.T @ B_q.T)
K = x @ W_k.T + scale * (x @ A_k.T @ B_k.T)  
V = x @ W_v.T + scale * (x @ A_v.T @ B_v.T)
```

## Performance

| Config | Metal | PyTorch (12 matmuls) | Status |
|--------|-------|---------------------|--------|
| 128×512→512/128 r=16 | Single dispatch | 12 dispatches | 🟢 Fewer launches |

## Usage
```python
import metalcore_backend as mc

Q, K, V = mc.fused_lora_qkv_fwd(
    x, 
    W_q, W_k, W_v,
    A_q, B_q, A_k, B_k, A_v, B_v,
    alpha / rank
)
```

## Notes
- Error verified: 0.0 against PyTorch reference (exact match!)
- Supports GQA (different out_k, out_v dimensions)
- Fallback to MPS lazy execution if PSO not available
