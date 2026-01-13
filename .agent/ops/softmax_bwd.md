# Softmax Backward

## Overview
Computes gradient through softmax for attention backward pass.

## Why We Built It
- **GPU Memory Locality**: Keeps attention backward entirely GPU-resident
- **Avoids CPU transfers**: Critical for training pipelines where memory tagging switches cause latency
- **Attention backward**: Required for computing d_scores from d_attn_output

## Formula
```
d_logits = softmax_probs * (d_probs - sum(softmax_probs * d_probs))
```

## Performance

| Config | Metal | PyTorch | Status |
|--------|-------|---------|--------|
| [8, 32, 64] (B, H, L) | ~0.1ms | ~0.1ms | 🔵 Parity |

## Usage
```python
import metalcore_backend as mc

# In attention backward:
d_scores = mc.softmax_bwd(attention_probs, d_attention_weights)
```

## Notes
- Error verified: 2.98e-08 against PyTorch reference
- Supports FP32 and FP16 (half precision)
- Critical for keeping entire backward pass GPU-resident
