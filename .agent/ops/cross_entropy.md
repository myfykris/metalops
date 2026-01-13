# Cross-Entropy Loss

## Overview
Fused log-softmax + NLL loss for language model training.

## Why We Built It
- **GPU Memory Locality**: Keeps data GPU-resident for full training pipeline
- **Avoids CPU transfers**: PyTorch may invoke CPU for certain loss operations
- **Pipeline integration**: Seamless integration with other metalcore ops in command buffers

## Performance

| Config | Metal | PyTorch | Status |
|--------|-------|---------|--------|
| 128×32K (Llama vocab) | ~0.6ms | ~0.6ms | 🔵 Parity |
| 512×128K (Llama-3 vocab) | ~4ms | ~4ms | 🔵 Parity |

**Note**: Performance matches PyTorch MPS. The value is GPU memory locality, not raw speed.

## Algorithm
```
1. Find max(logits) for numerical stability
2. Compute logsumexp = max + log(sum(exp(logits - max)))
3. loss = logsumexp - logits[target]
```

## When To Use
- Training pipelines where data must stay GPU-resident
- Combined with other metalcore ops (RMSNorm, AdamW) in same command buffer
- Memory-constrained workflows where CPU<->GPU transfers are expensive

