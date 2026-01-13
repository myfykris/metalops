# KL Divergence

## Overview
Computes KL(P || Q) for distillation training.

## Why We Built It
- **GPU Memory Locality**: Keeps data on GPU throughout distillation pipeline
- **Top-K Memory Savings**: Process only top-k tokens, skip 99%+ of vocabulary
- **Pipeline Integration**: Works with metalcore cross_entropy, AdamW in same command buffer

## Functions
1. `kl_div_fwd(log_p, log_q)` - Full vocabulary KL divergence
2. `kl_div_topk_fwd(log_p, log_q, topk_indices, k)` - Top-K efficient variant

## Performance

| Op | Config | Metal | PyTorch | Status |
|----|--------|-------|---------|--------|
| Full KL | 128×32K | ~0.6ms | ~0.6ms | 🔵 Parity |
| Top-K KL | 128×32K, k=100 | ~0.3ms | ~0.3ms | 🔵 Parity |

## Top-K Memory Savings
For k=100 on vocab=32K: **99.7% of vocab not processed**

This is the key win - distillation doesn't need full vocabulary, just teacher's top tokens.

## When To Use
- Distillation training where student learns from teacher
- Memory-constrained workflows (Top-K reduces from 16MB to 50KB per sample)
- Full GPU pipeline with other metalcore ops

