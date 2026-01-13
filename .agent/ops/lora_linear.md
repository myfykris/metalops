# LoRA Linear Forward

## Overview
Computes `y = Wx + scale * (B @ A @ x)` for LoRA-adapted linear layers.

## Why We Built It
- **LoRA fine-tuning**: Standard approach for efficient LLM adaptation
- **Fused dispatch**: Combines base linear + LoRA in single function call
- **Rank-efficient**: Low-rank matrices A (rank×in) and B (out×rank) add minimal params

## Formula
```
y = W @ x + alpha/rank * (B @ A @ x)
```

Where:
- W: Frozen base weights [out_features, in_features]
- A: LoRA down-projection [rank, in_features]  
- B: LoRA up-projection [out_features, rank]
- alpha: LoRA scaling factor

## Performance

| Config | Metal | PyTorch | Status |
|--------|-------|---------|--------|
| 128×4096→4096 r=16 | ~0.9ms | ~1.0ms | 🔵 Close |
| 128×4096→11008 r=8 | ~1.5ms | ~1.6ms | 🔵 Close |

## Usage
```python
import metalcore_backend as mc

# LoRA-adapted attention projection
y = mc.lora_linear_fwd(x, W_q, A_q, B_q, alpha/rank)
```

## Notes
- Uses MPS matmuls internally - already highly optimized
- Command buffer batching happens automatically via lazy execution
- Typical LoRA ranks: 8, 16, 32 (very small vs hidden dim 4096)
