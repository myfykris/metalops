# LoRA Add

## Overview
Computes `y = base + scale * lora` for combining base output with LoRA adaptation.

## Why We Built It
- **Final LoRA step**: Combines frozen base layer output with LoRA delta
- **Single kernel**: Fused add + scale in one operation
- **Memory efficient**: No intermediate tensor for scaled lora

## Formula
```
y = base + alpha * lora
```

## Performance

| Config | Metal | PyTorch | Status |
|--------|-------|---------|--------|
| 128×4096 | ~0.2ms | ~0.2ms | 🔵 Close |

## Usage
```python
import metalcore_backend as mc

# Apply LoRA delta to base output
base = x @ W.T
lora = x @ A.T @ B.T
y = mc.lora_add_fwd(base, lora, alpha/rank)
```

## Notes
- Simple elementwise fusion
- PyTorch's lazy execution already fuses this pattern
- Provided for API consistency with other metalcore LoRA ops
