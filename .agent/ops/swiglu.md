# SwiGLU Activation

## Overview
Computes `silu(gate) * up` for SwiGLU MLPs used in Llama/Mistral.

## Why We Built It
- **LLM MLP standard**: SwiGLU is the activation in Llama, Mistral, Qwen, Gemma
- **Fused elementwise**: Combines silu and multiply into single operation
- **Memory efficient**: Avoids intermediate tensor for silu output

## Formula
```
swiglu(gate, up) = silu(gate) * up
                 = (gate * sigmoid(gate)) * up
```

## Performance

| Config | Metal | PyTorch | Status |
|--------|-------|---------|--------|
| 128×11008 (Llama-7B) | ~0.5ms | ~0.4ms | 🔵 Close |
| 256×14336 (Llama-3) | ~1ms | ~0.9ms | 🔵 Close |

## Usage
```python
import metalcore_backend as mc

# In SwiGLU MLP forward:
gate = x @ W_gate.T
up = x @ W_up.T
hidden = mc.swiglu_fwd(gate, up)  # Fused silu(gate) * up
out = hidden @ W_down.T
```

## Notes
- PyTorch's MPS backend already fuses elementwise ops via lazy execution
- Metal kernel provides explicit control for custom dispatch patterns
- Main matmuls (gate_proj, up_proj, down_proj) still use MPS for optimal performance
