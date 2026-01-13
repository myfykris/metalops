# AdamW Optimizer Step

## Overview
Fused AdamW update with ILP4 (instruction-level parallelism) for 4x throughput.

## Why We Built It
- **Training bottleneck**: Optimizer step runs on every backward pass
- **Memory bandwidth bound**: Standard implementation limited by memory access
- **ILP optimization**: Process 4 elements per thread for better instruction utilization

## Performance

| Config | Metal | CPU | Speedup |
|--------|-------|-----|---------|
| 1M params | ~0.1ms | ~0.9ms | **9.4x** ✓ |
| 10M params | ~0.5ms | ~4.5ms | **9x** ✓ |
| 16M params | ~0.8ms | ~7ms | **8.7x** ✓ |

## Algorithm
```python
# Standard AdamW per-parameter:
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
```

## ILP4 Optimization
Kernel processes 4 elements per thread:
1. Load 4 grads, 4 ms, 4 vs, 4 params
2. Compute all 4 updates in parallel
3. Write 4 updated values

## Usage
```python
from metalcore.optim import MetalAdamW

optimizer = MetalAdamW(model.parameters(), lr=1e-3)
# ... training loop
optimizer.step()  # Uses fused Metal kernel
```

## Notes
- **9x speedup** over CPU PyTorch AdamW
- Critical for LoRA training - optimizer runs every step
- Automatically dispatches vectorized kernel for large tensors
