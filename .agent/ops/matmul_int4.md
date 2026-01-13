# INT4 Quantized Matrix Multiplication

## Overview
Matrix multiplication with INT4 quantized weights for efficient LLM inference.

## Why We Built It
- **4x memory reduction**: INT4 vs FP16 means 4x more model fits in memory
- **Large models on Mac**: Run 70B+ models on unified memory
- **Fused activation**: Combined with SiLU/GELU for MLP layers

## Formula
```
y = x @ dequant(W_int4)

dequant(W) = (W - zero) * scale  # Per-group dequantization
```

## Functions
1. `matmul_int4(x, W_packed, scales, zeros, group_size)` - Base matmul
2. `matmul_int4_silu(x, W_packed, scales, zeros)` - Fused with SiLU
3. `matmul_int4_gelu(x, W_packed, scales, zeros)` - Fused with GELU

## Performance

| Config | Metal | CPU FP32 | Speedup |
|--------|-------|----------|---------|
| 4096x4096 | ~2ms | ~4ms | **2x** ✓ |
| 4096x11008 | ~5ms | ~10ms | **2x** ✓ |

## Notes
- Uses standard GPTQ/AWQ format (separate scales/zeros)
- Group size typically 128 for good accuracy
- Dequantization is compute-bound, not memory-bound
- Vectorized kernel processes 4 output columns per thread
