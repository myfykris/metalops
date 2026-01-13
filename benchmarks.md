# Metalops Benchmark Results

*Generated: 2026-01-13 15:26:26*

**Legend:** 💚 GPU wins big (>3x) | 🟢 GPU wins | 🔵 Close | ⚪ CPU wins | 🟠 CPU wins big (>3x)

## QR Batched (metalcore) ⭐ GPU WINS

*Batched QR - GPU processes all matrices in parallel, single dispatch*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 50×8×8 | Tiny 8x8 | 709.7µs | 2.3ms | 0.31x | 🟢 | ✓ 6e-07 |
| 100×8×8 | Batch 100 tiny | 701.3µs | 4.6ms | 0.15x | 💚 | ✓ 4e-07 |
| 500×8×8 | Batch 500 tiny | 714.9µs | 22.4ms | 0.03x | 💚 | ✓ 6e-07 |
| 50×16×16 | ML mini-batch 16 | 601.5µs | 2.3ms | 0.26x | 💚 | ✓ 1e-06 |
| 100×16×16 | Batch 100 16x16 | 558.9µs | 4.7ms | 0.12x | 💚 | ✓ 1e-06 |
| 200×16×16 | Batch 200 16x16 | 610.8µs | 9.4ms | 0.07x | 💚 | ✓ 6e-07 |
| 500×16×16 | Batch 500 16x16 | 594.0µs | 23.6ms | 0.03x | 💚 | ✓ 1e-06 |
| 1000×16×16 | Batch 1000 16x16 | 722.1µs | 47.0ms | 0.02x | 💚 | ✓ 9e-07 |
| 50×32×32 | ML mini-batch 32 | 638.8µs | 2.7ms | 0.24x | 💚 | ✓ 1e-06 |
| 100×32×32 | Batch 100 32x32 | 581.3µs | 5.4ms | 0.11x | 💚 | ✓ 1e-06 |
| 200×32×32 | Batch 200 32x32 | 531.5µs | 10.8ms | 0.05x | 💚 | ✓ 1e-06 |
| 500×32×32 | Batch 500 32x32 | 807.6µs | 27.0ms | 0.03x | 💚 | ✓ 1e-06 |
| 50×48×48 | Batch 50 48x48 | 32.2ms | 3.1ms | 10.30x | 🟠 | ✓ 1e-06 |
| 100×48×48 | Batch 100 48x48 | 63.8ms | 6.5ms | 9.83x | 🟠 | ✓ 1e-06 |
| 100×64×32 | Tall batch | 62.9ms | 5.8ms | 10.80x | 🟠 | ✓ 1e-06 |
| 100×32×64 | Wide batch | 899.4µs | 5.6ms | 0.16x | 💚 | ✗ nan |
| 200×64×32 | Large tall batch | 125.3ms | 12.2ms | 10.30x | 🟠 | ✓ 1e-06 |

## Cholesky (metalcore) ⭐ GPU WINS

*Batched Cholesky decomposition with MAGMA-style shared memory*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 100×16×16 | Tiny batched | 399.0µs | 4.4ms | 0.09x | 💚 | ✓ 6e-06 |
| 500×16×16 | Large batch tiny | 573.5µs | 21.4ms | 0.03x | 💚 | ✓ 4e-06 |
| 100×32×32 | Small batched | 588.0µs | 4.3ms | 0.14x | 💚 | ✓ 8e-06 |
| 200×48×48 | Medium batched | 908.2µs | 8.7ms | 0.10x | 💚 | ✓ 2e-05 |
| 100×64×64 | Larger batched | 1.3ms | 4.5ms | 0.29x | 💚 | ✓ 2e-05 |

## RMSNorm (metalcore) ⭐ GPU WINS

*Fused RMSNorm kernel vs torch.nn.RMSNorm*

| Shape | Config | Metal | CPU | Ratio | Status |
|---|---|---|---|---|---|
| 32x4096 | Fwd+Bwd fp32 | 465.1µs | 506.7µs | 0.92x | 🔵 |
| 1x4096 | Fwd+Bwd fp32 | 424.3µs | 449.5µs | 0.94x | 🔵 |
| 1024x1024 | Fwd+Bwd fp32 | 792.6µs | 909.2µs | 0.87x | 🔵 |
| 4096x4096 | Fwd+Bwd fp32 | 5.9ms | 8.6ms | 0.69x | 🟢 |

## AdamW (metalcore) ⭐ GPU WINS

*Fused AdamW optimizer step vs torch.optim.AdamW*

| Params | Size | Metal | CPU | Ratio | Status |
|---|---|---|---|---|---|
| 1M Params | N=1048576 fp32 | 330.9µs | 525.3µs | 0.63x | 🟢 |
| 10M Params | N=10485760 fp32 | 1.1ms | 2.9ms | 0.38x | 🟢 |
| 16M Params | N=16777216 fp32 | 1.6ms | 4.5ms | 0.36x | 🟢 |

## Pipeline Operations ⭐ GPU WINS (No Transfer)

*Chained operations where data stays on GPU - avoids costly memory transfers*

| Pipeline | Shape | GPU | Comparison | Ratio | Status |
|---|---|---|---|---|---|
| QR -> QR -> QR | 200×32×32 | 904.2µs | 32.0ms | 0.03x | 💚 |
| SVD -> truncate (PCA) | 50×128×64 | 11.1ms | 16.0ms | 0.69x | 🟢 |
| QR -> matmul (ML) | 1000×16×16 | 1.3ms | 51.5ms | 0.03x | 💚 |
| Fast+Slow+Fast (GPU all) | 200×32×32 | 2.0ms | 1.2ms | 1.65x vs hybrid | 🟠 |
| Fast+Slow+Fast (vs CPU) | 200×32×32 | 2.0ms | 12.0ms | 0.17x vs CPU | 💚 |

## LLM: Llama

*SVD performance on Llama weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 4096×4096 | Attention (7B) | 2.85s | 5.00s | 0.57x | 🟢 | ✓ 3e-05 |
| 4096×11008 | MLP up (7B) | 2.92s | 10.00s | 0.29x | 💚 | ✓ 3e-05 |
| 8192×8192 | Attention (70B) | 21.90s | 36.95s | 0.59x | 🟢 | ~ 1e-04 |

## LLM: Mistral

*SVD performance on Mistral weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 4096×4096 | Attention | 2.89s | 5.09s | 0.57x | 🟢 | ✓ 3e-05 |
| 4096×14336 | MLP up | 2.99s | 11.93s | 0.25x | 💚 | ✓ 3e-05 |

## LLM: Qwen

*SVD performance on Qwen weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 4096×4096 | Attention | 2.89s | 5.10s | 0.57x | 🟢 | ✓ 3e-05 |
| 4096×11008 | MLP up | 2.99s | 10.17s | 0.29x | 💚 | ✓ 3e-05 |

## LLM: Gemma

*SVD performance on Gemma weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 3072×3072 | Attention | 927.5ms | 1.60s | 0.58x | 🟢 | ✓ 2e-05 |
| 3072×24576 | MLP up | 997.8ms | 9.25s | 0.11x | 💚 | ✓ 3e-05 |

## LLM: Phi

*SVD performance on Phi weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 3072×3072 | Attention | 919.7ms | 1.61s | 0.57x | 🟢 | ✓ 2e-05 |
| 3072×8192 | MLP up | 937.5ms | 3.74s | 0.25x | 💚 | ✓ 2e-05 |

## Activations (metalcore)

*GELU/SiLU activations with float4 vectorization*

| Op | Shape | Metal | Torch | Ratio | Status |
|---|---|---|---|---|---|
| GELU Small (256x1024) | 256x1024 fp32 | 182.0µs | 206.5µs | 0.88x | 🔵 |
| SiLU Small (256x1024) | 256x1024 fp32 | 187.2µs | 203.6µs | 0.92x | 🔵 |
| GELU Medium (1024x4096) | 1024x4096 fp32 | 286.6µs | 272.5µs | 1.05x | 🔵 |
| SiLU Medium (1024x4096) | 1024x4096 fp32 | 261.4µs | 275.3µs | 0.95x | 🔵 |
| GELU Large (4096x4096) | 4096x4096 fp32 | 643.6µs | 632.8µs | 1.02x | 🔵 |
| SiLU Large (4096x4096) | 4096x4096 fp32 | 631.2µs | 650.1µs | 0.97x | 🔵 |

## Eigendecomposition (metaleig)

*Symmetric eigenvalue decomposition*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 32×32 | Tiny | 837.3µs | 49.2µs | 17.01x | 🟠 | ✓ 3e-06 |
| 64×64 | Small | 776.8µs | 176.9µs | 4.39x | 🟠 | ✓ 2e-06 |
| 128×128 | Medium | 1.2ms | 624.8µs | 1.96x | ⚪ | ✓ 4e-06 |
| 256×256 | Large | 4.0ms | 2.7ms | 1.45x | ⚪ | ✓ 7e-06 |
| 512×512 | Very large | 13.8ms | 12.1ms | 1.14x | 🔵 | ✓ 1e-05 |
| 1024×1024 | Huge | 67.3ms | 64.9ms | 1.04x | 🔵 | ✓ 3e-05 |
| 100×32×32 | Batch 100 tiny | 5.7ms | 4.7ms | 1.22x | 🔵 | ✗ 2e+00 |
| 50×64×64 | Batch 50 small | 4.5ms | 8.9ms | 0.51x | 🟢 | ~ 2e-03 |
| 100×64×64 | Batch 100 small | 7.9ms | 17.8ms | 0.45x | 🟢 | ~ 2e-04 |
| 200×64×64 | Batch 200 small | 14.4ms | 35.4ms | 0.41x | 🟢 | ~ 3e-04 |
| 20×128×128 | Batch 20 medium | 5.1ms | 11.3ms | 0.45x | 🟢 | ~ 2e-03 |
| 50×128×128 | Batch 50 medium | 9.2ms | 28.3ms | 0.33x | 🟢 | ~ 1e-03 |
| 10×256×256 | Batch 10 large | 28.0ms | 26.2ms | 1.07x | 🔵 | ✓ 7e-06 |

## SVD (metalsvd)

*Singular Value Decomposition using Jacobi algorithm on GPU*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 32×32 | Tiny | 945.1µs | 77.2µs | 12.24x | 🟠 | ✓ 2e-06 |
| 64×64 | Small square | 1.0ms | 230.1µs | 4.56x | 🟠 | ✓ 4e-06 |
| 128×128 | Medium square | 1.6ms | 781.4µs | 2.07x | ⚪ | ✓ 5e-06 |
| 256×256 | Large square | 5.3ms | 4.0ms | 1.31x | ⚪ | ✓ 8e-06 |
| 512×512 | Very large | 14.4ms | 17.5ms | 0.82x | 🔵 | ✓ 9e-06 |
| 1024×1024 | Huge square | 67.1ms | 92.3ms | 0.73x | 🔵 | ✓ 1e-05 |
| 2048×2048 | Massive square | 338.1ms | 515.2ms | 0.66x | 🟢 | ✓ 2e-05 |
| 256×128 | Tall 2:1 | 2.3ms | 1.7ms | 1.33x | ⚪ | ✓ 5e-06 |
| 512×256 | Tall 2:1 large | 7.4ms | 8.5ms | 0.87x | 🔵 | ✓ 9e-06 |
| 1024×512 | Tall matrix | 14.6ms | 39.1ms | 0.37x | 🟢 | ✓ 1e-05 |
| 2048×512 | Very tall | 14.3ms | 105.0ms | 0.14x | 💚 | ✓ 9e-06 |
| 128×256 | Wide 1:2 | 2.3ms | 1.8ms | 1.27x | 🔵 | ✓ 6e-06 |
| 4096×4096 | Llama-7B attn (4096x4096) | 2.83s | 4.95s | 0.57x | 🟢 | ~ 3e-04 |
| 4096×11008 | Llama-2-7B MLP (4096x11008) | 2.89s | 12.18s | 0.24x | 💚 | ✓ 3e-05 |
| 4096×14336 | Llama-3-8B MLP (4096x14336) | 2.96s | 17.14s | 0.17x | 💚 | ✓ 3e-05 |
| 8192×8192 | Llama-70B attn (8192x8192) | 22.28s | 37.94s | 0.59x | 🟢 | ~ 2e-04 |
| 4096×14336 | Mistral-7B MLP (4096x14336) | 2.94s | 17.14s | 0.17x | 💚 | ✓ 4e-05 |
| 4096×11008 | Qwen-7B MLP (4096x11008) | 2.94s | 11.73s | 0.25x | 💚 | ✓ 3e-05 |
| 3072×24576 | Gemma-7B MLP (3072x24576) | 1.00s | 28.22s | 0.04x | 💚 | ✓ 2e-05 |
| 3072×8192 | Phi-3-mini MLP (3072x8192) | 933.2ms | 5.29s | 0.18x | 💚 | ✓ 2e-05 |
| 100×32×32 | Batch 100 tiny | 5.5ms | 7.1ms | 0.77x | 🔵 | ✓ 3e-06 |
| 50×64×64 | Batch 50 small | 5.7ms | 11.7ms | 0.49x | 🟢 | ✓ 5e-06 |
| 100×64×64 | Batch 100 small | 8.6ms | 23.3ms | 0.37x | 🟢 | ✓ 6e-06 |
| 200×64×64 | Batch 200 small | 14.6ms | 46.5ms | 0.31x | 🟢 | ✓ 5e-06 |
| 20×128×128 | Batch 20 medium | 7.4ms | 16.1ms | 0.46x | 🟢 | ✓ 8e-06 |
| 50×128×128 | Batch 50 medium | 10.5ms | 40.2ms | 0.26x | 💚 | ✓ 9e-06 |
| 10×256×256 | Batch 10 large | 47.0ms | 41.9ms | 1.12x | 🔵 | ✓ 2e-05 |
| 5×512×512 | Batch 5 huge | 61.3ms | 89.3ms | 0.69x | 🟢 | ✓ 1e-05 |

## QR Single Matrix (metalcore)

*Single matrix QR - CPU typically wins due to sequential dependencies*

| Shape | Config | Metal | CPU | Ratio | Status | Recon | Ortho |
|---|---|---|---|---|---|---|---|
| 16×16 | Tiny | 1.1ms | 40.7µs | 27.74x | 🟠 | ✓ 7e-07 | ✓ 2e-07 |
| 32×32 | Small 32 | 576.2µs | 55.0µs | 10.48x | 🟠 | ✓ 1e-06 | ✓ 3e-07 |
| 64×64 | Small 64 | 791.4µs | 89.9µs | 8.80x | 🟠 | ✓ 1e-06 | ✓ 6e-07 |
| 128×128 | Medium | 942.3µs | 227.1µs | 4.15x | 🟠 | ✓ 4e-06 | ✓ 6e-07 |
| 256×256 | Large 256 | 1.8ms | 995.6µs | 1.77x | ⚪ | ✓ 5e-06 | ✓ 8e-07 |
| 512×512 | Large 512 | 5.3ms | 4.1ms | 1.30x | 🔵 | ✓ 7e-06 | ✓ 1e-06 |
| 1024×1024 | Huge 1024 | 26.7ms | 24.5ms | 1.09x | 🔵 | ✓ 1e-05 | ✓ 2e-06 |
| 256×64 | Tall 4:1 | 1.5ms | 223.4µs | 6.80x | 🟠 | ✓ 2e-06 | ✓ 7e-07 |
| 256×128 | Tall 2:1 | 1.2ms | 538.6µs | 2.25x | ⚪ | ✓ 3e-06 | ✓ 7e-07 |
| 512×128 | Tall 4:1 large | 1.7ms | 1.0ms | 1.68x | ⚪ | ✓ 6e-06 | ✓ 8e-07 |
| 512×256 | Tall 2:1 large | 3.0ms | 1.8ms | 1.63x | ⚪ | ✓ 6e-06 | ✓ 1e-06 |
| 1000×200 | Tall 5:1 | 3.8ms | 2.5ms | 1.51x | ⚪ | ✓ 9e-06 | ✓ 1e-06 |
| 2000×500 | Huge tall | 15.4ms | 13.7ms | 1.13x | 🔵 | ✓ 1e-05 | ✓ 2e-06 |
| 4000×1000 | Massive | 79.7ms | 77.4ms | 1.03x | 🔵 | ✓ 2e-05 | ✓ 3e-06 |

## SDPA (metalcore)

*Scaled Dot Product Attention with Flash Attention v2 tiling*

| Config | Shape | Metal | Torch | Ratio | Status | Error |
|---|---|---|---|---|---|---|
| Small (B=2, H=8, N=64, D=64) | B=2, H=8, N=64, D=64 fp32 | 233.0µs | 302.0µs | 0.77x | 🔵 | ✓ 7e-07 |
| Small (B=2, H=8, N=64, D=64) (causal) | B=2, H=8, N=64, D=64 fp32 | 671.4µs | 286.6µs | 2.34x | ⚪ | ✓ 3e-06 |
| Medium (B=2, H=8, N=256, D=64) | B=2, H=8, N=256, D=64 fp32 | 929.4µs | 394.6µs | 2.36x | ⚪ | ✗ 1e-01 |
| Medium (B=2, H=8, N=256, D=64) (causal) | B=2, H=8, N=256, D=64 fp32 | 3.4ms | 406.7µs | 8.42x | 🟠 | ✓ 3e-06 |
| Large (B=1, H=8, N=512, D=64) | B=1, H=8, N=512, D=64 fp32 | 1.7ms | 454.8µs | 3.71x | 🟠 | ✗ 4e-02 |
| Large (B=1, H=8, N=512, D=64) (causal) | B=1, H=8, N=512, D=64 fp32 | 5.8ms | 479.1µs | 12.17x | 🟠 | ✓ 3e-06 |

## Usage Recommendations

| Operation | When to Use Metal | When to Use CPU |
|---|---|---|
| EIGH | Batched symmetric matrices | Single large matrices |
| Pipeline | Keep data on GPU to avoid transfer cost | Single ops on CPU-resident data |
| QR (batched) | Many small matrices (10x speedup!) | Few matrices |
| QR (single) | — | Always (sequential dependencies) |
| SVD | Batched small/medium matrices | Single large matrices |

## Linear Solve (metalcore)

*Batched linear system solve using QR + TRSM*

| Shape | Config | Metal | CPU | Ratio | Status | Residual |
|---|---|---|---|---|---|---|
| 100×16×16 | Tiny batched fp32 | 313.3µs | 1.3ms | 0.25x | 💚 | ✓ 4e-05 |
| 500×16×16 | Large batch tiny fp32 | 291.1µs | 6.1ms | 0.05x | 💚 | ✗ 1e-02 |
| 100×32×32 | Small batched fp32 | 856.9µs | 1.6ms | 0.52x | 🟢 | ~ 2e-03 |
| 200×48×48 | Medium batched fp32 | 878.1µs | 3.8ms | 0.23x | 💚 | ~ 2e-04 |

## Fused Softmax (metalcore)

*Online softmax algorithm with SIMD reductions*

| Config | Shape | Metal | Torch | Ratio | Status | Error |
|---|---|---|---|---|---|---|
| Small | 32x1024 fp32 | 165.1µs | 186.4µs | 0.89x | 🔵 | 3.73e-09 |
| Medium | 64x4096 fp32 | 159.8µs | 201.8µs | 0.79x | 🔵 | 9.31e-10 |
| Large | 128x8192 fp32 | 182.1µs | 220.1µs | 0.83x | 🔵 | 6.98e-10 |
| Very Large | 256x16384 fp32 | 259.9µs | 290.1µs | 0.90x | 🔵 | 9.31e-10 |
| Huge | 512x32768 fp32 | 950.9µs | 831.6µs | 1.14x | 🔵 | 2.33e-10 |
| LLM Vocab | 32x32000 fp32 | 216.9µs | 248.9µs | 0.87x | 🔵 | 2.33e-10 |
| LLM Vocab Large | 128x128000 fp32 | 937.1µs | 824.8µs | 1.14x | 🔵 | 1.16e-10 |

## LayerNorm (metalcore)

*Welford's algorithm for fused mean/variance*

| Config | Shape | Metal | Torch | Ratio | Status | Error |
|---|---|---|---|---|---|---|
| Tiny | 32x512 fp32 | 171.5µs | 169.6µs | 1.01x | 🔵 | 4.77e-07 |
| Small | 64x1024 fp32 | 163.9µs | 169.9µs | 0.96x | 🔵 | 4.77e-07 |
| Llama-7B | 32x4096 fp32 | 175.7µs | 170.2µs | 1.03x | 🔵 | 4.77e-07 |
| Llama-13B | 32x5120 fp32 | 173.6µs | 170.8µs | 1.02x | 🔵 | 4.77e-07 |
| Llama-70B | 16x8192 fp32 | 179.0µs | 173.1µs | 1.03x | 🔵 | 4.77e-07 |
| Large Batch | 256x4096 fp32 | 188.4µs | 194.2µs | 0.97x | 🔵 | 9.54e-07 |
| Huge Batch | 1024x4096 fp32 | 242.2µs | 268.6µs | 0.90x | 🔵 | 4.77e-07 |

## Embedding Bag (metalcore)

*Coalesced reads for embedding lookups and aggregation*

| Config | Shape | Metal | Torch | Ratio | Status |
|---|---|---|---|---|---|
| Small Vocab | 10000x64, B=32 | 227.0µs | 1.3ms | 0.17x | 💚 |
| Medium Vocab | 50000x128, B=64 | 232.7µs | 1.7ms | 0.14x | 💚 |
| Large Vocab | 100000x256, B=32 | 274.4µs | 6.5ms | 0.04x | 💚 |
| LLM Embedding | 32000x4096, B=16 | 348.1µs | 20.6ms | 0.02x | 💚 |
| Huge Vocab | 250000x512, B=16 | 246.1µs | 19.8ms | 0.01x | 💚 |

## Scatter/Gather (metalcore)

*Atomic scatter_add and vectorized gather operations*

| Op | Shape | Metal | Torch | Ratio | Status |
|---|---|---|---|---|---|
| Gather Small | src=10000, idx=1000 | 204.1µs | 223.9µs | 0.91x | 🔵 |
| Gather Medium | src=100000, idx=10000 | 210.6µs | 229.0µs | 0.92x | 🔵 |
| Gather Large | src=1000000, idx=100000 | 219.1µs | 238.9µs | 0.92x | 🔵 |
| Gather Huge | src=10000000, idx=1000000 | 353.6µs | 357.2µs | 0.99x | 🔵 |
| ScatterAdd Small | dst=10000, idx=1000 | 213.1µs | 228.0µs | 0.93x | 🔵 |
| ScatterAdd Medium | dst=100000, idx=10000 | 185.6µs | 217.0µs | 0.86x | 🔵 |
| ScatterAdd Large | dst=1000000, idx=100000 | 253.5µs | 269.4µs | 0.94x | 🔵 |
| ScatterAdd Huge | dst=10000000, idx=1000000 | 953.6µs | 995.0µs | 0.96x | 🔵 |

## LoRA Training Ops (metalcore)

*Fused operations for LoRA fine-tuning: cross-entropy, KL divergence, SwiGLU, LoRA linear*

| Op | Config | Metal | Torch | Ratio | Status |
|---|---|---|---|---|---|
| CrossEntropy | 128x32000 Llama vocab | 244.0µs | 431.0µs | 0.57x | 🟢 |
| CrossEntropy | 256x32000 Large batch | 310.6µs | 665.9µs | 0.47x | 🟢 |
| CrossEntropy | 512x128256 Llama-3 vocab | 1.6ms | 3.2ms | 0.51x | 🟢 |
| KL Divergence | 128x32000 Full vocab | 3.1ms | 650.4µs | 4.81x | 🟠 |
| KL Divergence | 256x32000 Large batch | 4.5ms | 1.1ms | 3.98x | 🟠 |
| KL-TopK | 128x32000 k=100 (100% saved) | 269.0µs | 340.5µs | 0.79x | 🔵 |
| KL-TopK | 256x32000 k=50 (100% saved) | 252.3µs | 304.0µs | 0.83x | 🔵 |
| SwiGLU | 128x11008 Llama-7B hidden | 230.2µs | 320.5µs | 0.72x | 🔵 |
| SwiGLU | 256x14336 Llama-3 hidden | 363.8µs | 387.8µs | 0.94x | 🔵 |
| LoRA Linear | 128x4096→4096 r=16 Llama attn r=16 | 881.9µs | 885.2µs | 1.00x | 🔵 |
| LoRA Linear | 128x4096→11008 r=8 Llama MLP r=8 | 1.6ms | 1.6ms | 1.01x | 🔵 |

## Fused Backward Operations (Phase 3)

*Benchmarks run in Lite mode (fewer iterations)*

| Op | Config | Metal | Torch | Ratio | Status |
|---|---|---|---|---|---|
| FusedAtt Bwd | Lite Attn fp32 | 4.6ms | 272.2ms | 0.02x | 💚 |
| FusedAtt Bwd | Lite Attn fp16 | 6.7ms | 266.2ms | 0.03x | 💚 |
| FusedAtt Bwd | Lite Attn bf16 | 4.3ms | 245.6ms | 0.02x | 💚 |
| FusedMLP Bwd | Lite MLP fp32 | 38.6ms | 228.2ms | 0.17x | 💚 |
| FusedMLP Bwd | Lite MLP fp16 | 96.7ms | 267.4ms | 0.36x | 🟢 |
| FusedMLP Bwd | Lite MLP bf16 | 39.8ms | 210.7ms | 0.19x | 💚 |

## Fused Attention Backward (metalcore)

*Fused Bwd: SDPA Grads -> RoPE Bwd -> QKV Bwd -> RMSNorm Bwd*

| Op | Config | Metal | Torch | Ratio | Status |
|---|---|---|---|---|---|
| FusedAtt Bwd | 32x128 Llama-7B Attn fp32 | 77.6ms | 170.4ms | 0.46x | 🟢 |
| FusedAtt Bwd | 8x128 Large Head Count fp32 | 74.3ms | 159.9ms | 0.46x | 🟢 |
