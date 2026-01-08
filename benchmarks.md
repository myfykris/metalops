# Metalops Benchmark Results

*Generated: 2026-01-07 21:03:04*

**Legend:** 💚 GPU wins big (>3x) | 🟢 GPU wins | 🔵 Close | ⚪ CPU wins | 🟠 CPU wins big (>3x)

## SVD (metalsvd)

*Singular Value Decomposition using Jacobi algorithm on GPU*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 32×32 | Tiny | 922.8µs | 78.4µs | 11.77x | 🟠 | ✓ 2e-06 |
| 64×64 | Small square | 1.1ms | 229.1µs | 4.86x | 🟠 | ✓ 4e-06 |
| 128×128 | Medium square | 1.7ms | 793.0µs | 2.08x | ⚪ | ✓ 5e-06 |
| 256×256 | Large square | 5.4ms | 4.1ms | 1.30x | ⚪ | ✓ 7e-06 |
| 512×512 | Very large | 14.4ms | 17.5ms | 0.83x | 🔵 | ✓ 9e-06 |
| 1024×1024 | Huge square | 67.5ms | 86.6ms | 0.78x | 🔵 | ✓ 1e-05 |
| 2048×2048 | Massive square | 341.0ms | 508.8ms | 0.67x | 🟢 | ✓ 2e-05 |
| 256×128 | Tall 2:1 | 2.3ms | 1.7ms | 1.35x | ⚪ | ✓ 6e-06 |
| 512×256 | Tall 2:1 large | 7.3ms | 8.4ms | 0.87x | 🔵 | ✓ 9e-06 |
| 1024×512 | Tall matrix | 14.8ms | 39.8ms | 0.37x | 🟢 | ✓ 9e-06 |
| 2048×512 | Very tall | 15.3ms | 104.0ms | 0.15x | 💚 | ✓ 9e-06 |
| 128×256 | Wide 1:2 | 2.5ms | 1.9ms | 1.34x | ⚪ | ✓ 5e-06 |
| 4096×4096 | Llama-7B attn (4096x4096) | 2.86s | 4.89s | 0.58x | 🟢 | ✓ 6e-05 |
| 4096×11008 | Llama-2-7B MLP (4096x11008) | 2.92s | 11.82s | 0.25x | 💚 | ✓ 3e-05 |
| 4096×14336 | Llama-3-8B MLP (4096x14336) | 2.99s | 17.09s | 0.18x | 💚 | ✓ 3e-05 |
| 8192×8192 | Llama-70B attn (8192x8192) | 21.70s | 36.90s | 0.59x | 🟢 | ✓ 9e-05 |
| 4096×14336 | Mistral-7B MLP (4096x14336) | 2.98s | 16.93s | 0.18x | 💚 | ✓ 3e-05 |
| 4096×11008 | Qwen-7B MLP (4096x11008) | 2.91s | 11.67s | 0.25x | 💚 | ✓ 3e-05 |
| 3072×24576 | Gemma-7B MLP (3072x24576) | 998.9ms | 28.26s | 0.04x | 💚 | ✓ 3e-05 |
| 3072×8192 | Phi-3-mini MLP (3072x8192) | 939.4ms | 5.39s | 0.17x | 💚 | ✓ 3e-05 |
| 100×32×32 | Batch 100 tiny | 5.6ms | 7.2ms | 0.78x | 🔵 | ✓ 3e-06 |
| 50×64×64 | Batch 50 small | 6.3ms | 11.9ms | 0.53x | 🟢 | ✓ 6e-06 |
| 100×64×64 | Batch 100 small | 8.3ms | 23.6ms | 0.35x | 🟢 | ✓ 4e-06 |
| 200×64×64 | Batch 200 small | 14.1ms | 47.2ms | 0.30x | 💚 | ✓ 5e-06 |
| 20×128×128 | Batch 20 medium | 6.6ms | 16.6ms | 0.40x | 🟢 | ✓ 1e-05 |
| 50×128×128 | Batch 50 medium | 11.1ms | 41.2ms | 0.27x | 💚 | ✓ 1e-05 |
| 10×256×256 | Batch 10 large | 47.3ms | 44.7ms | 1.06x | 🔵 | ✓ 2e-05 |
| 5×512×512 | Batch 5 huge | 62.5ms | 89.4ms | 0.70x | 🟢 | ✓ 9e-06 |

## QR Single Matrix (metalcore)

*Single matrix QR - CPU typically wins due to sequential dependencies*

| Shape | Config | Metal | CPU | Ratio | Status | Recon | Ortho |
|---|---|---|---|---|---|---|---|
| 16×16 | Tiny | 1.1ms | 46.3µs | 23.85x | 🟠 | ✓ 7e-07 | ✓ 3e-07 |
| 32×32 | Small 32 | 572.3µs | 60.2µs | 9.51x | 🟠 | ✓ 1e-06 | ✓ 3e-07 |
| 64×64 | Small 64 | 727.8µs | 86.7µs | 8.39x | 🟠 | ✓ 1e-06 | ✓ 6e-07 |
| 128×128 | Medium | 773.6µs | 282.6µs | 2.74x | ⚪ | ✓ 3e-06 | ✓ 6e-07 |
| 256×256 | Large 256 | 2.0ms | 964.5µs | 2.05x | ⚪ | ✓ 4e-06 | ✓ 8e-07 |
| 512×512 | Large 512 | 5.4ms | 4.3ms | 1.28x | 🔵 | ✓ 9e-06 | ✓ 1e-06 |
| 1024×1024 | Huge 1024 | 28.2ms | 25.2ms | 1.12x | 🔵 | ✓ 1e-05 | ✓ 2e-06 |
| 256×64 | Tall 4:1 | 1.3ms | 211.4µs | 6.29x | 🟠 | ✓ 2e-06 | ✓ 5e-07 |
| 256×128 | Tall 2:1 | 1.4ms | 576.4µs | 2.40x | ⚪ | ✓ 4e-06 | ✓ 7e-07 |
| 512×128 | Tall 4:1 large | 1.9ms | 1.1ms | 1.76x | ⚪ | ✓ 7e-06 | ✓ 7e-07 |
| 512×256 | Tall 2:1 large | 3.1ms | 2.1ms | 1.49x | ⚪ | ✓ 6e-06 | ✓ 1e-06 |
| 1000×200 | Tall 5:1 | 3.7ms | 2.6ms | 1.44x | ⚪ | ✓ 8e-06 | ✓ 2e-06 |
| 2000×500 | Huge tall | 16.3ms | 13.5ms | 1.20x | 🔵 | ✓ 2e-05 | ✓ 2e-06 |
| 4000×1000 | Massive | 82.5ms | 76.2ms | 1.08x | 🔵 | ✓ 2e-05 | ✓ 3e-06 |

## QR Batched (metalcore) ⭐ GPU WINS

*Batched QR - GPU processes all matrices in parallel, single dispatch*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 50×8×8 | Tiny 8x8 | 911.7µs | 2.1ms | 0.44x | 🟢 | ✓ 1e-06 |
| 100×8×8 | Batch 100 tiny | 867.2µs | 4.5ms | 0.19x | 💚 | ✓ 5e-07 |
| 500×8×8 | Batch 500 tiny | 1.4ms | 23.0ms | 0.06x | 💚 | ✓ 4e-07 |
| 50×16×16 | ML mini-batch 16 | 1.7ms | 2.4ms | 0.70x | 🟢 | ✓ 1e-06 |
| 100×16×16 | Batch 100 16x16 | 1.3ms | 5.0ms | 0.26x | 💚 | ✓ 1e-06 |
| 200×16×16 | Batch 200 16x16 | 1.7ms | 9.8ms | 0.17x | 💚 | ✓ 1e-06 |
| 500×16×16 | Batch 500 16x16 | 2.4ms | 24.2ms | 0.10x | 💚 | ✓ 6e-07 |
| 1000×16×16 | Batch 1000 16x16 | 3.9ms | 47.9ms | 0.08x | 💚 | ✓ 7e-07 |
| 50×32×32 | ML mini-batch 32 | 2.8ms | 2.7ms | 1.04x | 🔵 | ✓ 1e-06 |
| 100×32×32 | Batch 100 32x32 | 2.3ms | 5.5ms | 0.41x | 🟢 | ✓ 1e-06 |
| 200×32×32 | Batch 200 32x32 | 2.8ms | 11.0ms | 0.25x | 💚 | ✓ 1e-06 |
| 500×32×32 | Batch 500 32x32 | 6.3ms | 27.7ms | 0.23x | 💚 | ✓ 1e-06 |
| 50×48×48 | Batch 50 48x48 | 5.0ms | 3.4ms | 1.46x | ⚪ | ✓ 1e-06 |
| 100×48×48 | Batch 100 48x48 | 4.9ms | 6.7ms | 0.74x | 🔵 | ✓ 2e-06 |
| 100×64×32 | Tall batch | 2.6ms | 6.0ms | 0.44x | 🟢 | ✓ 1e-06 |
| 100×32×64 | Wide batch | 3.9ms | 6.0ms | 0.65x | 🟢 | ✓ 1e-06 |
| 200×64×32 | Large tall batch | 4.4ms | 12.5ms | 0.35x | 🟢 | ✓ 2e-06 |

## Cholesky (metalcore) ⭐ GPU WINS

*Batched Cholesky decomposition with MAGMA-style shared memory*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 100×16×16 | Tiny batched | 224.2µs | 4.4ms | 0.05x | 💚 | ✓ 8e-06 |
| 500×16×16 | Large batch tiny | 832.6µs | 23.3ms | 0.04x | 💚 | ✓ 8e-06 |
| 100×32×32 | Small batched | 841.7µs | 4.4ms | 0.19x | 💚 | ✓ 8e-06 |
| 200×48×48 | Medium batched | 849.5µs | 9.4ms | 0.09x | 💚 | ✓ 2e-05 |
| 100×64×64 | Larger batched | 1.0ms | 4.6ms | 0.23x | 💚 | ✓ 2e-05 |

## Linear Solve (metalcore)

*Batched linear system solve using QR + TRSM*

| Shape | Config | Metal | CPU | Ratio | Status | Residual |
|---|---|---|---|---|---|---|
| 100×16×16 | Tiny batched fp32 | 221.2µs | 1.3ms | 0.17x | 💚 | ~ 3e-04 |
| 500×16×16 | Large batch tiny fp32 | 324.3µs | 6.2ms | 0.05x | 💚 | ~ 2e-03 |
| 100×32×32 | Small batched fp32 | 601.4µs | 1.7ms | 0.36x | 🟢 | ~ 3e-04 |
| 200×48×48 | Medium batched fp32 | 870.8µs | 4.1ms | 0.21x | 💚 | ~ 8e-03 |

## Eigendecomposition (metaleig)

*Symmetric eigenvalue decomposition*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 32×32 | Tiny | 773.3µs | 52.3µs | 14.79x | 🟠 | ✓ 2e-06 |
| 64×64 | Small | 899.5µs | 179.8µs | 5.00x | 🟠 | ✓ 3e-06 |
| 128×128 | Medium | 1.5ms | 632.9µs | 2.43x | ⚪ | ✓ 4e-06 |
| 256×256 | Large | 3.9ms | 3.0ms | 1.30x | ⚪ | ✓ 8e-06 |
| 512×512 | Very large | 13.9ms | 12.1ms | 1.15x | 🔵 | ✓ 1e-05 |
| 1024×1024 | Huge | 69.1ms | 65.9ms | 1.05x | 🔵 | ✓ 2e-05 |
| 100×32×32 | Batch 100 tiny | 5.7ms | 4.8ms | 1.20x | 🔵 | ✗ 2e+00 |
| 50×64×64 | Batch 50 small | 4.6ms | 9.6ms | 0.47x | 🟢 | ~ 5e-04 |
| 100×64×64 | Batch 100 small | 7.8ms | 17.8ms | 0.44x | 🟢 | ~ 2e-03 |
| 200×64×64 | Batch 200 small | 13.5ms | 35.3ms | 0.38x | 🟢 | ~ 9e-04 |
| 20×128×128 | Batch 20 medium | 5.5ms | 11.6ms | 0.48x | 🟢 | ~ 8e-04 |
| 50×128×128 | Batch 50 medium | 9.1ms | 28.5ms | 0.32x | 🟢 | ~ 7e-03 |
| 10×256×256 | Batch 10 large | 28.7ms | 27.0ms | 1.06x | 🔵 | ✓ 8e-06 |

## RMSNorm (metalcore) ⭐ GPU WINS

*Fused RMSNorm kernel vs torch.nn.RMSNorm*

| Shape | Config | Metal | CPU | Ratio | Status |
|---|---|---|---|---|---|
| 32x4096 | Fwd+Bwd fp32 | 440.3µs | 499.4µs | 0.88x | 🔵 |
| 1x4096 | Fwd+Bwd fp32 | 427.3µs | 448.3µs | 0.95x | 🔵 |
| 1024x1024 | Fwd+Bwd fp32 | 854.7µs | 932.4µs | 0.92x | 🔵 |
| 4096x4096 | Fwd+Bwd fp32 | 6.0ms | 8.7ms | 0.68x | 🟢 |

## AdamW (metalcore) ⭐ GPU WINS

*Fused AdamW optimizer step vs torch.optim.AdamW*

| Params | Size | Metal | CPU | Ratio | Status |
|---|---|---|---|---|---|
| 1M Params | N=1048576 fp32 | 298.6µs | 496.4µs | 0.60x | 🟢 |
| 10M Params | N=10485760 fp32 | 1.1ms | 3.0ms | 0.37x | 🟢 |
| 16M Params | N=16777216 fp32 | 1.7ms | 4.5ms | 0.37x | 🟢 |

## Activations (metalcore)

*GELU/SiLU activations with float4 vectorization*

| Op | Shape | Metal | Torch | Ratio | Status |
|---|---|---|---|---|---|
| GELU Small (256x1024) | 256x1024 fp32 | 207.0µs | 210.8µs | 0.98x | 🔵 |
| SiLU Small (256x1024) | 256x1024 fp32 | 187.6µs | 189.6µs | 0.99x | 🔵 |
| GELU Medium (1024x4096) | 1024x4096 fp32 | 262.9µs | 246.8µs | 1.07x | 🔵 |
| SiLU Medium (1024x4096) | 1024x4096 fp32 | 231.4µs | 261.6µs | 0.88x | 🔵 |
| GELU Large (4096x4096) | 4096x4096 fp32 | 595.7µs | 605.4µs | 0.98x | 🔵 |
| SiLU Large (4096x4096) | 4096x4096 fp32 | 607.9µs | 592.3µs | 1.03x | 🔵 |

## SDPA (metalcore)

*Scaled Dot Product Attention with Flash Attention v2 tiling*

| Config | Shape | Metal | Torch | Ratio | Status | Error |
|---|---|---|---|---|---|---|
| Small (B=2, H=8, N=64, D=64) | B=2, H=8, N=64, D=64 fp32 | 254.1µs | 301.4µs | 0.84x | 🔵 | ✓ 1e-06 |
| Small (B=2, H=8, N=64, D=64) (causal) | B=2, H=8, N=64, D=64 fp32 | 671.4µs | 291.7µs | 2.30x | ⚪ | ✓ 4e-06 |
| Medium (B=2, H=8, N=256, D=64) | B=2, H=8, N=256, D=64 fp32 | 902.8µs | 394.8µs | 2.29x | ⚪ | ✗ 1e-01 |
| Medium (B=2, H=8, N=256, D=64) (causal) | B=2, H=8, N=256, D=64 fp32 | 3.5ms | 398.7µs | 8.85x | 🟠 | ✓ 3e-06 |
| Large (B=1, H=8, N=512, D=64) | B=1, H=8, N=512, D=64 fp32 | 1.7ms | 457.6µs | 3.72x | 🟠 | ✗ 4e-02 |
| Large (B=1, H=8, N=512, D=64) (causal) | B=1, H=8, N=512, D=64 fp32 | 6.0ms | 446.9µs | 13.44x | 🟠 | ✓ 3e-06 |

## Fused Softmax (metalcore)

*Online softmax algorithm with SIMD reductions*

| Config | Shape | Metal | Torch | Ratio | Status | Error |
|---|---|---|---|---|---|---|
| Small | 32x1024 fp32 | 151.6µs | 180.6µs | 0.84x | 🔵 | 3.73e-09 |
| Medium | 64x4096 fp32 | 153.6µs | 201.4µs | 0.76x | 🔵 | 1.40e-09 |
| Large | 128x8192 fp32 | 169.3µs | 202.6µs | 0.84x | 🔵 | 9.31e-10 |
| Very Large | 256x16384 fp32 | 228.5µs | 292.8µs | 0.78x | 🔵 | 9.31e-10 |
| Huge | 512x32768 fp32 | 908.7µs | 809.9µs | 1.12x | 🔵 | 6.98e-10 |
| LLM Vocab | 32x32000 fp32 | 205.4µs | 234.4µs | 0.88x | 🔵 | 2.33e-10 |
| LLM Vocab Large | 128x128000 fp32 | 915.4µs | 804.5µs | 1.14x | 🔵 | 1.16e-10 |

## LayerNorm (metalcore)

*Welford's algorithm for fused mean/variance*

| Config | Shape | Metal | Torch | Ratio | Status | Error |
|---|---|---|---|---|---|---|
| Tiny | 32x512 fp32 | 162.1µs | 162.6µs | 1.00x | 🔵 | 4.77e-07 |
| Small | 64x1024 fp32 | 184.5µs | 180.5µs | 1.02x | 🔵 | 4.77e-07 |
| Llama-7B | 32x4096 fp32 | 193.3µs | 208.6µs | 0.93x | 🔵 | 4.77e-07 |
| Llama-13B | 32x5120 fp32 | 175.0µs | 180.6µs | 0.97x | 🔵 | 4.77e-07 |
| Llama-70B | 16x8192 fp32 | 180.6µs | 182.3µs | 0.99x | 🔵 | 4.77e-07 |
| Large Batch | 256x4096 fp32 | 193.6µs | 207.1µs | 0.93x | 🔵 | 4.77e-07 |
| Huge Batch | 1024x4096 fp32 | 257.7µs | 333.7µs | 0.77x | 🔵 | 9.54e-07 |

## Embedding Bag (metalcore)

*Coalesced reads for embedding lookups and aggregation*

| Config | Shape | Metal | Torch | Ratio | Status |
|---|---|---|---|---|---|
| Small Vocab | 10000x64, B=32 | 248.9µs | 1.2ms | 0.21x | 💚 |
| Medium Vocab | 50000x128, B=64 | 234.1µs | 1.6ms | 0.14x | 💚 |
| Large Vocab | 100000x256, B=32 | 305.0µs | 8.5ms | 0.04x | 💚 |
| LLM Embedding | 32000x4096, B=16 | 316.3µs | 20.8ms | 0.02x | 💚 |
| Huge Vocab | 250000x512, B=16 | 253.7µs | 19.2ms | 0.01x | 💚 |

## Scatter/Gather (metalcore)

*Atomic scatter_add and vectorized gather operations*

| Op | Shape | Metal | Torch | Ratio | Status |
|---|---|---|---|---|---|
| Gather Small | src=10000, idx=1000 | 209.5µs | 230.6µs | 0.91x | 🔵 |
| Gather Medium | src=100000, idx=10000 | 204.1µs | 217.3µs | 0.94x | 🔵 |
| Gather Large | src=1000000, idx=100000 | 217.3µs | 249.3µs | 0.87x | 🔵 |
| Gather Huge | src=10000000, idx=1000000 | 355.8µs | 360.7µs | 0.99x | 🔵 |
| ScatterAdd Small | dst=10000, idx=1000 | 214.6µs | 244.5µs | 0.88x | 🔵 |
| ScatterAdd Medium | dst=100000, idx=10000 | 240.5µs | 264.0µs | 0.91x | 🔵 |
| ScatterAdd Large | dst=1000000, idx=100000 | 251.8µs | 264.9µs | 0.95x | 🔵 |
| ScatterAdd Huge | dst=10000000, idx=1000000 | 960.7µs | 993.4µs | 0.97x | 🔵 |

## Pipeline Operations ⭐ GPU WINS (No Transfer)

*Chained operations where data stays on GPU - avoids costly memory transfers*

| Pipeline | Shape | GPU | Comparison | Ratio | Status |
|---|---|---|---|---|---|
| QR -> QR -> QR | 200×32×32 | 8.2ms | 31.8ms | 0.26x | 💚 |
| SVD -> truncate (PCA) | 50×128×64 | 10.5ms | 16.0ms | 0.66x | 🟢 |
| QR -> matmul (ML) | 1000×16×16 | 5.5ms | 52.1ms | 0.11x | 💚 |
| Fast+Slow+Fast (GPU all) | 200×32×32 | 5.1ms | 3.2ms | 1.59x vs hybrid | 🟠 |
| Fast+Slow+Fast (vs CPU) | 200×32×32 | 5.1ms | 12.1ms | 0.42x vs CPU | 🟢 |

## LLM: Llama

*SVD performance on Llama weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 4096×4096 | Attention (7B) | 2.88s | 5.02s | 0.57x | 🟢 | ✓ 3e-05 |
| 4096×11008 | MLP up (7B) | 2.93s | 9.93s | 0.30x | 💚 | ✓ 3e-05 |
| 8192×8192 | Attention (70B) | 21.35s | 36.82s | 0.58x | 🟢 | ✓ 9e-05 |

## LLM: Mistral

*SVD performance on Mistral weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 4096×4096 | Attention | 2.86s | 4.97s | 0.57x | 🟢 | ✓ 3e-05 |
| 4096×14336 | MLP up | 2.99s | 11.68s | 0.26x | 💚 | ✓ 4e-05 |

## LLM: Qwen

*SVD performance on Qwen weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 4096×4096 | Attention | 2.87s | 5.03s | 0.57x | 🟢 | ✓ 3e-05 |
| 4096×11008 | MLP up | 2.92s | 10.11s | 0.29x | 💚 | ✓ 4e-05 |

## LLM: Gemma

*SVD performance on Gemma weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 3072×3072 | Attention | 936.9ms | 1.57s | 0.60x | 🟢 | ~ 1e-04 |
| 3072×24576 | MLP up | 1.00s | 9.24s | 0.11x | 💚 | ✓ 3e-05 |

## LLM: Phi

*SVD performance on Phi weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 3072×3072 | Attention | 928.4ms | 1.60s | 0.58x | 🟢 | ~ 2e-04 |
| 3072×8192 | MLP up | 944.1ms | 3.68s | 0.26x | 💚 | ✓ 2e-05 |

## Usage Recommendations

| Operation | When to Use Metal | When to Use CPU |
|---|---|---|
| SVD | Batched small/medium matrices | Single large matrices |
| QR (single) | — | Always (sequential dependencies) |
| QR (batched) | Many small matrices (10x speedup!) | Few matrices |
| EIGH | Batched symmetric matrices | Single large matrices |
| Pipeline | Keep data on GPU to avoid transfer cost | Single ops on CPU-resident data |
