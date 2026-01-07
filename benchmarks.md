# Metalops Benchmark Results

*Generated: 2026-01-07 18:19:55*

**Legend:** 💚 GPU wins big (>3x) | 🟢 GPU wins | 🔵 Close | ⚪ CPU wins | 🟠 CPU wins big (>3x)

## SVD (metalsvd)

*Singular Value Decomposition using Jacobi algorithm on GPU*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 32×32 | Tiny | 1.4ms | 80.3µs | 17.90x | 🟠 | ✓ 3e-06 |
| 64×64 | Small square | 1.1ms | 226.1µs | 5.00x | 🟠 | ✓ 3e-06 |
| 128×128 | Medium square | 1.6ms | 783.0µs | 2.10x | ⚪ | ✓ 5e-06 |
| 256×256 | Large square | 5.3ms | 4.0ms | 1.33x | ⚪ | ✓ 8e-06 |
| 512×512 | Very large | 14.1ms | 17.9ms | 0.79x | 🔵 | ✓ 9e-06 |
| 1024×1024 | Huge square | 68.7ms | 90.0ms | 0.76x | 🔵 | ✓ 1e-05 |
| 2048×2048 | Massive square | 341.1ms | 526.4ms | 0.65x | 🟢 | ✓ 2e-05 |
| 256×128 | Tall 2:1 | 3.2ms | 1.7ms | 1.88x | ⚪ | ✓ 5e-06 |
| 512×256 | Tall 2:1 large | 7.1ms | 8.2ms | 0.87x | 🔵 | ✓ 1e-05 |
| 1024×512 | Tall matrix | 14.1ms | 43.2ms | 0.33x | 🟢 | ✓ 9e-06 |
| 2048×512 | Very tall | 15.3ms | 103.5ms | 0.15x | 💚 | ✓ 9e-06 |
| 128×256 | Wide 1:2 | 3.2ms | 1.9ms | 1.74x | ⚪ | ✓ 7e-06 |
| 4096×4096 | Llama-7B attn (4096x4096) | 2.87s | 5.11s | 0.56x | 🟢 | ✓ 6e-05 |
| 4096×11008 | Llama-2-7B MLP (4096x11008) | 2.96s | 11.87s | 0.25x | 💚 | ✓ 3e-05 |
| 4096×14336 | Llama-3-8B MLP (4096x14336) | 2.97s | 17.17s | 0.17x | 💚 | ✓ 3e-05 |
| 8192×8192 | Llama-70B attn (8192x8192) | 22.02s | 37.14s | 0.59x | 🟢 | ✓ 9e-05 |
| 4096×14336 | Mistral-7B MLP (4096x14336) | 2.98s | 17.17s | 0.17x | 💚 | ✓ 4e-05 |
| 4096×11008 | Qwen-7B MLP (4096x11008) | 2.94s | 11.98s | 0.25x | 💚 | ✓ 3e-05 |
| 3072×24576 | Gemma-7B MLP (3072x24576) | 1.00s | 28.30s | 0.04x | 💚 | ✓ 2e-05 |
| 3072×8192 | Phi-3-mini MLP (3072x8192) | 933.1ms | 5.27s | 0.18x | 💚 | ✓ 2e-05 |
| 100×32×32 | Batch 100 tiny | 5.7ms | 6.9ms | 0.83x | 🔵 | ✓ 3e-06 |
| 50×64×64 | Batch 50 small | 5.8ms | 11.6ms | 0.50x | 🟢 | ✓ 7e-06 |
| 100×64×64 | Batch 100 small | 8.6ms | 23.3ms | 0.37x | 🟢 | ✓ 7e-06 |
| 200×64×64 | Batch 200 small | 15.4ms | 46.2ms | 0.33x | 🟢 | ✓ 5e-06 |
| 20×128×128 | Batch 20 medium | 7.2ms | 16.0ms | 0.45x | 🟢 | ✓ 8e-06 |
| 50×128×128 | Batch 50 medium | 10.7ms | 40.0ms | 0.27x | 💚 | ✓ 1e-05 |
| 10×256×256 | Batch 10 large | 46.9ms | 40.8ms | 1.15x | 🔵 | ✓ 2e-05 |
| 5×512×512 | Batch 5 huge | 60.7ms | 87.8ms | 0.69x | 🟢 | ✓ 8e-06 |

## QR Single Matrix (metalcore)

*Single matrix QR - CPU typically wins due to sequential dependencies*

| Shape | Config | Metal | CPU | Ratio | Status | Recon | Ortho |
|---|---|---|---|---|---|---|---|
| 16×16 | Tiny | 1.3ms | 53.0µs | 24.95x | 🟠 | ✓ 5e-07 | ✓ 6e-07 |
| 32×32 | Small 32 | 702.7µs | 58.4µs | 12.04x | 🟠 | ✓ 1e-06 | ✓ 3e-07 |
| 64×64 | Small 64 | 788.7µs | 87.8µs | 8.98x | 🟠 | ✓ 2e-06 | ✓ 5e-07 |
| 128×128 | Medium | 905.9µs | 262.4µs | 3.45x | 🟠 | ✓ 3e-06 | ✓ 7e-07 |
| 256×256 | Large 256 | 1.8ms | 921.7µs | 1.92x | ⚪ | ✓ 4e-06 | ✓ 6e-07 |
| 512×512 | Large 512 | 5.4ms | 4.1ms | 1.32x | ⚪ | ✓ 9e-06 | ✓ 1e-06 |
| 1024×1024 | Huge 1024 | 26.7ms | 23.8ms | 1.12x | 🔵 | ✓ 1e-05 | ✓ 2e-06 |
| 256×64 | Tall 4:1 | 1.6ms | 248.0µs | 6.40x | 🟠 | ✓ 2e-06 | ✓ 6e-07 |
| 256×128 | Tall 2:1 | 1.3ms | 579.9µs | 2.22x | ⚪ | ✓ 5e-06 | ✓ 7e-07 |
| 512×128 | Tall 4:1 large | 1.8ms | 1.0ms | 1.77x | ⚪ | ✓ 5e-06 | ✓ 1e-06 |
| 512×256 | Tall 2:1 large | 2.9ms | 1.9ms | 1.54x | ⚪ | ✓ 9e-06 | ✓ 1e-06 |
| 1000×200 | Tall 5:1 | 3.8ms | 2.6ms | 1.45x | ⚪ | ✓ 9e-06 | ✓ 1e-06 |
| 2000×500 | Huge tall | 15.4ms | 13.2ms | 1.17x | 🔵 | ✓ 2e-05 | ✓ 2e-06 |
| 4000×1000 | Massive | 79.4ms | 73.1ms | 1.09x | 🔵 | ✓ 3e-05 | ✓ 3e-06 |

## QR Batched (metalcore) ⭐ GPU WINS

*Batched QR - GPU processes all matrices in parallel, single dispatch*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 50×8×8 | Tiny 8x8 | 653.1µs | 2.1ms | 0.31x | 🟢 | ✓ 1e-06 |
| 100×8×8 | Batch 100 tiny | 727.7µs | 4.3ms | 0.17x | 💚 | ✓ 4e-07 |
| 500×8×8 | Batch 500 tiny | 1.6ms | 22.2ms | 0.07x | 💚 | ✓ 5e-07 |
| 50×16×16 | ML mini-batch 16 | 1.3ms | 2.3ms | 0.56x | 🟢 | ✓ 7e-07 |
| 100×16×16 | Batch 100 16x16 | 1.5ms | 4.7ms | 0.32x | 🟢 | ✓ 8e-07 |
| 200×16×16 | Batch 200 16x16 | 1.6ms | 9.3ms | 0.17x | 💚 | ✓ 7e-07 |
| 500×16×16 | Batch 500 16x16 | 2.9ms | 23.1ms | 0.13x | 💚 | ✓ 1e-06 |
| 1000×16×16 | Batch 1000 16x16 | 3.5ms | 47.2ms | 0.07x | 💚 | ✓ 7e-07 |
| 50×32×32 | ML mini-batch 32 | 2.9ms | 2.8ms | 1.06x | 🔵 | ✓ 2e-06 |
| 100×32×32 | Batch 100 32x32 | 2.3ms | 5.3ms | 0.44x | 🟢 | ✓ 1e-06 |
| 200×32×32 | Batch 200 32x32 | 2.9ms | 10.6ms | 0.28x | 💚 | ✓ 1e-06 |
| 500×32×32 | Batch 500 32x32 | 6.2ms | 26.5ms | 0.23x | 💚 | ✓ 1e-06 |
| 50×48×48 | Batch 50 48x48 | 4.9ms | 3.2ms | 1.52x | ⚪ | ✓ 1e-06 |
| 100×48×48 | Batch 100 48x48 | 4.9ms | 6.4ms | 0.77x | 🔵 | ✓ 1e-06 |
| 100×64×32 | Tall batch | 2.6ms | 6.0ms | 0.43x | 🟢 | ✓ 1e-06 |
| 100×32×64 | Wide batch | 3.7ms | 5.7ms | 0.65x | 🟢 | ✓ 1e-06 |
| 200×64×32 | Large tall batch | 4.5ms | 11.9ms | 0.38x | 🟢 | ✓ 1e-06 |

## Cholesky (metalcore) ⭐ GPU WINS

*Batched Cholesky decomposition with MAGMA-style shared memory*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 100×16×16 | Tiny batched | 830.9µs | 4.3ms | 0.19x | 💚 | ✓ 6e-06 |
| 500×16×16 | Large batch tiny | 702.5µs | 21.1ms | 0.03x | 💚 | ✓ 6e-06 |
| 100×32×32 | Small batched | 876.6µs | 4.3ms | 0.21x | 💚 | ✓ 8e-06 |
| 200×48×48 | Medium batched | 910.5µs | 8.7ms | 0.11x | 💚 | ✓ 8e-06 |
| 100×64×64 | Larger batched | 930.5µs | 4.4ms | 0.21x | 💚 | ✓ 2e-05 |

## Linear Solve (metalcore)

*Batched linear system solve using QR + TRSM*

| Shape | Config | Metal | CPU | Ratio | Status | Residual |
|---|---|---|---|---|---|---|
| 100×16×16 | Tiny batched fp32 | 705.1µs | 1.1ms | 0.62x | 🟢 | ~ 1e-04 |
| 500×16×16 | Large batch tiny fp32 | 745.9µs | 5.7ms | 0.13x | 💚 | ~ 6e-04 |
| 100×32×32 | Small batched fp32 | 1.3ms | 1.6ms | 0.83x | 🔵 | ~ 1e-03 |
| 200×48×48 | Medium batched fp32 | 1.7ms | 4.0ms | 0.43x | 🟢 | ~ 9e-04 |

## Eigendecomposition (metaleig)

*Symmetric eigenvalue decomposition*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 32×32 | Tiny | 745.0µs | 48.6µs | 15.31x | 🟠 | ✓ 2e-06 |
| 64×64 | Small | 810.1µs | 176.4µs | 4.59x | 🟠 | ✓ 2e-06 |
| 128×128 | Medium | 1.1ms | 556.0µs | 2.02x | ⚪ | ✓ 4e-06 |
| 256×256 | Large | 3.9ms | 2.7ms | 1.48x | ⚪ | ✓ 9e-06 |
| 512×512 | Very large | 13.9ms | 11.9ms | 1.16x | 🔵 | ✓ 1e-05 |
| 1024×1024 | Huge | 67.4ms | 64.8ms | 1.04x | 🔵 | ✓ 2e-05 |
| 100×32×32 | Batch 100 tiny | 6.4ms | 4.7ms | 1.35x | ⚪ | ✗ 3e+00 |
| 50×64×64 | Batch 50 small | 4.6ms | 8.8ms | 0.52x | 🟢 | ~ 5e-04 |
| 100×64×64 | Batch 100 small | 7.9ms | 17.8ms | 0.45x | 🟢 | ~ 2e-04 |
| 200×64×64 | Batch 200 small | 14.1ms | 35.4ms | 0.40x | 🟢 | ~ 5e-04 |
| 20×128×128 | Batch 20 medium | 5.4ms | 11.3ms | 0.48x | 🟢 | ~ 3e-04 |
| 50×128×128 | Batch 50 medium | 9.5ms | 28.0ms | 0.34x | 🟢 | ~ 6e-03 |
| 10×256×256 | Batch 10 large | 27.9ms | 25.6ms | 1.09x | 🔵 | ✓ 6e-06 |

## RMSNorm (metalcore) ⭐ GPU WINS

*Fused RMSNorm kernel vs torch.nn.RMSNorm*

| Shape | Config | Metal | CPU | Ratio | Status |
|---|---|---|---|---|---|
| 32x4096 | Fwd+Bwd fp32 | 580.5µs | 488.2µs | 1.19x | 🔵 |
| 1x4096 | Fwd+Bwd fp32 | 511.8µs | 429.8µs | 1.19x | 🔵 |
| 1024x1024 | Fwd+Bwd fp32 | 946.5µs | 923.8µs | 1.02x | 🔵 |
| 4096x4096 | Fwd+Bwd fp32 | 6.2ms | 8.6ms | 0.72x | 🔵 |

## AdamW (metalcore) ⭐ GPU WINS

*Fused AdamW optimizer step vs torch.optim.AdamW*

| Params | Size | Metal | CPU | Ratio | Status |
|---|---|---|---|---|---|
| 1M Params | N=1048576 fp32 | 324.5µs | 499.7µs | 0.65x | 🟢 |
| 10M Params | N=10485760 fp32 | 1.1ms | 2.9ms | 0.37x | 🟢 |
| 16M Params | N=16777216 fp32 | 1.6ms | 4.5ms | 0.36x | 🟢 |

## Activations (metalcore)

*GELU/SiLU activations with float4 vectorization*

| Op | Shape | Metal | Torch | Ratio | Status |
|---|---|---|---|---|---|
| GELU Small (256x1024) | 256x1024 fp32 | 185.2µs | 184.3µs | 1.00x | 🔵 |
| SiLU Small (256x1024) | 256x1024 fp32 | 154.4µs | 148.2µs | 1.04x | 🔵 |
| GELU Medium (1024x4096) | 1024x4096 fp32 | 231.5µs | 225.3µs | 1.03x | 🔵 |
| SiLU Medium (1024x4096) | 1024x4096 fp32 | 220.8µs | 213.7µs | 1.03x | 🔵 |
| GELU Large (4096x4096) | 4096x4096 fp32 | 567.6µs | 565.8µs | 1.00x | 🔵 |
| SiLU Large (4096x4096) | 4096x4096 fp32 | 558.1µs | 578.7µs | 0.96x | 🔵 |

## SDPA (metalcore)

*Scaled Dot Product Attention with Flash Attention v2 tiling*

| Config | Shape | Metal | Torch | Ratio | Status | Error |
|---|---|---|---|---|---|---|
| Small (B=2, H=8, N=64, D=64) | B=2, H=8, N=64, D=64 fp32 | 192.0µs | 228.1µs | 0.84x | 🔵 | ✓ 9e-07 |
| Small (B=2, H=8, N=64, D=64) (causal) | B=2, H=8, N=64, D=64 fp32 | 613.9µs | 278.4µs | 2.21x | ⚪ | ✓ 3e-06 |
| Medium (B=2, H=8, N=256, D=64) | B=2, H=8, N=256, D=64 fp32 | 862.1µs | 352.8µs | 2.44x | ⚪ | ✗ 5e-02 |
| Medium (B=2, H=8, N=256, D=64) (causal) | B=2, H=8, N=256, D=64 fp32 | 3.4ms | 383.7µs | 8.97x | 🟠 | ✓ 4e-06 |
| Large (B=1, H=8, N=512, D=64) | B=1, H=8, N=512, D=64 fp32 | 1.6ms | 436.1µs | 3.74x | 🟠 | ✗ 5e-02 |
| Large (B=1, H=8, N=512, D=64) (causal) | B=1, H=8, N=512, D=64 fp32 | 6.0ms | 463.2µs | 12.86x | 🟠 | ✓ 3e-06 |

## Fused Softmax (metalcore)

*Online softmax algorithm with SIMD reductions*

| Config | Shape | Metal | Torch | Ratio | Status | Error |
|---|---|---|---|---|---|---|
| Small | 32x1024 fp32 | 155.3µs | 180.2µs | 0.86x | 🔵 | 3.73e-09 |
| Medium | 64x4096 fp32 | 152.8µs | 182.2µs | 0.84x | 🔵 | 1.86e-09 |
| Large | 128x8192 fp32 | 164.5µs | 213.0µs | 0.77x | 🔵 | 9.31e-10 |
| Very Large | 256x16384 fp32 | 262.4µs | 279.2µs | 0.94x | 🔵 | 4.66e-10 |
| Huge | 512x32768 fp32 | 948.5µs | 833.0µs | 1.14x | 🔵 | 3.49e-10 |
| LLM Vocab | 32x32000 fp32 | 202.7µs | 228.4µs | 0.89x | 🔵 | 2.33e-10 |
| LLM Vocab Large | 128x128000 fp32 | 945.0µs | 830.8µs | 1.14x | 🔵 | 1.16e-10 |

## LayerNorm (metalcore)

*Welford's algorithm for fused mean/variance*

| Config | Shape | Metal | Torch | Ratio | Status | Error |
|---|---|---|---|---|---|---|
| Tiny | 32x512 fp32 | 166.0µs | 164.0µs | 1.01x | 🔵 | 4.77e-07 |
| Small | 64x1024 fp32 | 165.9µs | 159.6µs | 1.04x | 🔵 | 4.77e-07 |
| Llama-7B | 32x4096 fp32 | 164.8µs | 168.7µs | 0.98x | 🔵 | 4.77e-07 |
| Llama-13B | 32x5120 fp32 | 172.9µs | 166.3µs | 1.04x | 🔵 | 4.77e-07 |
| Llama-70B | 16x8192 fp32 | 173.9µs | 165.7µs | 1.05x | 🔵 | 9.54e-07 |
| Large Batch | 256x4096 fp32 | 188.4µs | 185.8µs | 1.01x | 🔵 | 9.54e-07 |
| Huge Batch | 1024x4096 fp32 | 256.4µs | 256.6µs | 1.00x | 🔵 | 7.15e-07 |

## Embedding Bag (metalcore)

*Coalesced reads for embedding lookups and aggregation*

| Config | Shape | Metal | Torch | Ratio | Status |
|---|---|---|---|---|---|
| Small Vocab | 10000x64, B=32 | 215.7µs | 1.2ms | 0.18x | 💚 |
| Medium Vocab | 50000x128, B=64 | 221.0µs | 1.6ms | 0.14x | 💚 |
| Large Vocab | 100000x256, B=32 | 233.8µs | 6.6ms | 0.04x | 💚 |
| LLM Embedding | 32000x4096, B=16 | 353.9µs | 18.7ms | 0.02x | 💚 |
| Huge Vocab | 250000x512, B=16 | 244.7µs | 17.1ms | 0.01x | 💚 |

## Scatter/Gather (metalcore)

*Atomic scatter_add and vectorized gather operations*

| Op | Shape | Metal | Torch | Ratio | Status |
|---|---|---|---|---|---|
| Gather Small | src=10000, idx=1000 | 186.1µs | 191.6µs | 0.97x | 🔵 |
| Gather Medium | src=100000, idx=10000 | 180.8µs | 192.3µs | 0.94x | 🔵 |
| Gather Large | src=1000000, idx=100000 | 203.9µs | 223.9µs | 0.91x | 🔵 |
| Gather Huge | src=10000000, idx=1000000 | 331.0µs | 344.5µs | 0.96x | 🔵 |
| ScatterAdd Small | dst=10000, idx=1000 | 207.7µs | 220.3µs | 0.94x | 🔵 |
| ScatterAdd Medium | dst=100000, idx=10000 | 209.0µs | 222.7µs | 0.94x | 🔵 |
| ScatterAdd Large | dst=1000000, idx=100000 | 240.4µs | 257.9µs | 0.93x | 🔵 |
| ScatterAdd Huge | dst=10000000, idx=1000000 | 935.5µs | 986.2µs | 0.95x | 🔵 |

## Pipeline Operations ⭐ GPU WINS (No Transfer)

*Chained operations where data stays on GPU - avoids costly memory transfers*

| Pipeline | Shape | GPU | Comparison | Ratio | Status |
|---|---|---|---|---|---|
| QR -> QR -> QR | 200×32×32 | 8.1ms | 31.6ms | 0.26x | 💚 |
| SVD -> truncate (PCA) | 50×128×64 | 9.7ms | 15.8ms | 0.61x | 🟢 |
| QR -> matmul (ML) | 1000×16×16 | 4.0ms | 50.4ms | 0.08x | 💚 |
| Fast+Slow+Fast (GPU all) | 200×32×32 | 4.8ms | 3.7ms | 1.29x vs hybrid | 🟠 |
| Fast+Slow+Fast (vs CPU) | 200×32×32 | 4.8ms | 11.9ms | 0.40x vs CPU | 🟢 |

## LLM: Llama

*SVD performance on Llama weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 4096×4096 | Attention (7B) | 2.85s | 5.04s | 0.57x | 🟢 | ✓ 3e-05 |
| 4096×11008 | MLP up (7B) | 2.94s | 10.12s | 0.29x | 💚 | ✓ 3e-05 |
| 8192×8192 | Attention (70B) | 21.50s | 37.15s | 0.58x | 🟢 | ✓ 5e-05 |

## LLM: Mistral

*SVD performance on Mistral weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 4096×4096 | Attention | 2.84s | 4.95s | 0.57x | 🟢 | ✓ 3e-05 |
| 4096×14336 | MLP up | 2.94s | 11.71s | 0.25x | 💚 | ✓ 3e-05 |

## LLM: Qwen

*SVD performance on Qwen weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 4096×4096 | Attention | 2.84s | 4.97s | 0.57x | 🟢 | ✓ 3e-05 |
| 4096×11008 | MLP up | 2.90s | 9.99s | 0.29x | 💚 | ✓ 3e-05 |

## LLM: Gemma

*SVD performance on Gemma weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 3072×3072 | Attention | 929.5ms | 1.56s | 0.60x | 🟢 | ✓ 3e-05 |
| 3072×24576 | MLP up | 991.0ms | 9.25s | 0.11x | 💚 | ✓ 3e-05 |

## LLM: Phi

*SVD performance on Phi weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 3072×3072 | Attention | 922.3ms | 1.55s | 0.60x | 🟢 | ✓ 2e-05 |
| 3072×8192 | MLP up | 941.7ms | 3.78s | 0.25x | 💚 | ✓ 2e-05 |

## Usage Recommendations

| Operation | When to Use Metal | When to Use CPU |
|---|---|---|
| SVD | Batched small/medium matrices | Single large matrices |
| QR (single) | — | Always (sequential dependencies) |
| QR (batched) | Many small matrices (10x speedup!) | Few matrices |
| EIGH | Batched symmetric matrices | Single large matrices |
| Pipeline | Keep data on GPU to avoid transfer cost | Single ops on CPU-resident data |
