# Metalops Benchmark Results

*Generated: 2026-01-05 02:16:47*

**Legend:** 💚 GPU wins big (>3x) | 🟢 GPU wins | ⚪ Close | 🟠 CPU wins | 🔴 CPU wins big (>3x)

## Activations (metalcore)

*GELU/SiLU activations with float4 vectorization*

| Op | Shape | Metal | Torch | Ratio | Status |
|---|---|---|---|---|---|
| GELU Small (256x1024) | 256x1024 | 506.1µs | 409.4µs | 1.24x | ⚪ |
| SiLU Small (256x1024) | 256x1024 | 326.7µs | 1.4ms | 0.23x | 💚 |
| GELU Medium (1024x4096) | 1024x4096 | 536.6µs | 602.6µs | 0.89x | ⚪ |
| SiLU Medium (1024x4096) | 1024x4096 | 552.8µs | 670.0µs | 0.83x | ⚪ |

## SDPA (metalcore)

*Scaled Dot Product Attention with Flash Attention v2 tiling*

| Config | Shape | Metal | Torch | Ratio | Status | Error |
|---|---|---|---|---|---|---|
| Small (B=2, H=8, N=64, D=64) | B=2, H=8, N=64, D=64 | 2.1ms | 714.4µs | 2.98x | 🟠 | ✓ 4e-07 |
| Small (B=2, H=8, N=64, D=64) (causal) | B=2, H=8, N=64, D=64 | 1.4ms | 883.4µs | 1.59x | 🟠 | ✓ 3e-06 |
| Medium (B=2, H=8, N=256, D=64) | B=2, H=8, N=256, D=64 | 7.2ms | 509.6µs | 14.06x | 🔴 | ✓ 4e-07 |
| Medium (B=2, H=8, N=256, D=64) (causal) | B=2, H=8, N=256, D=64 | 4.9ms | 1.1ms | 4.66x | 🔴 | ✓ 3e-06 |

## SVD (metalcore) ⭐ GPU WINS

*Batched SVD with hybrid Gram strategy for tall matrices*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 50×64×64 | Batch 50 small | 5.6ms | 11.9ms | 0.47x | 🟢 | ✓ 5e-06 |
| 100×64×64 | Batch 100 small | 8.4ms | 23.4ms | 0.36x | 🟢 | ✓ 6e-06 |
| 200×64×64 | Batch 200 small | 14.4ms | 46.6ms | 0.31x | 🟢 | ✓ 5e-06 |
| 20×128×128 | Batch 20 medium | 7.3ms | 16.6ms | 0.44x | 🟢 | ✓ 9e-06 |
| 50×128×128 | Batch 50 medium | 10.6ms | 41.3ms | 0.26x | 💚 | ✓ 1e-05 |
| 10×256×256 | Batch 10 large | 47.6ms | 42.2ms | 1.13x | ⚪ | ✓ 1e-05 |
| 5×512×512 | Batch 5 huge | 64.0ms | 89.6ms | 0.71x | ⚪ | ✓ 9e-06 |

## QR Single Matrix (metalcore)

*Single matrix QR - CPU typically wins due to sequential dependencies*

| Shape | Config | Metal | CPU | Ratio | Status | Recon | Ortho |
|---|---|---|---|---|---|---|---|
| 16×16 | Tiny | 1.7ms | 49.1µs | 33.80x | 🔴 | ✓ 1e-06 | ✓ 3e-07 |
| 32×32 | Small 32 | 1.8ms | 64.7µs | 27.35x | 🔴 | ✓ 2e-06 | ✓ 6e-07 |
| 64×64 | Small 64 | 3.5ms | 93.1µs | 37.88x | 🔴 | ✓ 2e-06 | ✓ 8e-07 |
| 128×128 | Medium | 7.3ms | 344.5µs | 21.13x | 🔴 | ✓ 4e-06 | ✓ 6e-07 |
| 256×256 | Large 256 | 16.7ms | 1.1ms | 15.52x | 🔴 | ✓ 5e-06 | ✓ 7e-07 |
| 512×512 | Large 512 | 33.0ms | 4.4ms | 7.47x | 🔴 | ✓ 6e-06 | ✓ 1e-06 |
| 1024×1024 | Huge 1024 | 84.7ms | 25.0ms | 3.38x | 🔴 | ✓ 1e-05 | ✓ 1e-06 |
| 256×64 | Tall 4:1 | 5.5ms | 230.1µs | 23.73x | 🔴 | ✓ 4e-06 | ✓ 8e-07 |
| 256×128 | Tall 2:1 | 9.5ms | 741.2µs | 12.81x | 🔴 | ✓ 4e-06 | ✓ 7e-07 |
| 512×128 | Tall 4:1 large | 10.1ms | 1.2ms | 8.67x | 🔴 | ✓ 7e-06 | ✓ 8e-07 |
| 512×256 | Tall 2:1 large | 25.8ms | 2.3ms | 11.26x | 🔴 | ✓ 7e-06 | ✓ 1e-06 |
| 1000×200 | Tall 5:1 | 21.6ms | 2.6ms | 8.28x | 🔴 | ✓ 9e-06 | ✓ 1e-06 |
| 2000×500 | Huge tall | 66.5ms | 13.3ms | 5.01x | 🔴 | ✓ 2e-05 | ✓ 2e-06 |
| 4000×1000 | Massive | 297.8ms | 78.8ms | 3.78x | 🔴 | ✓ 2e-05 | ✓ 3e-06 |

## QR Batched (metalcore) ⭐ GPU WINS

*Batched QR - GPU processes all matrices in parallel, single dispatch*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 50×8×8 | Tiny 8x8 | 995.1µs | 2.2ms | 0.45x | 🟢 | ✓ 7e-07 |
| 100×8×8 | Batch 100 tiny | 921.8µs | 5.0ms | 0.18x | 💚 | ✓ 4e-07 |
| 500×8×8 | Batch 500 tiny | 1.3ms | 22.7ms | 0.06x | 💚 | ✓ 6e-07 |
| 50×16×16 | ML mini-batch 16 | 1.6ms | 2.5ms | 0.61x | 🟢 | ✓ 1e-06 |
| 100×16×16 | Batch 100 16x16 | 1.4ms | 4.9ms | 0.28x | 💚 | ✓ 1e-06 |
| 200×16×16 | Batch 200 16x16 | 1.8ms | 9.8ms | 0.18x | 💚 | ✓ 1e-06 |
| 500×16×16 | Batch 500 16x16 | 2.3ms | 24.7ms | 0.09x | 💚 | ✓ 6e-07 |
| 1000×16×16 | Batch 1000 16x16 | 3.5ms | 47.1ms | 0.07x | 💚 | ✓ 5e-07 |
| 50×32×32 | ML mini-batch 32 | 2.6ms | 2.7ms | 0.97x | ⚪ | ✓ 1e-06 |
| 100×32×32 | Batch 100 32x32 | 2.3ms | 5.3ms | 0.43x | 🟢 | ✓ 1e-06 |
| 200×32×32 | Batch 200 32x32 | 2.8ms | 11.0ms | 0.26x | 💚 | ✓ 2e-06 |
| 500×32×32 | Batch 500 32x32 | 6.2ms | 27.2ms | 0.23x | 💚 | ✓ 2e-06 |
| 50×48×48 | Batch 50 48x48 | 4.8ms | 3.2ms | 1.51x | 🟠 | ✓ 1e-06 |
| 100×48×48 | Batch 100 48x48 | 4.9ms | 6.6ms | 0.75x | ⚪ | ✓ 1e-06 |
| 100×64×32 | Tall batch | 2.6ms | 6.0ms | 0.43x | 🟢 | ✓ 1e-06 |
| 100×32×64 | Wide batch | 3.7ms | 5.8ms | 0.64x | 🟢 | ✓ 1e-06 |
| 200×64×32 | Large tall batch | 4.3ms | 12.0ms | 0.36x | 🟢 | ✓ 1e-06 |

## Cholesky (metalcore) ⭐ GPU WINS

*Batched Cholesky decomposition with MAGMA-style shared memory*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 100×16×16 | Tiny batched | 467.6µs | 4.3ms | 0.11x | 💚 | ✓ 4e-06 |
| 500×16×16 | Large batch tiny | 582.9µs | 22.6ms | 0.03x | 💚 | ✓ 8e-06 |
| 100×32×32 | Small batched | 724.3µs | 4.4ms | 0.17x | 💚 | ✓ 8e-06 |
| 200×48×48 | Medium batched | 873.6µs | 9.5ms | 0.09x | 💚 | ✓ 2e-05 |
| 100×64×64 | Larger batched | 1.3ms | 4.5ms | 0.29x | 💚 | ✓ 2e-05 |

## Linear Solve (metalcore)

*Batched linear system solve using QR + TRSM*

| Shape | Config | Metal | CPU | Ratio | Status | Residual |
|---|---|---|---|---|---|---|
| 100×16×16 | Tiny batched | 1.6ms | 1.2ms | 1.38x | 🟠 | ✓ 9e-05 |
| 500×16×16 | Large batch tiny | 2.4ms | 5.9ms | 0.41x | 🟢 | ~ 3e-04 |
| 100×32×32 | Small batched | 2.6ms | 1.6ms | 1.62x | 🟠 | ~ 5e-03 |
| 200×48×48 | Medium batched | 8.9ms | 4.4ms | 2.04x | 🟠 | ~ 3e-03 |

## Eigendecomposition (metaleig)

*Symmetric eigenvalue decomposition*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 32×32 | Tiny | 746.6µs | 49.7µs | 15.02x | 🔴 | ✓ 2e-06 |
| 64×64 | Small | 762.9µs | 173.5µs | 4.40x | 🔴 | ✓ 3e-06 |
| 128×128 | Medium | 1.2ms | 621.8µs | 1.85x | 🟠 | ✓ 5e-06 |
| 256×256 | Large | 4.0ms | 2.7ms | 1.49x | 🟠 | ✓ 7e-06 |
| 512×512 | Very large | 14.1ms | 12.4ms | 1.14x | ⚪ | ✓ 1e-05 |
| 1024×1024 | Huge | 66.5ms | 64.1ms | 1.04x | ⚪ | ✓ 2e-05 |
| 100×32×32 | Batch 100 tiny | 6.2ms | 4.7ms | 1.31x | 🟠 | ✗ 3e+00 |
| 50×64×64 | Batch 50 small | 5.1ms | 9.0ms | 0.57x | 🟢 | ~ 2e-03 |
| 100×64×64 | Batch 100 small | 7.9ms | 20.7ms | 0.38x | 🟢 | ~ 3e-04 |
| 200×64×64 | Batch 200 small | 13.2ms | 35.3ms | 0.37x | 🟢 | ~ 4e-04 |
| 20×128×128 | Batch 20 medium | 5.1ms | 11.6ms | 0.43x | 🟢 | ~ 2e-03 |
| 50×128×128 | Batch 50 medium | 9.2ms | 28.5ms | 0.32x | 🟢 | ~ 4e-03 |
| 10×256×256 | Batch 10 large | 28.9ms | 26.9ms | 1.07x | ⚪ | ✓ 7e-06 |

## Pipeline Operations ⭐ GPU WINS (No Transfer)

*Chained operations where data stays on GPU - avoids costly memory transfers*

| Pipeline | Shape | GPU | Comparison | Ratio | Status |
|---|---|---|---|---|---|
| QR -> QR -> QR | 200×32×32 | 9.7ms | 31.2ms | 0.31x | 🟢 |
| SVD -> truncate (PCA) | 50×128×64 | 18.1ms | 16.5ms | 1.10x | ⚪ |
| QR -> matmul (ML) | 1000×16×16 | 5.3ms | 52.7ms | 0.10x | 💚 |
| Fast+Slow+Fast (GPU all) | 200×32×32 | 6.4ms | 3.4ms | 1.87x vs hybrid | 🟠 |
| Fast+Slow+Fast (vs CPU) | 200×32×32 | 6.4ms | 11.6ms | 0.55x vs CPU | 🟢 |

## LLM: Llama

*SVD performance on Llama weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 4096×4096 | Attention (7B) | 2.86s | 4.99s | 0.57x | 🟢 | ~ 1e-04 |
| 4096×11008 | MLP up (7B) | 2.94s | 10.03s | 0.29x | 💚 | ✓ 3e-05 |
| 8192×8192 | Attention (70B) | 21.27s | 36.23s | 0.59x | 🟢 | ~ 1e-04 |

## LLM: Mistral

*SVD performance on Mistral weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 4096×4096 | Attention | 2.90s | 4.93s | 0.59x | 🟢 | ✓ 3e-05 |
| 4096×14336 | MLP up | 2.95s | 11.73s | 0.25x | 💚 | ✓ 3e-05 |

## LLM: Qwen

*SVD performance on Qwen weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 4096×4096 | Attention | 2.85s | 4.99s | 0.57x | 🟢 | ✓ 4e-05 |
| 4096×11008 | MLP up | 2.91s | 10.04s | 0.29x | 💚 | ✓ 3e-05 |

## LLM: Gemma

*SVD performance on Gemma weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 3072×3072 | Attention | 933.7ms | 1.61s | 0.58x | 🟢 | ✓ 2e-05 |
| 3072×24576 | MLP up | 1.01s | 9.16s | 0.11x | 💚 | ✓ 2e-05 |

## LLM: Phi

*SVD performance on Phi weight matrix sizes*

| Shape | Layer | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 3072×3072 | Attention | 926.7ms | 1.56s | 0.59x | 🟢 | ✓ 2e-05 |
| 3072×8192 | MLP up | 935.5ms | 3.67s | 0.26x | 💚 | ✓ 2e-05 |

## Usage Recommendations

| Operation | When to Use Metal | When to Use CPU |
|---|---|---|
| SVD | Batched small/medium matrices | Single large matrices |
| QR (single) | — | Always (sequential dependencies) |
| QR (batched) | Many small matrices (10x speedup!) | Few matrices |
| EIGH | Batched symmetric matrices | Single large matrices |
| Pipeline | Keep data on GPU to avoid transfer cost | Single ops on CPU-resident data |
