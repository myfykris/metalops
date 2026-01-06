# Metalops Benchmark Results

*Last updated: 2026-01-05T16:46:09.956753*

**Legend:** 💚 GPU wins big (>3x) | 🟢 GPU wins | ⚪ Close | 🟠 CPU wins | 🔴 CPU wins big (>3x)

## Cholesky (metalcore) ⭐ GPU WINS

*Batched Cholesky decomposition with MAGMA-style shared memory*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 100×16×16 | Tiny batched | 283.5µs | 4.9ms | 0.06x | 💚 | ✓ 4e-06 |
| 100×32×32 | Small batched | 434.2µs | 4.9ms | 0.09x | 💚 | ✓ 1e-05 |
| 100×64×64 | Larger batched | 1.1ms | 5.0ms | 0.22x | 💚 | ✓ 2e-05 |
| 200×48×48 | Medium batched | 1.1ms | 10.1ms | 0.10x | 💚 | ✓ 2e-05 |
| 500×16×16 | Large batch tiny | 457.2µs | 24.2ms | 0.02x | 💚 | ✓ 4e-06 |

## AdamW (metalcore) ⭐ GPU WINS

*Fused AdamW optimizer step vs torch.optim.AdamW*

| Params | Size | Metal | CPU | Ratio | Status |
|---|---|---|---|---|---|
| 10M Params | N=10485760 | 1.1ms | 3.1ms | 0.36x | 🟢 |
| 16M Params | N=16777216 | 1.6ms | 4.7ms | 0.35x | 🟢 |
| 1M Params | N=1048576 | 327.9µs | 592.9µs | 0.55x | 🟢 |
| 1M Params | N=1048576 bf16 | 341.5µs | 1.0ms | 0.33x | 🟢 |
| 1M Params | N=1048576 fp16 | 319.8µs | 1.0ms | 0.32x | 🟢 |
| 1M Params | N=1048576 fp32 | 296.4µs | 913.0µs | 0.32x | 🟢 |

## QR Batched (metalcore) ⭐ GPU WINS

*Batched QR via Householder reflections*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 1000×16×16 | Batch 1000 16x16 | 3.4ms | 51.4ms | 0.07x | 💚 | ✓ 7e-07 |
| 100×16×16 | Batch 100 16x16 | 1.0ms | 5.1ms | 0.20x | 💚 | ✓ 2e-06 |
| 100×32×32 | Batch 100 32x32 | 2.3ms | 5.7ms | 0.40x | 🟢 | ✓ 1e-06 |
| 100×32×64 | Wide batch | 4.2ms | 5.9ms | 0.71x | ⚪ | ✓ 1e-06 |
| 100×48×48 | Batch 100 48x48 | 4.9ms | 6.7ms | 0.73x | ⚪ | ✓ 1e-06 |
| 100×64×32 | Tall batch | 2.8ms | 6.2ms | 0.45x | 🟢 | ✓ 1e-06 |
| 100×8×8 | Batch 100 tiny | 791.8µs | 4.9ms | 0.16x | 💚 | ✓ 5e-07 |
| 200×16×16 | Batch 200 16x16 | 1.3ms | 11.1ms | 0.12x | 💚 | ✓ 8e-07 |
| 200×32×32 | Batch 200 32x32 | 3.0ms | 11.2ms | 0.27x | 💚 | ✓ 1e-06 |
| 200×64×32 | Large tall batch | 4.9ms | 12.4ms | 0.40x | 🟢 | ✓ 1e-06 |
| 500×16×16 | Batch 500 16x16 | 3.1ms | 25.7ms | 0.12x | 💚 | ✓ 7e-07 |
| 500×32×32 | Batch 500 32x32 | 6.8ms | 28.1ms | 0.24x | 💚 | ✓ 1e-06 |
| 500×8×8 | Batch 500 tiny | 2.1ms | 24.7ms | 0.09x | 💚 | ✓ 6e-07 |
| 50×16×16 | ML mini-batch 16 | 971.8µs | 2.6ms | 0.38x | 🟢 | ✓ 2e-06 |
| 50×32×32 | ML mini-batch 32 | 2.4ms | 2.8ms | 0.85x | ⚪ | ✓ 8e-07 |
| 50×48×48 | Batch 50 48x48 | 5.0ms | 3.4ms | 1.47x | 🟠 | ✓ 1e-06 |
| 50×8×8 | Tiny 8x8 | 714.0µs | 2.5ms | 0.29x | 💚 | ✓ 8e-07 |

## Linear Solve (metalcore) ⭐ GPU WINS

*Fused LU decomposition with forward/back substitution*

| Shape | Config | Metal | CPU | Ratio | Status | Residual |
|---|---|---|---|---|---|---|
| 100×16×16 | Tiny batched | 1.6ms | 1.2ms | 1.38x | 🟠 | ✓ 9e-05 |
| 100×16×16 | Tiny batched bf16 | 651.1µs | 1.2ms | 0.55x | 🟢 | ✗ nan |
| 100×16×16 | Tiny batched fp16 | 368.0µs | 1.2ms | 0.30x | 💚 | ✗ nan |
| 100×16×16 | Tiny batched fp32 | 464.6µs | 1.3ms | 0.37x | 🟢 | ~ 1e-04 |
| 100×32×32 | Small batched | 2.6ms | 1.6ms | 1.62x | 🟠 | ~ 5e-03 |
| 100×32×32 | Small batched bf16 | 1.4ms | 1.6ms | 0.84x | ⚪ | ✗ nan |
| 100×32×32 | Small batched fp16 | 1.2ms | 1.6ms | 0.76x | ⚪ | ✗ nan |
| 100×32×32 | Small batched fp32 | 1.5ms | 1.6ms | 0.92x | ⚪ | ~ 1e-04 |
| 200×48×48 | Medium batched | 8.9ms | 4.4ms | 2.04x | 🟠 | ~ 3e-03 |
| 200×48×48 | Medium batched bf16 | 2.2ms | 4.1ms | 0.53x | 🟢 | ✗ nan |
| 200×48×48 | Medium batched fp16 | 2.4ms | 4.1ms | 0.59x | 🟢 | ✗ nan |
| 200×48×48 | Medium batched fp32 | 2.4ms | 4.3ms | 0.56x | 🟢 | ~ 6e-03 |
| 500×16×16 | Large batch tiny | 2.4ms | 5.9ms | 0.41x | 🟢 | ~ 3e-04 |
| 500×16×16 | Large batch tiny bf16 | 672.1µs | 6.0ms | 0.11x | 💚 | ✗ nan |
| 500×16×16 | Large batch tiny fp16 | 442.3µs | 6.1ms | 0.07x | 💚 | ✗ nan |
| 500×16×16 | Large batch tiny fp32 | 471.4µs | 6.0ms | 0.08x | 💚 | ~ 6e-04 |

## RMSNorm (metalcore) ⭐ GPU WINS

*Fused RMSNorm kernel vs torch.nn.RMSNorm*

| Shape | Config | Metal | CPU | Ratio | Status |
|---|---|---|---|---|---|
| 1024x1024 | Fwd+Bwd | 1.0ms | 947.5µs | 1.08x | ⚪ |
| 1x4096 | Fwd+Bwd | 584.1µs | 592.1µs | 0.99x | ⚪ |
| 1x4096 | Fwd+Bwd bf16 | 754.7µs | 703.8µs | 1.07x | ⚪ |
| 1x4096 | Fwd+Bwd fp16 | 706.1µs | 721.3µs | 0.98x | ⚪ |
| 1x4096 | Fwd+Bwd fp32 | 667.6µs | 579.4µs | 1.15x | ⚪ |
| 32x4096 | Fwd+Bwd | 648.8µs | 586.3µs | 1.11x | ⚪ |
| 32x4096 | Fwd+Bwd bf16 | 856.6µs | 789.8µs | 1.08x | ⚪ |
| 32x4096 | Fwd+Bwd fp16 | 750.2µs | 797.7µs | 0.94x | ⚪ |
| 32x4096 | Fwd+Bwd fp32 | 763.6µs | 753.3µs | 1.01x | ⚪ |
| 4096x4096 | Fwd+Bwd | 3.9ms | 9.9ms | 0.40x | 🟢 |

## SVD (metalcore) ⭐ GPU WINS

*Singular Value Decomposition using metalcore*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 100×32×32 | Batch 100 tiny | 6.9ms | 7.2ms | 0.97x | ⚪ | ✓ 3e-06 |
| 100×64×64 | Batch 100 small | 15.9ms | 23.8ms | 0.67x | 🟢 | ✓ 7e-06 |
| 1024×1024 | Huge square | 69.6ms | 91.1ms | 0.76x | ⚪ | ✓ 1e-05 |
| 1024×512 | Tall matrix | 16.9ms | 47.3ms | 0.36x | 🟢 | ✓ 9e-06 |
| 10×256×256 | Batch 10 large | 47.7ms | 40.4ms | 1.18x | ⚪ | ✓ 1e-05 |
| 128×128 | Medium square | 1.7ms | 852.7µs | 2.02x | 🟠 | ✓ 5e-06 |
| 128×256 | Wide 1:2 | 2.9ms | 2.0ms | 1.47x | 🟠 | ✓ 7e-06 |
| 200×64×64 | Batch 200 small | 17.2ms | 47.5ms | 0.36x | 🟢 | ✓ 5e-06 |
| 2048×2048 | Massive square | 385.7ms | 571.3ms | 0.68x | 🟢 | ✓ 7e-05 |
| 2048×512 | Very tall | 15.0ms | 118.6ms | 0.13x | 💚 | ✓ 9e-06 |
| 20×128×128 | Batch 20 medium | 9.4ms | 16.9ms | 0.56x | 🟢 | ✓ 9e-06 |
| 256×128 | Tall 2:1 | 3.7ms | 1.8ms | 2.07x | 🟠 | ✓ 7e-06 |
| 256×256 | Large square | 5.5ms | 4.1ms | 1.33x | 🟠 | ✓ 8e-06 |
| 3072×24576 | Gemma-7B MLP (3072x24576) | 1.00s | 28.31s | 0.04x | 💚 | ✓ 4e-05 |
| 3072×8192 | Phi-3-mini MLP (3072x8192) | 941.0ms | 5.29s | 0.18x | 💚 | ✓ 2e-05 |
| 32×32 | Tiny | 906.6µs | 79.4µs | 11.42x | 🔴 | ✓ 3e-06 |
| 4096×11008 | Llama-2-7B MLP (4096x11008) | 3.59s | 17.06s | 0.21x | 💚 | ✓ 3e-05 |
| 4096×11008 | Qwen-7B MLP (4096x11008) | 2.90s | 11.67s | 0.25x | 💚 | ✓ 3e-05 |
| 4096×14336 | Llama-3-8B MLP (4096x14336) | 3.77s | 21.56s | 0.17x | 💚 | ✓ 3e-05 |
| 4096×14336 | Mistral-7B MLP (4096x14336) | 3.65s | 19.54s | 0.19x | 💚 | ✓ 3e-05 |
| 4096×4096 | Llama-7B attn (4096x4096) | 4.02s | 7.83s | 0.51x | 🟢 | ✓ 6e-05 |
| 50×128×128 | Batch 50 medium | 11.3ms | 42.2ms | 0.27x | 💚 | ✓ 9e-06 |
| 50×64×64 | Batch 50 small | 8.8ms | 12.0ms | 0.74x | ⚪ | ✓ 5e-06 |
| 512×256 | Tall 2:1 large | 7.5ms | 8.5ms | 0.88x | ⚪ | ✓ 1e-05 |
| 512×512 | Very large | 14.8ms | 21.3ms | 0.69x | 🟢 | ✓ 8e-06 |
| 5×512×512 | Batch 5 huge | 63.6ms | 90.8ms | 0.70x | ⚪ | ~ 1e-04 |
| 64×64 | Small square | 1.1ms | 244.3µs | 4.49x | 🔴 | ✓ 4e-06 |
| 8192×8192 | Llama-70B attn (8192x8192) | 28.10s | 47.81s | 0.59x | 🟢 | ~ 1e-04 |

## Eigendecomposition (metaleig)

*Symmetric eigenvalue decomposition*

| Shape | Config | Metal | CPU | Ratio | Status | Recon Error |
|---|---|---|---|---|---|---|
| 100×32×32 | Batch 100 tiny | 6.2ms | 4.9ms | 1.27x | ⚪ | ✗ 3e+00 |
| 100×64×64 | Batch 100 small | 8.3ms | 17.8ms | 0.47x | 🟢 | ~ 1e-03 |
| 1024×1024 | Huge | 66.5ms | 64.1ms | 1.04x | ⚪ | ✓ 2e-05 |
| 10×256×256 | Batch 10 large | 28.3ms | 25.7ms | 1.10x | ⚪ | ✓ 6e-06 |
| 128×128 | Medium | 1.2ms | 584.9µs | 2.13x | 🟠 | ✓ 5e-06 |
| 200×64×64 | Batch 200 small | 14.7ms | 35.0ms | 0.42x | 🟢 | ~ 2e-04 |
| 20×128×128 | Batch 20 medium | 7.4ms | 11.4ms | 0.65x | 🟢 | ~ 2e-03 |
| 256×256 | Large | 3.7ms | 2.6ms | 1.42x | 🟠 | ✓ 6e-06 |
| 32×32 | Tiny | 1.1ms | 50.2µs | 21.37x | 🔴 | ✓ 3e-06 |
| 50×128×128 | Batch 50 medium | 9.2ms | 28.7ms | 0.32x | 🟢 | ~ 3e-03 |
| 50×64×64 | Batch 50 small | 4.5ms | 8.8ms | 0.52x | 🟢 | ~ 1e-04 |
| 512×512 | Very large | 14.1ms | 12.4ms | 1.14x | ⚪ | ✓ 1e-05 |
| 64×64 | Small | 939.6µs | 177.5µs | 5.30x | 🔴 | ✓ 3e-06 |

## QR Single Matrix (metalcore)

*Single matrix QR factorization*

| Shape | Config | Metal | CPU | Ratio | Status | Q Error | R Error |
|---|---|---|---|---|---|---|---|
| 1000×200 | Tall 5:1 | 21.6ms | 2.6ms | 8.28x | 🔴 | ✓ 9e-06 | ✓ 1e-06 |
| 1024×1024 | Huge 1024 | 84.7ms | 25.0ms | 3.38x | 🔴 | ✓ 1e-05 | ✓ 1e-06 |
| 128×128 | Medium | 909.0µs | 231.7µs | 3.92x | 🔴 | ✓ 4e-06 | ✓ 6e-07 |
| 16×16 | Tiny | 1.5ms | 54.6µs | 27.76x | 🔴 | ✓ 6e-07 | ✓ 2e-07 |
| 2000×500 | Huge tall | 66.5ms | 13.3ms | 5.01x | 🔴 | ✓ 2e-05 | ✓ 2e-06 |
| 256×128 | Tall 2:1 | 1.3ms | 545.3µs | 2.33x | 🟠 | ✓ 5e-06 | ✓ 6e-07 |
| 256×256 | Large 256 | 1.6ms | 904.3µs | 1.77x | 🟠 | ✓ 5e-06 | ✓ 8e-07 |
| 256×64 | Tall 4:1 | 804.6µs | 215.4µs | 3.74x | 🔴 | ✓ 2e-06 | ✓ 7e-07 |
| 32×32 | Small 32 | 1.0ms | 59.4µs | 17.36x | 🔴 | ✓ 1e-06 | ✓ 4e-07 |
| 4000×1000 | Massive | 297.8ms | 78.8ms | 3.78x | 🔴 | ✓ 2e-05 | ✓ 3e-06 |
| 512×128 | Tall 4:1 large | 1.9ms | 1.0ms | 1.86x | 🟠 | ✓ 6e-06 | ✓ 8e-07 |
| 512×256 | Tall 2:1 large | 2.9ms | 1.9ms | 1.56x | 🟠 | ✓ 5e-06 | ✓ 8e-07 |
| 512×512 | Large 512 | 5.3ms | 4.2ms | 1.27x | ⚪ | ✓ 6e-06 | ✓ 1e-06 |
| 64×64 | Small 64 | 732.4µs | 91.1µs | 8.04x | 🔴 | ✓ 2e-06 | ✓ 5e-07 |

## Usage Recommendations

| Operation | When to Use Metal | When to Use CPU |
|---|---|---|
| EIGH | Batched symmetric matrices | Single large matrices |
| Pipeline | Keep data on GPU to avoid transfer cost | Single ops on CPU-resident data |
| QR (batched) | Many small matrices (10x speedup!) | Few matrices |
| QR (single) | — | Always (sequential dependencies) |
| SVD | Batched small/medium matrices | Single large matrices |
