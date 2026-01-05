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

## Usage Recommendations

| Operation | When to Use Metal | When to Use CPU |
|---|---|---|
| SVD | Batched small/medium matrices | Single large matrices |
| QR (single) | — | Always (sequential dependencies) |
| QR (batched) | Many small matrices (10x speedup!) | Few matrices |
| EIGH | Batched symmetric matrices | Single large matrices |
| Pipeline | Keep data on GPU to avoid transfer cost | Single ops on CPU-resident data |
