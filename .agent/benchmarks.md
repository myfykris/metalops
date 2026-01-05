# Performance & Benchmarks

## Benchmark Results (metalcore consolidated)

Run with: `python benchmark.py --quick`

### Best GPU Speedups

| Operation | Size | GPU vs CPU |
|-----------|------|------------|
| **Gemma-7B MLP SVD** | 3072×24576 | **25x** (0.04x ratio) |
| **Cholesky batched** | 500×16×16 | **33x** (0.03x ratio) |
| **QR batched** | 1000×16×16 | **14x** (0.07x ratio) |
| **Llama-3-8B SVD** | 4096×14336 | **5.9x** (0.17x ratio) |

### Where GPU Wins
- Batched operations (QR, Cholesky, SVD, Eigh)
- Large tall matrices (M >> N)
- LLM weight matrices

### Where CPU Wins  
- Small single matrices (N < 64)
- Sequential dependencies (single QR)
- Small batch solve (overhead dominates)

## Benchmark Comparison
Use `--compare` flag to see delta from previous run:
```bash
python benchmark.py --quick --compare
```

Results saved to `benchmark_history.jsonl` with runtime tracking.
