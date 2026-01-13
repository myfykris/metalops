#!/usr/bin/env python3
"""
Competitive Benchmark: MetalCore vs MLX vs PyTorch MPS
======================================================
Run this to verify marketing claims about MetalCore performance.

FAIR BENCHMARKING METHODOLOGY:
- Each device operates on tensors created in its NATIVE memory space
- No tensor copies are timed - only the operation itself
- GPU sync (torch.mps.synchronize()) called before/after timing
- MLX uses mx.eval() for fair async completion

Requirements:
  pip install mlx torch metalcore

Usage:
  python competitive_benchmark.py
"""

import time
import torch
import numpy as np

# Check for MLX
try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("⚠️  MLX not installed. Install with: pip install mlx")

# Check for MetalCore
try:
    import metalcore
    HAS_METALCORE = True
except ImportError:
    HAS_METALCORE = False
    print("⚠️  MetalCore not installed")


def benchmark(fn, warmup=3, iters=10, sync_fn=None):
    """Benchmark a function with warmup and timing."""
    for _ in range(warmup):
        fn()
    if sync_fn:
        sync_fn()
    
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if sync_fn:
        sync_fn()
    return (time.perf_counter() - start) / iters


def print_result(name, times):
    """Print formatted benchmark result."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    # Sort by time
    sorted_times = sorted(times.items(), key=lambda x: x[1])
    fastest = sorted_times[0][1]
    
    for lib, t in sorted_times:
        ratio = t / fastest
        marker = "🏆" if ratio == 1.0 else ""
        print(f"  {lib:15s}: {t*1000:8.2f}ms  ({ratio:.2f}x) {marker}")


# =============================================================================
# Benchmark 1: Batched QR Decomposition
# =============================================================================
def benchmark_batched_qr():
    print("\n" + "="*60)
    print(" BENCHMARK: Batched QR Decomposition")
    print(" Shape: [500, 32, 32] (500 matrices of 32x32)")
    print(" NOTE: Each device uses tensor created in its native memory")
    print("="*60)
    
    times = {}
    
    # PyTorch CPU - tensor created on CPU
    A_cpu = torch.randn(500, 32, 32, device='cpu')
    times["PyTorch CPU"] = benchmark(
        lambda: torch.linalg.qr(A_cpu),
        sync_fn=lambda: None
    )
    
    # PyTorch MPS - tensor created on MPS (fair comparison)
    A_mps = torch.randn(500, 32, 32, device='mps')
    torch.mps.synchronize()  # Ensure tensor is ready
    def pytorch_mps_qr():
        return torch.linalg.qr(A_mps)
    times["PyTorch MPS"] = benchmark(
        pytorch_mps_qr,
        sync_fn=torch.mps.synchronize
    )
    
    # MetalCore - same MPS tensor
    if HAS_METALCORE:
        def metalcore_qr():
            return metalcore.qr(A_mps)
        times["MetalCore"] = benchmark(
            metalcore_qr,
            sync_fn=torch.mps.synchronize
        )
    
    # MLX - create native MLX array
    if HAS_MLX:
        # MLX array created natively (fair comparison)
        A_mlx = mx.random.normal(shape=(500, 32, 32))
        mx.eval(A_mlx)
        def mlx_qr():
            Q, R = mx.linalg.qr(A_mlx)
            mx.eval(Q, R)
            return Q, R
        times["MLX"] = benchmark(mlx_qr)
    
    print_result("Batched QR [500, 32, 32]", times)
    return times


# =============================================================================
# Benchmark 2: Batched Cholesky Decomposition
# =============================================================================
def benchmark_batched_cholesky():
    print("\n" + "="*60)
    print(" BENCHMARK: Batched Cholesky Decomposition")
    print(" Shape: [200, 48, 48] (200 SPD matrices of 48x48)")
    print(" NOTE: Each device uses tensor created in its native memory")
    print("="*60)
    
    times = {}
    
    # PyTorch CPU - create SPD matrices on CPU
    A_cpu = torch.randn(200, 48, 48, device='cpu')
    A_cpu = A_cpu @ A_cpu.transpose(-1, -2) + torch.eye(48) * 48
    times["PyTorch CPU"] = benchmark(
        lambda: torch.linalg.cholesky(A_cpu),
        sync_fn=lambda: None
    )
    
    # PyTorch MPS - create SPD matrices on MPS (fair comparison)
    A_mps = torch.randn(200, 48, 48, device='mps')
    A_mps = A_mps @ A_mps.transpose(-1, -2) + torch.eye(48, device='mps') * 48
    torch.mps.synchronize()
    def pytorch_mps_chol():
        return torch.linalg.cholesky(A_mps)
    times["PyTorch MPS"] = benchmark(
        pytorch_mps_chol,
        sync_fn=torch.mps.synchronize
    )
    
    # MetalCore
    if HAS_METALCORE:
        def metalcore_chol():
            return metalcore.cholesky(A_mps)
        times["MetalCore"] = benchmark(
            metalcore_chol,
            sync_fn=torch.mps.synchronize
        )
    
    # MLX - create native MLX array
    if HAS_MLX:
        A_mlx = mx.random.normal(shape=(200, 48, 48))
        A_mlx = A_mlx @ mx.transpose(A_mlx, axes=(0, 2, 1)) + mx.eye(48) * 48
        mx.eval(A_mlx)
        def mlx_chol():
            L = mx.linalg.cholesky(A_mlx)
            mx.eval(L)
            return L
        times["MLX"] = benchmark(mlx_chol)
    
    print_result("Batched Cholesky [200, 48, 48]", times)
    return times


# =============================================================================
# Benchmark 3: EmbeddingBag (sum aggregation)
# =============================================================================
def benchmark_embedding_bag():
    print("\n" + "="*60)
    print(" BENCHMARK: EmbeddingBag (sum mode)")
    print(" Vocab: 32000x4096, Batch: 16 sequences, 128 tokens each")
    print(" NOTE: Each device uses tensor created in its native memory")
    print("="*60)
    
    times = {}
    
    vocab_size, embed_dim = 32000, 4096
    batch_size, seq_len = 16, 128
    
    # PyTorch CPU - tensors created on CPU
    weight_cpu = torch.randn(vocab_size, embed_dim, device='cpu')
    indices_cpu = torch.randint(0, vocab_size, (batch_size * seq_len,), device='cpu')
    offsets_cpu = torch.arange(0, batch_size * seq_len + 1, seq_len, device='cpu')
    times["PyTorch CPU"] = benchmark(
        lambda: torch.nn.functional.embedding_bag(
            indices_cpu, weight_cpu, offsets_cpu, mode='sum'
        )
    )
    
    # PyTorch MPS - tensors created on MPS (fair comparison)
    weight_mps = torch.randn(vocab_size, embed_dim, device='mps')
    indices_mps = torch.randint(0, vocab_size, (batch_size * seq_len,), device='mps')
    offsets_mps = torch.arange(0, batch_size * seq_len + 1, seq_len, device='mps')
    torch.mps.synchronize()
    
    def pytorch_mps_emb():
        return torch.nn.functional.embedding_bag(
            indices_mps, weight_mps, offsets_mps, mode='sum'
        )
    times["PyTorch MPS"] = benchmark(
        pytorch_mps_emb,
        sync_fn=torch.mps.synchronize
    )
    
    # MetalCore
    if HAS_METALCORE:
        def metalcore_emb():
            return metalcore.embedding_bag(weight_mps, indices_mps, offsets_mps, mode='sum')
        times["MetalCore"] = benchmark(
            metalcore_emb,
            sync_fn=torch.mps.synchronize
        )
    
    # Note: MLX doesn't have a direct EmbeddingBag equivalent
    
    print_result("EmbeddingBag [32000x4096, B=16]", times)
    return times


# =============================================================================
# Benchmark 4: SVD on LLM-sized matrices
# =============================================================================
def benchmark_llm_svd():
    print("\n" + "="*60)
    print(" BENCHMARK: SVD on LLM-sized matrix")
    print(" Shape: [4096, 11008] (Llama-7B MLP shape)")
    print(" NOTE: Each device uses tensor created in its native memory")
    print("="*60)
    
    times = {}
    
    # PyTorch CPU - tensor created on CPU
    A_cpu = torch.randn(4096, 11008, device='cpu')
    times["PyTorch CPU"] = benchmark(
        lambda: torch.linalg.svd(A_cpu, full_matrices=False),
        iters=3
    )
    
    # PyTorch MPS - tensor created on MPS (fair comparison)
    A_mps = torch.randn(4096, 11008, device='mps')
    torch.mps.synchronize()
    def pytorch_mps_svd():
        return torch.linalg.svd(A_mps, full_matrices=False)
    times["PyTorch MPS"] = benchmark(
        pytorch_mps_svd,
        sync_fn=torch.mps.synchronize,
        iters=3
    )
    
    # MetalCore
    if HAS_METALCORE:
        def metalcore_svd():
            return metalcore.svd(A_mps)
        times["MetalCore"] = benchmark(
            metalcore_svd,
            sync_fn=torch.mps.synchronize,
            iters=3
        )
    
    # MLX - create native MLX array
    if HAS_MLX:
        A_mlx = mx.random.normal(shape=(4096, 11008))
        mx.eval(A_mlx)
        def mlx_svd():
            U, S, Vt = mx.linalg.svd(A_mlx, stream=mx.cpu)
            mx.eval(U, S, Vt)
            return U, S, Vt
        times["MLX"] = benchmark(mlx_svd, iters=3)
    
    print_result("SVD [4096, 11008]", times)
    return times


# =============================================================================
# Benchmark 5: AdamW Optimizer Step
# =============================================================================
def benchmark_adamw():
    print("\n" + "="*60)
    print(" BENCHMARK: AdamW Optimizer Step")
    print(" Params: 10M parameters")
    print("="*60)
    
    times = {}
    n_params = 10_000_000
    
    # PyTorch MPS
    param = torch.randn(n_params, device="mps", requires_grad=True)
    grad = torch.randn(n_params, device="mps")
    
    # Create optimizer
    opt_torch = torch.optim.AdamW([param], lr=1e-3)
    
    def pytorch_adamw():
        param.grad = grad.clone()
        opt_torch.step()
    
    times["PyTorch MPS"] = benchmark(
        pytorch_adamw,
        sync_fn=torch.mps.synchronize
    )
    
    # MetalCore
    if HAS_METALCORE:
        param2 = torch.randn(n_params, device="mps", requires_grad=True)
        opt_metal = metalcore.MetalAdamW([param2], lr=1e-3)
        
        def metalcore_adamw():
            param2.grad = grad.clone()
            opt_metal.step()
        
        times["MetalCore"] = benchmark(
            metalcore_adamw,
            sync_fn=torch.mps.synchronize
        )
    
    # Note: MLX uses different optimizer API, not directly comparable
    
    print_result("AdamW Step [10M params]", times)
    return times


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print(" COMPETITIVE BENCHMARK: MetalCore vs MLX vs PyTorch")
    print(" Testing claims for fastest operations on Apple Silicon")
    print("="*60)
    print(f"\n Libraries detected:")
    print(f"   PyTorch: ✅ {torch.__version__}")
    print(f"   MLX:     {'✅' if HAS_MLX else '❌'}")
    print(f"   MetalCore: {'✅' if HAS_METALCORE else '❌'}")
    
    results = {}
    
    try:
        results["Batched QR"] = benchmark_batched_qr()
    except Exception as e:
        print(f"  ❌ Batched QR failed: {e}")
    
    try:
        results["Batched Cholesky"] = benchmark_batched_cholesky()
    except Exception as e:
        print(f"  ❌ Batched Cholesky failed: {e}")
    
    try:
        results["EmbeddingBag"] = benchmark_embedding_bag()
    except Exception as e:
        print(f"  ❌ EmbeddingBag failed: {e}")
    
    try:
        results["LLM SVD"] = benchmark_llm_svd()
    except Exception as e:
        print(f"  ❌ LLM SVD failed: {e}")
    
    try:
        results["AdamW"] = benchmark_adamw()
    except Exception as e:
        print(f"  ❌ AdamW failed: {e}")
    
    print("\n" + "="*60)
    print(" SUMMARY: Can We Claim 'Fastest'?")
    print("="*60)
    
    for test_name, times in results.items():
        if not times:
            continue
        sorted_times = sorted(times.items(), key=lambda x: x[1])
        winner = sorted_times[0][0]
        
        if "MetalCore" in times:
            mc_time = times["MetalCore"]
            best_time = sorted_times[0][1]
            if winner == "MetalCore":
                print(f"  ✅ {test_name}: MetalCore WINS")
            else:
                ratio = mc_time / best_time
                print(f"  ❌ {test_name}: {winner} wins ({ratio:.1f}x faster than MetalCore)")
        else:
            print(f"  ⚠️  {test_name}: MetalCore not tested")
