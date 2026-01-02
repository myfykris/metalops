import torch
import metalsvd
import time
import math

class BenchmarkRunner:
    def __init__(self):
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        print(f"Running Benchmarks on: {self.device}")
        
        # Explicit Library Load Warmup
        # This ensures all Metal kernels are compiled/loaded before any timing starts.
        if self.device.type == 'mps':
            print("Warming up Metal kernels to exclude laoding time...")
            dummy = torch.randn(1, 16, 16, device=self.device)
            metalsvd.svd(dummy) 
            torch.mps.synchronize()
            print("Warmup complete.\n")
            
        self.results = []

    def run_case(self, name, func, *args, warmups=3, iters=5):
        print(f"Benchmarking: {name}...")
        
        # Warmup
        for _ in range(warmups):
            func(*args)
        torch.mps.synchronize()
        
        # Timing
        start = time.time()
        for _ in range(iters):
            func(*args)
            torch.mps.synchronize()
        end = time.time()
        
        avg_time = (end - start) / iters
        self.results.append((name, avg_time))
        print(f"  -> {avg_time*1000:.2f} ms")
        return avg_time

    def compare(self, name, baseline_func, target_func, *args, iters=5):
        print(f"Comparing: {name}")
        t_base = self.run_case(name + " (Baseline)", baseline_func, *args, iters=iters)
        t_target = self.run_case(name + " (MetalSVD)", target_func, *args, iters=iters)
        
        speedup = t_base / t_target
        print(f"  => Speedup: {speedup:.2f}x\n")
        return speedup

    def print_summary(self):
        print("\n" + "="*40)
        print("BENCHMARK SUMMARY")
        print("="*40)
        print(f"{'Benchmark Name':<30} | {'Time (ms)':<10}")
        print("-" * 45)
        for name, t in self.results:
            print(f"{name:<30} | {t*1000:10.2f}")
        print("="*40)

def benchmark_suite():
    runner = BenchmarkRunner()
    device = runner.device
    
    # 1. Batched Small Matrices (Attention Heads / LoRA)
    # 64 x 128 x 128
    B, M, N = 64, 128, 128
    A_small = torch.randn(B, M, N, device=device)
    
    runner.compare(
        "Batched SVD (64x128x128)",
        lambda: torch.linalg.svd(A_small),
        lambda: metalsvd.svd(A_small)
    )
    
    # 2. Medium Square Matrix
    # 1024 x 1024
    # Full Decomposition
    A_med = torch.randn(1024, 1024, device=device)
    # Baseline on CPU might be slow? MPS usually falls back.
    # We compare single run.
    runner.compare(
        "Square SVD (1024x1024)",
        lambda: torch.linalg.svd(A_med),
        lambda: metalsvd.svd(A_med)
    )
    
    # 3. Large Matrix Randomized SVD
    # 4096 x 4096, Rank 100
    M_large = 4096
    A_large = torch.randn(M_large, M_large, device=device)
    
    def rsvd_run():
        metalsvd.randomized_svd(A_large, k=100, n_iter=2)
        
    def torch_svd_run():
        # Torch doesn't have rSVD easily accessible on MPS usually.
        # We verify full SVD time (truncating later)
        torch.linalg.svd(A_large)
        
    runner.compare(
        "Large rSVD vs Full SVD (4096^2)",
        torch_svd_run,
        rsvd_run,
        iters=2 # Huge, fewer iters
    )
    
    # 4. FP16 vs FP32 Performance
    # Using 10k x 10k rSVD for max load
    M_huge = 8192
    A_fp32 = torch.randn(M_huge, M_huge, device=device, dtype=torch.float32)
    A_fp16 = A_fp32.to(torch.float16)
    
    runner.run_case("Huge rSVD FP32 (8192^2)", lambda: metalsvd.randomized_svd(A_fp32, k=100), iters=3)
    runner.run_case("Huge rSVD FP16 (8192^2)", lambda: metalsvd.randomized_svd(A_fp16, k=100), iters=3)
    
    # 5. Correctness Check (Sanity)
    print("\nVerifying Accuracy on Huge Matrix (FP16)...")
    # Generate Low Rank for valid reconstruction check
    K_sanity = 50
    U_h = torch.randn(M_huge, K_sanity, device=device, dtype=torch.float32)
    U_h = torch.linalg.qr(U_h)[0].to(torch.float16)
    V_h = torch.randn(M_huge, K_sanity, device=device, dtype=torch.float32)
    V_h = torch.linalg.qr(V_h)[0].to(torch.float16)
    S_h = torch.linspace(100, 10, K_sanity, device=device, dtype=torch.float16)
    A_sanity = U_h @ torch.diag(S_h) @ V_h.T
    
    U, S, V = metalsvd.randomized_svd(A_sanity, k=K_sanity, n_iter=2)
    A_approx = U @ torch.diag(S) @ V.T
    
    diff = (A_sanity - A_approx).norm()
    norm = A_sanity.norm()
    print(f"Relative Error: {diff/norm:.6f}")
    
    runner.print_summary()

if __name__ == "__main__":
    benchmark_suite()
