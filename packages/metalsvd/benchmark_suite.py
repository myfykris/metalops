import sys
import torch
import metalsvd
import time
import math

class BenchmarkRunner:
    def __init__(self):
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        print(f"Running Benchmarks on: {self.device}")
        
        self.filter_str = ""
        self.lite_mode = False
        
        for arg in sys.argv[1:]:
             if arg == "--lite":
                  self.lite_mode = True
                  print("Running in LITE MODE (Reduced iterations)")
             else:
                  self.filter_str = arg

        if self.filter_str:
            print(f"Filtering benchmarks by: '{self.filter_str}'")

        # Explicit Library Load Warmup
        if self.device.type == 'mps':
            print("Warming up Metal kernels to exclude laoding time...")
            dummy = torch.randn(1, 16, 16, device=self.device)
            metalsvd.svd(dummy) 
            torch.mps.synchronize()
            print("Warmup complete.\n")
            
        self.results = []

    def run_case(self, name, func, *args, warmups=3, iters=5, record=False):
        if self.filter_str and self.filter_str.lower() not in name.lower():
             return 0.0 # Skip run_case if called directly too? No, mainly for compare.

        print(f"Benchmarking: {name}...")
        
        # Lite Mode Overrides
        if self.lite_mode:
             warmups = 0
             iters = 1
             
        # Warmup
        for _ in range(warmups):
            func(*args)
            torch.mps.synchronize()
        
        time.sleep(0.1)
        
        start = time.time()
        for i in range(iters):
            func(*args)
            torch.mps.synchronize()
        end = time.time()
        
        avg_time = (end - start) / iters
        if record:
             self.results.append((name, avg_time))
        print(f"  -> {avg_time*1000:.2f} ms")
        return avg_time

    def compare(self, name, baseline_func, target_func, *args, iters=5):
        if self.filter_str and self.filter_str.lower() not in name.lower():
            return 1.0 # Skip
            
        print(f"Comparing: {name}")
        t_base = self.run_case(name + " (Baseline)", baseline_func, *args, iters=iters)
        t_target = self.run_case(name + " (MetalSVD)", target_func, *args, iters=iters)
        
        speedup = t_base / t_target
        self.results.append((name, t_base, t_target))
        
        pct = (speedup - 1.0) * 100.0
        if pct >= 0:
            print(f"  => +{pct:.2f}% FASTER\n")
        else:
            print(f"  => {pct:.2f}% SLOWER\n")

        return speedup

    def print_summary(self):
        print("\n" + "="*85)
        print("BENCHMARK SUMMARY")
        print("="*85)
        print(f"{'Test Name':<35} | {'Baseline (ms)':<15} | {'Metal (ms)':<15} | {'Change':<15}")
        print("-" * 85)
        for name, t_base, t_metal in self.results:
            speedup = t_base / t_metal
            pct = (speedup - 1.0) * 100.0
            if pct >= 0:
                res = f"+{pct:.2f}%"
            else:
                res = f"{pct:.2f}%" # Includes negative sign
            print(f"{name:<35} | {t_base*1000:15.2f} | {t_metal*1000:15.2f} | {res:<15}")
        print("="*85)

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
        lambda: metalsvd.svd(A_small),
        iters=50
    )
    
    # 2. Medium Square Matrix
    # 256 x 256
    # Full Decomposition
    A_med = torch.randn(256, 256, device=device)
    # Baseline on CPU might be slow? MPS usually falls back.
    # We compare single run.
    runner.compare(
        "Square SVD (256x256)",
        lambda: torch.linalg.svd(A_med),
        lambda: metalsvd.svd(A_med),
        iters=3
    )

    # 3. Batched Medium Matrix (Proof of Specialized Kernel Throughput)
    # 64 x 256 x 256
    A_med_batch = torch.randn(64, 256, 256, device=device)
    runner.compare(
        "Batched SVD (64x256x256)",
        lambda: torch.linalg.svd(A_med_batch),
        lambda: metalsvd.svd(A_med_batch),
        iters=10
    )

    # User Priority 1: Huge Matrices
    A_huge_fat = torch.randn(4096, 11008, device=device)
    runner.compare("Huge Fat (4096x11008)", 
        lambda: torch.linalg.svd(A_huge_fat),
        lambda: metalsvd.svd(A_huge_fat),
        iters=2 # Low iters for huge tests
    )

    A_huge_tall = torch.randn(11008, 4096, device=device)
    runner.compare("Huge Tall (11008x4096)", 
        lambda: torch.linalg.svd(A_huge_tall),
        lambda: metalsvd.svd(A_huge_tall),
        iters=2
    )
    
    # 3. Large Matrix Randomized SVD
    # 1024 x 1024, Rank 100
    M_large = 1024
    A_large = torch.randn(M_large, M_large, device=device)
    
    def rsvd_run():
        metalsvd.randomized_svd(A_large, k=100, n_iter=2)
        
    def torch_svd_run():
        # Torch doesn't have rSVD easily accessible on MPS usually.
        # We verify full SVD time (truncating later)
        torch.linalg.svd(A_large)
        
    runner.compare(
        "Large rSVD vs Full SVD (1024^2)",
        torch_svd_run,
        rsvd_run,
        iters=5 # Huge, fewer iters
    )
    
    # 4. FP16 vs FP32 Performance
    # Using 2048 x 2048 rSVD for max load (Reduced from 8192)
    M_huge = 2048
    A_fp32 = torch.randn(M_huge, M_huge, device=device, dtype=torch.float32)
    A_fp16 = A_fp32.to(torch.float16)
    
    runner.run_case("Huge rSVD FP32 (8192^2)", lambda: metalsvd.randomized_svd(A_fp32, k=100), iters=3)
    runner.run_case("Huge rSVD FP16 (8192^2)", lambda: metalsvd.randomized_svd(A_fp16, k=100), iters=3)
    
    
    # 6. Exhaustive Permutations (User Request)
    print("\n" + "="*40)
    print("EXHAUSTIVE PERMUTATIONS (Size x Shape)")
    print("="*40)
    
    sizes = [32, 64, 128, 256, 512] # Reduced max size
    shapes = [
        ("Square", lambda N: (1, N, N)), 
        ("Tall", lambda N: (1, 2*N, N)),
        ("Wide", lambda N: (1, N, 2*N)) # Library should transpose internally or error? Library supports N<=M.
        # metalsvd handles Wide by check or transpose?
        # Standard SVD requires M >= N. If Wide, we just transpose, SVD, then swap U/V.
        # metalsvd.svd currently: N must be even. M >= N.
    ]
    
    for size in sizes:
        for shape_name, shape_fn in shapes:
            B, M, N = shape_fn(size)
            
            # Skip if N > M (Wide) if handled poorly, but let's test it.
            # Usually users want M >= N.
            if N > M:
                # Transpose for native compat test
                pass
                
            name = f"{shape_name} ({M}x{N})"
            
            # Generate input
            try:
                A = torch.randn(B, M, N, device=device)
            except:
                print(f"Skipping {name} (OOM or Error)")
                continue

            # Check logic for Wide support in metalsvd wrapper?
            # If naive implementation crashes on Wide, we skip/note it.
            # torch.linalg.svd handles it.
            
            def run_ours():
                metalsvd.svd(A) # Will error if not implemented for Wide?
                                
            def run_torch():
                torch.linalg.svd(A)
            
            # Fewer iters for larger ones
            n_iters = 50 if M <= 256 else (20 if M <= 1024 else 5)
                
            try:
                runner.compare(name, run_torch, run_ours, iters=n_iters)
            except Exception as e:
                print(f"  [FAIL] {name}: {e}\n")

    runner.print_summary()

if __name__ == "__main__":
    benchmark_suite()
