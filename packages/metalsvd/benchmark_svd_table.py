import torch
import metalsvd
import time


def run_benchmark():
    device = torch.device('mps')
    print(f"Benchmarking MetalSVD on {device}...")
    
    # Warmup
    print("Warming up...")
    dummy = torch.randn(1, 64, 64, device=device)
    metalsvd.svd(dummy)
    torch.mps.synchronize()
    
    sizes = [64, 128, 256, 512, 1024]
    batches = [1, 64, 256]
    
    results = []
    
    print(f"{'Size (NxN)':<15} | {'Batch':<10} | {'CPU (ms)':<15} | {'Metal (ms)':<15} | {'Speedup':<10}")
    print("-" * 80)
    
    for N in sizes:
        for B in batches:
            if N >= 1024 and B > 8:
                # Reduce batch for large N to avoid OOM/Time limits
                current_B = 8
            else:
                current_B = B
                
            label = f"{N}x{N}"
            
            # Prepare Data
            # CPU tensor must be on CPU for fairness if PyTorch defaults to CPU SVD
            # But typically we compare "End-to-End" on device.
            # PyTorch `svd` on MPS often falls back to CPU internally or uses MPS primitives.
            # For a fair "Baselines", we should check if PyTorch supports MPS SVD directly.
            # It usually does via library fallback. We will use `input.to(device)`.
            
            # CPU Baseline: explicitly move to CPU to measure pure CPU alg speed?
            # Or MPS baseline? PyTorch `linalg.svd` on MPS is often slower/fallback.
            # Standard practice: Compare against "Best Alternative". 
            # If user is on Mac, they might run on CPU.
            
            A_gpu = torch.randn(current_B, N, N, device=device)
            A_cpu = A_gpu.cpu()
            
            # 1. Baseline (PyTorch CPU - usually fastest standard option on Mac for small/medium)
            # We measure pure computation time.
            
            # Warmup
            torch.linalg.svd(A_cpu)
            
            iters = 10 if N < 512 else 3
            if N >= 1024: iters = 2
            
            start_cpu = time.time()
            for _ in range(iters):
                torch.linalg.svd(A_cpu)
            end_cpu = time.time()
            avg_cpu = (end_cpu - start_cpu) / iters * 1000.0
            
            # 2. MetalSVD
            metalsvd.svd(A_gpu) # Warmup
            torch.mps.synchronize()
            
            start_metal = time.time()
            for _ in range(iters):
                metalsvd.svd(A_gpu)
                torch.mps.synchronize()
            end_metal = time.time()
            
            avg_metal = (end_metal - start_metal) / iters * 1000.0
            
            speedup = avg_cpu / avg_metal
            
            print(f"{label:<15} | {current_B:<10} | {avg_cpu:<15.2f} | {avg_metal:<15.2f} | {speedup:<10.2f}x")
            
            results.append({
                "Size": N,
                "Batch": current_B,
                "CPU (ms)": avg_cpu,
                "Metal (ms)": avg_metal,
                "Speedup": speedup
            })
            
    print("-" * 80)
    return results

if __name__ == "__main__":
    run_benchmark()
