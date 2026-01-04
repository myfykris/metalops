import torch
import time

def benchmark_svd_vs_eigh():
    N = 4096
    print(f"Benchmarking SVD vs EIGH for {N}x{N} Symmetric Matrix on CPU...")
    
    # Create symmetric matrix
    torch.manual_seed(42)
    # A random
    X = torch.randn(N, N)
    # K symmetric
    K = X.T @ X
    K = K.float() # Ensure float32
    
    print("Matrix created.")
    
    # 1. SVD
    print("Running torch.linalg.svd (CPU)...")
    start = time.time()
    try:
        U, S, Vt = torch.linalg.svd(K)
        t_svd = time.time() - start
        print(f"SVD Time: {t_svd:.4f}s")
    except Exception as e:
        print(f"SVD Failed: {e}")
        t_svd = float('inf')

    # 2. EIGH
    print("Running torch.linalg.eigh (CPU)...")
    start = time.time()
    try:
        # eigh returns eigenvalues in ascending order
        L, Q = torch.linalg.eigh(K)
        t_eigh = time.time() - start
        print(f"EIGH Time: {t_eigh:.4f}s")
        
        # Verify reconstruction?
        # K ~= Q * diag(L) * Q.T
        
    except Exception as e:
        print(f"EIGH Failed: {e}")
        t_eigh = float('inf')

    if t_svd != float('inf'):
        print(f"Speedup: {t_svd / t_eigh:.2f}x")

if __name__ == "__main__":
    benchmark_svd_vs_eigh()
