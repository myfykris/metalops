import torch
import metalsvd
import time
import math

def benchmark_float32():
    M, N = 4096, 14336 # Huge Fat
    print(f"Benchmarking Huge Fat ({M}x{N}) in Float32...")
    
    device = torch.device('mps')
    
    # Init Float32
    torch.manual_seed(42)
    A = torch.randn(M, N, device=device, dtype=torch.float32)
    A_cpu = A.cpu()
    
    # 1. CPU Baseline
    print("Running CPU Baseline (Float32)...")
    start = time.time()
    try:
        torch.linalg.svd(A_cpu, full_matrices=False)
        cpu_time = time.time() - start
        print(f"CPU Time: {cpu_time:.4f}s")
    except:
        cpu_time = float('inf')
        
    # 2. Metal Hybrid
    print("Running Metal Hybrid Gram (Float32)...")
    torch.mps.synchronize()
    start = time.time()
    try:
        U, S, Vt = metalsvd.svd(A, strategy='gram')
        torch.mps.synchronize()
        metal_time = time.time() - start
        print(f"Metal Time: {metal_time:.4f}s")
        
        # Validate
        A_rec = U @ torch.diag(S) @ Vt
        diff = torch.norm(A - A_rec) / torch.norm(A)
        print(f"Reconstruction Error: {diff:.8f}")
        
        # Orthogonality
        ortho = torch.norm(U.T @ U - torch.eye(U.shape[1], device=U.device)) / math.sqrt(U.shape[1])
        print(f"Orthogonality Error: {ortho:.8f}")
        
        if diff < 0.0001:
            print("Status: ✅ PASS (Strict Accuracy)")
        else:
            print(f"Status: ❌ FAIL (Error {diff} > 0.0001)")
            
    except Exception as e:
        print(f"Metal Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    benchmark_float32()
