import torch
import time
import metaleig

def benchmark():
    N_values = [32, 64, 128, 256, 512, 768, 1024, 2048]
    device = torch.device("mps")
    
    print(f"{'N':<6} | {'CPU (ms)':<8} | {'Metal (ms)':<10} | {'Speedup':<7} |Accuracy metrics")
    print("-" * 80)
    
    for N in N_values:
        torch.manual_seed(42)
        A = torch.randn(N, N, device=device)
        A = A + A.mT
        
        # Warmup
        print(f"Warmup N={N}...", end="", flush=True)
        _ = metaleig.eigh(A)
        torch.mps.synchronize()
        print("Done.")
        
        # Metal Benchmark
        torch.mps.synchronize()
        start = time.time()
        L, Q = metaleig.eigh(A)
        torch.mps.synchronize()
        metal_time = (time.time() - start) * 1000 # ms
        
        # CPU Benchmark
        A_cpu = A.cpu()
        start = time.time()
        torch.linalg.eigh(A_cpu)
        cpu_time = (time.time() - start) * 1000 # ms
        
        # Accuracy Check
        # Reconstruction: A = Q * diag(L) * Q^T
        L_diag = torch.diag_embed(L)
        A_recon = Q @ L_diag @ Q.mT
        recon_err = (A - A_recon).abs().max().item()
        
        # Orthogonality: Q^T * Q = I
        I = torch.eye(N, device=device)
        ortho_err = (Q.mT @ Q - I).abs().max().item()
        
        # Eigenvalue Diff vs CPU
        L_metal_sorted, _ = torch.sort(L)
        L_cpu_sorted, _ = torch.sort(torch.linalg.eigvalsh(A_cpu))
        eig_err = (L_metal_sorted.cpu() - L_cpu_sorted).abs().max().item()

        print(f"{N:<6} | {cpu_time:<8.2f} | {metal_time:<10.2f} | {cpu_time/metal_time:<7.2f}x | Rec:{recon_err:.1e} Ort:{ortho_err:.1e} Eig:{eig_err:.1e}")

    # Batched Benchmarks (The real GPU use case)
    print("\nBatched Benchmarks (Batch Size = 64)")
    print("-" * 80)
    B = 64
    for N in [32, 64, 128, 256, 512]:
        torch.manual_seed(42)
        A = torch.randn(B, N, N, device=device)
        A = A + A.mT
        
        # Warmup
        _ = metaleig.eigh(A)
        torch.mps.synchronize()
        
        # Metal
        torch.mps.synchronize()
        start = time.time()
        metaleig.eigh(A)
        torch.mps.synchronize()
        metal_time = (time.time() - start) * 1000
        
        # CPU
        A_cpu = A.cpu()
        start = time.time()
        torch.linalg.eigh(A_cpu)
        cpu_time = (time.time() - start) * 1000
        
        print(f"B={B}, N={N:<4} | {cpu_time:<8.2f} | {metal_time:<10.2f} | {cpu_time/metal_time:<7.2f}x")

if __name__ == "__main__":
    benchmark()
