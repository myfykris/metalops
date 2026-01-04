import torch
import metalsvd
import time

def benchmark():
    device = torch.device('mps')
    print(f"Device: {device}")
    
    M, N = 4096, 11008
    print(f"Creating Huge Fat Matrix ({M}x{N})...")
    # A = torch.randn(M, N, device=device).to(torch.bfloat16) # Huge allocation
    # Use empty/uniform to match benchmark suite cost approx? randn is fine.
    A = torch.randn(M, N, device=device).to(torch.bfloat16)
    
    print("Warming up...")
    # Warmup with a smaller one to ensure library loaded?
    # Or just run once.
    try:
        metalsvd.svd(torch.randn(128, 128, device=device))
    except Exception as e:
        print(f"Warmup failed: {e}")

    torch.mps.synchronize()
    
    print(f"Benchmarking metalsvd.svd({M}x{N})...")
    start = time.time()
    
    # Run once is likely enough for this huge size if it takes 30s+
    res = metalsvd.svd(A)
    torch.mps.synchronize()
    
    end = time.time()
    duration = end - start
    print(f"Duration: {duration:.4f} seconds")
    
    # Output shapes
    U, S, Vt = res
    print(f"Output shapes: U={U.shape}, S={S.shape}, Vt={Vt.shape}")

    # Validation
    print("\nValidating results...")
    A_rec = U @ torch.diag(S) @ Vt
    # Compute error in Float32 to avoid overflow/underflow masking
    A_f = A.float()
    A_rec_f = A_rec.float()
    recon_err = torch.norm(A_f - A_rec_f) / torch.norm(A_f)
    print(f"Reconstruction Error ||A - USV|| / ||A||: {recon_err:.6f}")
    
    I_U = torch.eye(U.shape[1], device=A.device)
    ortho_err_U = torch.norm(U.T @ U - I_U) / torch.norm(I_U)
    print(f"Orthogonality Error U: {ortho_err_U:.6f}")
    
    # Check if Acceptable
    if recon_err < 0.1:
        print("Status: VALID (Acceptable Error)")
    else:
        print("Status: INVALID (High Error)")

if __name__ == "__main__":
    benchmark()
