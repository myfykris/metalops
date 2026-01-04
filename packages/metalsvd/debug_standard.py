import torch
import metalsvd
import time

def debug():
    device = torch.device('mps')
    # Use a small fat matrix (Medium) to fail fast if it crashes
    M, N = 1024, 1024 
    print(f"Creating {M}x{N}...")
    A = torch.randn(M, N, device=device).to(torch.float32)

    print("Running Standard Strategy...")
    U, S, Vt = metalsvd.svd(A, strategy='standard')
    print("Execution Success. Checking Accuracy...")
    
    A_cpu = A.float().cpu()
    U_f = U.float().cpu()
    S_f = S.float().cpu()
    Vt_f = Vt.float().cpu()
    
    A_rec = U_f @ torch.diag(S_f) @ Vt_f
    recon_err = torch.norm(A_cpu - A_rec) / torch.norm(A_cpu)
    print(f"Reconstruction Error: {recon_err:.6f}")
    
    print(f"S[0:5]: {S_f[0:5]}")
    print(f"U[0,0:5]: {U_f[0,0:5]}")

if __name__ == "__main__":
    debug()
