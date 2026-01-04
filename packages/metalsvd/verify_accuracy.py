import torch
import metalsvd
import time

def verify_accuracy():
    print("Verifying accuracy of Optimized Metal SVD...")
    device = torch.device('mps')
    cpu_device = torch.device('cpu')
    
    # Use the target dimensions
    M, N = 4096, 11008
    # Transposed logic internally handles this, but let's test the "Huge Fat" shape directly 
    # as the user inputs it.
    
    # Create random matrix
    torch.manual_seed(42)
    print(f"Creating matrix {M}x{N}...")
    A_cpu = torch.randn(M, N, dtype=torch.float32)
    A_mps = A_cpu.to(device).to(torch.bfloat16) # Our optimized path uses bfloat16 input
    
    # 1. CPU Baseline (Gold Standard)
    print("Running CPU SVD (Float32)...")
    U_cpu, S_cpu, Vt_cpu = torch.linalg.svd(A_cpu, full_matrices=False)
    
    # 2. Metal Optimized
    print("Running Metal SVD (BFloat16 + Gram)...")
    # metalsvd.svd should trigger the optimized path for this shape
    U_mps, S_mps, Vt_mps = metalsvd.svd(A_mps)
    
    # Move to CPU/Float32 for comparison
    U_mps = U_mps.float().cpu()
    S_mps = S_mps.float().cpu()
    Vt_mps = Vt_mps.float().cpu()
    
    # Check Shapes
    print(f"Shapes CPU: U{U_cpu.shape} S{S_cpu.shape} Vt{Vt_cpu.shape}")
    print(f"Shapes MPS: U{U_mps.shape} S{S_mps.shape} Vt{Vt_mps.shape}")

    # Reconstruction Error
    # A approx = U S Vt
    A_rec_mps = U_mps @ torch.diag(S_mps) @ Vt_mps
    recon_err = torch.norm(A_cpu - A_rec_mps) / torch.norm(A_cpu)
    print(f"\nReconstruction Error ||A - USV|| / ||A||: {recon_err:.6f}")
    
    # Singular Values Error
    s_err = torch.norm(S_cpu - S_mps) / torch.norm(S_cpu)
    print(f"Singular Values Error ||S_cpu - S_mps|| / ||S_cpu||: {s_err:.6f}")
    
    # Orthogonality Error
    # U^T U should be I
    I_U = torch.eye(U_mps.shape[1])
    ortho_err_U = torch.norm(U_mps.T @ U_mps - I_U) / torch.norm(I_U)
    print(f"Orthogonality Error U (||U^T U - I||): {ortho_err_U:.6f}")
    
    # V^T V should be I
    I_V = torch.eye(Vt_mps.shape[0])
    ortho_err_V = torch.norm(Vt_mps @ Vt_mps.T - I_V) / torch.norm(I_V) 
    print(f"Orthogonality Error V (||V V^T - I||): {ortho_err_V:.6f}")

    print("\nInterpretation:")
    print("- BFloat16 precision is roughly 1e-2 to 1e-3.")
    print("- Gram strategy squares condition number, potentially reducing precision further.")
    print("- Epsilon was tuned to 1e-3.")
    
    if recon_err < 0.05: 
        print("-> Result: ACCEPTABLE for ML training (typically robust to noise).")
    else:
        print("-> Result: POOR. May require falling back to float32 or QR if precision is critical.")

if __name__ == "__main__":
    verify_accuracy()
