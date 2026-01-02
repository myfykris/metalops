import torch
import metalsvd

def debug_rsvd():
    print("Debugging rSVD Accuracy...")
    device = torch.device('mps')
    
    # Test 1: Low Rank FP32
    M, N = 512, 512
    K_true = 10
    
    # Generate Rank-K matrix
    torch.manual_seed(42)
    U_true = torch.randn(M, K_true, device=device)
    U_true = torch.linalg.qr(U_true)[0]
    V_true = torch.randn(N, K_true, device=device)
    V_true = torch.linalg.qr(V_true)[0]
    S_true = torch.linspace(10, 1, K_true, device=device)
    
    A = U_true @ torch.diag(S_true) @ V_true.T
    
    U, S, V = metalsvd.randomized_svd(A, k=K_true, n_iter=2)
    A_approx = U @ torch.diag(S) @ V.T
    
    # Check shape
    print(f"U: {U.shape}, S: {S.shape}, V: {V.shape}")
    
    diff = (A - A_approx).norm()
    norm = A.norm()
    err = diff / norm
    print(f"FP32 (512x512) Rel Error: {err:.6f}")
    
    # Test 2: Low Rank Huge FP16
    M_huge = 2048
    K_true = 10
    
    U_h = torch.randn(M_huge, K_true, device=device, dtype=torch.float16)
    # QR might fallback to CPU or fail for half? 
    # Let's generate in float then cast.
    U_h = torch.randn(M_huge, K_true, device=device, dtype=torch.float32)
    U_h = torch.linalg.qr(U_h)[0].to(torch.float16)
    V_h = torch.randn(M_huge, K_true, device=device, dtype=torch.float32)
    V_h = torch.linalg.qr(V_h)[0].to(torch.float16)
    S_h = torch.linspace(100, 10, K_true, device=device, dtype=torch.float16)
    
    A_fp16 = U_h @ torch.diag(S_h) @ V_h.T
    
    U, S, V = metalsvd.randomized_svd(A_fp16, k=K_true, n_iter=2)
    A_approx = U @ torch.diag(S) @ V.T
    
    diff = (A_fp16 - A_approx).norm()
    norm = A_fp16.norm()
    err = diff / norm
    print(f"FP16 (2048x2048) Rel Error: {err:.6f}")

if __name__ == "__main__":
    debug_rsvd()
