import torch
import metalsvd
import time

def test_svd_correctness():
    if not torch.backends.mps.is_available():
        print("Skipping test: MPS not available")
        return

    device = torch.device('mps')
    # Small batch for correctness
    sizes = [64, 128, 256]
    
    for val in sizes:
        M, N = val, val
        # Small batch
        B = 2
        
        print(f"\n--- Testing N={N} ---")
        torch.manual_seed(42)
        A = torch.randn(B, M, N, device=device)
        
        start = time.time()
        U, S, V = metalsvd.svd(A)
        torch.mps.synchronize()
        end = time.time()
        print(f"Custom SVD took: {(end - start)*1000:.2f} ms")
        
        S_diag = torch.zeros(B, N, N, device=device)
        for b in range(B):
            S_diag[b] = torch.diag(S[b])
            
        A_recon = U @ S_diag @ V.transpose(-2, -1)
        
        diff = (A - A_recon).abs().max().item()
        print(f"Reconstruction Max Error: {diff:.6f}")
        
        if diff > 1e-3: # Slightly relaxed for float
            print("FAIL: Reconstruction error too high")
        else:
            print("PASS: Reconstruction OK")
            
        UTU = U.transpose(-2, -1) @ U
        I = torch.eye(N, device=device).unsqueeze(0).expand(B, N, N)
        ort_err = (UTU - I).abs().max().item()
        print(f"U Orthogonality Error: {ort_err:.6f}")


def test_svd_autograd():
    print("\nTesting Autograd (Backward Pass)...")
    if not torch.backends.mps.is_available():
        print("Skipping: MPS not available")
        return

    device = torch.device('mps')
    B, M, N = 2, 32, 32
    A = torch.randn(B, M, N, device=device, requires_grad=True)
    
    # Forward
    U, S, V = metalsvd.svd(A)
    
    # Loss: maximize sum of singular values (nuclear norm)
    loss = S.sum()
    
    # Backward
    loss.backward()
    
    print(f"A.grad shape: {A.grad.shape}")
    if A.grad is None:
        print("FAIL: Gradient is None")
    elif A.grad.abs().sum() == 0:
        print("FAIL: Gradient is zero (vanishing?)")
    else:
        print(f"PASS: Gradient computed. Norm: {A.grad.norm():.2f}")

if __name__ == "__main__":
    test_svd_correctness()
    test_svd_autograd()
