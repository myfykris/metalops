import torch

def verify_svd_to_eigh():
    torch.manual_seed(42)
    N = 10
    # Generate random symmetric matrix
    A = torch.randn(N, N)
    A = A + A.T
    
    # 1. True EIGH
    L_true, Q_true = torch.linalg.eigh(A)
    
    # 2. SVD
    U, S, Vh = torch.linalg.svd(A)
    V = Vh.mH # V in A = U S V^H
    
    # Recover signs
    # A v = lambda v
    # A v = sigma u
    # lambda v = sigma u => lambda = sigma * (u . v)
    
    # Check dot product of columns of U and V
    dots = (U * V).sum(dim=0)
    signs = dots.sign()
    
    L_pred = S * signs
    
    # SVD returns S in descending order of magnitude. 
    # EIGH returns L in ascending order of value.
    
    # Sort L_pred
    L_pred_sorted, indices = torch.sort(L_pred)
    Q_pred_sorted = V[:, indices]
    
    print("L True:", L_true)
    print("L Pred:", L_pred_sorted)
    
    err = (L_true - L_pred_sorted).abs().max()
    print("Max Eigenvalue Error:", err.item())
    
    # Check Eigenvectors (might differ by sign)
    # But Q_true * Q_true.T should equal Q_pred * Q_pred.T (Projection)
    # Actually just check A reconstruction
    A_recon = Q_pred_sorted @ torch.diag(L_pred_sorted) @ Q_pred_sorted.T
    recon_err = (A - A_recon).abs().max()
    print("Reconstruction Error:", recon_err.item())

    if err < 1e-5 and recon_err < 1e-5:
        print("SUCCESS: Algorithm verifies.")
    else:
        print("FAILURE.")

if __name__ == "__main__":
    verify_svd_to_eigh()
