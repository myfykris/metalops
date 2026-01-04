import torch
from .func import svd as metal_svd

def randomized_svd(A: torch.Tensor, k: int, n_iter: int = 2, q: int = None):
    """
    Computes Randomized SVD of matrix A fully on MPS.
    approximates A = U @ S @ V.T
    
    Args:
        A: Input matrix (M, N) on MPS.
        k: Target rank.
        n_iter: Number of power iterations (improves accuracy).
    
    Returns:
        U, S, V
    """
    # Support arbitrary batch dims: (..., M, N)
    shape = A.shape
    if len(shape) < 2:
        raise ValueError("A must be at least 2D")
        
    M, N = shape[-2], shape[-1]
    batch_dims = shape[:-2] # tuple
    device = A.device
    
    # 1. Random Projection Matrix Omega (..., N, k)
    p = 10 
    k_over = min(k + p, N)
    
    if k_over % 2 != 0:
        k_over += 1
    
    # Omega size: (*batch_dims, N, k_over)
    Omega = torch.randn(*batch_dims, N, k_over, device=device, dtype=A.dtype)
    
    # 2. Compute Sketch Y = A @ Omega (..., M, N) @ (..., N, k_over) -> (..., M, k_over)
    # Perform sketch in FP32 to avoid overflow/underflow if A is FP16.
    dtype_orig = A.dtype
    if dtype_orig == torch.float16 or dtype_orig == torch.bfloat16:
        # Cast inputs to float32 for the randomized loop
        A_f32 = A.to(torch.float32)
        Omega_f32 = Omega.to(torch.float32)
        Y = torch.matmul(A_f32, Omega_f32)
    else:
        A_f32 = A
        Y = torch.matmul(A, Omega)
    
    # 3. Power Iteration
    for _ in range(n_iter):
        # Z = A.T @ Y
        # A.transpose(-2, -1) handles batch transpose correctly
        At = A_f32.transpose(-2, -1)
        Z = torch.matmul(At, Y)
        
        # Orthogonalize Z
        # metal_svd handles batches if we pass correct structure.
        # But my mps_svd wrapper now squeezes 2D inputs?
        # If Z is batched, metal_svd works.
        # If Z is (M, k), metal_svd(Z) returns (k), etc.
        # Wait, metal_svd(Z) returns U, S, V.
        # if Z is (..., N, k_over)
        # We need to ensure Z is treated correctly.
        # metal_svd expects (..., Rows, Cols).
        Q_Z, _, _ = metal_svd(Z)
        
        # Y = A @ Q_Z
        Y = torch.matmul(A_f32, Q_Z)
        
        # Orthogonalize Y
        Q_Y, _, _ = metal_svd(Y)
        Y = Q_Y
        
    # 4. Orthogonalize Y to get Q
    Q, _, _ = metal_svd(Y)
    
    # 5. Form B = Q.T @ A
    # Q: (..., M, k_over). A: (..., M, N)
    # B = Q.transpose(-2, -1) @ A -> (..., k_over, N)
    Qt = Q.transpose(-2, -1)
    B = torch.matmul(Qt, A_f32)
    
    # 6. SVD of B
    # B.T: (..., N, k_over)
    Bt = B.transpose(-2, -1)
    U_BT, S_B, V_BT = metal_svd(Bt)
    
    # B approx U_BT S_B V_BT.T (if we did SVD of B directly)
    # But we did SVD of B.T = U_BT S_B V_BT.T
    # So B = (U_BT S_B V_BT.T).T = V_BT S_B U_BT.T
    # So left vectors of B are V_BT.
    
    # Final U = Q @ V_BT
    U_final = torch.matmul(Q, V_BT)
    
    # 8. Truncate
    # S_B: (..., k_over)
    # U_final: (..., M, k_over)
    # U_BT: (..., N, k_over) (This is actually V of final result)
    
    # Cast back if needed
    if dtype_orig != A_f32.dtype:
        U_final = U_final.to(dtype_orig)
        S_B = S_B.to(dtype_orig)
        U_BT = U_BT.to(dtype_orig)
        
    return U_final[..., :k], S_B[..., :k], U_BT[..., :k]
