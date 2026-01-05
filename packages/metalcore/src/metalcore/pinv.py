"""
Pseudo-Inverse using SVD

Computes the Moore-Penrose pseudo-inverse: A⁺ = V @ diag(1/S) @ U.T

Based on ROCm's approach using gesvd/gesvdj for SVD computation.
"""

import torch
from typing import Optional

try:
    import metalsvd
    HAS_SVD = True
except ImportError:
    HAS_SVD = False


def pinv(A: torch.Tensor, *, rcond: float = 1e-15, hermitian: bool = False) -> torch.Tensor:
    """
    Compute the Moore-Penrose pseudo-inverse of a matrix.
    
    GPU-accelerated using Metal SVD for MPS tensors.
    
    Args:
        A: Input tensor of shape (..., M, N)
        rcond: Cutoff ratio for small singular values. Singular values
               smaller than rcond * max(S) are set to zero.
        hermitian: If True, A is assumed to be Hermitian (symmetric if real),
                   allowing for faster computation (not yet implemented).
    
    Returns:
        A_pinv: Pseudo-inverse of shape (..., N, M)
    
    Example:
        >>> A = torch.randn(100, 64, 32, device='mps')
        >>> A_pinv = pinv(A)  # shape (100, 32, 64)
        >>> # A @ A_pinv @ A ≈ A
    """
    if A.device.type != 'mps' or not HAS_SVD:
        return torch.linalg.pinv(A, rcond=rcond, hermitian=hermitian)
    
    # Use our GPU SVD
    U, S, V = metalsvd.svd(A)
    
    # Apply rcond threshold
    max_s = S.amax(dim=-1, keepdim=True)
    threshold = rcond * max_s
    S_inv = torch.where(S > threshold, 1.0 / S, torch.zeros_like(S))
    
    # Compute pseudo-inverse: V @ diag(S_inv) @ U.T
    # V: (..., N, K), S_inv: (..., K), U: (..., M, K)
    # Result: (..., N, M)
    
    # Efficient: V @ (S_inv.unsqueeze(-1) * U.T)
    # = V @ diag(S_inv) @ U.T
    S_inv_diag = S_inv.unsqueeze(-2)  # (..., 1, K)
    U_T = U.transpose(-2, -1)  # (..., K, M)
    
    # (..., N, K) @ (..., K, M) -> (..., N, M)
    A_pinv = torch.matmul(V * S_inv.unsqueeze(-2), U_T)
    
    return A_pinv
