"""
Cholesky Decomposition and Solve

Cholesky decomposition for symmetric positive-definite matrices: A = L @ L.T

Cholesky solve for SPD systems: A @ x = b
1. L = cholesky(A)
2. y = trsm(L, b, lower=True)
3. x = trsm(L.T, y, upper=True)

Based on ROCm's potrf (Cholesky factorization) + potrs (solve).
"""

import torch
from typing import Optional, Tuple

try:
    import metalcore_backend as mc
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False

from . import config


def cholesky(A: torch.Tensor, *, upper: bool = False) -> torch.Tensor:
    """
    Compute the Cholesky decomposition of a symmetric positive-definite matrix.
    
    For batched operations on small matrices, uses GPU.
    Otherwise falls back to CPU (optimized LAPACK).
    
    Args:
        A: Symmetric positive-definite matrix of shape (..., N, N)
        upper: If True, return upper triangular U such that A = U.T @ U
               If False (default), return lower triangular L such that A = L @ L.T
    
    Returns:
        L or U: Triangular Cholesky factor of shape (..., N, N)
    
    Example:
        >>> A = torch.randn(100, 32, 32, device='mps')
        >>> A = A @ A.transpose(-2, -1) + 0.1 * torch.eye(32, device='mps')  # Make SPD
        >>> L = cholesky(A)
        >>> # A ≈ L @ L.T
    """
    if A.device.type != 'mps':
        return torch.linalg.cholesky(A, upper=upper)
    
    # Check dimensions
    is_batched = A.dim() > 2
    batch_size = A.shape[0] if is_batched else 1
    N = A.shape[-1]
    
    # Smart routing: GPU Cholesky for batched small matrices
    use_gpu = (
        is_batched and
        HAS_BACKEND and
        N <= 64  # Unblocked algorithm efficient for small N
    )
    
    if use_gpu:
        L = mc.cholesky_batched(A)
        if upper:
            L = L.transpose(-2, -1)
        return L
    else:
        return _cholesky_cpu(A, upper=upper)


def _cholesky_cpu(A: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """CPU fallback for Cholesky (faster due to optimized LAPACK)."""
    A_cpu = A.cpu()
    L = torch.linalg.cholesky(A_cpu, upper=upper)
    return L.to(A.device)


def cholesky_solve(b: torch.Tensor, L: torch.Tensor, *, upper: bool = False) -> torch.Tensor:
    """
    Solve a linear system using pre-computed Cholesky factor.
    
    For A = L @ L.T (lower=True) or A = U.T @ U (upper=True),
    solves A @ x = b efficiently using forward and back substitution.
    
    GPU-accelerated for batched operations using TRSM.
    
    Args:
        b: Right-hand side of shape (..., N) or (..., N, K)
        L: Cholesky factor from cholesky(), shape (..., N, N)
        upper: If True, L is upper triangular (A = L.T @ L)
               If False, L is lower triangular (A = L @ L.T)
    
    Returns:
        x: Solution of shape (..., N) or (..., N, K)
    
    Example:
        >>> A = torch.randn(100, 32, 32, device='mps')
        >>> A = A @ A.transpose(-2, -1) + 0.1 * torch.eye(32, device='mps')
        >>> b = torch.randn(100, 32, device='mps')
        >>> L = cholesky(A)
        >>> x = cholesky_solve(b, L)
        >>> # A @ x ≈ b
    """
    if L.device.type != 'mps':
        return torch.cholesky_solve(b, L, upper=upper)
    
    # Handle dimensions
    is_batched = L.dim() > 2
    if not is_batched:
        L = L.unsqueeze(0)
        b_was_1d = b.dim() == 1
        if b_was_1d:
            b = b.unsqueeze(0).unsqueeze(-1)
        else:
            b = b.unsqueeze(0)
    else:
        b_was_1d = b.dim() == L.dim() - 1
        if b_was_1d:
            b = b.unsqueeze(-1)
    
    batch_size = L.shape[0] if is_batched or L.dim() == 3 else 1
    N = L.shape[-1]
    
    # Smart routing: GPU TRSM for batched small matrices
    use_gpu = (
        batch_size > 1 and 
        HAS_BACKEND and 
        N <= 48
    )
    
    if use_gpu:
        x = _cholesky_solve_batched_gpu(b, L, upper=upper)
    elif batch_size > 1:
        x = _cholesky_solve_batched_cpu(b, L, upper=upper)
    else:
        x = _cholesky_solve_single_cpu(b, L, upper=upper)
    
    # Unwrap
    if not is_batched:
        x = x.squeeze(0)
        if b_was_1d:
            x = x.squeeze(-1)
    elif b_was_1d:
        x = x.squeeze(-1)
    
    return x


def _cholesky_solve_batched_gpu(b: torch.Tensor, L: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """GPU-accelerated batched Cholesky solve using fused C++ kernel."""
    if upper:
        # A = U.T @ U, solve U.T @ y = b, then U @ x = y
        # For upper, pass the transpose as the lower factor
        x = mc.cholesky_solve_batched(L.transpose(-2, -1), b)
    else:
        # A = L @ L.T, solve L @ y = b, then L.T @ x = y
        x = mc.cholesky_solve_batched(L, b)
    
    return x


def _cholesky_solve_batched_cpu(b: torch.Tensor, L: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """CPU fallback for batched Cholesky solve."""
    b_cpu = b.cpu()
    L_cpu = L.cpu()
    
    solutions = []
    for i in range(L.shape[0]):
        x = torch.cholesky_solve(b_cpu[i], L_cpu[i], upper=upper)
        solutions.append(x)
    
    return torch.stack(solutions).to(L.device)


def _cholesky_solve_single_cpu(b: torch.Tensor, L: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """CPU fallback for single matrix Cholesky solve."""
    b_cpu = b.cpu().squeeze(0)
    L_cpu = L.cpu().squeeze(0)
    
    x = torch.cholesky_solve(b_cpu, L_cpu, upper=upper)
    
    return x.unsqueeze(0).to(L.device)
