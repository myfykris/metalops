"""
Linear System Solver using QR Factorization

Solves Ax = b for square systems using QR decomposition:
1. Q, R = qr(A)
2. c = Q.T @ b
3. x = trsm(R, c)

For batched operations, uses GPU-accelerated qr_batched + trsm_batched.
"""

import torch
from typing import Optional

try:
    import metalcore_backend as mc
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False

from . import config


def solve(A: torch.Tensor, b: torch.Tensor, *, fast: bool = False) -> torch.Tensor:
    """
    Solve the linear system Ax = b.
    
    GPU-accelerated for batched operations using QR factorization.
    For single matrices, uses CPU (faster due to optimized LAPACK LU).
    
    Args:
        A: Input matrix of shape (..., N, N) - must be square
        b: Right-hand side of shape (..., N) or (..., N, K)
        fast: If True, uses less accurate but faster path
    
    Returns:
        x: Solution of shape (..., N) or (..., N, K)
    
    Example:
        >>> A = torch.randn(100, 32, 32, device='mps')
        >>> b = torch.randn(100, 32, device='mps')
        >>> x = solve(A, b)  # x such that A @ x = b
    """
    if A.device.type != 'mps':
        return torch.linalg.solve(A, b)
    
    # Check dimensions
    is_batched = A.dim() > 2
    if not is_batched:
        A = A.unsqueeze(0)
        b_was_1d = b.dim() == 1
        if b_was_1d:
            b = b.unsqueeze(0).unsqueeze(-1)
        else:
            b = b.unsqueeze(0)
    else:
        b_was_1d = b.dim() == A.dim() - 1
        if b_was_1d:
            b = b.unsqueeze(-1)
    
    M, N = A.shape[-2], A.shape[-1]
    if M != N:
        raise ValueError(f"solve requires square matrix, got {M}×{N}")
    
    batch_size = A.shape[0] if is_batched or A.dim() == 3 else 1
    
    # Smart routing:
    # - Batched: Use fused GPU LU-solve for all sizes
    # - Single: CPU LU is faster due to LAPACK optimization
    use_gpu = batch_size > 1 and HAS_BACKEND
    
    if use_gpu:
        x = _solve_batched_gpu(A, b)
    elif batch_size > 1:
        x = _solve_batched_cpu(A, b)
    else:
        x = _solve_single_cpu(A, b)
    
    # Unwrap
    if not is_batched:
        x = x.squeeze(0)
        if b_was_1d:
            x = x.squeeze(-1)
    elif b_was_1d:
        x = x.squeeze(-1)
    
    return x


def _solve_batched_gpu(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated batched solve using fused LU decomposition."""
    # Use fused LU-based solve kernel for maximum performance
    return mc.solve(A, b)


def _solve_batched_cpu(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """CPU fallback for batched solve (LU is faster for large N)."""
    A_cpu = A.cpu()
    b_cpu = b.cpu()
    
    solutions = []
    for i in range(A.shape[0]):
        x = torch.linalg.solve(A_cpu[i], b_cpu[i])
        solutions.append(x)
    
    return torch.stack(solutions).to(A.device)


def _solve_single_cpu(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """CPU fallback for single matrix solve."""
    A_cpu = A.cpu().squeeze(0)
    b_cpu = b.cpu().squeeze(0)
    
    x = torch.linalg.solve(A_cpu, b_cpu)
    
    return x.unsqueeze(0).to(A.device)
