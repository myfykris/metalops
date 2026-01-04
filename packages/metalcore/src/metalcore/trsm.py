"""
Triangular Solve (trsm) - Solve Ax = b where A is triangular.

This is a foundational operation used by:
- QR solve (after QR decomposition)
- Cholesky solve
- LU solve
"""

import torch

# Try to import Metal backend, fall back to None if not available
try:
    import metalcore_backend
except ImportError:
    metalcore_backend = None


def trsm(A, b, lower=True, transpose=False, unit_diagonal=False):
    """
    Solve a triangular system Ax = b.
    
    Args:
        A: Triangular matrix (..., N, N)
        b: Right-hand side (..., N) or (..., N, K)
        lower: If True, A is lower triangular; else upper triangular
        transpose: If True, solve A^T x = b instead
        unit_diagonal: If True, assume diagonal elements are 1
        
    Returns:
        x: Solution (..., N) or (..., N, K)
    """
    if A.device.type != 'mps':
        # Use PyTorch CPU implementation
        return torch.linalg.solve_triangular(A, b, upper=not lower, left=True, unitriangular=unit_diagonal)
    
    # Check if we have batched input
    is_batched = A.dim() > 2
    b_is_matrix = b.dim() == A.dim()
    
    # For now, use a Python implementation with GPU matmuls
    # This will be replaced with a proper Metal kernel
    return _trsm_python(A, b, lower, transpose, unit_diagonal)


def trsm_batched(A, B, lower=True, transpose=False, unit_diagonal=False):
    """
    Batched triangular solve.
    
    Args:
        A: Triangular matrices (batch, N, N)
        B: Right-hand sides (batch, N, K)
        
    Returns:
        X: Solutions (batch, N, K)
    """
    return trsm(A, B, lower, transpose, unit_diagonal)


def _trsm_python(A, b, lower=True, transpose=False, unit_diagonal=False):
    """
    Python implementation of trsm using back/forward substitution.
    
    For GPU efficiency, we use a blocked algorithm:
    1. Divide matrix into blocks
    2. Solve diagonal block (small trsm)
    3. Update remaining blocks with matmul
    
    This converts O(N²) sequential operations into O(N) sequential + O(N³/B) parallel.
    """
    device = A.device
    dtype = A.dtype
    
    # Handle transpose
    if transpose:
        A = A.transpose(-2, -1)
        lower = not lower
    
    # Ensure b has matrix shape
    b_was_1d = b.dim() == A.dim() - 1
    if b_was_1d:
        b = b.unsqueeze(-1)
    
    # Get dimensions
    N = A.shape[-1]
    K = b.shape[-1]
    
    # Block size for GPU efficiency
    BLOCK_SIZE = min(64, N)
    
    x = b.clone()
    
    if lower:
        # Forward substitution (process rows top to bottom)
        for i in range(0, N, BLOCK_SIZE):
            end_i = min(i + BLOCK_SIZE, N)
            block_size = end_i - i
            
            # Solve the diagonal block
            A_diag = A[..., i:end_i, i:end_i]
            x_block = x[..., i:end_i, :]
            
            # Simple back-substitution for the block
            for j in range(block_size):
                if not unit_diagonal:
                    x[..., i+j, :] = x[..., i+j, :] / A[..., i+j, i+j:i+j+1]
                
                if j + 1 < block_size:
                    # Update remaining rows in block
                    x[..., i+j+1:end_i, :] -= A[..., i+j+1:end_i, i+j:i+j+1] * x[..., i+j:i+j+1, :]
            
            # Update remaining rows with matmul (GPU efficient!)
            if end_i < N:
                x[..., end_i:, :] -= torch.matmul(A[..., end_i:, i:end_i], x[..., i:end_i, :])
    else:
        # Back substitution (process rows bottom to top)
        for i in range(N - 1, -1, -BLOCK_SIZE):
            start_i = max(0, i - BLOCK_SIZE + 1)
            block_size = i - start_i + 1
            
            # Solve the diagonal block (backwards)
            for j in range(block_size - 1, -1, -1):
                idx = start_i + j
                if not unit_diagonal:
                    x[..., idx, :] = x[..., idx, :] / A[..., idx, idx:idx+1]
                
                if j > 0:
                    # Update remaining rows in block
                    x[..., start_i:idx, :] -= A[..., start_i:idx, idx:idx+1] * x[..., idx:idx+1, :]
            
            # Update remaining rows with matmul
            if start_i > 0:
                x[..., :start_i, :] -= torch.matmul(A[..., :start_i, start_i:i+1], x[..., start_i:i+1, :])
    
    if b_was_1d:
        x = x.squeeze(-1)
    
    return x
