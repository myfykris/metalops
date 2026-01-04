"""
Householder Reflections - Core orthogonalization primitive.

A Householder reflection is an orthogonal transformation:
    H = I - tau * v @ v.T
    
where v is the Householder vector and tau = 2 / (v.T @ v).

Given a vector x, we can find v such that H @ x = ||x|| * e_1,
zeroing out all elements below the first.

This is the building block for:
- QR decomposition
- Tridiagonalization (for eigendecomposition)
- Bidiagonalization (for SVD)
"""

import torch


def householder_vector(x):
    """
    Compute Householder vector v and scalar tau such that:
        H @ x = ||x|| * e_1
        H = I - tau * v @ v.T
    
    Args:
        x: Input vector (..., N) - will reflect first element
        
    Returns:
        v: Householder vector (..., N), normalized so v[0] = 1
        tau: Scalar (...,) for the reflection
    """
    device = x.device
    dtype = x.dtype
    
    # Compute norm of x[1:]
    sigma = torch.sum(x[..., 1:] ** 2, dim=-1)
    
    # Compute norm of full x
    norm_x = torch.sqrt(x[..., 0] ** 2 + sigma)
    
    # Initialize v = x
    v = x.clone()
    
    # Handle the case where sigma = 0 (x is already along e_1)
    # In this case, H = I (tau = 0)
    zero_sigma = sigma < 1e-10
    
    # Compute v[0] = x[0] - sign(x[0]) * ||x||
    # Using sign(x[0]) to avoid cancellation
    sign = torch.where(x[..., 0] >= 0, 
                       torch.ones_like(x[..., 0]), 
                       -torch.ones_like(x[..., 0]))
    v[..., 0] = x[..., 0] + sign * norm_x
    
    # Normalize v so v[0] = 1 (standard form)
    v_norm_sq = v[..., 0:1] ** 2 + sigma.unsqueeze(-1)
    tau = 2.0 * v[..., 0] ** 2 / (v[..., 0] ** 2 + sigma + 1e-10)
    
    # Normalize v by v[0]
    v = v / (v[..., 0:1] + 1e-10)
    
    # Set tau = 0 for zero sigma case
    tau = torch.where(zero_sigma, torch.zeros_like(tau), tau)
    
    return v, tau


def apply_householder(A, v, tau, side='left'):
    """
    Apply Householder reflection to a matrix.
    
    If side='left':  A = H @ A = A - tau * v @ (v.T @ A)
    If side='right': A = A @ H = A - tau * (A @ v) @ v.T
    
    This is a rank-1 update and can be done with two matmuls.
    
    Args:
        A: Matrix (..., M, N)
        v: Householder vector (..., M) for left, (..., N) for right
        tau: Householder scalar (...)
        side: 'left' or 'right'
        
    Returns:
        A_transformed: Transformed matrix (..., M, N)
    """
    if side == 'left':
        # A = A - tau * v @ (v.T @ A)
        # v: (..., M), A: (..., M, N)
        vT_A = torch.einsum('...m,...mn->...n', v, A)  # (..., N)
        update = tau.unsqueeze(-1).unsqueeze(-1) * torch.einsum('...m,...n->...mn', v, vT_A)
        return A - update
    else:
        # A = A - tau * (A @ v) @ v.T
        # A: (..., M, N), v: (..., N)
        A_v = torch.einsum('...mn,...n->...m', A, v)  # (..., M)
        update = tau.unsqueeze(-1).unsqueeze(-1) * torch.einsum('...m,...n->...mn', A_v, v)
        return A - update


def apply_householder_inplace(A, v, tau, side='left', start_row=0, start_col=0):
    """
    Apply Householder reflection in place to a submatrix.
    
    More efficient for iterative QR where we only transform
    the trailing submatrix.
    
    Args:
        A: Matrix (M, N) - modified in place
        v: Householder vector (M - start_row,) for left
        tau: Householder scalar
        side: 'left' or 'right'
        start_row: Starting row for the submatrix
        start_col: Starting column for the submatrix
    """
    if side == 'left':
        # Apply to A[start_row:, start_col:]
        sub_A = A[start_row:, start_col:]
        vT_A = v @ sub_A  # (N - start_col,)
        sub_A -= tau * torch.outer(v, vT_A)
    else:
        # Apply to A[start_row:, start_col:]
        sub_A = A[start_row:, start_col:]
        A_v = sub_A @ v  # (M - start_row,)
        sub_A -= tau * torch.outer(A_v, v)


def householder_wy(V, T):
    """
    WY representation of accumulated Householder reflections.
    
    Q = I - V @ T @ V.T
    
    where V is (M, K) with K Householder vectors as columns,
    and T is (K, K) upper triangular.
    
    This allows applying multiple Householder reflections with
    a single level-3 BLAS operation:
        Q @ A = A - V @ (T @ (V.T @ A))
    
    Args:
        V: Matrix of Householder vectors (M, K)
        T: Upper triangular accumulator (K, K)
        
    Returns:
        Q: Orthogonal matrix (M, M)
    """
    M, K = V.shape
    I = torch.eye(M, device=V.device, dtype=V.dtype)
    return I - V @ T @ V.T


def apply_householder_wy(A, V, T, side='left'):
    """
    Apply accumulated Householder reflections using WY representation.
    
    Q @ A = A - V @ T @ (V.T @ A)  (left)
    A @ Q = A - (A @ V) @ T.T @ V.T  (right)
    
    This is much more efficient than applying reflections one by one
    because it converts K rank-1 updates into two matrix multiplications.
    
    Args:
        A: Matrix (..., M, N)
        V: Householder vectors (..., M, K)
        T: Upper triangular (..., K, K)
        side: 'left' or 'right'
        
    Returns:
        Transformed matrix
    """
    if side == 'left':
        # Q @ A = A - V @ T @ (V.T @ A)
        VT_A = torch.matmul(V.transpose(-2, -1), A)  # (K, N)
        T_VT_A = torch.matmul(T, VT_A)  # (K, N)
        return A - torch.matmul(V, T_VT_A)  # (M, N)
    else:
        # A @ Q = A - (A @ V) @ T.T @ V.T
        A_V = torch.matmul(A, V)  # (M, K)
        A_V_TT = torch.matmul(A_V, T.transpose(-2, -1))  # (M, K)
        return A - torch.matmul(A_V_TT, V.transpose(-2, -1))  # (M, N)
