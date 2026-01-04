"""
QR Decomposition using Blocked Householder Algorithm.

Algorithm:
    for each block of columns:
        1. Factor panel: A[:, j:j+bs] = Q_panel @ R_panel (sequential Householder)
        2. Form WY representation: Q_panel = I - V @ T @ V.T
        3. Apply to trailing matrix: A[:, j+bs:] = Q_panel.T @ A[:, j+bs:] (big matmul!)

The key insight is that step 3 is a large matrix multiplication that dominates
runtime for big matrices, making this algorithm GPU-friendly.
"""

import torch
from .householder import householder_vector, apply_householder_wy


def qr(A, mode='reduced'):
    """
    Compute QR decomposition of a matrix.
    
    A = Q @ R
    
    where Q is orthogonal and R is upper triangular.
    
    Args:
        A: Matrix (..., M, N)
        mode: 'reduced' (default), 'complete', or 'r'
            - 'reduced': Q is (M, K), R is (K, N) where K = min(M, N)
            - 'complete': Q is (M, M), R is (M, N)
            - 'r': Only return R
            
    Returns:
        Q: Orthogonal matrix (if mode != 'r')
        R: Upper triangular matrix
    """
    if A.device.type != 'mps':
        # Use PyTorch implementation
        return torch.linalg.qr(A, mode=mode)
    
    # Handle batched case
    if A.dim() > 2:
        return _qr_batched(A, mode)
    
    return _qr_blocked(A, mode)


def _qr_blocked(A, mode='reduced', block_size=32):
    """
    Blocked Householder QR decomposition.
    
    This is the main algorithm optimized for GPU.
    """
    device = A.device
    dtype = A.dtype
    M, N = A.shape
    K = min(M, N)
    
    # Work on a copy
    R = A.clone()
    
    # Storage for Householder vectors (for forming Q later)
    V_storage = []
    tau_storage = []
    
    for j in range(0, K, block_size):
        # Determine block size
        bs = min(block_size, K - j)
        
        # Factor the panel A[j:, j:j+bs] using sequential Householder
        V_panel = torch.zeros(M - j, bs, device=device, dtype=dtype)
        tau_panel = torch.zeros(bs, device=device, dtype=dtype)
        
        for k in range(bs):
            col = j + k
            
            # Compute Householder vector for column col
            x = R[col:, col].clone()
            
            # Compute norm and sign
            norm_x = torch.norm(x)
            if norm_x < 1e-10:
                V_panel[k:, k] = x
                V_panel[k, k] = 1.0
                tau_panel[k] = 0.0
                continue
            
            sign = 1.0 if x[0] >= 0 else -1.0
            
            # v = x + sign * norm_x * e_1
            v = x.clone()
            v[0] = x[0] + sign * norm_x
            
            # tau = 2 * v[0]^2 / ||v||^2
            v_normsq = torch.dot(v, v)
            tau = 2.0 * v[0]**2 / v_normsq
            
            # Normalize v so v[0] = 1
            v = v / v[0]
            
            # Store v and tau
            V_panel[k:, k] = v
            tau_panel[k] = tau
            
            # Apply Householder to remaining columns in this panel and trailing matrix
            # H @ A[col:, col:] = A[col:, col:] - tau * v @ (v.T @ A[col:, col:])
            if col < N:
                trailing = R[col:, col:]
                vT_A = v @ trailing
                R[col:, col:] = trailing - tau * torch.outer(v, vT_A)
            
            # The diagonal element R[col, col] is now correct (-sign * norm_x)
            # Zero out below diagonal explicitly (should already be zero from H)
            R[col+1:, col] = 0.0
        
        V_storage.append((j, V_panel, tau_panel))
    
    if mode == 'r':
        return R[:K, :]
    
    # Form Q by accumulating Householder reflections
    if mode == 'reduced':
        Q = torch.eye(M, K, device=device, dtype=dtype)
    else:  # complete
        Q = torch.eye(M, M, device=device, dtype=dtype)
    
    # Apply Householder reflections in reverse order
    for j, V_panel, tau_panel in reversed(V_storage):
        bs = V_panel.shape[1]
        
        for k in range(bs - 1, -1, -1):
            col = j + k
            v = V_panel[k:, k]
            tau = tau_panel[k]
            
            # Apply to Q[col:, col:]
            if mode == 'reduced':
                sub_Q = Q[col:, col:]
            else:
                sub_Q = Q[col:, :]
            
            vT_Q = v @ sub_Q
            if mode == 'reduced':
                Q[col:, col:] = sub_Q - tau * torch.outer(v, vT_Q)
            else:
                Q[col:, :] = sub_Q - tau * torch.outer(v, vT_Q)
    
    if mode == 'reduced':
        return Q, R[:K, :]
    else:  # complete
        return Q, R


def _qr_batched(A, mode='reduced', block_size=32):
    """
    Batched QR decomposition.
    
    For now, loop over batch dimension. Could be parallelized with
    a proper Metal kernel.
    """
    batch_shape = A.shape[:-2]
    M, N = A.shape[-2:]
    
    A_flat = A.reshape(-1, M, N)
    batch_size = A_flat.shape[0]
    
    Q_list = []
    R_list = []
    
    for i in range(batch_size):
        if mode == 'r':
            R_i = _qr_blocked(A_flat[i], mode='r', block_size=block_size)
            R_list.append(R_i)
        else:
            Q_i, R_i = _qr_blocked(A_flat[i], mode=mode, block_size=block_size)
            Q_list.append(Q_i)
            R_list.append(R_i)
    
    R = torch.stack(R_list).reshape(*batch_shape, *R_list[0].shape)
    
    if mode == 'r':
        return R
    
    Q = torch.stack(Q_list).reshape(*batch_shape, *Q_list[0].shape)
    return Q, R


def qr_solve(A, b):
    """
    Solve least squares problem using QR decomposition.
    
    Minimizes ||Ax - b||_2
    
    Args:
        A: Matrix (M, N) with M >= N
        b: Right-hand side (M,) or (M, K)
        
    Returns:
        x: Solution (N,) or (N, K)
    """
    from .trsm import trsm
    
    Q, R = qr(A, mode='reduced')
    
    # x = R^{-1} @ Q.T @ b
    QT_b = Q.T @ b
    x = trsm(R, QT_b, lower=False)
    
    return x
