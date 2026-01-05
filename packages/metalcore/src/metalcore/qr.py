"""
QR Decomposition using Blocked Householder Algorithm.

Based on ROCm/rocSOLVER patterns:
1. For small matrices: Use unblocked algorithm (BLAS-2)
2. For large matrices: 
   - Factor panel with unblocked Householder (geqr2)
   - Build T matrix for WY representation (larft)
   - Apply block reflector to trailing matrix (larfb) - BIG GEMM!

The key insight is that larfb converts O(nb) rank-1 updates into
two large matrix multiplications, which is where GPU shines.
"""

import torch
from . import config


# Threshold for switching between blocked and unblocked
QR_BLOCKSIZE = 32
QR_SWITCH_SIZE = 64


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
        
    Note:
        For single matrices, CPU is used (LAPACK is 3-50x faster).
        For batched matrices, use qr_batched() from metalcore_backend.
    """
    if A.device.type != 'mps':
        return torch.linalg.qr(A, mode=mode)
    
    if A.dim() > 2:
        return _qr_batched(A, mode)
    
    # SMART FALLBACK: For single matrices, CPU always wins
    # due to sequential column dependencies in Householder QR.
    # CPU LAPACK achieves 3-50x faster performance than GPU.
    # Can be disabled via config.ENABLE_CPU_FALLBACK = False
    if config.ENABLE_CPU_FALLBACK:
        A_cpu = A.cpu()
        result = torch.linalg.qr(A_cpu, mode=mode)
        
        if mode == 'r':
            _, R = result
            return R.to(A.device)
        else:
            Q, R = result
            return Q.to(A.device), R.to(A.device)
    
    # Fallback disabled - use GPU implementation
    M, N = A.shape
    if M <= QR_SWITCH_SIZE or N <= QR_SWITCH_SIZE:
        return _qr_unblocked(A, mode)
    return _qr_blocked(A, mode)


def _geqr2(A):
    """
    Unblocked Householder QR (panel factorization).
    
    Computes QR of A in-place, storing Householder vectors below diagonal
    and R on/above diagonal. Returns tau values.
    
    This is LAPACK's geqr2: sequential Householder for each column.
    """
    device = A.device
    dtype = A.dtype
    M, N = A.shape
    K = min(M, N)
    
    R = A.clone()
    taus = torch.zeros(K, device=device, dtype=dtype)
    
    for k in range(K):
        # Get column k below diagonal
        x = R[k:, k].clone()
        
        # Compute Householder vector
        norm_x = torch.norm(x)
        if norm_x < 1e-10:
            taus[k] = 0.0
            continue
        
        sign = 1.0 if x[0] >= 0 else -1.0
        
        # v = x + sign * norm_x * e_1
        v = x.clone()
        v[0] = x[0] + sign * norm_x
        
        # tau = 2 / (v^T v / v[0]^2) = 2 * v[0]^2 / ||v||^2
        v_normsq = torch.dot(v, v)
        tau = 2.0 * v[0]**2 / v_normsq
        
        # Normalize v so v[0] = 1
        v = v / v[0]
        taus[k] = tau
        
        # Apply H to trailing matrix: R[k:, k:] = H @ R[k:, k:]
        # H @ R = R - tau * v @ (v^T @ R)
        trailing = R[k:, k:]
        vT_R = v @ trailing  # (N-k,) - this is a reduction
        R[k:, k:] = trailing - tau * torch.outer(v, vT_R)
        
        # Store v below diagonal (for later reconstruction of Q)
        R[k+1:, k] = v[1:]
    
    return R, taus


def _larft(V, tau):
    """
    Form the triangular factor T of a block reflector H = I - V @ T @ V^T.
    
    This is LAPACK's larft: builds the T matrix for WY representation.
    
    Args:
        V: Matrix (M, K) with Householder vectors as columns (v_i stored below diagonal)
           V[i, i] is implicitly 1
        tau: Householder scalars (K,)
        
    Returns:
        T: Upper triangular matrix (K, K)
    """
    device = V.device
    dtype = V.dtype
    M, K = V.shape
    
    T = torch.zeros(K, K, device=device, dtype=dtype)
    
    for i in range(K):
        T[i, i] = tau[i]
        
        if i > 0:
            # T[0:i, i] = -tau[i] * T[0:i, 0:i] @ V[:, 0:i]^T @ V[:, i]
            # Build V column properly (v[j] = 1 at position j)
            v_i = V[:, i].clone()
            v_i[i] = 1.0  # Implicit 1 on diagonal
            
            V_prev = V[:, :i].clone()
            for j in range(i):
                V_prev[j, j] = 1.0  # Implicit 1s
            
            VT_vi = V_prev.T @ v_i  # (i,)
            T[:i, i] = -tau[i] * (T[:i, :i] @ VT_vi)
    
    return T


def _larfb(C, V, T, side='left', trans=True):
    """
    Apply block Householder reflector to a matrix.
    
    This is LAPACK's larfb: the performance-critical operation.
    
    If side='left' and trans=True:
        C = H^T @ C = (I - V @ T^T @ V^T) @ C = C - V @ (T^T @ (V^T @ C))
    
    This is TWO matrix multiplications - GPU efficient!
    
    Args:
        C: Matrix (M, N) to transform
        V: Householder vectors (M, K) - stored below diagonal, implicit 1 on diagonal
        T: Triangular factor (K, K)
        side: 'left' or 'right'
        trans: If True, apply H^T; if False, apply H
        
    Returns:
        Transformed C
    """
    device = C.device
    dtype = C.dtype
    M, N = C.shape
    _, K = V.shape
    
    # Build proper V with implicit 1s on diagonal
    V_full = V.clone()
    for i in range(K):
        V_full[i, i] = 1.0
    
    if side == 'left':
        if trans:
            # C = C - V @ T^T @ V^T @ C
            # Step 1: W = V^T @ C  (K x N) - matmul
            W = torch.matmul(V_full.T, C)
            # Step 2: W = T^T @ W  (K x N) - triangular matmul
            W = torch.matmul(T.T, W)
            # Step 3: C = C - V @ W  (M x N) - matmul
            C = C - torch.matmul(V_full, W)
        else:
            # C = C - V @ T @ V^T @ C
            W = torch.matmul(V_full.T, C)
            W = torch.matmul(T, W)
            C = C - torch.matmul(V_full, W)
    else:  # side == 'right'
        if trans:
            # C = C @ H^T = C - C @ V @ T^T @ V^T
            W = torch.matmul(C, V_full)  # (M x K)
            W = torch.matmul(W, T.T)
            C = C - torch.matmul(W, V_full.T)
        else:
            W = torch.matmul(C, V_full)
            W = torch.matmul(W, T)
            C = C - torch.matmul(W, V_full.T)
    
    return C


def _qr_blocked(A, mode='reduced'):
    """
    Blocked Householder QR decomposition.
    
    Algorithm (follows rocSOLVER pattern):
        for each block of columns:
            1. Factor panel with geqr2 (unblocked)
            2. Build T matrix with larft
            3. Apply block reflector with larfb (big GEMM!)
    """
    device = A.device
    dtype = A.dtype
    M, N = A.shape
    K = min(M, N)
    nb = QR_BLOCKSIZE
    
    R = A.clone()
    
    # Storage for V and tau to reconstruct Q later
    V_storage = []
    T_storage = []
    
    j = 0
    while j < K:
        jb = min(nb, K - j)  # Block size
        
        # Factor panel A[j:, j:j+jb] using unblocked algorithm
        panel = R[j:, j:j+jb].clone()
        panel_R, panel_tau = _geqr2(panel)
        
        # Copy R part (on and above diagonal)
        R[j:, j:j+jb] = panel_R
        
        # Extract V part (below diagonal, with implicit 1s)
        V_panel = panel_R.clone()
        for k in range(jb):
            V_panel[:k+1, k] = 0.0  # Zero above diagonal (will be filled by R)
            V_panel[k, k] = 0.0  # Will be implicit 1
        
        # Build T matrix
        T_panel = _larft(V_panel, panel_tau)
        
        # Apply block reflector to trailing matrix
        if j + jb < N:
            trailing = R[j:, j+jb:]
            R[j:, j+jb:] = _larfb(trailing, V_panel, T_panel, side='left', trans=True)
        
        # Store for Q reconstruction
        V_storage.append((j, jb, V_panel, T_panel))
        
        j += jb
    
    # Zero below diagonal explicitly
    R = torch.triu(R[:K, :])
    
    if mode == 'r':
        return R
    
    # Reconstruct Q by applying reflectors in reverse
    if mode == 'reduced':
        Q = torch.eye(M, K, device=device, dtype=dtype)
    else:  # complete
        Q = torch.eye(M, M, device=device, dtype=dtype)
    
    for j, jb, V_panel, T_panel in reversed(V_storage):
        # Apply H (not H^T) to Q[j:, j:] 
        if mode == 'reduced':
            sub_Q = Q[j:, j:]
            Q[j:, j:] = _larfb(sub_Q, V_panel, T_panel, side='left', trans=False)
        else:
            sub_Q = Q[j:, :]
            Q[j:, :] = _larfb(sub_Q, V_panel, T_panel, side='left', trans=False)
    
    return Q, R


def _qr_unblocked(A, mode='reduced'):
    """
    Unblocked Householder QR for small matrices.
    """
    device = A.device
    dtype = A.dtype
    M, N = A.shape
    K = min(M, N)
    
    R, taus = _geqr2(A)
    
    # Zero below diagonal
    R_out = torch.triu(R[:K, :])
    
    if mode == 'r':
        return R_out
    
    # Build Q by applying Householder reflectors
    if mode == 'reduced':
        Q = torch.eye(M, K, device=device, dtype=dtype)
    else:
        Q = torch.eye(M, M, device=device, dtype=dtype)
    
    for k in range(K - 1, -1, -1):
        v = torch.zeros(M - k, device=device, dtype=dtype)
        v[0] = 1.0
        v[1:] = R[k+1:, k]  # Stored below diagonal
        tau = taus[k]
        
        if mode == 'reduced':
            sub_Q = Q[k:, k:]
            vT_Q = v @ sub_Q
            Q[k:, k:] = sub_Q - tau * torch.outer(v, vT_Q)
        else:
            sub_Q = Q[k:, :]
            vT_Q = v @ sub_Q
            Q[k:, :] = sub_Q - tau * torch.outer(v, vT_Q)
    
    return Q, R_out


def _qr_batched(A, mode='reduced'):
    """Batched QR decomposition."""
    batch_shape = A.shape[:-2]
    M, N = A.shape[-2:]
    
    A_flat = A.reshape(-1, M, N)
    batch_size = A_flat.shape[0]
    
    Q_list = []
    R_list = []
    
    for i in range(batch_size):
        if mode == 'r':
            R_i = qr(A_flat[i], mode='r')
            R_list.append(R_i)
        else:
            Q_i, R_i = qr(A_flat[i], mode=mode)
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
    """
    from .trsm import trsm
    
    Q, R = qr(A, mode='reduced')
    QT_b = Q.T @ b
    x = trsm(R, QT_b, lower=False)
    
    return x
