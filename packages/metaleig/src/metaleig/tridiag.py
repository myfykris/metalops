"""
Tridiagonalization-based eigendecomposition for large matrices.

Uses Householder reflections to reduce A to tridiagonal form T,
then solves the tridiagonal eigenvalue problem on CPU.

Optimized version using matrix operations for Householder application.
"""

import torch


def tridiagonalize_fast(A):
    """
    Tridiagonalize symmetric matrix A using Householder reflections.
    Optimized to use matrix operations instead of element-by-element.
    
    Returns:
        d: Diagonal elements (N,)
        e: Off-diagonal elements (N-1,)
        Q: Orthogonal transformation matrix such that T = Q.T @ A @ Q
    """
    device = A.device
    dtype = A.dtype
    n = A.shape[-1]
    
    # Work in float32 for stability
    A_work = A.float().clone()
    Q = torch.eye(n, device=device, dtype=torch.float32)
    
    for k in range(n - 2):
        # Extract column below diagonal
        x = A_work[k+1:, k].clone()
        m = len(x)
        
        if m == 0:
            continue
            
        # Compute Householder vector using the standard formula
        norm_x = torch.norm(x)
        if norm_x < 1e-10:
            continue
        
        # v = x + sign(x[0]) * norm(x) * e_1
        sign = 1.0 if x[0] >= 0 else -1.0
        v = x.clone()
        v[0] = v[0] + sign * norm_x
        
        # Normalize v
        norm_v = torch.norm(v)
        if norm_v < 1e-10:
            continue
        v = v / norm_v
        
        # Apply Householder: H = I - 2 * v @ v.T
        # To A: A' = H @ A @ H = A - 2*v@(v.T@A) - 2*(A@v)@v.T + 4*(v.T@A@v)*v@v.T
        # Simplified: A[k+1:, :] -= 2*v@(v.T@A[k+1:, :])
        #             A[:, k+1:] -= 2*(A[:, k+1:]@v)@v.T
        
        # Apply from left to A[k+1:, k:]
        sub = A_work[k+1:, k:]
        vT_sub = v @ sub  # (n-k-1,) @ (m, n-k) = (n-k,)
        A_work[k+1:, k:] = sub - 2.0 * torch.outer(v, vT_sub)
        
        # Apply from right to A[k:, k+1:]  
        sub = A_work[k:, k+1:]
        sub_v = sub @ v  # (n-k, m) @ (m,) = (n-k,)
        A_work[k:, k+1:] = sub - 2.0 * torch.outer(sub_v, v)
        
        # Accumulate Q: Q[:, k+1:] = Q[:, k+1:] @ H
        sub_Q = Q[:, k+1:]
        Qv = sub_Q @ v
        Q[:, k+1:] = sub_Q - 2.0 * torch.outer(Qv, v)
    
    # Extract diagonal and off-diagonal
    d = torch.diag(A_work)
    e = torch.diag(A_work, 1)
    
    return d.to(dtype), e.to(dtype), Q.to(dtype)


def eigh_tridiag(A):
    """
    Compute eigendecomposition using tridiagonalization strategy.
    
    Pipeline:
    1. Tridiagonalize: A -> (d, e, Q) where T = Q.T @ A @ Q
    2. Solve tridiag: T = V_t @ D @ V_t.T (CPU, very fast O(N²))  
    3. Transform back: V = Q @ V_t (GPU matmul)
    
    Args:
        A: Symmetric matrix (..., N, N) on MPS
        
    Returns:
        eigenvalues: (..., N)
        eigenvectors: (..., N, N)
    """
    device = A.device
    dtype = A.dtype
    
    # Handle batched case
    if A.dim() > 2:
        batch_shape = A.shape[:-2]
        n = A.shape[-1]
        A_flat = A.reshape(-1, n, n)
        
        eigenvalues_list = []
        eigenvectors_list = []
        
        for i in range(A_flat.shape[0]):
            L, V = eigh_tridiag(A_flat[i])
            eigenvalues_list.append(L)
            eigenvectors_list.append(V)
        
        eigenvalues = torch.stack(eigenvalues_list).reshape(*batch_shape, n)
        eigenvectors = torch.stack(eigenvectors_list).reshape(*batch_shape, n, n)
        
        return eigenvalues, eigenvectors
    
    n = A.shape[-1]
    
    # Step 1: Tridiagonalize
    d, e, Q = tridiagonalize_fast(A)
    
    # Step 2: Build tridiagonal matrix and solve on CPU
    # This is the fast part - CPU tridiagonal solver is O(N²)
    T_cpu = torch.diag(d.cpu()) + torch.diag(e.cpu(), 1) + torch.diag(e.cpu(), -1)
    eigenvalues_cpu, V_t_cpu = torch.linalg.eigh(T_cpu)
    
    # Step 3: Transfer back and transform
    eigenvalues = eigenvalues_cpu.to(device=device, dtype=dtype)
    V_t = V_t_cpu.to(device=device, dtype=dtype)
    
    # Step 4: Transform eigenvectors: V = Q @ V_t
    eigenvectors = torch.matmul(Q, V_t)
    
    return eigenvalues, eigenvectors
