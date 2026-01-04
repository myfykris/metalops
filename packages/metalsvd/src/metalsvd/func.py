import torch
import metalsvd_backend
from . import config

# Import Metal eigh for GPU-accelerated Gram SVD
try:
    import metaleig
    HAS_METALEIG = True
except ImportError:
    HAS_METALEIG = False

def svd(A: torch.Tensor):
    """
    Computes the Singular Value Decomposition of a batch of matrices A (M x N).
    A = U @ diag(S) @ V.T
    
    Args:
        A: (..., M, N) tensor. Currently supports 2D or 3D tensors.
    
    Returns:
        U: (..., M, N) orthogonal
        S: (..., N) singular values
        V: (..., N, N) orthogonal
    """
    # Basic shape checks
class SVDAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        # A: (B, M, N)
        # Pad if N is odd (Odd dimension handling)
        orig_N = A.size(-1)
        if orig_N % 2 != 0:
            # Pad with 1 column of zeros
            A_pad = torch.nn.functional.pad(A, (0, 1), value=0)
        else:
            A_pad = A
            
        A_pad = A_pad.contiguous()

        if config.ENABLE_DE_RIJK_OPT:
            # De Rijk Strategy: Sort columns by norm descending to improve convergence
            # 1. Compute column norms
            col_norms = torch.linalg.norm(A_pad, dim=-2) # (B, N)
            
            # 2. Get sort indices
            _, perm = torch.sort(col_norms, dim=-1, descending=True) # (B, N)
            
            # 3. Permute A columns
            # perm: (B, N) -> (B, 1, N) for gather
            perm_expanded = perm.unsqueeze(1).expand(A_pad.shape)
            A_sorted = torch.gather(A_pad, -1, perm_expanded)
            
            # Run Backend SVD on Sorted Matrix
            U, S, V = metalsvd_backend.svd_forward(A_sorted)
            
            # 4. Restore V rows (Scatter back)
            # V is (B, N, N). We permuted columns of A, which corresponds to rows of V in SVD def?
            # A_sorted = A * P.  A = U S V_sorted.T. 
            # A * P = U S V_sorted.T  => A = U S V_sorted.T * P.T = U S (P * V_sorted).T
            # So V_true = P * V_sorted.
            # Only rows of V_sorted need to be permuted back to original indices.
            
            # We want V_final such that V_final[perm[i]] = V_sorted[i] (conceptually)
            # Actually simpler: V_final.gather(1, perm_expanded_V) = V_sorted is not quite right for scatter.
            # We use scatter explicitly.
            
            V_final = torch.empty_like(V)
            # perm for V: (B, N, N) expanding along dim 2
            perm_v = perm.unsqueeze(-1).expand(V.shape)
            
            # Scatter src=V into index=perm_v along dim=1 (rows)
            # V_final[b, perm[b, i], j] = V[b, i, j]
            V_final.scatter_(1, perm_v, V)
            V = V_final
            
        else:
            # Run Backend SVD Normally
            U, S, V = metalsvd_backend.svd_forward(A_pad)
        
        # Sort (Backend might not sort)
        S_sorted, indices = torch.sort(S, dim=-1, descending=True)
        idx_u = indices.unsqueeze(1).expand(U.shape)
        U_sorted = torch.gather(U, -1, idx_u)
        idx_v = indices.unsqueeze(1).expand(V.shape)
        V_sorted = torch.gather(V, -1, idx_v)
        
        # Unpad if needed
        # If padded, the last column of U/V and last element of S correspond to the dummy column (usually singular value 0).
        # We slice back to orig_N.
        if orig_N % 2 != 0:
            U_sorted = U_sorted[..., :orig_N] 
            S_sorted = S_sorted[..., :orig_N]
            U_sorted = U_sorted[..., :orig_N] 
            V_sorted = V_sorted[..., :orig_N, :orig_N]

        # Sign Canonicalization
        # Flip signs so that the element with max magnitude in each column of U is positive.
        # This is more stable than just checking the first element.
        
        # U_sorted: (B, M, N)
        # Find max abs index along dim 1
        _, max_idx = torch.max(U_sorted.abs(), dim=1) # (B, N)
        
        # Gather the actual values at those indices
        # max_idx: (B, N) -> (B, 1, N)
        max_idx_expanded = max_idx.unsqueeze(1)
        # Gather from U
        max_vals = torch.gather(U_sorted, 1, max_idx_expanded).squeeze(1) # (B, N)
        
        signs = torch.sign(max_vals)
        signs[signs == 0] = 1.0
        
        # Expand signs to broadcast
        U_sorted = U_sorted * signs.unsqueeze(1)
        V_sorted = V_sorted * signs.unsqueeze(1)
        
        ctx.save_for_backward(U_sorted, S_sorted, V_sorted)
        return U_sorted, S_sorted, V_sorted

    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_V):
        U, S, V = ctx.saved_tensors
        # Standard SVD Backward
        # Based on "Backpropagation for a Linear Layer" / PyTorch internals
        # dA = U ( dS + F * (U.T dU - V.T dV) ) V.T
        # Where F_ij = 1 / (s_i^2 - s_j^2)
        
        Vt = V.transpose(-2, -1)
        Ut = U.transpose(-2, -1)
        
        # S is vector (B, N). Make diagonal (B, N, N)
        S_diag = torch.diag_embed(S)
        
        # F matrix
        # s_i^2 - s_j^2
        S2 = S * S
        # Broadcast subtraction
        F = S2.unsqueeze(-1) - S2.unsqueeze(-2) # (B, N, N)
        
        # Safe inverse
        eps = 1e-6
        # Mask diagonal and close values
        # F = 1/F. 
        # For i=j, F is infinite. But the term multiplying it is 0 (diagonal of skew symmetric).
        # We fill diagonal with 0.
        F_inv = F.clone()
        mask = F_inv.abs() < eps
        F_inv[mask] = float('inf') 
        F = 1.0 / F_inv
        F[mask] = 0.0
        
        # Terms
        # U.T @ grad_U
        Ut_gU = torch.matmul(Ut, grad_U)
        # V.T @ grad_V
        Vt_gV = torch.matmul(Vt, grad_V)
        
        # Make them skew-symmetric parts?
        # The formula usually assumes U, V are orthogonal.
        # Term J = F * (Ut_gU - Vt_gV)
        # Note: In some derivations, sym/skew-sym logic applies.
        # Specifically: (U.T dU)_sym = 0.
        # We use the standard full rank formula.
        
        J = F * (Ut_gU - Vt_gV)
        
        # Correct diagonal of J is 0. Mmasked above.
        
        # dA = U @ (diag(grad_S) + J) @ V.T
        # Wait, shape of J is (N, N).
        # U is (M, N). V.T is (N, N).
        # term inner: (B, N, N).
        
        # Contribution from grad_S
        term_S = torch.diag_embed(grad_S)
        
        inner = term_S + J
        
        # If M > N (Tall matrix), there's a correction term for (I - U U.T) grad_U (part orthogonal to U).
        # dA += (I - U U.T) grad_U S^-1 V.T
        # My MPS SVD returns 'thin' U (M x N).
        
        dA = torch.matmul(torch.matmul(U, inner), Vt)
        
        if U.size(-2) > U.size(-1): # M > N
             # Add projection term
             # (grad_U - U @ Ut_gU) @ diag(1/S) @ V.T
             # This handles the component of grad_U orthogonal to U.
             
             inv_S = 1.0 / (S + eps)
             inv_S_mat = torch.diag_embed(inv_S)
             
             # Project grad_U onto U perp
             # P_U = U U.T
             # grad_U_perp = grad_U - U @ (U.T @ grad_U)
             grad_U_perp = grad_U - torch.matmul(U, Ut_gU)
             
             term_perp = torch.matmul(torch.matmul(grad_U_perp, inv_S_mat), Vt)
             dA = dA + term_perp
             
        # Restore batch dim logic if needed?
        # A was (B, M, N). dA is (B, M, N).
        # If input was 2D, A was promoted to 3D by svd() wrapper? 
        # No, forward received unsqueezed?
        # Let's check svd() wrapper.
             
        return dA

def svd(A, full_matrices=False, compute_uv=True, strategy='auto'):
    """
    Computes SVD of A (..., M, N) -> U, S, V.
    Differentiable.
    
    Args:
        A: Input tensor
        full_matrices: (Ignored for now, always returns thin SVD)
        compute_uv: (Ignored for now, always computes UV)
        strategy: 'auto', 'gram', or 'standard'
    """
    if A.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")
    
    is_batched = A.dim() > 2
    if not is_batched:
        A = A.unsqueeze(0)
    
    # Check CPU Fallback for Single SMALL Matrix (optimization for latency)
    # For large matrices (N >= 1024), the Gram path is faster than CPU SVD.
    # For smaller matrices (N < 1024), CPU is faster (Metal overhead dominates).
    N = A.shape[-1]
    is_small = N < 1024
    if config.ENABLE_CPU_FALLBACK and A.size(0) == 1 and is_small:
         U, S, V = _svd_cpu_fallback(A) # returns (1, M, N)
         if not is_batched:
             U = U.squeeze(0)
             S = S.squeeze(0)
             V = V.squeeze(0)
         return U, S, V

    # Check Wide Matrix (M < N)
    # One-sided Jacobi requires M >= N.
    # If M < N, compute SVD(A.T) = V S U.T
    # A = U S V.T
    M, N = A.shape[-2], A.shape[-1]
    is_wide = M < N
    
    if is_wide:
        # Recursive call specific to wide: compute SVD(A.T) = U_t S V_t.T
        # Then A = (A.T).T = (V_t S U_t.T).T = U_t S V_t.T swapped
        # So: A's U = V_t, A's S = S, A's V = U_t
        Ut, S, Vt = svd(A.transpose(-2, -1), full_matrices, compute_uv, strategy)
        
        if not is_batched:
            Ut = Ut.squeeze(0)
            S = S.squeeze(0)
            Vt = Vt.squeeze(0)
        
        # For A = U S V.T: U is Vt (N_orig x K), V is Ut (M_orig x K)
        # Wait no - check dimensions carefully:
        # A is (M, N) with M < N (wide)
        # A.T is (N, M) tall
        # svd(A.T) returns: Ut (N, M), S (M,), Vt (M, M)
        # A = (A.T).T = (Ut @ diag(S) @ Vt.T).T = Vt @ diag(S) @ Ut.T
        # So for A = U @ S @ V.T: U = Vt (M, M), V = Ut (N, M)
        return Vt, S, Ut

    # Forward expects 3D (B, Rows, Cols)
    
    # Check Tall Matrix Strategy (Optimization for M >> N)
    # Reduces complexity from O(M N^2) to O(N^3) + Matmul cost.
    # Heuristic: 
    # - Float32: M > 1.5 * N (Performance tradeoff)
    # - BFloat16: M > 1.2 * N (Stability preference for Hybrid Gram over native Jacobi)
    ratio = A.shape[-2] / A.shape[-1]
    
    # Heuristic for Gram Strategy:
    # 1. Use for Tall matrices (M >> N) for performance.
    # 2. Use for Large Square Matrices (N >= GRAM_THRESHOLD) for speed.
    # 3. Use for ALL BFloat16 matrices to ensure accuracy (avoiding sensitive Jacobi kernel).
    
    threshold = 1.5 # Std Float32 threshold for tall matrices
    is_large_square = (A.shape[-1] >= config.GRAM_THRESHOLD)
    is_bf16 = (A.dtype == torch.bfloat16)
    
    should_use_gram = (ratio > threshold) or is_large_square or is_bf16
    
    use_gram = False
    if strategy == 'gram':
        use_gram = True
    elif strategy == 'standard':
        use_gram = False
    elif strategy == 'auto':
        use_gram = should_use_gram
    
    if use_gram:
       # Gram Matrix Strategy (A^T A)
       # Faster than QR for tall matrices on Metal due to optimized Matmul.
       try:
           # 1. Precision Promotion
           # Gram matrix formation squares the condition number. 
           # We MUST accumulate in Float32 to preserve structure.
           orig_dtype = A.dtype
           using_low_precision = (orig_dtype == torch.bfloat16 or orig_dtype == torch.float16)
           
           if using_low_precision:
               A_compute = A.float()
           else:
               A_compute = A

           # 2. Form K (N x N) in Float32
           # K = A^T @ A
           K = torch.matmul(A_compute.transpose(-2, -1), A_compute)
           
           # 3. Eigendecomposition of K
           # K is symmetric positive semi-definite.
           # CPU eigh is used because:
           # 1. Single large matrices: CPU LAPACK is faster than Metal eigh (~0.9x)
           # 2. Transfer overhead is minimal since A^T@A on GPU produces K on GPU
           # 3. CPU eigh (Divide & Conquer) is O(N^3) vs Jacobi O(k*N^3)
           K_cpu = K.cpu()
           L, V_k = torch.linalg.eigh(K_cpu)
           V_k = V_k.to(A.device)
           L = L.to(A.device)
           
           # eigh returns eigenvalues in ASCENDING order - flip to descending
           L = L.flip(-1)
           V_k = V_k.flip(-1)
           
           # Compute Singular Values: S = sqrt(L)
           S_sq = torch.clamp(L, min=0.0)
           S_k = torch.sqrt(S_sq)
           
           # 4. Reconstruct U
           # U = A V S^-1
           
           # Compute A @ V
           AV = torch.matmul(A_compute, V_k) 
           
           # Scale by 1/S
           epsilon = 1e-6 if not using_low_precision else 1e-4
           inv_S = torch.where(S_k > epsilon, 1.0 / S_k, torch.zeros_like(S_k))
           
           U = AV * inv_S.unsqueeze(-2)
           
           if not is_batched:
               U = U.squeeze(0)
               S_k = S_k.squeeze(0)
               V_k = V_k.squeeze(0)
               
           return U.to(orig_dtype), S_k.to(orig_dtype), V_k.to(orig_dtype)
           
       except Exception as e:
           # Fallback
           import traceback
           traceback.print_exc()
           pass

    U_raw, S_raw, V_raw = SVDAutograd.apply(A)
    U, S, V = U_raw, S_raw, V_raw
    
    if not is_batched:
        U = U.squeeze(0)
        S = S.squeeze(0)
        V = V.squeeze(0)
        
    return U, S, V


def _svd_cpu_fallback(A):
    # Move to CPU, compute, move back
    # Note: torch.linalg.svd on CPU is very fast (LAPACK Divide & Conquer)
    # A is (1, M, N) here because of unsqueeze(0) in caller? 
    # Wait, in caller we did A = A.unsqueeze(0) BEFORE calling this.
    # So A is (1, M, N).
    
    A_cpu = A.detach().cpu()
    # torch.linalg.svd returns U, S, Vh (transposed V)
    # But our API returns U, S, V (not transposed)?
    # Let's check docstring at line 6: "A = U @ diag(S) @ V.T"
    # Return: "V: (..., N, N) orthogonal"
    # torch.linalg.svd returns Vh (adjoint).
    # So V = Vh.mH (conjugate transpose). Since real, just transpose.
    
    U_cpu, S_cpu, Vh_cpu = torch.linalg.svd(A_cpu, full_matrices=False)
    
    V_cpu = Vh_cpu.transpose(-2, -1)
    
    # Return same shape (1, M, N) - caller handles squeeze if needed
    U = U_cpu.to(A.device)
    S = S_cpu.to(A.device)
    V = V_cpu.to(A.device)
    
    return U, S, V

def _randomized_svd(A, n_components=None, n_oversamples=10, n_iter=2):
    """
    Randomized SVD for large matrices.
    Uses Power Iteration method for improved accuracy.
    
    Args:
        A: Input tensor (B, M, N)
        n_components: Target rank (default: min(M, N))
        n_oversamples: Extra samples for accuracy (default: 10)
        n_iter: Power iteration count (default: 2)
    
    Returns:
        U, S, V same as regular SVD
    """
    device = A.device
    dtype = A.dtype
    B, M, N = A.shape
    
    # Target rank 
    if n_components is None:
        n_components = min(M, N)
    k = min(n_components + n_oversamples, min(M, N))
    
    # 1. Random projection: Y = A @ Ω
    Omega = torch.randn(B, N, k, device=device, dtype=dtype)
    Y = torch.matmul(A, Omega)  # (B, M, k)
    
    # 2. Power iteration for improved accuracy
    for _ in range(n_iter):
        Y = torch.matmul(A, torch.matmul(A.transpose(-2, -1), Y))
    
    # 3. Orthogonalize Y -> Q
    Q, _ = torch.linalg.qr(Y)  # (B, M, k)
    
    # 4. Form smaller matrix B = Q^T @ A
    B_small = torch.matmul(Q.transpose(-2, -1), A)  # (B, k, N)
    
    # 5. SVD of smaller matrix (on CPU for robustness)
    B_cpu = B_small.cpu()
    U_b_cpu, S_cpu, Vh_cpu = torch.linalg.svd(B_cpu, full_matrices=False)
    
    # 6. Recover U = Q @ U_b
    U_b = U_b_cpu.to(device)
    S = S_cpu.to(device)
    V = Vh_cpu.transpose(-2, -1).to(device)
    
    U = torch.matmul(Q, U_b)  # (B, M, k)
    
    # Truncate to n_components
    if n_components < k:
        U = U[..., :n_components]
        S = S[..., :n_components]
        V = V[..., :n_components]
    
    return U, S, V
