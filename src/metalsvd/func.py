import torch
import metalsvd_backend

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

        # Run Backend SVD
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

def svd(A):
    """
    Computes SVD of A (..., M, N) -> U, S, V.
    Differentiable.
    """
    if A.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")
    
    is_batched = A.dim() > 2
    if not is_batched:
        A = A.unsqueeze(0)
    
    # Check Wide Matrix (M < N)
    # One-sided Jacobi requires M >= N.
    # If M < N, compute SVD(A.T) = V S U.T
    # A = U S V.T
    M, N = A.shape[-2], A.shape[-1]
    is_wide = M < N
    
    if is_wide:
        # Use recursive call to handle transpose
        # A (Wide) -> A.T (Tall)
        # svd(A.T) -> Ut, St, Vt
        # A.T = Ut St Vt.T
        # A = Vt St Ut.T
        Ut, S, Vt = svd(A.transpose(-2, -1))
        
        if not is_batched:
            Ut = Ut.squeeze(0)
            S = S.squeeze(0)
            Vt = Vt.squeeze(0)
            
        return Vt, S, Ut

    # Forward expects 3D (B, Rows, Cols)
    U_raw, S_raw, V_raw = SVDAutograd.apply(A)
    U, S, V = U_raw, S_raw, V_raw
    
    if not is_batched:
        U = U.squeeze(0)
        S = S.squeeze(0)
        V = V.squeeze(0)
        
    return U, S, V

