import torch
from .func import svd as metal_svd

def lanczos_svd(A: torch.Tensor, k: int, n_iter: int = None, tol: float = 1e-5):
    """
    Computes truncated SVD using Golub-Kahan-Lanczos Bidiagonalization with Full Re-orthogonalization.
    
    A: (M, N) matrix on MPS.
    k: Number of singular values to find.
    n_iter: Size of Krylov subspace (defaults to 2*k).
    
    Returns: U, S, V
    """
    if A.dim() > 2:
         raise NotImplementedError("Batched Lanczos SVD is not yet supported. Please loop over batch dimension.")

    M, N = A.shape
         
    device = A.device
    
    if n_iter is None:
        n_iter = min(min(M, N), 2 * k + 1)
        
    # Storage for Lanczos vectors
    # U_lanczos: (M, n_iter + 1)
    # V_lanczos: (N, n_iter + 1) -> Actually V is size (N, n_iter) usually?
    # Bidiagonal B: alphas (diagonal), betas (superdiagonal)
    
    # Pre-allocate to avoid cat lists (performance)
    # Note: On MPS, pre-allocation is better.
    U_basis = torch.zeros(M, n_iter + 1, device=device)
    V_basis = torch.zeros(N, n_iter + 1, device=device) # Store v_1 ... v_{k+1}
    
    alphas = []
    betas = []
    
    # 1. Initialization
    # Start with random vector
    # b is typically u_1
    # u_1 = random / norm
    u = torch.randn(M, 1, device=device)
    u_norm = u.norm()
    u = u / (u_norm + 1e-12)
    
    U_basis[:, 0] = u.squeeze()
    
    # v_0 = 0
    beta = 0.0
    
    # Loop
    # We follow the notation:
    # beta_{i-1} u_i = A v_{i-1} - alpha_{i-1} u_{i-1} ... different variants.
    # Variant:
    # A v_i = alpha_i u_i + beta_i u_{i+1} ?? No that's Arnoldi.
    
    # Golub-Kahan Bidiagonlization Algorithm:
    # 1. Choose b (starting vector). u_1 = b / ||b||. v_0 = 0. beta_0 = 0?
    # 2. For j = 1 to k:
    #      v_j = A.T u_j - beta_{j-1} v_{j-1}
    #      alpha_j = ||v_j||
    #      v_j = v_j / alpha_j
    #      
    #      u_{j+1} = A v_j - alpha_j u_j
    #      beta_j = ||u_{j+1}||
    #      u_{j+1} = u_{j+1} / beta_j
    
    # We need to re-orthogonalize v_j against V_{1..j-1} AND u_{j+1} against U_{1..j}.
    
    # First step setup
    # u_1 is set. v_0 is 0. beta_0 is not really used in first step A.T u - 0.
    
    # But usually implementation stores beta_{j} which connects u_{j+1}.
    
    # To handle indices in loop:
    # U_basis[:, 0] holds u_1.
    # V_basis will hold v_1 .. v_k.
    
    # Let's map indices:
    # j goes from 0 to n_iter-1.
    # u_curr = U_basis[:, j]
    # v_prev is V_basis[:, j-1] if j>0 else 0.
    
    for j in range(n_iter):
        u_curr = U_basis[:, j].unsqueeze(1) # (M, 1)
        
        # 1. Compute v_unnorm
        # v = A^T u_j - beta_{j-1} v_{j-1}
        v = torch.mm(A.T, u_curr)
        if j > 0:
            v_prev = V_basis[:, j-1].unsqueeze(1)
            v = v - betas[-1] * v_prev
            
        # Full Re-orthogonalization of v against V_basis[:, :j]
        # v = v - V V^T v
        if j > 0:
            # V_basis_active: (N, j)
            V_active = V_basis[:, :j] 
            # Project
            # coeff = V^T v -> (j, 1)
            coeffs = torch.mm(V_active.T, v)
            # v = v - V coeff
            v = v - torch.mm(V_active, coeffs)
            
            # Double re-orthogonalization (standard for Lanczos stability)
            coeffs2 = torch.mm(V_active.T, v)
            v = v - torch.mm(V_active, coeffs2)
            
        alpha = v.norm()
        alphas.append(alpha)
        
        # Check breakdown
        if alpha < 1e-12:
            break
            
        v = v / alpha
        V_basis[:, j] = v.squeeze()
        
        # 2. Compute u_next
        # u_{j+1} = A v_j - alpha_j u_j
        u_next = torch.mm(A, v)
        u_next = u_next - alpha * u_curr
        
        # Full Re-orthogonalization of u_next against U_basis[:, :j+1]
        # U_active: (M, j+1)
        U_active = U_basis[:, :j+1]
        
        coeffs_u = torch.mm(U_active.T, u_next)
        u_next = u_next - torch.mm(U_active, coeffs_u)
        
        # Double
        coeffs_u2 = torch.mm(U_active.T, u_next)
        u_next = u_next - torch.mm(U_active, coeffs_u2)
        
        beta = u_next.norm()
        betas.append(beta)
        
        if beta < 1e-12:
            break
            
        u_next = u_next / beta
        # Store u_{j+1} at index j+1
        if j + 1 < n_iter + 1:
            U_basis[:, j+1] = u_next.squeeze()
            
    # Form Bidiagonal Matrix B
    # Diagonal = alphas. Superdiagonal = betas.
    # B is (j+1) x (j+1) or (j+1) x j ?
    # Dimensions:
    # A V_k = U_{k+1} B_k.
    # V_k is (N, k). U_{k+1} is (M, k+1).
    # B_k i (k+1, k).
    # Diag is alphas. Lower diag is betas?
    # Wait, my loop:
    # v_j from u_j. alpha_j
    # u_{j+1} from v_j. beta_j.
    # So alpha is diagonal term. beta is sub-diagonal term?
    # Let's verify relation: u_{j+1} beta_j = A v_j - alpha_j u_j
    # => A v_j = alpha_j u_j + beta_j u_{j+1}.
    # So A [v_1 ... v_k] = [u_1 ... u_{k+1}] B.
    # Col j of LHS is A v_j.
    # Col j of RHS is alpha_j u_j + beta_j u_{j+1}.
    # So B is (k+1) x k.
    # B[j, j] = alpha_j. B[j+1, j] = beta_j.
    
    n_found = len(alphas)
    
    B = torch.zeros(n_found + 1, n_found, device=device)
    B[torch.arange(n_found), torch.arange(n_found)] = torch.tensor(alphas, device=device)
    B[torch.arange(n_found) + 1, torch.arange(n_found)] = torch.tensor(betas, device=device)
    
    # SVD of B
    # B is small. Use mps_svd.
    # B is lower bidiagonal.
    # mps_svd handles general matrices.
    # But B is (K+1, K).
    Ub, Sb, Vb = metal_svd(B.unsqueeze(0))
    Ub = Ub.squeeze(0) # (K+1, K+1) or (K+1, K) depending on implementation (metal_svd returns full U)
    Sb = Sb.squeeze(0) # (K)
    Vb = Vb.squeeze(0) # (K, K)
    
    # Final U = U_basis_active @ Ub
    # Final V = V_basis_active @ Vb
    
    # U_basis used: indices 0..n_found (k+1 vectors)
    U_final = torch.mm(U_basis[:, :n_found+1], Ub)
    V_final = torch.mm(V_basis[:, :n_found], Vb) # Vb columns are singular vectors of B.
    
    # Note: Vb from metal_svd is V (not V.T).
    # B approx Ub Sb Vb.T
    # A V = U B approx U (Ub Sb Vb.T)
    # A (V Vb) = (U Ub) Sb
    # So Right Vectors = V Vb.
    
    # Truncate to k
    return U_final[:, :k], Sb[:k], V_final[:, :k]
