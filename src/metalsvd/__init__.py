from .func import svd
from .randomized import randomized_svd
from .lanczos import lanczos_svd

_original_svd = None

def patch_torch():
    """
    Monkeypatches torch.linalg.svd to use metalsvd.svd for MPS tensors.
    """
    global _original_svd
    import torch
    
    if _original_svd is not None:
        return # Already patched
        
    _original_svd = torch.linalg.svd
    
    def mps_aware_svd(A, full_matrices=True, *args, **kwargs):
        if A.device.type == 'mps':
             # Our SVD always returns thin U (M, N) if M >= N.
             # torch.linalg.svd full_matrices=True returns U (M, M).
             # full_matrices=False returns U (M, N).
             # We effectively implement full_matrices=False.
             # If user asks for full_matrices=True, we should warn or error?
             # For now, let's just delegate to our SVD and maybe note it.
             # Or check full_matrices.
             
             if full_matrices:
                 # We don't support full matrices yet (U is M x M).
                 # Fallback to CPU if they strictly need it?
                 # Or just run ours and let them deal with it?
                 # Better: Fallback if they insist on full_matrices.
                 pass 
             else:
                 # Use ours!
                 # mpssvd returns U, S, V. 
                 # torch.linalg.svd returns U, S, Vh (adjoint of V).
                 # Wait, torch returns Vh (V.T).
                 # mpssvd.svd returns V (not transposed). Columns are vectors.
                 # So we must return V.T.
                 
                 # NOTE: Our func.py returns V.
                 # Standard SVD A = U S Vh.
                 try:
                     U, S, V = svd(A)
                     # V in our output is (N, N) where columns are singular vectors.
                     # Vh should be V.T.
                     
                     # Check batch dims
                     if A.dim() > 2:
                         Vh = V.transpose(-2, -1)
                     else:
                         Vh = V.t()
                         
                     return U, S, Vh
                     
                 except Exception:
                     # Fallback on error (e.g. Wide matrix NotImplemented)
                     pass
        
        return _original_svd(A, full_matrices=full_matrices, *args, **kwargs)

    torch.linalg.svd = mps_aware_svd
    # Also patch torch.svd? (Deprecated but used)
    # torch.svd returns U, S, V (not Vh).
    
    print("Metal SVD patched into torch.linalg.svd")

