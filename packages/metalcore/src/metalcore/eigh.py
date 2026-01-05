import torch
from . import config

try:
    import metalcore_backend as mc
    HAS_BACKEND = True
except ImportError:
    mc = None
    HAS_BACKEND = False


def eigh(A, strategy='auto'):
    """
    Computes eigenvalues and eigenvectors of a real symmetric matrix A.
    
    Args:
        A: Tensor (..., N, N) symmetric, on MPS device
        strategy: Algorithm selection
            - 'auto': Automatically select best strategy based on size/batch
            - 'jacobi': One-sided Jacobi (current Metal kernels)
            - 'tridiag': Tridiagonalization + CPU solver
            - 'cpu': Force CPU fallback
    
    Returns:
        eigenvalues: (..., N) in ascending order
        eigenvectors: (..., N, N)
    """
    if A.device.type != 'mps':
        return torch.linalg.eigh(A)
    
    is_batched = A.dim() > 2
    N = A.shape[-1]
    batch_size = A.shape[0] if is_batched else 1
    
    # Strategy selection
    if strategy == 'auto':
        strategy = _select_strategy(N, batch_size, is_batched)
    
    if strategy == 'cpu':
        return _eigh_cpu_fallback(A)
    
    if strategy == 'tridiag':
        return _eigh_tridiag(A)
    
    if strategy == 'jacobi':
        return _eigh_jacobi(A, is_batched)
    
    raise ValueError(f"Unknown strategy: {strategy}")


def _select_strategy(N, batch_size, is_batched):
    """
    Auto-select best strategy based on matrix geometry.
    
    Based on benchmarks (vs CPU with transfer included):
    - Single matrices: CPU is always faster (LAPACK D&C is very optimized)
    - Batched N=64: Jacobi wins (~2.9x speedup)
    - Batched N=128: Jacobi wins (~2.4x speedup)  
    - Batched N>=256: CPU is faster (Jacobi O(N²) per pair becomes expensive)
    
    Note: CPU fallback can be disabled via config.ENABLE_CPU_FALLBACK = False
    """
    if not config.ENABLE_CPU_FALLBACK:
        # CPU fallback disabled - always use GPU
        return 'jacobi'
    
    if not is_batched:
        # Single matrix: CPU is always faster
        return 'cpu'
    else:
        # Batched: Jacobi only wins for small N
        if N <= 128:
            return 'jacobi'  # Clear win (2.4-2.9x speedup)
        else:
            return 'cpu'  # CPU is faster for larger N


def _eigh_jacobi(A, is_batched):
    """Use current Metal Jacobi implementation."""
    if mc is None:
        raise ImportError("metalcore_backend not available. Please rebuild.")
    
    L, Q = mc.eigh_forward(A)
    
    # Jacobi returns eigenvalues in arbitrary order
    # Sort them in ascending order to match torch.linalg.eigh convention
    if is_batched:
        # Batched case: sort along last dimension
        sorted_indices = torch.argsort(L, dim=-1)
        L = torch.gather(L, -1, sorted_indices)
        # Reorder eigenvectors: Q[..., :, sorted_indices]
        Q = torch.gather(Q, -1, sorted_indices.unsqueeze(-2).expand_as(Q))
    else:
        L = L.squeeze(0)
        Q = Q.squeeze(0)
        # Single matrix: sort
        sorted_indices = torch.argsort(L)
        L = L[sorted_indices]
        Q = Q[:, sorted_indices]
        
    return L, Q


def _eigh_tridiag(A):
    """Use tridiagonalization + CPU solver."""
    from .tridiag import eigh_tridiag
    return eigh_tridiag(A)


def _eigh_cpu_fallback(A):
    """Move to CPU, compute, move back."""
    A_cpu = A.detach().cpu()
    L_cpu, Q_cpu = torch.linalg.eigh(A_cpu)
    return L_cpu.to(A.device), Q_cpu.to(A.device)


def eigvalsh(A, strategy='auto'):
    """Compute eigenvalues only (faster than full eigh)."""
    return eigh(A, strategy=strategy)[0]
