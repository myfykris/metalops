import torch
import os
from torch.utils.cpp_extension import load
from . import config

# Load Extension
def get_cpp_source_path():
    package_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming standard install structure, navigate to native/
    # For development (editable install), it might be different.
    # We'll try to locate it relative to source root
    root_dir = os.path.abspath(os.path.join(package_dir, "../../.."))
    native_dir = os.path.join(root_dir, "native")
    return os.path.join(native_dir, "eigh_mps.mm")

# Logic to compile/load
# For now, rely on `pip install .` having built it as `metaleig_backend`
try:
    import metaleig_backend
except ImportError:
    # Fallback or JIT compile?
    # raise ImportError("metaleig_backend not available. Please install package.")
    pass

def eigh(A):
    """
    Computes eigenvalues and eigenvectors of a real symmetric matrix A.
    A: Tensor (..., N, N) symmetric
    Returns: (eigenvalues, eigenvectors)
             eigenvalues: (..., N)
             eigenvectors: (..., N, N)
    """
    if A.device.type != 'mps':
        return torch.linalg.eigh(A)
    
    is_batched = A.dim() > 2

    # CPU Fallback Mode
    # If enabled and input is a single matrix, run on CPU.
    # Why?
    # 1. Latency: Metal driver overhead (~5ms) dominates small single ops (CPU ~0.1ms).
    # 2. Algorithm: CPU uses Divide & Conquer (O(N^3)) which is faster than Jacobi (O(k*N^3)) for single large N.
    # Metal wins on THROUGHPUT (Batched), so we keep Metal for batched.
    if config.ENABLE_CPU_FALLBACK and not is_batched:
        return _eigh_cpu_fallback(A)
    
    # Backend always returns batched form
    L, Q = metaleig_backend.eigh_forward(A)
    
    if not is_batched:
        L = L.squeeze(0)
        Q = Q.squeeze(0)
        
    return L, Q

def eigvalsh(A):
    return eigh(A)[0]

def _eigh_cpu_fallback(A):
    # Move to CPU, compute, move back
    # Note: torch.linalg.eigh on CPU is very fast
    A_cpu = A.detach().cpu()
    L_cpu, Q_cpu = torch.linalg.eigh(A_cpu)
    return L_cpu.to(A.device), Q_cpu.to(A.device)
