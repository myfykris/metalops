"""
metalcore: Foundational Metal Linear Algebra Primitives

This package provides GPU-accelerated building blocks for linear algebra:
- trsm: Triangular solve
- householder: Householder reflections
- qr: QR decomposition
- lstsq: Least squares solver
- pinv: Pseudo-inverse
- solve: Linear system solver
- cholesky: Cholesky decomposition
- cholesky_solve: Cholesky-based solve
- svd: Singular Value Decomposition
- eigh: Eigendecomposition (symmetric)
- High-impact: lu_batched, syrk_batched, frobenius_norm_batched, softmax_batched, trace_batched
"""

from .trsm import trsm, trsm_batched
from .householder import householder_vector, apply_householder
from .qr import qr, qr_solve
from .lstsq import lstsq
from .pinv import pinv
from .solve import solve
from .cholesky import cholesky, cholesky_solve
from .svd import svd
from .eigh import eigh
from .rmsnorm import RMSNormFunction, MetalRMSNorm, fused_add_rmsnorm, FusedAddRMSNorm
from .optim import MetalAdamW
from .activations import metal_gelu, metal_silu, MetalGELU, MetalSiLU
from .sdpa import metal_scaled_dot_product_attention
from .comfy_patch import patch_comfy, unpatch_comfy, is_patched
from .ops import (
    fused_softmax, MetalSoftmax,
    layer_norm, MetalLayerNorm,
    embedding_bag, MetalEmbeddingBag,
    gather, scatter_add, index_select
)

# High-impact ops (direct backend exports)
_HAS_METAL_KERNELS = False
try:
    import metalcore_backend as _mc
    # Check if kernels actually loaded by looking for a required function
    if hasattr(_mc, 'qr_single'):
        _HAS_METAL_KERNELS = True
    # lu_batched = _mc.lu_batched
    # syrk_batched = _mc.syrk_batched
    # frobenius_norm_batched = _mc.frobenius_norm_batched
    # softmax_batched = _mc.softmax_batched
    # trace_batched = _mc.trace_batched
except ImportError:
    pass

def has_metal_kernels() -> bool:
    """Check if Metal kernels are available (for test skipping)."""
    return _HAS_METAL_KERNELS

# SDPA enable/disable flags
_sdpa_enabled = False
_sdpa_original_fn = None

def enable_slow_metal_sdpa():
    """
    Enable Metal SDPA by monkeypatching torch.nn.functional.scaled_dot_product_attention.
    
    WARNING: Our custom Metal SDPA is ~6x SLOWER than PyTorch's native implementation!
    This is for non-PyTorch use cases only. Use PyTorch's native SDPA when possible.
    
    Usage:
        import metalcore
        metalcore.enable_slow_metal_sdpa()
    """
    global _sdpa_enabled, _sdpa_original_fn
    import torch.nn.functional as F
    
    if _sdpa_enabled:
        return  # Already enabled
    
    _sdpa_original_fn = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = metal_scaled_dot_product_attention
    _sdpa_enabled = True
    print("metalcore: Metal SDPA enabled (WARNING: ~6x slower than PyTorch native)")

def disable_slow_metal_sdpa():
    """Disable Metal SDPA and restore PyTorch's native implementation."""
    global _sdpa_enabled, _sdpa_original_fn
    import torch.nn.functional as F
    
    if not _sdpa_enabled or _sdpa_original_fn is None:
        return
    
    F.scaled_dot_product_attention = _sdpa_original_fn
    _sdpa_enabled = False
    _sdpa_original_fn = None
    print("metalcore: Metal SDPA disabled (restored native)")

def is_slow_metal_sdpa_enabled():
    """Check if (slow) Metal SDPA is currently enabled."""
    return _sdpa_enabled

__version__ = "0.1.11"
__all__ = [
    "trsm",
    "trsm_batched", 
    "householder_vector",
    "apply_householder",
    "qr",
    "qr_solve",
    "lstsq",
    "pinv",
    "solve",
    "cholesky",
    "cholesky_solve",
    "svd",
    "eigh",
    # Training Ops
    "RMSNormFunction",
    "MetalRMSNorm",
    "fused_add_rmsnorm",
    "FusedAddRMSNorm",
    "MetalAdamW",
    # Activations
    "metal_gelu",
    "metal_silu",
    "MetalGELU",
    "MetalSiLU",
    # Attention (explicit enable required - WARNING: slower than PyTorch native!)
    "metal_scaled_dot_product_attention",
    "enable_slow_metal_sdpa",
    "disable_slow_metal_sdpa",
    "is_slow_metal_sdpa_enabled",
    # ComfyUI patches
    "patch_comfy",
    "unpatch_comfy",
    "is_patched",
    # High-performance ops
    "fused_softmax",
    "MetalSoftmax",
    "layer_norm",
    "MetalLayerNorm",
    "embedding_bag",
    "MetalEmbeddingBag",
    "gather",
    "scatter_add",
    "index_select",
]
