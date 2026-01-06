"""
ComfyUI Metal Patch

Monkeypatches key PyTorch functions to use Metal-accelerated implementations.
This is particularly useful for ComfyUI where GELU/SiLU activations are hot paths.

Usage:
    import metalcore
    metalcore.patch_comfy()  # Apply patches
    
    # Run ComfyUI normally - it will now use Metal kernels
    
    metalcore.unpatch_comfy()  # Restore original (optional)
"""

import torch
import torch.nn.functional as F

# Store original functions
_originals = {}
_patched = False


def patch_comfy(activations: bool = True, sdpa: bool = False, verbose: bool = True):
    """
    Monkeypatch PyTorch functions to use Metal-accelerated implementations.
    
    Args:
        activations: Patch GELU and SiLU activations (default: True)
        sdpa: Patch scaled_dot_product_attention (default: False, as PyTorch's MPS is faster)
        verbose: Print status messages (default: True)
    
    Example:
        import metalcore
        metalcore.patch_comfy()
        
        # Now run ComfyUI - it will use Metal GELU/SiLU
    """
    global _patched, _originals
    
    if _patched:
        if verbose:
            print("metalcore: Already patched")
        return
    
    from .activations import metal_gelu, metal_silu
    
    if activations:
        # Store originals
        _originals['gelu'] = F.gelu
        _originals['silu'] = F.silu
        
        # Create wrapper that handles the approximate argument for GELU
        def patched_gelu(input, approximate='none'):
            if input.device.type == 'mps':
                return metal_gelu(input)
            return _originals['gelu'](input, approximate=approximate)
        
        def patched_silu(input, inplace=False):
            if input.device.type == 'mps':
                return metal_silu(input)
            return _originals['silu'](input, inplace=inplace)
        
        F.gelu = patched_gelu
        F.silu = patched_silu
        
        if verbose:
            print("metalcore: Patched F.gelu and F.silu")
    
    if sdpa:
        from .sdpa import metal_scaled_dot_product_attention
        _originals['sdpa'] = F.scaled_dot_product_attention
        
        def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
            if query.device.type == 'mps':
                return metal_scaled_dot_product_attention(
                    query, key, value, 
                    attn_mask=attn_mask, 
                    dropout_p=dropout_p, 
                    is_causal=is_causal, 
                    scale=scale
                )
            return _originals['sdpa'](query, key, value, attn_mask, dropout_p, is_causal, scale)
        
        F.scaled_dot_product_attention = patched_sdpa
        
        if verbose:
            print("metalcore: Patched F.scaled_dot_product_attention")
    
    # Always patch torch.linalg.solve (MPS doesn't support it)
    from .solve import solve as metal_solve
    _originals['linalg_solve'] = torch.linalg.solve
    
    def patched_linalg_solve(A, B, *, left=True, out=None):
        if A.device.type == 'mps' and left and out is None:
            return metal_solve(A, B)
        return _originals['linalg_solve'](A, B, left=left, out=out)
    
    torch.linalg.solve = patched_linalg_solve
    
    if verbose:
        print("metalcore: Patched torch.linalg.solve")
    
    _patched = True
    if verbose:
        print("metalcore: ComfyUI patches applied successfully")


def unpatch_comfy(verbose: bool = True):
    """
    Remove all monkeypatches and restore original PyTorch functions.
    """
    global _patched, _originals
    
    if not _patched:
        if verbose:
            print("metalcore: Not patched, nothing to unpatch")
        return
    
    if 'gelu' in _originals:
        F.gelu = _originals['gelu']
    if 'silu' in _originals:
        F.silu = _originals['silu']
    if 'sdpa' in _originals:
        F.scaled_dot_product_attention = _originals['sdpa']
    if 'linalg_solve' in _originals:
        torch.linalg.solve = _originals['linalg_solve']
    
    _originals.clear()
    _patched = False
    
    if verbose:
        print("metalcore: Patches removed, original functions restored")


def is_patched() -> bool:
    """Check if ComfyUI patches are currently applied."""
    return _patched
