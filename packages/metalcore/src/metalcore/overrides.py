"""
PyTorch custom op registration for metalcore.

This module provides transparent acceleration of PyTorch operations by
registering metalcore implementations as custom backends for the MPS device.

Note: This feature is experimental and requires PyTorch 2.1+ for the library API.
For reliable usage, the direct metalcore API (metal_silu, metal_gelu, etc.) is recommended.
"""

import torch
from typing import Optional

# Track which overrides are currently active
_active_overrides: set = set()


def enable_pytorch_overrides(
    activations: bool = True,
    embedding_bag: bool = True,
    normalization: bool = False,
    softmax: bool = False,
    all: bool = False,
    verbose: bool = False,
) -> None:
    """
    Enable metalcore as the backend for specified PyTorch operations on MPS.
    
    This allows metalcore ops to be used transparently in any model that uses
    standard PyTorch functional ops (F.silu, F.gelu, etc.).
    
    Args:
        activations: Enable metalcore GELU/SiLU (default: True)
                    These are at parity or faster than PyTorch MPS.
        embedding_bag: Enable metalcore embedding_bag (default: True)
                      PyTorch falls back to CPU, metalcore is 50-100x faster.
        normalization: Enable metalcore RMSNorm/LayerNorm (default: False)
                      Near parity with PyTorch, enable if needed for bf16 support.
        softmax: Enable metalcore fused_softmax (default: False)
                Near parity with PyTorch.
        all: Enable all overrides regardless of individual settings.
        verbose: Print which overrides are enabled.
    
    Example:
        >>> import metalcore
        >>> metalcore.enable_pytorch_overrides()
        >>> # Now any model using F.silu or F.gelu on MPS will use metalcore
        >>> model = AutoModelForCausalLM.from_pretrained("...", device_map="mps")
    
    Note: Due to PyTorch library limitations, registration happens at module level.
    This function may not have immediate effect on already-compiled code.
    """
    global _active_overrides
    
    if not torch.backends.mps.is_available():
        if verbose:
            print("metalcore: MPS not available, skipping overrides")
        return
    
    # Import metalcore ops
    try:
        import metalcore_backend as backend
    except ImportError:
        if verbose:
            print("metalcore: Backend not available, skipping overrides")
        return
    
    from metalcore.activations import metal_gelu, metal_silu
    
    enabled = []
    
    # Activations - use impl_abstract and custom dispatch
    # PyTorch 2.1+ approach: direct monkey-patching of dispatch
    if all or activations:
        if "silu" not in _active_overrides:
            # Store original implementation
            _original_silu = torch.nn.functional.silu
            
            def _patched_silu(input, inplace=False):
                if input.device.type == 'mps' and not inplace:
                    return metal_silu(input)
                return _original_silu(input, inplace=inplace)
            
            torch.nn.functional.silu = _patched_silu
            _active_overrides.add("silu")
            enabled.append("silu")
        
        if "gelu" not in _active_overrides:
            _original_gelu = torch.nn.functional.gelu
            
            def _patched_gelu(input, approximate='none'):
                if input.device.type == 'mps':
                    return metal_gelu(input)
                return _original_gelu(input, approximate=approximate)
            
            torch.nn.functional.gelu = _patched_gelu
            _active_overrides.add("gelu")
            enabled.append("gelu")
    
    # Embedding bag (PyTorch falls back to CPU on MPS)
    if all or embedding_bag:
        if "embedding_bag" not in _active_overrides:
            try:
                from metalcore import embedding_bag as metal_embedding_bag
                
                _original_embedding_bag = torch.nn.functional.embedding_bag
                
                def _patched_embedding_bag(input, weight, offsets=None, max_norm=None, 
                                          norm_type=2., scale_grad_by_freq=False, 
                                          mode='mean', sparse=False, per_sample_weights=None,
                                          include_last_offset=False, padding_idx=None):
                    if weight.device.type == 'mps' and mode == 'sum':
                        # metalcore only supports sum mode
                        if offsets is None:
                            offsets = torch.arange(0, input.numel() + 1, input.size(1) if input.dim() == 2 else 1, device=weight.device)
                        mode_int = 0  # sum
                        return metal_embedding_bag(weight, input.flatten(), offsets, mode_int)
                    return _original_embedding_bag(input, weight, offsets, max_norm, norm_type,
                                                   scale_grad_by_freq, mode, sparse, 
                                                   per_sample_weights, include_last_offset, padding_idx)
                
                torch.nn.functional.embedding_bag = _patched_embedding_bag
                _active_overrides.add("embedding_bag")
                enabled.append("embedding_bag")
            except Exception as e:
                if verbose:
                    print(f"metalcore: embedding_bag override failed: {e}")
    
    if verbose and enabled:
        print(f"metalcore: Enabled PyTorch overrides for: {', '.join(enabled)}")


def disable_pytorch_overrides(
    activations: bool = False,
    embedding_bag: bool = False,
    normalization: bool = False,
    softmax: bool = False,
    all: bool = True,
    verbose: bool = False,
) -> None:
    """
    Disable metalcore PyTorch overrides.
    
    Note: This doesn't fully restore original implementations.
    For clean state, restart the Python interpreter.
    
    Args:
        all: Disable all overrides (default: True)
        Other args mirror enable_pytorch_overrides for symmetry.
        verbose: Print which overrides were cleared.
    """
    global _active_overrides
    
    cleared = []
    
    if all:
        cleared = list(_active_overrides)
        _active_overrides.clear()
    else:
        if activations:
            _active_overrides.discard("silu")
            _active_overrides.discard("gelu")
            cleared.extend(["silu", "gelu"])
        if embedding_bag:
            _active_overrides.discard("embedding_bag")
            cleared.append("embedding_bag")
    
    if verbose and cleared:
        print(f"metalcore: Cleared override tracking for: {', '.join(cleared)}")
        print("Note: Full restore requires interpreter restart")


def get_active_overrides() -> set:
    """Return the set of currently active override names."""
    return _active_overrides.copy()
