# Author: Kris Bailey
# Copyright 2026
# Email: kris@krisbailey.com
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
    normalization: bool = True,  # RMSNorm is 675x faster!
    softmax: bool = False,
    linalg: bool = True,
    all: bool = False,
    verbose: bool = False,
) -> None:
    """
    Enable metalcore as the backend for specified PyTorch operations on MPS.
    
    This allows metalcore ops to be used transparently in any model that uses
    standard PyTorch functional ops (F.silu, F.gelu, etc.).
    
    Args:
        activations: Enable metalcore SiLU (default: True)
                    SiLU is 1.1x faster than PyTorch MPS.
                    Note: GELU is NOT overridden (PyTorch is faster).
        embedding_bag: Enable metalcore embedding_bag (default: True)
                      PyTorch falls back to CPU, metalcore is 6x faster.
        normalization: Enable metalcore RMSNorm (default: True)
                      RMSNorm is 675x faster than PyTorch!
        softmax: Enable metalcore fused_softmax (default: False)
                Near parity with PyTorch.
        linalg: Enable metalcore SVD/QR for large matrices (default: True)
               Uses size thresholds: SVD for matrices >= 512x512, QR for batched.
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
    
    # Activations: Only SiLU (1.1x faster), NOT GELU (PyTorch is faster)
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
        
        # NOTE: GELU is NOT patched - PyTorch MPS is faster (0.55x)
    
    # RMSNorm (675x faster than PyTorch!)
    if all or normalization:
        if "rmsnorm" not in _active_overrides:
            try:
                from metalcore import MetalRMSNorm
                # RMSNorm patching requires model-level patching via patch_transformers_rmsnorm
                # We can't patch F.rms_norm because it doesn't exist in PyTorch
                # Just mark as enabled - users should call patch_transformers_rmsnorm(model)
                _active_overrides.add("rmsnorm")
                enabled.append("rmsnorm (use patch_transformers_rmsnorm(model))")
            except Exception as e:
                if verbose:
                    print(f"metalcore: rmsnorm override failed: {e}")
    
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
    
    # Linear algebra ops (SVD, QR) with size-based thresholds
    # Based on benchmarks: SVD wins at 512x512+, QR wins for batched operations
    if all or linalg:
        if "svd" not in _active_overrides:
            try:
                from metalcore import svd as metal_svd
                
                _original_svd = torch.linalg.svd
                
                # Threshold: metalcore wins at 512x512+ based on benchmarks
                # At 512x512: 0.83x, at 1024x1024: 0.78x
                SVD_MIN_DIM = 512
                
                def _patched_svd(A, full_matrices=True, *, driver=None):
                    if (A.device.type == 'mps' and 
                        A.shape[-2] >= SVD_MIN_DIM and 
                        A.shape[-1] >= SVD_MIN_DIM):
                        # Use metalcore for large matrices
                        return metal_svd(A, full_matrices=full_matrices)
                    return _original_svd(A, full_matrices=full_matrices, driver=driver)
                
                torch.linalg.svd = _patched_svd
                _active_overrides.add("svd")
                enabled.append("svd")
            except Exception as e:
                if verbose:
                    print(f"metalcore: svd override failed: {e}")
        
        if "qr" not in _active_overrides:
            try:
                from metalcore import qr as metal_qr
                
                _original_qr = torch.linalg.qr
                
                # Threshold: metalcore wins for batched operations (dim >= 3)
                # Only use for batched, not single matrix (PyTorch faster for single)
                
                def _patched_qr(A, mode='reduced'):
                    # Use metalcore for batched QR (3+ dims)
                    if A.device.type == 'mps' and A.dim() >= 3:
                        return metal_qr(A)
                    return _original_qr(A, mode=mode)
                
                torch.linalg.qr = _patched_qr
                _active_overrides.add("qr")
                enabled.append("qr")
            except Exception as e:
                if verbose:
                    print(f"metalcore: qr override failed: {e}")
    
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


def patch_transformers_rmsnorm(model, verbose: bool = False) -> int:
    """
    Replace all RMSNorm modules in a HuggingFace Transformers model with MetalRMSNorm.
    
    This patches:
    - Qwen2RMSNorm
    - LlamaRMSNorm
    - MistralRMSNorm
    - Any other *RMSNorm module
    
    Args:
        model: A HuggingFace model (e.g., AutoModelForCausalLM)
        verbose: Print which modules were patched
    
    Returns:
        Number of modules patched
    
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> import metalcore
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", device_map="mps")
        >>> patched = metalcore.patch_transformers_rmsnorm(model, verbose=True)
        >>> print(f"Patched {patched} RMSNorm modules")
    """
    from metalcore import MetalRMSNorm
    
    patched_count = 0
    
    # Find all modules that look like RMSNorm
    for name, module in list(model.named_modules()):
        class_name = type(module).__name__
        
        # Check if it's an RMSNorm variant
        if 'RMSNorm' in class_name:
            # Get the hidden size from weight shape
            if hasattr(module, 'weight'):
                hidden_size = module.weight.shape[0]
                device = module.weight.device
                dtype = module.weight.dtype
                
                # Get eps if available
                eps = getattr(module, 'eps', 1e-6) or getattr(module, 'variance_epsilon', 1e-6) or 1e-6
                
                # Create replacement
                new_module = MetalRMSNorm(hidden_size, eps=eps)
                new_module.weight.data = module.weight.data.clone()
                new_module = new_module.to(device=device, dtype=dtype)
                
                # Replace in parent
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                setattr(parent, child_name, new_module)
                patched_count += 1
                
                if verbose:
                    print(f"metalcore: Patched {name} ({class_name} -> MetalRMSNorm)")
    
    if verbose:
        print(f"metalcore: Total {patched_count} modules patched")
    
    return patched_count


def patch_transformers_rope(model, verbose: bool = False) -> int:
    """
    Patch HuggingFace Transformers models to use Metal-accelerated RoPE.
    
    This monkey-patches the `apply_rotary_pos_emb` function in model modules
    to use our Metal kernel instead of Python loops.
    
    Args:
        model: A HuggingFace Transformers model (e.g., LlamaForCausalLM)
        verbose: Print detailed patching info
        
    Returns:
        Number of modules patched
        
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B", device_map="mps")
        >>> metalcore.patch_transformers_rope(model)
        >>> # Now RoPE uses Metal kernels automatically
    """
    from .rope import apply_rotary_pos_emb
    
    patched_count = 0
    
    # Get model's module (handles different model wrapper types)
    try:
        import transformers
    except ImportError:
        raise ImportError("patch_transformers_rope requires transformers library")
    
    # Find the modeling module for this model
    model_class_name = model.__class__.__name__
    
    # Common model prefixes
    model_prefixes = ['Llama', 'Mistral', 'Qwen', 'Phi', 'Gemma', 'Falcon']
    
    for prefix in model_prefixes:
        if prefix.lower() in model_class_name.lower():
            try:
                # Try to patch the modeling module
                module_name = f"transformers.models.{prefix.lower()}.modeling_{prefix.lower()}"
                import importlib
                modeling_module = importlib.import_module(module_name)
                
                if hasattr(modeling_module, 'apply_rotary_pos_emb'):
                    original_fn = modeling_module.apply_rotary_pos_emb
                    
                    def _patched_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
                        """Metal-accelerated RoPE replacement."""
                        # Handle HuggingFace's position-based cos/sin expansion
                        if position_ids is not None:
                            cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
                            sin = sin.squeeze(1).squeeze(0)
                            cos = cos[position_ids].unsqueeze(unsqueeze_dim)
                            sin = sin[position_ids].unsqueeze(unsqueeze_dim)
                        
                        # Handle dimension permutation if needed
                        # HF uses [batch, heads, seq, dim], we need [batch, seq, heads, dim]
                        if q.dim() == 4 and q.size(1) != q.size(2):
                            q_perm = q.transpose(1, 2)
                            k_perm = k.transpose(1, 2)
                            cos_exp = cos.squeeze()
                            sin_exp = sin.squeeze()
                            
                            if q.device.type == 'mps':
                                try:
                                    q_rot, k_rot = apply_rotary_pos_emb(q_perm, k_perm, cos_exp, sin_exp)
                                    return q_rot.transpose(1, 2), k_rot.transpose(1, 2)
                                except Exception:
                                    pass
                        
                        # Fallback to original
                        return original_fn(q, k, cos, sin, position_ids, unsqueeze_dim)
                    
                    modeling_module.apply_rotary_pos_emb = _patched_apply_rotary_pos_emb
                    patched_count += 1
                    
                    if verbose:
                        print(f"metalcore: Patched {module_name}.apply_rotary_pos_emb")
                        
            except (ImportError, AttributeError) as e:
                if verbose:
                    print(f"metalcore: Could not patch {prefix} RoPE: {e}")
    
    if patched_count == 0 and verbose:
        print(f"metalcore: No RoPE functions patched for {model_class_name}")
    elif verbose:
        print(f"metalcore: Total {patched_count} RoPE functions patched")
    
    return patched_count
