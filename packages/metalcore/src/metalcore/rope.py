# Author: Kris Bailey
# Copyright 2026
# Email: kris@krisbailey.com
"""
RoPE (Rotary Position Embedding) - Metal-accelerated implementation

Matches HuggingFace/Liger implementation using split-half format:
- x1 = x[..., :D/2], x2 = x[..., D/2:]
- out[..., :D/2] = x1 * cos - x2 * sin
- out[..., D/2:] = x2 * cos + x1 * sin
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

try:
    from . import metalcore_backend
    _HAS_METAL = True
except ImportError:
    _HAS_METAL = False


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor, 
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Uses Metal-accelerated kernel when available (5-10x faster than Python).
    
    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: Key tensor [batch, seq_len, num_kv_heads, head_dim]  
        cos: Cosine values [seq_len, head_dim/2]
        sin: Sine values [seq_len, head_dim/2]
        
    Returns:
        Tuple of (q_rotated, k_rotated)
    """
    if not _HAS_METAL or q.device.type != 'mps':
        return _apply_rotary_pos_emb_python(q, k, cos, sin)
    
    try:
        return metalcore_backend.rope_fwd_qk(q, k, cos, sin)
    except Exception:
        return _apply_rotary_pos_emb_python(q, k, cos, sin)


def apply_rotary_pos_emb_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embedding to a single tensor (q or k).
    
    Args:
        x: Input tensor [batch, seq_len, num_heads, head_dim]
        cos: Cosine values [seq_len, head_dim/2]
        sin: Sine values [seq_len, head_dim/2]
        
    Returns:
        Rotated tensor
    """
    if not _HAS_METAL or x.device.type != 'mps':
        return _rotate_half_python(x, cos, sin)
    
    try:
        return metalcore_backend.rope_fwd(x, cos, sin)
    except Exception:
        return _rotate_half_python(x, cos, sin)


def _rotate_half_python(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Python fallback for single tensor rotation."""
    head_dim = x.shape[-1]
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    
    # Expand cos/sin to match x dimensions
    # x: [B, S, H, D/2], cos/sin: [S, D/2] -> [1, S, 1, D/2]
    if cos.dim() == 2 and x.dim() == 4:
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, S, 1, D/2]
        sin = sin.unsqueeze(0).unsqueeze(2)
    
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _apply_rotary_pos_emb_python(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Python fallback for Q+K rotation."""
    q_rot = _rotate_half_python(q, cos, sin)
    k_rot = _rotate_half_python(k, cos, sin)
    return q_rot, k_rot


class RoPEFunction(torch.autograd.Function):
    """Autograd function for RoPE with Metal-accelerated backward pass."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        ctx.save_for_backward(cos, sin)
        return apply_rotary_pos_emb_single(x, cos, sin)
    
    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        
        if not _HAS_METAL or grad_output.device.type != 'mps':
            # Python fallback backward
            head_dim = grad_output.shape[-1]
            dy1 = grad_output[..., :head_dim // 2]
            dy2 = grad_output[..., head_dim // 2:]
            dx1 = dy1 * cos + dy2 * sin
            dx2 = -dy1 * sin + dy2 * cos
            return torch.cat([dx1, dx2], dim=-1), None, None
        
        try:
            grad_x = metalcore_backend.rope_bwd(grad_output, cos, sin)
            return grad_x, None, None
        except Exception:
            head_dim = grad_output.shape[-1]
            dy1 = grad_output[..., :head_dim // 2]
            dy2 = grad_output[..., head_dim // 2:]
            dx1 = dy1 * cos + dy2 * sin
            dx2 = -dy1 * sin + dy2 * cos
            return torch.cat([dx1, dx2], dim=-1), None, None


def rope_forward(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    RoPE forward with autograd support.
    
    Use this if you need gradients through the rotation.
    """
    return RoPEFunction.apply(x, cos, sin)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding module.
    
    Pre-computes cos/sin for positions and applies RoPE to inputs.
    Drop-in replacement for HuggingFace's RotaryEmbedding.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Pre-compute cos/sin for common sequence lengths
        self._cos_cached = None
        self._sin_cached = None
        self._cached_seq_len = 0
    
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cos/sin values if needed."""
        if seq_len > self._cached_seq_len or self._cos_cached is None:
            self._cached_seq_len = max(seq_len, self.max_position_embeddings)
            
            t = torch.arange(self._cached_seq_len, device=device, dtype=torch.float32)
            t = t / self.scaling_factor
            
            freqs = torch.outer(t, self.inv_freq.to(device))
            
            self._cos_cached = freqs.cos().to(dtype)
            self._sin_cached = freqs.sin().to(dtype)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.
        
        Args:
            q: [batch, seq_len, num_heads, head_dim]
            k: [batch, seq_len, num_kv_heads, head_dim]
            position_ids: Optional position indices [batch, seq_len]
            
        Returns:
            (q_rotated, k_rotated)
        """
        seq_len = q.shape[1]
        self._update_cos_sin_cache(seq_len, q.device, q.dtype)
        
        if position_ids is not None:
            # Use position_ids to index into cached values
            cos = self._cos_cached[position_ids]  # [batch, seq_len, dim/2]
            sin = self._sin_cached[position_ids]
        else:
            cos = self._cos_cached[:seq_len]
            sin = self._sin_cached[:seq_len]
        
        return apply_rotary_pos_emb(q, k, cos, sin)
