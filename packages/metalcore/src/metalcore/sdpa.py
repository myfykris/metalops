"""
Metal-accelerated Scaled Dot Product Attention (SDPA).

Supports:
- Arbitrary sequence lengths (tiled Flash Attention v2 for large sequences)
- Causal masking for autoregressive models
- Both 3D (B*H, N, D) and 4D (B, H, N, D) input formats
"""
import torch
import torch.nn.functional as F
import math

try:
    import metalcore_backend
    _HAS_METAL = True
except ImportError:
    _HAS_METAL = False


def metal_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float = None,
    is_causal: bool = False
) -> torch.Tensor:
    """
    Metal-accelerated Scaled Dot Product Attention.
    
    Uses Flash Attention v2 algorithm with tiling and online softmax for
    memory-efficient attention on large sequences.
    
    Args:
        query: (B, H, N, D) or (B*H, N, D) query tensor
        key: (B, H, N, D) or (B*H, N, D) key tensor
        value: (B, H, N, D) or (B*H, N, D) value tensor
        scale: Optional scale factor (default: 1/sqrt(D))
        is_causal: If True, applies causal mask (upper triangular)
    
    Returns:
        Output tensor of same shape as query.
    
    Performance:
        - Uses Flash Attention v2 (tiled) for seq_len > 256 or is_causal=True
        - Uses optimized naive kernel for small sequences
        - Falls back to PyTorch for unsupported cases
    """
    if not _HAS_METAL or query.device.type != 'mps':
        return F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
    
    # Handle 4D input: (B, H, N, D) -> (B*H, N, D)
    original_shape = query.shape
    is_4d = query.dim() == 4
    
    if is_4d:
        B, H, N, D = query.shape
        query = query.reshape(B * H, N, D)
        key = key.reshape(B * H, N, D)
        value = value.reshape(B * H, N, D)
    else:
        N = query.size(1)
        D = query.size(2)
    
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    
    # Call Metal kernel (handles both Flash Attention v2 and naive internally)
    output = metalcore_backend.sdpa_fwd(query, key, value, scale, is_causal)
    
    if is_4d:
        output = output.reshape(original_shape)
    
    return output


class ScaledDotProductAttentionFunction(torch.autograd.Function):
    """Custom autograd function for Metal SDPA with backward pass."""
    
    @staticmethod
    def forward(ctx, query, key, value, scale, is_causal):
        if not _HAS_METAL or query.device.type != 'mps':
            # Fallback - will have standard autograd backward
            return F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
        
        # Handle 4D input: (B, H, N, D) -> (B*H, N, D)
        original_shape = query.shape
        is_4d = query.dim() == 4
        
        if is_4d:
            B, H, N, D = query.shape
            query = query.reshape(B * H, N, D)
            key = key.reshape(B * H, N, D)
            value = value.reshape(B * H, N, D)
        else:
            N = query.size(1)
            D = query.size(2)
        
        if scale is None:
            scale = 1.0 / math.sqrt(D)
        
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        
        # Forward pass returns output; logsumexp L is computed internally
        output = metalcore_backend.sdpa_fwd(query, key, value, scale, is_causal)
        
        # We need to recompute L for backward, so save inputs
        # The forward kernel stores L internally, but we need to access it
        # For now, we'll recompute logsumexp in backward
        ctx.save_for_backward(query, key, value, output)
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.is_4d = is_4d
        ctx.original_shape = original_shape
        
        if is_4d:
            output = output.reshape(original_shape)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, output = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal
        is_4d = ctx.is_4d
        original_shape = ctx.original_shape
        
        if is_4d:
            grad_output = grad_output.reshape(query.shape)
        
        grad_output = grad_output.contiguous()
        
        # Compute logsumexp for backward (needed by the kernel)
        # L = max_j(s_ij) + log(sum_j exp(s_ij - max))
        # We recompute this from Q, K
        batch_heads = query.size(0)
        seq_len = query.size(1)
        head_dim = query.size(2)
        
        # Compute attention scores to get logsumexp
        # This is O(N^2) but only for backward - could optimize later
        scores = torch.bmm(query, key.transpose(1, 2)) * scale
        if is_causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
        L = torch.logsumexp(scores, dim=-1)  # (batch_heads, seq_len)
        
        # Call Metal backward
        dQ, dK, dV = metalcore_backend.sdpa_bwd(
            query, key, value, output, grad_output, L, scale, is_causal
        )
        
        if is_4d:
            dQ = dQ.reshape(original_shape)
            dK = dK.reshape(original_shape)
            dV = dV.reshape(original_shape)
        
        return dQ, dK, dV, None, None


# Convenience function that uses the autograd function
def metal_sdpa_with_backward(query, key, value, scale=None, is_causal=False):
    """SDPA with Metal-accelerated backward pass."""
    if scale is None:
        D = query.size(-1)
        scale = 1.0 / math.sqrt(D)
    return ScaledDotProductAttentionFunction.apply(query, key, value, scale, is_causal)

