"""
Metal-accelerated high-performance operations:
- Fused Softmax (online algorithm)
- LayerNorm (Welford's algorithm)
- Embedding Bag (coalesced reads)
- Scatter/Gather (atomic ops)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    import metalcore_backend
    _HAS_METAL = True
except ImportError:
    _HAS_METAL = False


# =============================================================================
# Fused Softmax
# =============================================================================

def fused_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Metal-accelerated fused softmax using online algorithm.
    
    Single-pass numerical stability with SIMD reductions.
    Falls back to PyTorch if Metal not available.
    
    Args:
        x: Input tensor
        dim: Dimension to softmax over (default: -1)
    
    Returns:
        Softmax output
    """
    if not _HAS_METAL or x.device.type != 'mps':
        return F.softmax(x, dim=dim)
    return metalcore_backend.fused_softmax(x.contiguous(), dim)


class MetalSoftmax(nn.Module):
    """Drop-in replacement for nn.Softmax using Metal kernels."""
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_softmax(x, self.dim)


# =============================================================================
# LayerNorm
# =============================================================================

class LayerNormFunction(torch.autograd.Function):
    """Custom autograd function for Metal LayerNorm."""
    
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        if not _HAS_METAL or x.device.type != 'mps':
            return F.layer_norm(x, [x.size(-1)], weight, bias, eps)
        
        output, mean, rstd = metalcore_backend.layernorm_fwd(
            x.contiguous(), weight.contiguous(), bias.contiguous(), eps
        )
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.eps = eps
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, mean, rstd = ctx.saved_tensors
        # Fallback to PyTorch backward for now
        x_cpu = x.detach().cpu().requires_grad_(True)
        w_cpu = weight.detach().cpu().requires_grad_(True)
        b_cpu = torch.zeros_like(weight).cpu().requires_grad_(True)
        
        y_cpu = F.layer_norm(x_cpu, [x_cpu.size(-1)], w_cpu, b_cpu, ctx.eps)
        y_cpu.backward(grad_output.cpu())
        
        return x_cpu.grad.to(x.device), w_cpu.grad.to(weight.device), b_cpu.grad.to(weight.device), None


def layer_norm(x: torch.Tensor, normalized_shape: list, weight: torch.Tensor, 
               bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Metal-accelerated Layer Normalization.
    
    Uses Welford's algorithm for fused mean/variance computation.
    
    Args:
        x: Input tensor
        normalized_shape: Shape to normalize over (last N dims)
        weight: Learnable weight (gamma)
        bias: Learnable bias (beta)
        eps: Small constant for numerical stability
    
    Returns:
        Normalized output
    """
    if not _HAS_METAL or x.device.type != 'mps':
        return F.layer_norm(x, normalized_shape, weight, bias, eps)
    return LayerNormFunction.apply(x, weight, bias, eps)


class MetalLayerNorm(nn.Module):
    """Drop-in replacement for nn.LayerNorm using Metal kernels."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5, 
                 elementwise_affine: bool = True, device=None, dtype=None):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.elementwise_affine:
            return layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return F.layer_norm(x, self.normalized_shape, None, None, self.eps)


# =============================================================================
# Embedding Bag
# =============================================================================

def embedding_bag(weight: torch.Tensor, indices: torch.Tensor, 
                  offsets: torch.Tensor, mode: str = 'sum') -> torch.Tensor:
    """
    Metal-accelerated Embedding Bag operation.
    
    Efficiently looks up and aggregates embeddings.
    
    Args:
        weight: Embedding table [num_embeddings, embedding_dim]
        indices: Indices to look up [total_indices]
        offsets: Start offsets for each bag [batch_size + 1]
        mode: Aggregation mode: 'sum', 'mean', or 'max'
    
    Returns:
        Aggregated embeddings [batch_size, embedding_dim]
    """
    mode_map = {'sum': 0, 'mean': 1, 'max': 2}
    mode_int = mode_map.get(mode, 0)
    
    if not _HAS_METAL or weight.device.type != 'mps':
        result, _, _, _ = torch.embedding_bag(weight, indices, offsets, False, mode_int)
        return result
    
    return metalcore_backend.embedding_bag(weight, indices, offsets, mode_int)


class MetalEmbeddingBag(nn.Module):
    """Drop-in replacement for nn.EmbeddingBag using Metal kernels."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 mode: str = 'sum', device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))
    
    def forward(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return embedding_bag(self.weight, indices, offsets, self.mode)


# =============================================================================
# Scatter / Gather
# =============================================================================

def gather(src: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Metal-accelerated gather operation.
    
    out[i] = src[index[i]] for 1D case.
    
    Args:
        src: Source tensor
        index: Indices to gather
        dim: Dimension to gather along
    
    Returns:
        Gathered tensor
    """
    if not _HAS_METAL or src.device.type != 'mps':
        return torch.gather(src, dim, index.to(torch.long))
    return metalcore_backend.gather(src, index, dim)


def scatter_add(dst: torch.Tensor, index: torch.Tensor, 
                src: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Metal-accelerated scatter add operation.
    
    dst[index[i]] += src[i] for 1D case (uses atomics).
    
    Args:
        dst: Destination tensor
        index: Indices to scatter to
        src: Source values to add
        dim: Dimension to scatter along
    
    Returns:
        Updated tensor
    """
    if not _HAS_METAL or dst.device.type != 'mps':
        return dst.scatter_add(dim, index.to(torch.long), src)
    return metalcore_backend.scatter_add(dst, index, src, dim)


def index_select(src: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """
    Metal-accelerated index select operation.
    
    Args:
        src: Source tensor
        dim: Dimension to index
        index: Indices to select
    
    Returns:
        Selected tensor
    """
    if not _HAS_METAL or src.device.type != 'mps':
        return torch.index_select(src, dim, index.to(torch.long))
    return metalcore_backend.index_select(src, dim, index)
