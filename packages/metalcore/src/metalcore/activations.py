"""
Metal-accelerated activation functions (GELU, SiLU).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import metalcore_backend
    _HAS_METAL = True
except ImportError:
    _HAS_METAL = False


class GELUFunction(torch.autograd.Function):
    """Custom autograd function for Metal GELU."""
    
    @staticmethod
    def forward(ctx, x):
        if not _HAS_METAL or x.device.type != 'mps':
            return F.gelu(x)
        
        x_contig = x.contiguous()
        y = metalcore_backend.gelu_fwd(x_contig)
        ctx.save_for_backward(x_contig)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        
        if not _HAS_METAL or x.device.type != 'mps':
            # CPU fallback
            x_cpu = x.detach().cpu().requires_grad_(True)
            y_cpu = F.gelu(x_cpu)
            y_cpu.backward(grad_output.cpu())
            return x_cpu.grad.to(x.device)
        
        grad_output_contig = grad_output.contiguous()
        dx = metalcore_backend.gelu_bwd(grad_output_contig, x)
        return dx


class SiLUFunction(torch.autograd.Function):
    """Custom autograd function for Metal SiLU (Swish)."""
    
    @staticmethod
    def forward(ctx, x):
        if not _HAS_METAL or x.device.type != 'mps':
            return F.silu(x)
        
        x_contig = x.contiguous()
        y = metalcore_backend.silu_fwd(x_contig)
        ctx.save_for_backward(x_contig)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        
        if not _HAS_METAL or x.device.type != 'mps':
            x_cpu = x.detach().cpu().requires_grad_(True)
            y_cpu = F.silu(x_cpu)
            y_cpu.backward(grad_output.cpu())
            return x_cpu.grad.to(x.device)
        
        grad_output_contig = grad_output.contiguous()
        dx = metalcore_backend.silu_bwd(grad_output_contig, x)
        return dx


def metal_gelu(x):
    """
    Metal-accelerated GELU activation.
    
    Faster than torch.nn.functional.gelu on MPS for large tensors.
    Falls back to PyTorch GELU if Metal is not available.
    """
    return GELUFunction.apply(x)


def metal_silu(x):
    """
    Metal-accelerated SiLU (Swish) activation.
    
    Faster than torch.nn.functional.silu on MPS for large tensors.
    Falls back to PyTorch SiLU if Metal is not available.
    """
    return SiLUFunction.apply(x)


class MetalGELU(nn.Module):
    """Drop-in replacement for torch.nn.GELU using Metal kernels."""
    
    def __init__(self, approximate='none'):
        super().__init__()
        self.approximate = approximate
        if approximate != 'none':
            # Fallback for 'tanh' approximation mode
            self._fallback = nn.GELU(approximate=approximate)
        else:
            self._fallback = None
    
    def forward(self, x):
        if self._fallback is not None:
            return self._fallback(x)
        return metal_gelu(x)


class MetalSiLU(nn.Module):
    """Drop-in replacement for torch.nn.SiLU using Metal kernels."""
    
    def forward(self, x):
        return metal_silu(x)
