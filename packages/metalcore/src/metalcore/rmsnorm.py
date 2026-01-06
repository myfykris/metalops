
import torch
import torch.nn as nn
from torch.autograd import Function

try:
    import metalcore_backend
except ImportError:
    metalcore_backend = None

class RMSNormFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        if not x.is_mps or metalcore_backend is None:
            # Fallback to PyTorch native
            # torch.nn.functional.rms_norm is available in recent PyTorch (2.4+)
            # For older versions, we might need manual implementation, but let's assume recent for now
            # or use simple fallback implementation.
            if hasattr(torch.nn.functional, 'rms_norm'):
                return torch.nn.functional.rms_norm(x, x.size()[1:], weight, eps)
            else:
                # Manual fallback
                # cast to float32 for stability
                var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
                x_norm = x * torch.rsqrt(var + eps)
                return weight * x_norm.to(x.dtype)

        # Metal implementation
        # rmsnorm_fwd returns (Y, Rstd)
        # We need Rstd for backward
        y, rstd = metalcore_backend.rmsnorm_fwd(x, weight, eps)
        
        ctx.save_for_backward(x, rstd, weight)
        ctx.eps = eps
        
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, rstd, weight = ctx.saved_tensors
        if not x.is_mps or metalcore_backend is None:
             # This path shouldn't be hit if forward was MPS, but for safety:
             # Manual backward (slow)
             # Use autograd on fallback implementation
             with torch.enable_grad():
                x_ = x.detach().requires_grad_(True)
                w_ = weight.detach().requires_grad_(True)
                # re-compute forward
                var = x_.to(torch.float32).pow(2).mean(-1, keepdim=True)
                y = w_ * x_ * torch.rsqrt(var + ctx.eps)
                y.backward(grad_output)
                return x_.grad, w_.grad, None

        # Metal backward
        # Returns (dX, dW)
        dx, dw = metalcore_backend.rmsnorm_bwd(grad_output, x, rstd, weight)
        
        return dx, dw, None

class MetalRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return RMSNormFunction.apply(x, self.weight, self.eps)


def fused_add_rmsnorm(input, residual, weight, eps=1e-6):
    """Fused Add + RMSNorm: residual = input + residual; output = rmsnorm(residual)
    
    Saves one memory round-trip compared to separate operations.
    Modifies residual in-place.
    
    Args:
        input: [B, N] tensor to add to residual
        residual: [B, N] tensor, modified in-place
        weight: [N] RMSNorm weight
        eps: epsilon for numerical stability
        
    Returns:
        tuple of (normalized_output, rstd)
    """
    if not input.is_mps or metalcore_backend is None:
        # Fallback
        residual.add_(input)
        var = residual.to(torch.float32).pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        output = residual * rstd.to(residual.dtype) * weight
        return output, rstd.squeeze(-1)
    
    return metalcore_backend.fused_add_rmsnorm(input, residual, weight, eps)


class FusedAddRMSNorm(nn.Module):
    """Module wrapper for fused add + RMSNorm.
    
    Usage in transformer blocks:
        hidden = self.fused_norm(attn_output, residual)
        # residual is updated in-place
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, input, residual):
        """Returns normalized output. Updates residual in-place."""
        output, _ = fused_add_rmsnorm(input, residual, self.weight, self.eps)
        return output
