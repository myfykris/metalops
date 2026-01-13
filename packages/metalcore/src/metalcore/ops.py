# Author: Kris Bailey
# Copyright 2026
# Email: kris@krisbailey.com
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


def fused_add_layernorm(input: torch.Tensor, residual: torch.Tensor,
                         weight: torch.Tensor, bias: torch.Tensor,
                         eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused Add + LayerNorm: y = layernorm(input + residual, weight, bias)
    
    Eliminates intermediate tensor allocation for the residual add.
    Common pattern in transformer blocks.
    
    Args:
        input: Input tensor [*, hidden_size]
        residual: Residual tensor [*, hidden_size]
        weight: LayerNorm weight (gamma)
        bias: LayerNorm bias (beta)
        eps: Epsilon for numerical stability
    
    Returns:
        Tuple of (output, mean, rstd)
    """
    if not _HAS_METAL or input.device.type != 'mps':
        # Fallback to PyTorch
        x = input + residual
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        rstd = torch.rsqrt(var + eps)
        y = (x - mean) * rstd * weight + bias
        return y, mean.squeeze(-1), rstd.squeeze(-1)
    
    try:
        return metalcore_backend.fused_add_layernorm(
            input.contiguous(), residual.contiguous(),
            weight.contiguous(), bias.contiguous(), eps
        )
    except (AttributeError, RuntimeError):
        # Fallback
        x = input + residual
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        rstd = torch.rsqrt(var + eps)
        y = (x - mean) * rstd * weight + bias
        return y, mean.squeeze(-1), rstd.squeeze(-1)


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


# =============================================================================
# Meta-Fused Kernels
# =============================================================================


class FusedSwiGLUMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, residual, ln_weight, ln_eps, 
                W_gate, W_up, W_down, 
                A_gate, B_gate, A_up, B_up, A_down, B_down, 
                lora_scale, dropout_p, is_training):
        
        # Call C++ kernel which now returns intermediates
        out, dummy, x_norm, rstd, gate, up = metalcore_backend.fused_swiglu_mlp_fwd(
            hidden_states, residual, ln_weight, ln_eps,
            W_gate, W_up, W_down,
            A_gate, B_gate, A_up, B_up, A_down, B_down,
            lora_scale, dropout_p, is_training
        )
        
        if is_training:
            ctx.save_for_backward(x_norm, gate, up, W_gate, W_up, W_down, 
                                  A_gate, B_gate, A_up, B_up, A_down, B_down, 
                                  hidden_states, ln_weight, rstd)
            ctx.lora_scale = lora_scale
            
        return out, dummy

    @staticmethod
    def backward(ctx, grad_out, grad_dummy):
        if not hasattr(ctx, 'lora_scale'):
             return (None,) * 16 # Should not happen if training
             
        (x_norm, gate, up, W_gate, W_up, W_down, 
         A_gate, B_gate, A_up, B_up, A_down, B_down, 
         hidden_states, ln_weight, rstd) = ctx.saved_tensors
         
        # Call fused backward kernel
        grads = metalcore_backend.fused_mlp_bwd(
            grad_out, x_norm, gate, up,
            W_gate, W_up, W_down,
            A_gate, B_gate, A_up, B_up, A_down, B_down,
            ctx.lora_scale,
            hidden_states, ln_weight, rstd
        )
        
        # Map grads to inputs
        # d_hidden, dW_gate, dW_up, dW_down, dA_gate, dB_gate, dA_up, dB_up, dA_down, dB_down, d_rms_w
        (d_hidden, dW_gate, dW_up, dW_down, dA_gate, dB_gate, dA_up, dB_up, dA_down, dB_down, d_rms_w) = grads
        
        return (
            d_hidden,   # hidden_states
            grad_out,   # residual (since out = res + mlp, d_res = d_out)
            d_rms_w,    # ln_weight
            None,       # ln_eps
            dW_gate, dW_up, dW_down,
            dA_gate, dB_gate, dA_up, dB_up, dA_down, dB_down,
            None,       # lora_scale
            None,       # dropout_p
            None        # is_training
        )


def fused_swiglu_mlp(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_eps: float,
    W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor,
    A_gate: torch.Tensor, B_gate: torch.Tensor,
    A_up: torch.Tensor, B_up: torch.Tensor,
    A_down: torch.Tensor, B_down: torch.Tensor,
    lora_scale: float = 1.0,
    dropout_p: float = 0.0,
    is_training: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Meta-Fused SwiGLU MLP: RMSNorm -> Gate/Up -> SwiGLU -> Down -> Residual.
    
    Eliminates ~15 kernel launches per layer by fusing operations into a single command stream.
    
    Args:
        hidden_states: Input [B, L, D]
        residual: Residual connection [B, L, D] (added to output)
        ln_weight: RMSNorm gamma
        ln_eps: RMSNorm epsilon
        W_gate, W_up, W_down: Linear weights
        A_gate, B_gate, etc.: LoRA weights (pass empty tensor if no LoRA)
        lora_scale: Scaling factor for LoRA
        dropout_p: Dropout probability (currently unused in Metal path)
        is_training: Training mode flag
        
    Returns:
        Tuple (output, dummy) - dummy is for compatibility with attention signature
    """
    if not _HAS_METAL or hidden_states.device.type != 'mps':
        # Fallback to PyTorch
        # 1. RMSNorm
        h = hidden_states.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + ln_eps)
        normed = (h * ln_weight).to(hidden_states.dtype)
        
        # 2. Gate/Up
        gate = F.linear(normed, W_gate) + lora_scale * (normed @ A_gate.t() @ B_gate.t()) if A_gate.numel() > 0 else F.linear(normed, W_gate)
        up = F.linear(normed, W_up) + lora_scale * (normed @ A_up.t() @ B_up.t()) if A_up.numel() > 0 else F.linear(normed, W_up)
        
        # 3. SwiGLU
        act = F.silu(gate) * up
        
        # 4. Down
        out = F.linear(act, W_down) + lora_scale * (act @ A_down.t() @ B_down.t()) if A_down.numel() > 0 else F.linear(act, W_down)
        
        # 5. Residual
        return (residual + out), torch.empty(0)
        
    return FusedSwiGLUMLPFunction.apply(
        hidden_states, residual, ln_weight, ln_eps,
        W_gate, W_up, W_down,
        A_gate, B_gate, A_up, B_up, A_down, B_down,
        lora_scale, dropout_p, is_training
    )


def fused_attention_bwd(
    d_q: torch.Tensor, d_k: torch.Tensor, d_v: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    x_norm: torch.Tensor,
    W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor,
    A_q: torch.Tensor, B_q: torch.Tensor,
    A_k: torch.Tensor, B_k: torch.Tensor,
    A_v: torch.Tensor, B_v: torch.Tensor,
    scale: float,
    hidden_states: torch.Tensor,
    rms_weight: torch.Tensor,
    rstd: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused Backward for Attention Block: RoPE Bwd -> QKV Bwd -> RMSNorm Bwd.
    
    Args:
        d_q, d_k, d_v: Gradients from SDPA [B, S, H, D] (d_v unrotated, d_q/d_k rotated)
        cos, sin: RoPE constants
        x_norm: Input to QKV (after RMSNorm)
        W_q, W_k, ...: Weights
        hidden_states: Input to RMSNorm (before normalization)
        rms_weight: RMSNorm gamma
        rstd: Cached 1/sigma from RMSNorm forward
        
    Returns:
        Tuple of gradients: (d_hidden, dWq, dWk, dWv, dA_q, dB_q, ..., d_rms_w)
    """
    if not _HAS_METAL:
        raise NotImplementedError("Metal not available")
        
    return metalcore_backend.fused_attention_bwd(
        d_q, d_k, d_v, cos, sin, x_norm,
        W_q, W_k, W_v,
        A_q, B_q, A_k, B_k, A_v, B_v,
        scale,
        hidden_states, rms_weight, rstd
    )


class FusedAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, residual, ln_weight, ln_eps,
                W_q, W_k, W_v, 
                A_q, B_q, A_k, B_k, A_v, B_v,
                lora_scale, W_o, A_o, B_o,
                cos, sin, num_heads, num_kv_heads, dropout_p, is_causal, is_training):

        # Call C++: output, dummy, x_norm, rstd
        out, dummy, x_norm, rstd = metalcore_backend.fused_lora_attention_fwd(
            hidden_states, residual, ln_weight, ln_eps,
            W_q, W_k, W_v,
            A_q, B_q, A_k, B_k, A_v, B_v,
            lora_scale, W_o, A_o, B_o,
            cos, sin, num_heads, num_kv_heads, dropout_p, is_causal, is_training
        )
        
        ctx.save_for_backward(x_norm, W_q, W_k, W_v, A_q, B_q, A_k, B_k, A_v, B_v,
                              hidden_states, ln_weight, rstd, cos, sin)
        ctx.scale = lora_scale # Is this SDPA scale or LoRA scale? 
        # C++ kernel uses lora_scale. SDPA scale is implied 1/sqrt(head_dim).
        # fused_attention_bwd takes 'scale'. Checking signature... 
        # fused_attention_bwd docs say 'scale' (float). In C++ it's used for LoRA?
        # Let's check implementation.
        # Yes, line 6518: d_x_norm.add_(..., scale) -> LoRA scale.
        ctx.lora_scale = lora_scale
        
        return out, dummy

    @staticmethod
    def backward(ctx, grad_out, grad_dummy):
        # We need d_q, d_k, d_v.
        # CURRENT GAP: PyTorch SDPA backward computes d_q, d_k, d_v.
        # But our forward kernel fused SDPA!
        # Does our forward kernel return something to help compute SDPA grads?
        # The forward kernel called SDPA internally.
        # Unless we implement SDPA Backward MANUALLY in this function (using saved Q/K/V?), we can't get d_q/d_k/d_v.
        # This is a MAJOR problem.
        # Our fused_attention_bwd starts with d_q, d_k, d_v.
        # But where do they come from?
        # In the original plan, we assume PyTorch handles SDPA backward?
        # But if we used a fused Forward, PyTorch didn't see the SDPA call! It's buried in C++.
        # So PyTorch CANNOT compute d_q/d_k/d_v.
        
        # Solution:
        # We must invoke SDPA backward here.
        # But SDPA backward needs Q, K, V.
        # We didn't save Q, K, V (they were local in C++).
        # We MUST recompute Q, K, V here? OR save them in C++ (which we avoided).
        
        # If we recompute Q, K, V:
        # 1. We have x_norm (saved).
        # 2. We have weights (saved).
        # 3. Recompute Q, K, V.
        # 4. Invoke sdpa_bwd (or torch.nn.functional.scaled_dot_product_attention backward?).
        #    Torch doesn't expose sdpa_bwd easily.
        #    We rely on `metalcore_backend.sdpa_bwd` (if we exposed it).
        
        # I saw `m.def("sdpa_bwd", ...)` in Step 8963!
        # So we can call it.
        # So:
        # 1. Recompute Q, K, V from x_norm, W_q, etc. (using LoRA if needed).
        # 2. Apply RoPE (using cos, sin).
        # 3. Call `metalcore_backend.sdpa_bwd(grad_out?? no, grad_out is d_output of block)`.
        #    Wait. `grad_out` is passed to Output Projection backward first.
        #    d_attn_out = d_out @ W_o.T.
        #    Then `d_attn_out` is `grad` for SDPA.
        
        # So the flow is:
        # 1. `d_out` (gradient of block output).
        # 2. `d_out` -> Output Proj Backward -> `d_attn_out` (and dW_o).
        # 3. `d_attn_out` -> SDPA Backward -> `d_q`, `d_k`, `d_v`.
        # 4. `d_q/k/v` -> Fused Attn Bwd -> `d_x_norm`, `d_weights`.
        
        # This is creating the "Everything Hooked Up" logic.
        
        raise NotImplementedError("Full Attention Backward Wiring Logic Pending Recomputation of Q/K/V")
        
    
def fused_lora_attention(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_eps: float,
    W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor,
    A_q: torch.Tensor, B_q: torch.Tensor,
    A_k: torch.Tensor, B_k: torch.Tensor,
    A_v: torch.Tensor, B_v: torch.Tensor,
    lora_scale: float,
    W_o: torch.Tensor,
    A_o: torch.Tensor, B_o: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    num_heads: int, num_kv_heads: int, 
    dropout_p: float = 0.0,
    is_causal: bool = True,
    is_training: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    if not _HAS_METAL or hidden_states.device.type != 'mps':
        # Fallback needed... complex
        raise NotImplementedError("Fallback for fused_lora_attention not implemented")

    return FusedAttentionFunction.apply(
        hidden_states, residual, ln_weight, ln_eps,
        W_q, W_k, W_v,
        A_q, B_q, A_k, B_k, A_v, B_v,
        lora_scale, W_o, A_o, B_o,
        cos, sin, num_heads, num_kv_heads, dropout_p, is_causal, is_training
    )


def swiglu_bwd(d_h: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for SwiGLU.
    d_h: gradient w.r.t output (contiguous or strided)
    gate: forward input (contiguous or strided)
    up: forward input (contiguous or strided)
    """
    if not _HAS_METAL or d_h.device.type != 'mps':
        # Fallback (using autograd logic manually)
        sig = torch.sigmoid(gate)
        silu = gate * sig
        d_up = d_h * silu
        d_gate = d_h * up * (sig * (1 + gate * (1 - sig)))
        return d_gate, d_up
        
    return metalcore_backend.swiglu_bwd_strided(d_h, gate, up)


def fused_mlp_bwd(
    d_out: torch.Tensor,
    x_norm: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor,
    A_gate: torch.Tensor, B_gate: torch.Tensor,
    A_up: torch.Tensor, B_up: torch.Tensor,
    A_down: torch.Tensor, B_down: torch.Tensor,
    scale: float,
    hidden_states: torch.Tensor,
    rms_weight: torch.Tensor,
    rstd: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused Backward for MLP Block:
    DownPro Bwd -> SwiGLU Bwd -> Gate/UpProj Bwd -> RMSNorm Bwd.
    """
    if not _HAS_METAL or d_out.device.type != 'mps':
        raise NotImplementedError("Metal not available")
        
    return metalcore_backend.fused_mlp_bwd(
        d_out, x_norm, gate, up,
        W_gate, W_up, W_down,
        A_gate, B_gate, A_up, B_up, A_down, B_down,
        scale,
        hidden_states, rms_weight, rstd
    )



