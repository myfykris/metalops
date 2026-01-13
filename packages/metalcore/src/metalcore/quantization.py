"""
Author: Kris Bailey
Copyright 2026
Email: kris@krisbailey.com

Quantized Operations - INT8/INT4 quantization for memory-efficient LLM inference

Enables running larger models on Apple Silicon by storing weights in low-precision:
- INT8: 2x compression (7B model: 14GB -> 7GB)
- INT4: 4x compression (7B model: 14GB -> 3.5GB)

Uses on-the-fly dequantization during matmul for memory efficiency.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

try:
    from . import metalcore_backend
    _HAS_METAL = True
except ImportError:
    _HAS_METAL = False


def quantize_int4(
    weight: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a weight tensor to INT4 (4-bit) with per-group scale and zero-point.
    
    Args:
        weight: Weight tensor [K, N] to quantize
        group_size: Number of elements per quantization group (default: 128)
        
    Returns:
        Tuple of (weight_packed, scales, zeros):
        - weight_packed: [K/2, N] uint8 tensor (2 x 4-bit values per byte)
        - scales: [num_groups, N] float tensor
        - zeros: [num_groups, N] float tensor
        
    Example:
        >>> W = torch.randn(4096, 4096)
        >>> W_q, scales, zeros = quantize_int4(W)
        >>> # W_q is now 4x smaller than W
    """
    K, N = weight.shape
    num_groups = (K + group_size - 1) // group_size
    
    # Initialize outputs
    device = weight.device
    dtype = weight.dtype
    
    # Layout: [K/2, N] - standard row-major packing
    weight_packed = torch.zeros((K // 2, N), dtype=torch.uint8, device=device)
    scales = torch.zeros((num_groups, N), dtype=dtype, device=device)
    zeros = torch.zeros((num_groups, N), dtype=dtype, device=device)
    
    # Process each group
    for g in range(num_groups):
        k_start = g * group_size
        k_end = min(k_start + group_size, K)
        
        # Get group slice
        group = weight[k_start:k_end, :]  # [group_size, N]
        
        # Compute min/max per column
        min_val = group.min(dim=0).values  # [N]
        max_val = group.max(dim=0).values  # [N]
        
        # Compute scale and zero
        # INT4: 0 to 15, we use signed offset of -8
        scale = (max_val - min_val) / 15.0
        scale = scale.clamp(min=1e-8)  # Avoid division by zero
        zero = -min_val / scale  # Offset so min maps to 0
        
        scales[g] = scale
        zeros[g] = zero
        
        # Quantize
        for k in range(k_start, k_end):
            q = ((weight[k, :] / scale + zero).round().clamp(0, 15)).to(torch.uint8)
            
            # Pack into bytes (2 values per byte)
            byte_idx = k // 2
            if k % 2 == 0:
                weight_packed[byte_idx, :] = q  # Lower 4 bits
            else:
                weight_packed[byte_idx, :] |= (q << 4)  # Upper 4 bits
    
    return weight_packed, scales, zeros


def dequantize_int4(
    W_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Dequantize INT4 packed weights to floating point.
    
    This is the inverse of quantize_int4. Use this at model load time
    to convert INT4 weights to FP16 for fast MPS matmul.
    
    The hybrid approach:
    - Store on disk as INT4 (7x smaller than FP32)
    - Dequant once at load time
    - Use fast MPS matmul for inference (0.6ms vs 300ms)
    
    Args:
        W_packed: Packed INT4 weights [K/2, N]
        scales: Per-group scales [num_groups, N]
        zeros: Per-group zeros [num_groups, N]
        group_size: Quantization group size
        dtype: Output dtype (default: torch.float16 for fast matmul)
    
    Returns:
        Dequantized weights [K, N]
        
    Example:
        >>> W = torch.randn(4096, 4096)
        >>> W_packed, scales, zeros = quantize_int4(W)
        >>> W_recovered = dequantize_int4(W_packed, scales, zeros)
        >>> # W_recovered is FP16, use with standard matmul
    """
    K_half, N = W_packed.shape
    K = K_half * 2
    
    if dtype is None:
        dtype = torch.float16  # FP16 for fast MPS matmul
    
    device = W_packed.device
    W = torch.zeros((K, N), dtype=dtype, device=device)
    
    for k in range(K):
        byte_idx = k // 2
        packed = W_packed[byte_idx, :]
        
        if k % 2 == 0:
            w_q = (packed & 0x0F).to(torch.int32)
        else:
            w_q = (packed >> 4).to(torch.int32)
        
        group_idx = k // group_size
        scale = scales[group_idx, :]
        zero = zeros[group_idx, :]
        
        W[k, :] = ((w_q.float() - zero) * scale).to(dtype)
    
    return W


def matmul_int4(
    X: torch.Tensor,
    W_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Matrix multiplication with INT4 quantized weights.
    
    Y = X @ dequant(W_packed, scales, zeros)
    
    Args:
        X: Input activation [M, K] or [B, M, K]
        W_packed: Quantized weights [K/2, N] (packed INT4)
        scales: Per-group scales [num_groups, N]
        zeros: Per-group zero-points [num_groups, N]
        group_size: Quantization group size (default: 128)
        
    Returns:
        Y: Output [M, N] or [B, M, N]
    """
    # Handle batched input
    batched = X.dim() == 3
    if batched:
        B, M, K = X.shape
        X_flat = X.view(B * M, K)
    else:
        M, K = X.shape
        X_flat = X
    
    N = W_packed.shape[1]
    
    # For now, use Python fallback (dequantize and matmul)
    # TODO: Use Metal kernel when available
    if X.device.type == 'mps' and _HAS_METAL:
        try:
            # Metal kernel path
            Y = metalcore_backend.matmul_int4(X_flat, W_packed, scales, zeros, group_size)
        except (AttributeError, RuntimeError):
            # Fallback to Python
            Y = _matmul_int4_python(X_flat, W_packed, scales, zeros, group_size)
    else:
        Y = _matmul_int4_python(X_flat, W_packed, scales, zeros, group_size)
    
    if batched:
        Y = Y.view(B, M, N)
    
    return Y


def _matmul_int4_python(
    X: torch.Tensor,
    W_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Python fallback for INT4 matmul - dequantize then matmul."""
    M, K = X.shape
    N = W_packed.shape[1]
    
    # Dequantize weights
    W = torch.zeros((K, N), dtype=X.dtype, device=X.device)
    
    for k in range(K):
        byte_idx = k // 2
        packed = W_packed[byte_idx, :]
        
        if k % 2 == 0:
            w_q = (packed & 0x0F).to(torch.int32)  # [0,15]
        else:
            w_q = (packed >> 4).to(torch.int32)    # [0,15]
        
        group_idx = k // group_size
        scale = scales[group_idx, :]
        zero = zeros[group_idx, :]
        
        # Dequant formula: W = (q - zero) * scale
        # zero already accounts for the offset
        W[k, :] = (w_q.float() - zero) * scale
    
    # Standard matmul
    return X @ W


def matmul_int8(
    X: torch.Tensor,
    W_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Matrix multiplication with INT8 quantized weights.
    
    Y = X @ dequant(W_packed, scales, zeros)
    
    Uses Metal kernel for on-the-fly dequantization.
    
    Args:
        X: Input activation [M, K]
        W_packed: Quantized weights [K, N] in int8
        scales: Per-group scales [num_groups, N]
        zeros: Per-group zero-points [num_groups, N]
        group_size: Quantization group size (default: 128)
        
    Returns:
        Y: Output [M, N]
    """
    if X.device.type == 'mps' and _HAS_METAL:
        try:
            return metalcore_backend.matmul_int8(X.contiguous(), W_packed.contiguous(),
                                                  scales.contiguous(), zeros.contiguous(),
                                                  group_size)
        except (AttributeError, RuntimeError):
            pass
    
    # Python fallback - dequantize and matmul
    K, N = W_packed.shape
    num_groups = (K + group_size - 1) // group_size
    W = torch.zeros((K, N), dtype=X.dtype, device=X.device)
    
    for g in range(num_groups):
        k_start = g * group_size
        k_end = min(k_start + group_size, K)
        scale = scales[g]
        zero = zeros[g]
        W[k_start:k_end] = (W_packed[k_start:k_end].float() - zero) * scale
    
    return X @ W.to(X.dtype)


# =============================================================================
# GGML-COMPATIBLE QUANTIZATION (llama.cpp block_q4_0 format)
# =============================================================================
# Portions derived from llama.cpp (https://github.com/ggerganov/llama.cpp)
# Copyright (c) 2023-2024 The ggml authors
# Licensed under MIT License
#
# block_q4_0 format:
# - 32 values per block (QK4_0 = 32)
# - Each block: 2 bytes scale (half) + 16 bytes packed (32 nibbles)
# - Total: 18 bytes per 32 values = 4.5 bits/weight
# - Zero point is implicit at 8 (values centered at 8)
# - Dequant formula: W = d * (q - 8); where d = scale, q = [0,15]

QK4_0 = 32  # Block size for q4_0 format
BLOCK_Q4_0_SIZE = 2 + QK4_0 // 2  # 18 bytes: 2 for scale + 16 for packed


def quantize_ggml_q4_0(
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Quantize weights to GGML block_q4_0 format (llama.cpp compatible).
    
    Args:
        weight: Weight tensor [K, N] to quantize
        
    Returns:
        blocks: uint8 tensor [num_blocks, 18] where each block is:
                - bytes[0:2]: scale as float16 (little-endian)
                - bytes[2:18]: 16 bytes of packed INT4 (32 values)
        
    The output can be used directly with llama.cpp-style kernels.
    
    Example:
        >>> W = torch.randn(4096, 4096)
        >>> W_ggml = quantize_ggml_q4_0(W)
        >>> # W_ggml is llama.cpp compatible
    """
    K, N = weight.shape
    assert K % QK4_0 == 0, f"K ({K}) must be divisible by {QK4_0}"
    
    num_blocks = K // QK4_0
    device = weight.device
    
    # Reshape to [num_blocks, 32, N] for vectorized processing
    weight_blocks = weight.view(num_blocks, QK4_0, N)
    
    # Compute max absolute value per block: [num_blocks, N]
    amax = weight_blocks.abs().amax(dim=1)
    
    # Compute scale (d = amax / 7), avoid div by zero
    d = amax / 7.0
    d = torch.where(d > 0, d, torch.ones_like(d))  # [num_blocks, N]
    
    # Quantize: q = round(x/d + 8), clamped to [0, 15]
    # Expand d to [num_blocks, 32, N] for broadcasting
    d_expanded = d.unsqueeze(1)  # [num_blocks, 1, N]
    q = ((weight_blocks / d_expanded) + 8.0).round().clamp(0, 15).to(torch.uint8)
    # q is [num_blocks, 32, N]
    
    # Prepare output: [num_blocks, N, 18]
    blocks = torch.zeros((num_blocks, N, BLOCK_Q4_0_SIZE), dtype=torch.uint8, device=device)
    
    # Store scales as float16 bytes
    # d is [num_blocks, N], convert to half and view as bytes
    d_half = d.to(torch.float16)  # [num_blocks, N]
    d_bytes = d_half.view(torch.uint8).view(num_blocks, N, 2)  # [num_blocks, N, 2]
    blocks[:, :, 0:2] = d_bytes
    
    # Pack 32 nibbles into 16 bytes
    # q is [num_blocks, 32, N], reorder to [num_blocks, N, 32] then pack
    q_t = q.permute(0, 2, 1)  # [num_blocks, N, 32]
    
    # Split into lo and hi nibbles
    q_lo = q_t[:, :, 0::2]  # [num_blocks, N, 16] - even indices
    q_hi = q_t[:, :, 1::2]  # [num_blocks, N, 16] - odd indices
    
    # Pack: lo | (hi << 4)
    packed = q_lo | (q_hi << 4)  # [num_blocks, N, 16]
    blocks[:, :, 2:18] = packed
    
    return blocks


def dequantize_ggml_q4_0(
    blocks: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Dequantize GGML block_q4_0 format back to floating point.
    
    Args:
        blocks: [num_blocks_per_col, N, 18] uint8 tensor from quantize_ggml_q4_0
        dtype: Output dtype (default: float16)
        
    Returns:
        Weight tensor [K, N] where K = num_blocks_per_col * 32
    """
    if dtype is None:
        dtype = torch.float16
        
    num_blocks, N, _ = blocks.shape
    K = num_blocks * QK4_0
    device = blocks.device
    
    # Extract scales: bytes 0:2 as float16
    d_bytes = blocks[:, :, 0:2].contiguous()  # [num_blocks, N, 2]
    d = d_bytes.view(torch.float16).view(num_blocks, N)  # [num_blocks, N]
    
    # Extract packed data: bytes 2:18
    packed = blocks[:, :, 2:18]  # [num_blocks, N, 16]
    
    # Unpack nibbles
    q_lo = (packed & 0x0F).to(torch.int8)  # [num_blocks, N, 16]
    q_hi = (packed >> 4).to(torch.int8)     # [num_blocks, N, 16]
    
    # Interleave: create [num_blocks, N, 32]
    q = torch.zeros((num_blocks, N, QK4_0), dtype=torch.int8, device=device)
    q[:, :, 0::2] = q_lo
    q[:, :, 1::2] = q_hi
    
    # Dequantize: W = (q - 8) * d
    # q is [num_blocks, N, 32], d is [num_blocks, N]
    d_expanded = d.unsqueeze(2)  # [num_blocks, N, 1]
    W_blocks = (q.float() - 8.0) * d_expanded.float()  # [num_blocks, N, 32]
    
    # Reshape to [K, N]
    W = W_blocks.permute(0, 2, 1).reshape(K, N).to(dtype)
    
    return W


def matmul_ggml_q4_0(
    X: torch.Tensor,
    W_blocks: torch.Tensor,
) -> torch.Tensor:
    """
    Matrix multiplication with GGML block_q4_0 quantized weights.
    
    Y = X @ dequant(W_blocks)
    
    Args:
        X: Input activation [M, K]
        W_blocks: Quantized weights [num_blocks_per_col, N, 18] from quantize_ggml_q4_0
        
    Returns:
        Y: Output [M, N]
        
    Note: Uses optimized Metal kernel on MPS, Python fallback otherwise.
    """
    num_blocks_per_col, N, _ = W_blocks.shape
    K = num_blocks_per_col * QK4_0
    
    assert X.shape[1] == K, f"X K dim ({X.shape[1]}) must match weight K ({K})"
    
    # Use Metal kernel on MPS
    if X.device.type == 'mps' and _HAS_METAL:
        try:
            return metalcore_backend.matmul_ggml_q4_0(X.half(), W_blocks)
        except (AttributeError, RuntimeError):
            pass  # Fall back to Python
    
    # Python fallback: dequantize and matmul
    W = dequantize_ggml_q4_0(W_blocks, X.dtype)
    return X @ W


class Int4Linear(nn.Module):
    """
    INT4 quantized linear layer for memory-efficient inference.
    
    Stores weights in 4-bit format (7x smaller than fp32).
    
    Two modes:
    - dequant_on_load=False: Keeps INT4 weights, dequantizes on-the-fly (slow but memory efficient)
    - dequant_on_load=True: Dequants to FP16 at init for fast MPS matmul (recommended)
    
    The hybrid approach: Store on disk as INT4 (7x compression), dequant at load time.
    
    Example:
        >>> linear = nn.Linear(4096, 4096)
        >>> q_linear = Int4Linear.from_float(linear, dequant_on_load=True)
        >>> # Uses fast FP16 matmul (0.6ms) instead of slow INT4 kernel (300ms)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        dequant_on_load: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self._dequant_on_load = dequant_on_load
        
        # Always store packed weights for serialization
        self.register_buffer(
            "weight_packed",
            torch.zeros((in_features // 2, out_features), dtype=torch.uint8, device=device)
        )
        
        # Scales and zeros: [num_groups, out_features]
        num_groups = (in_features + group_size - 1) // group_size
        self.register_buffer(
            "scales",
            torch.ones((num_groups, out_features), dtype=dtype or torch.float32, device=device)
        )
        self.register_buffer(
            "zeros",
            torch.zeros((num_groups, out_features), dtype=dtype or torch.float32, device=device)
        )
        
        # Dequantized weights for fast matmul (optional, not saved)
        self._weight_dequant: Optional[torch.Tensor] = None
        
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, dtype=dtype or torch.float32, device=device)
            )
        else:
            self.register_parameter("bias", None)
    
    def dequantize(self, dtype: Optional[torch.dtype] = None) -> None:
        """
        Dequantize INT4 weights to FP16 for fast matmul.
        
        Call this after loading a model for fast inference.
        The dequantized weights are NOT saved when serializing.
        
        Args:
            dtype: Output dtype (default: torch.float16)
        """
        if dtype is None:
            dtype = torch.float16
        self._weight_dequant = dequantize_int4(
            self.weight_packed, self.scales, self.zeros, self.group_size, dtype
        )
        self._dequant_on_load = True
    
    def keep_quantized(self) -> None:
        """
        Discard dequantized weights and use INT4 kernel.
        
        Use this for memory-constrained scenarios where you can't
        afford to store FP16 weights (7x larger than INT4).
        """
        self._weight_dequant = None
        self._dequant_on_load = False
    
    @classmethod
    def from_float(
        cls,
        module: nn.Linear,
        group_size: int = 128,
        dequant_on_load: bool = True,
    ) -> "Int4Linear":
        """
        Create an Int4Linear from a standard nn.Linear module.
        
        Args:
            module: The nn.Linear to quantize
            group_size: Quantization group size
            dequant_on_load: If True, dequant to FP16 for fast matmul
            
        Returns:
            Int4Linear with quantized weights
        """
        q_linear = cls(
            in_features=module.in_features,
            out_features=module.out_features,
            group_size=group_size,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
            dequant_on_load=dequant_on_load,
        )
        
        # Quantize weights (Linear stores as [out, in], we need [in, out] = transposed)
        weight = module.weight.data.T.contiguous()  # [in, out]
        W_packed, scales, zeros = quantize_int4(weight, group_size)
        
        q_linear.weight_packed.copy_(W_packed)
        q_linear.scales.copy_(scales)
        q_linear.zeros.copy_(zeros)
        
        if module.bias is not None:
            q_linear.bias.copy_(module.bias.data)
        
        # Dequant for fast matmul if requested
        if dequant_on_load:
            q_linear.dequantize()
        
        return q_linear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with INT4 dequantization or fast FP16 matmul."""
        if self._weight_dequant is not None:
            # Fast path: use dequantized FP16 weights with MPS matmul
            x_half = x.half() if x.dtype != torch.float16 else x
            y = x_half @ self._weight_dequant
            y = y.to(x.dtype)
        else:
            # Slow path: INT4 kernel with on-the-fly dequant
            y = matmul_int4(x, self.weight_packed, self.scales, self.zeros, self.group_size)
        
        if self.bias is not None:
            y = y + self.bias
        
        return y
    
    def extra_repr(self) -> str:
        mode = "fast_fp16" if self._weight_dequant is not None else "int4_kernel"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, bias={self.bias is not None}, mode={mode}"
        )

