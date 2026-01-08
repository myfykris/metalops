"""
Test forward pass integration with PyTorch ops.

This test verifies that metalcore ops can be used in the same forward pass
as PyTorch ops without causing Metal command buffer collisions.
"""

import pytest
import torch


def test_mixed_forward_pass_silu():
    """Test metalcore SiLU works mid-forward-pass with PyTorch matmul."""
    try:
        from metalcore import metal_silu
    except ImportError:
        pytest.skip("metalcore not available")
    
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    
    device = torch.device('mps')
    
    # Create a mini forward pass: matmul -> silu -> matmul
    x = torch.randn(32, 512, device=device)
    w1 = torch.randn(512, 512, device=device)
    w2 = torch.randn(512, 512, device=device)
    
    # This should NOT cause "command encoder already encoding" error
    x = torch.matmul(x, w1)  # PyTorch op
    x = metal_silu(x)         # metalcore op
    x = torch.matmul(x, w2)  # PyTorch op
    
    torch.mps.synchronize()
    
    assert x.shape == (32, 512)
    assert not torch.isnan(x).any()


def test_mixed_forward_pass_gelu():
    """Test metalcore GELU works mid-forward-pass with PyTorch matmul."""
    try:
        from metalcore import metal_gelu
    except ImportError:
        pytest.skip("metalcore not available")
    
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    
    device = torch.device('mps')
    
    x = torch.randn(32, 512, device=device)
    w1 = torch.randn(512, 512, device=device)
    w2 = torch.randn(512, 512, device=device)
    
    x = torch.matmul(x, w1)
    x = metal_gelu(x)
    x = torch.matmul(x, w2)
    
    torch.mps.synchronize()
    
    assert x.shape == (32, 512)
    assert not torch.isnan(x).any()


def test_mixed_forward_pass_rmsnorm():
    """Test metalcore RMSNorm works mid-forward-pass."""
    try:
        from metalcore import MetalRMSNorm
    except ImportError:
        pytest.skip("metalcore not available")
    
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    
    device = torch.device('mps')
    
    norm = MetalRMSNorm(512).to(device)
    x = torch.randn(32, 512, device=device)
    w1 = torch.randn(512, 512, device=device)
    w2 = torch.randn(512, 512, device=device)
    
    x = torch.matmul(x, w1)
    x = norm(x)
    x = torch.matmul(x, w2)
    
    torch.mps.synchronize()
    
    assert x.shape == (32, 512)
    assert not torch.isnan(x).any()


def test_chained_metalcore_ops():
    """Test multiple metalcore ops in sequence."""
    try:
        from metalcore import metal_silu, metal_gelu, MetalRMSNorm
    except ImportError:
        pytest.skip("metalcore not available")
    
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    
    device = torch.device('mps')
    
    norm = MetalRMSNorm(512).to(device)
    x = torch.randn(32, 512, device=device)
    
    # Chain: norm -> silu -> gelu -> norm
    x = norm(x)
    x = metal_silu(x)
    x = metal_gelu(x)
    x = norm(x)
    
    torch.mps.synchronize()
    
    assert x.shape == (32, 512)
    assert not torch.isnan(x).any()


def test_metalcore_with_pytorch_sdpa():
    """Test metalcore ops work seamlessly with PyTorch's SDPA.
    
    This validates command buffer integration - metalcore ops using COMMIT
    should not conflict with PyTorch's active command encoder during SDPA.
    """
    try:
        from metalcore import metal_silu, MetalRMSNorm
    except ImportError:
        pytest.skip("metalcore not available")
    
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    
    device = torch.device('mps')
    
    # Simulate a transformer block: norm -> qkv_proj -> SDPA -> silu -> norm
    batch, seq_len, hidden = 2, 64, 512
    num_heads = 8
    head_dim = hidden // num_heads
    
    x = torch.randn(batch, seq_len, hidden, device=device)
    norm = MetalRMSNorm(hidden).to(device)
    qkv_proj = torch.nn.Linear(hidden, hidden * 3, device=device)
    
    # metalcore norm
    x = norm(x)
    
    # PyTorch QKV projection
    qkv = qkv_proj(x)
    q, k, v = qkv.chunk(3, dim=-1)
    
    # Reshape for multi-head attention
    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # PyTorch's native SDPA (highly optimized on MPS)
    attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    
    # Reshape back
    attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, hidden)
    
    # metalcore SiLU
    x = x + metal_silu(attn_out)
    
    # metalcore norm again
    x = norm(x)
    
    torch.mps.synchronize()
    
    assert x.shape == (batch, seq_len, hidden)
    assert not torch.isnan(x).any()


def test_pytorch_overrides_with_sdpa():
    """Test enable_pytorch_overrides() works with native SDPA.
    
    This simulates a HuggingFace-style model where SiLU is called via F.silu
    and attention uses PyTorch's native SDPA.
    """
    try:
        import metalcore
    except ImportError:
        pytest.skip("metalcore not available")
    
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    
    device = torch.device('mps')
    
    # Enable transparent acceleration
    metalcore.enable_pytorch_overrides(activations=True, embedding_bag=False)
    
    batch, seq_len, hidden = 2, 64, 512
    num_heads = 8
    head_dim = hidden // num_heads
    
    x = torch.randn(batch, seq_len, hidden, device=device)
    
    # F.silu now uses metalcore (via override)
    x = torch.nn.functional.silu(x)
    
    # SDPA still uses PyTorch native
    q = k = v = x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, hidden)
    
    # F.gelu now uses metalcore (via override)
    x = torch.nn.functional.gelu(attn_out)
    
    torch.mps.synchronize()
    
    assert x.shape == (batch, seq_len, hidden)
    assert not torch.isnan(x).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
