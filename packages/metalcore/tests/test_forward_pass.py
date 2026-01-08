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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
