"""
Unit tests for Metal activation functions (GELU, SiLU).
"""
import torch
import torch.nn.functional as F
import pytest


# Skip if not on macOS with MPS
pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available"
)


def test_gelu_forward():
    """Test GELU forward against PyTorch reference (tanh approximation)."""
    from metalcore.activations import metal_gelu
    
    x = torch.randn(1024, 4096, device='mps', dtype=torch.float32)
    
    y_metal = metal_gelu(x)
    # Our Metal kernel uses the tanh approximation, match it
    y_torch = F.gelu(x, approximate='tanh')
    
    torch.testing.assert_close(y_metal, y_torch, rtol=1e-4, atol=1e-4)


def test_gelu_backward():
    """Test GELU backward gradient (tanh approximation)."""
    from metalcore.activations import metal_gelu
    
    x = torch.randn(512, 2048, device='mps', dtype=torch.float32, requires_grad=True)
    
    # Metal path
    y_metal = metal_gelu(x)
    torch.mps.synchronize()  # Ensure forward encoder is finalized
    loss_metal = y_metal.sum()
    loss_metal.backward()
    torch.mps.synchronize()
    grad_metal = x.grad.clone()
    
    x.grad = None
    
    # Torch path (tanh approximation)
    y_torch = F.gelu(x, approximate='tanh')
    loss_torch = y_torch.sum()
    loss_torch.backward()
    torch.mps.synchronize()
    grad_torch = x.grad.clone()
    
    torch.testing.assert_close(grad_metal, grad_torch, rtol=1e-3, atol=1e-3)


def test_silu_forward():
    """Test SiLU forward against PyTorch reference."""
    from metalcore.activations import metal_silu
    
    x = torch.randn(1024, 4096, device='mps', dtype=torch.float32)
    
    y_metal = metal_silu(x)
    y_torch = F.silu(x)
    
    torch.testing.assert_close(y_metal, y_torch, rtol=1e-4, atol=1e-4)


def test_silu_backward():
    """Test SiLU backward gradient."""
    from metalcore.activations import metal_silu
    
    x = torch.randn(512, 2048, device='mps', dtype=torch.float32, requires_grad=True)
    
    # Metal path
    y_metal = metal_silu(x)
    torch.mps.synchronize()
    loss_metal = y_metal.sum()
    loss_metal.backward()
    torch.mps.synchronize()
    grad_metal = x.grad.clone()
    
    x.grad = None
    
    # Torch path
    y_torch = F.silu(x)
    loss_torch = y_torch.sum()
    loss_torch.backward()
    torch.mps.synchronize()
    grad_torch = x.grad.clone()
    
    torch.testing.assert_close(grad_metal, grad_torch, rtol=1e-3, atol=1e-3)


def test_gelu_module():
    """Test MetalGELU nn.Module (tanh approximation)."""
    from metalcore.activations import MetalGELU
    
    x = torch.randn(256, 1024, device='mps', dtype=torch.float32)
    
    gelu_metal = MetalGELU()
    gelu_torch = torch.nn.GELU(approximate='tanh')
    
    y_metal = gelu_metal(x)
    y_torch = gelu_torch(x)
    
    torch.testing.assert_close(y_metal, y_torch, rtol=1e-4, atol=1e-4)


def test_silu_module():
    """Test MetalSiLU nn.Module."""
    from metalcore.activations import MetalSiLU
    
    x = torch.randn(256, 1024, device='mps', dtype=torch.float32)
    
    silu_metal = MetalSiLU()
    silu_torch = torch.nn.SiLU()
    
    y_metal = silu_metal(x)
    y_torch = silu_torch(x)
    
    torch.testing.assert_close(y_metal, y_torch, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    print("Running GELU tests...")
    test_gelu_forward()
    print("  GELU Forward: PASSED")
    test_gelu_backward()
    print("  GELU Backward: PASSED")
    
    print("Running SiLU tests...")
    test_silu_forward()
    print("  SiLU Forward: PASSED")
    test_silu_backward()
    print("  SiLU Backward: PASSED")
    
    print("Running Module tests...")
    test_gelu_module()
    print("  MetalGELU: PASSED")
    test_silu_module()
    print("  MetalSiLU: PASSED")
    
    print("\nAll activation tests PASSED!")
