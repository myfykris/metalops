
import torch
import torch.nn as nn
import pytest
from metalcore.rmsnorm import MetalRMSNorm
from metalcore.optim import MetalAdamW

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_rmsnorm():
    device = torch.device('mps')
    torch.manual_seed(42)
    B, N = 32, 4096
    eps = 1e-5
    
    # Inputs
    x = torch.randn(B, N, device=device, dtype=torch.float32, requires_grad=True)
    
    # Metal Model
    model_metal = MetalRMSNorm(N, eps=eps).to(device)
    # Initialize weight strictly
    model_metal.weight.data.fill_(1.0)
    
    # Torch Model
    model_torch = nn.RMSNorm(N, eps=eps).to(device)
    model_torch.weight.data.fill_(1.0)
    
    # Forward
    y_metal = model_metal(x)
    y_torch = model_torch(x)
    
    # Check forward
    assert torch.allclose(y_metal, y_torch, atol=1e-5, rtol=1e-5), "RMSNorm forward mismatch"
    
    # Backward
    grad_y = torch.randn_like(y_metal)
    
    # Metal backward
    x.grad = None
    y_metal.backward(grad_y, retain_graph=True)
    grad_x_metal = x.grad.clone()
    grad_w_metal = model_metal.weight.grad.clone()
    
    # Torch backward
    x.grad = None
    model_torch.zero_grad()
    y_torch.backward(grad_y)
    grad_x_torch = x.grad.clone()
    grad_w_torch = model_torch.weight.grad.clone()
    
    # Check backward
    assert torch.allclose(grad_x_metal, grad_x_torch, atol=1e-4, rtol=1e-4), "RMSNorm grad_x mismatch"
    assert torch.allclose(grad_w_metal, grad_w_torch, atol=1e-4, rtol=1e-4), "RMSNorm grad_w mismatch"

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_adamw():
    device = torch.device('mps')
    torch.manual_seed(42)
    N = 1000
    
    # Inputs
    param_data = torch.randn(N, device=device)
    grad_data = torch.randn(N, device=device)
    
    # Metal Optimizer
    p_metal = param_data.clone().requires_grad_(True)
    p_metal.grad = grad_data.clone()
    opt_metal = MetalAdamW([p_metal], lr=1e-3, weight_decay=0.01)
    
    # Torch Optimizer
    p_torch = param_data.clone().requires_grad_(True)
    p_torch.grad = grad_data.clone()
    opt_torch = torch.optim.AdamW([p_torch], lr=1e-3, weight_decay=0.01)
    
    # Step
    opt_metal.step()
    opt_torch.step()
    
    # Check params
    assert torch.allclose(p_metal, p_torch, atol=1e-5, rtol=1e-5), "AdamW param mismatch"
    
    # Check state
    state_metal = opt_metal.state[p_metal]
    state_torch = opt_torch.state[p_torch]
    
    assert torch.allclose(state_metal['exp_avg'], state_torch['exp_avg'], atol=1e-5), "AdamW exp_avg mismatch"
    assert torch.allclose(state_metal['exp_avg_sq'], state_torch['exp_avg_sq'], atol=1e-5), "AdamW exp_avg_sq mismatch"
