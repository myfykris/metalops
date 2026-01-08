"""
Tests specifically designed to break command buffer integration.

These tests target the "command encoder already encoding" error that occurs
when metalcore ops conflict with PyTorch's active encoder.
"""

import pytest
import torch


def skip_if_no_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")


class TestCommandBufferIntegration:
    """Tests that would crash with the old encoder pattern."""
    
    def test_metalcore_between_pytorch_matmuls(self):
        """metalcore op sandwiched between PyTorch matmuls."""
        skip_if_no_mps()
        from metalcore import metal_silu
        
        device = torch.device('mps')
        x = torch.randn(32, 512, device=device)
        w1 = torch.randn(512, 512, device=device)
        w2 = torch.randn(512, 512, device=device)
        
        x = torch.matmul(x, w1)
        x = metal_silu(x)
        x = torch.matmul(x, w2)
        torch.mps.synchronize()
        
        assert x.shape == (32, 512)
    
    def test_metalcore_with_pytorch_sdpa(self):
        """metalcore ops interleaved with PyTorch's SDPA."""
        skip_if_no_mps()
        from metalcore import metal_silu, MetalRMSNorm
        
        device = torch.device('mps')
        batch, seq_len, hidden = 2, 64, 512
        num_heads, head_dim = 8, 64
        
        x = torch.randn(batch, seq_len, hidden, device=device)
        norm = MetalRMSNorm(hidden).to(device)
        
        x = norm(x)  # metalcore
        q = k = v = x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # PyTorch
        x = metal_silu(attn_out.transpose(1, 2).reshape(batch, seq_len, hidden))  # metalcore
        x = norm(x)  # metalcore
        
        torch.mps.synchronize()
        assert x.shape == (batch, seq_len, hidden)
    
    def test_multiple_metalcore_ops_no_sync(self):
        """Multiple metalcore ops without intermediate sync."""
        skip_if_no_mps()
        from metalcore import metal_silu, metal_gelu, MetalRMSNorm
        
        device = torch.device('mps')
        x = torch.randn(16, 256, device=device)
        norm = MetalRMSNorm(256).to(device)
        
        # Chain of metalcore ops - no intermediate sync
        for _ in range(5):
            x = norm(x)
            x = metal_silu(x)
            x = metal_gelu(x)
        
        torch.mps.synchronize()
        assert x.shape == (16, 256)
    
    def test_conv_metalcore_sdpa_mix(self):
        """Conv2D -> metalcore -> SDPA mix."""
        skip_if_no_mps()
        from metalcore import metal_silu, MetalRMSNorm
        
        device = torch.device('mps')
        conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
        norm = MetalRMSNorm(64).to(device)
        
        x = torch.randn(4, 3, 32, 32, device=device)
        x = conv(x)  # PyTorch
        x = metal_silu(x)  # metalcore
        x = x.flatten(2).transpose(1, 2)  # B, HW, C
        x = norm(x)  # metalcore
        
        q = k = v = x.view(4, 1024, 4, 16).transpose(1, 2)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # PyTorch
        
        torch.mps.synchronize()
        assert attn_out.shape == (4, 4, 1024, 16)
    
    def test_bf16_mixed_ops(self):
        """bf16 precision with mixed PyTorch/metalcore ops."""
        skip_if_no_mps()
        from metalcore import metal_silu, metal_gelu, MetalRMSNorm
        
        device = torch.device('mps')
        x = torch.randn(8, 64, 256, device=device, dtype=torch.bfloat16)
        norm = MetalRMSNorm(256).to(device).to(torch.bfloat16)
        
        x = norm(x)  # metalcore
        x = metal_silu(x)  # metalcore
        x = torch.matmul(x, torch.randn(256, 256, device=device, dtype=torch.bfloat16))  # PyTorch
        x = metal_gelu(x)  # metalcore
        
        torch.mps.synchronize()
        assert x.shape == (8, 64, 256)
    
    def test_pytorch_overrides_with_sdpa(self):
        """F.silu/F.gelu overrides work with SDPA."""
        skip_if_no_mps()
        import metalcore
        metalcore.enable_pytorch_overrides(activations=True, embedding_bag=False)
        
        device = torch.device('mps')
        batch, seq_len, hidden = 2, 64, 512
        num_heads, head_dim = 8, 64
        
        x = torch.randn(batch, seq_len, hidden, device=device)
        x = torch.nn.functional.silu(x)  # Uses metalcore via override
        
        q = k = v = x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = torch.nn.functional.gelu(attn_out.transpose(1, 2).reshape(batch, seq_len, hidden))  # Uses metalcore
        
        torch.mps.synchronize()
        assert x.shape == (batch, seq_len, hidden)
    
    def test_backward_pass_mixed_ops(self):
        """Gradient flow through mixed metalcore/PyTorch ops."""
        skip_if_no_mps()
        from metalcore import metal_silu, metal_gelu
        
        device = torch.device('mps')
        x = torch.randn(8, 256, device=device, requires_grad=True)
        
        y = metal_silu(x)  # metalcore with autograd
        y = torch.matmul(y, torch.randn(256, 256, device=device))  # PyTorch
        y = metal_gelu(y)  # metalcore with autograd
        
        loss = y.sum()
        loss.backward()
        
        torch.mps.synchronize()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestEdgeCases:
    """Edge cases that might expose encoder issues."""
    
    def test_empty_tensor(self):
        """Empty tensor shouldn't crash."""
        skip_if_no_mps()
        from metalcore import metal_silu
        
        device = torch.device('mps')
        x = torch.empty(0, 256, device=device)
        # This may skip or return empty - shouldn't crash
        try:
            y = metal_silu(x)
            torch.mps.synchronize()
        except RuntimeError:
            pass  # Expected for some ops
    
    def test_very_large_tensor(self):
        """Large tensor to stress GPU memory."""
        skip_if_no_mps()
        from metalcore import metal_silu
        
        device = torch.device('mps')
        x = torch.randn(1024, 4096, device=device)
        y = metal_silu(x)
        y = torch.matmul(y, torch.randn(4096, 1024, device=device))
        
        torch.mps.synchronize()
        assert y.shape == (1024, 1024)
    
    def test_non_contiguous_tensor(self):
        """Non-contiguous tensors."""
        skip_if_no_mps()
        from metalcore import metal_silu
        
        device = torch.device('mps')
        x = torch.randn(32, 512, device=device).t()  # Non-contiguous
        y = metal_silu(x)  # Should handle contiguity internally
        
        torch.mps.synchronize()
        assert y.shape == (512, 32)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
