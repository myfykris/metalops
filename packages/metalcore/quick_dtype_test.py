#!/usr/bin/env python3
"""
Lightweight test harness for rapid dtype testing across all kernels.
Usage: python quick_dtype_test.py [--activations] [--training] [--sdpa] [--all]

Runs fast correctness checks without benchmarking.
"""
import argparse
import sys
import torch

# Skip if MPS not available
if not torch.backends.mps.is_available():
    print("MPS not available, skipping tests")
    sys.exit(0)

DTYPES = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

def check_close(name, result, expected, dtype, rtol=1e-2, atol=1e-2):
    """Check if result is close to expected, with relaxed tolerances for fp16/bf16."""
    if dtype in (torch.float16, torch.bfloat16):
        rtol, atol = 5e-2, 5e-2  # Relaxed for half precision
    
    try:
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)
        return True, None
    except AssertionError as e:
        max_diff = (result - expected).abs().max().item()
        return False, f"max_diff={max_diff:.2e}"


def test_activations():
    """Test GELU and SiLU activations across dtypes."""
    print("\n=== Activation Tests ===")
    
    try:
        from metalcore import metal_gelu, metal_silu
    except ImportError:
        print("  metalcore not available, skipping")
        return 0, 0
    
    passed, failed = 0, 0
    
    for dtype_name, dtype in DTYPES.items():
        # Generate test input
        x = torch.randn(256, 512, device='mps', dtype=dtype)
        
        # GELU forward
        try:
            y_metal = metal_gelu(x)
            # For comparison, use fp32 if bf16 (since Metal promotes bf16 to fp32)
            x_ref = x.float() if dtype == torch.bfloat16 else x
            y_torch = torch.nn.functional.gelu(x_ref, approximate='tanh')
            y_expected = y_torch.to(dtype)
            
            ok, err = check_close(f"GELU {dtype_name}", y_metal, y_expected, dtype)
            if ok:
                print(f"  ✓ GELU fwd {dtype_name}")
                passed += 1
            else:
                print(f"  ✗ GELU fwd {dtype_name}: {err}")
                failed += 1
        except Exception as e:
            print(f"  ✗ GELU fwd {dtype_name}: {e}")
            failed += 1
        
        # SiLU forward
        try:
            y_metal = metal_silu(x)
            x_ref = x.float() if dtype == torch.bfloat16 else x
            y_torch = torch.nn.functional.silu(x_ref)
            y_expected = y_torch.to(dtype)
            
            ok, err = check_close(f"SiLU {dtype_name}", y_metal, y_expected, dtype)
            if ok:
                print(f"  ✓ SiLU fwd {dtype_name}")
                passed += 1
            else:
                print(f"  ✗ SiLU fwd {dtype_name}: {err}")
                failed += 1
        except Exception as e:
            print(f"  ✗ SiLU fwd {dtype_name}: {e}")
            failed += 1
    
    return passed, failed


def test_training():
    """Test RMSNorm and AdamW across dtypes."""
    print("\n=== Training Ops Tests ===")
    
    try:
        from metalcore.rmsnorm import MetalRMSNorm
        from metalcore.optim import MetalAdamW
    except ImportError:
        print("  metalcore training ops not available, skipping")
        return 0, 0
    
    passed, failed = 0, 0
    
    for dtype_name, dtype in DTYPES.items():
        # RMSNorm forward
        try:
            x = torch.randn(16, 256, device='mps', dtype=dtype)
            model = MetalRMSNorm(256).to('mps')
            y = model(x)
            
            if y.dtype == dtype and y.shape == x.shape:
                print(f"  ✓ RMSNorm fwd {dtype_name}")
                passed += 1
            else:
                print(f"  ✗ RMSNorm fwd {dtype_name}: wrong dtype/shape")
                failed += 1
        except Exception as e:
            print(f"  ✗ RMSNorm fwd {dtype_name}: {e}")
            failed += 1
    
    # AdamW step (only fp32 typically supported for optimizer state)
    try:
        p = torch.randn(1024, device='mps', dtype=torch.float32, requires_grad=True)
        p.grad = torch.randn_like(p)
        opt = MetalAdamW([p], lr=1e-3)
        opt.step()
        print(f"  ✓ AdamW step fp32")
        passed += 1
    except Exception as e:
        print(f"  ✗ AdamW step fp32: {e}")
        failed += 1
    
    return passed, failed


def test_sdpa():
    """Test SDPA across dtypes."""
    print("\n=== SDPA Tests ===")
    
    try:
        from metalcore import metal_scaled_dot_product_attention
    except ImportError:
        print("  metalcore SDPA not available, skipping")
        return 0, 0
    
    passed, failed = 0, 0
    
    for dtype_name, dtype in DTYPES.items():
        try:
            B, H, N, D = 2, 4, 32, 32
            Q = torch.randn(B, H, N, D, device='mps', dtype=dtype)
            K = torch.randn(B, H, N, D, device='mps', dtype=dtype)
            V = torch.randn(B, H, N, D, device='mps', dtype=dtype)
            
            O = metal_scaled_dot_product_attention(Q, K, V, is_causal=False)
            
            if O.dtype == dtype and O.shape == Q.shape:
                print(f"  ✓ SDPA {dtype_name}")
                passed += 1
            else:
                print(f"  ✗ SDPA {dtype_name}: wrong dtype/shape")
                failed += 1
        except Exception as e:
            print(f"  ✗ SDPA {dtype_name}: {e}")
            failed += 1
    
    return passed, failed


def test_solve():
    """Test linear solve across various input configurations."""
    print("\n=== Linear Solve Tests ===")
    
    try:
        from metalcore import solve
    except ImportError:
        print("  metalcore solve not available, skipping")
        return 0, 0
    
    passed, failed = 0, 0
    
    # Test configurations: (batch, N, K, description)
    configs = [
        # Single matrix cases
        (None, 16, 1, "Single 16x16, 1D RHS"),
        (None, 32, 1, "Single 32x32, 1D RHS"),
        (None, 64, 4, "Single 64x64, 4 RHS"),
        # Batched cases
        (8, 16, 1, "Batch=8, 16x16"),
        (32, 32, 1, "Batch=32, 32x32"),
        (16, 48, 1, "Batch=16, 48x48"),
        (4, 64, 1, "Batch=4, 64x64"),
        (2, 128, 1, "Batch=2, 128x128"),
        # Multiple RHS
        (8, 32, 4, "Batch=8, 32x32, 4 RHS"),
    ]
    
    for batch, N, K, desc in configs:
        try:
            if batch is None:
                # Single matrix
                A = torch.randn(N, N, device='mps')
                if K == 1:
                    b = torch.randn(N, device='mps')
                else:
                    b = torch.randn(N, K, device='mps')
            else:
                # Batched
                A = torch.randn(batch, N, N, device='mps')
                if K == 1:
                    b = torch.randn(batch, N, device='mps')
                else:
                    b = torch.randn(batch, N, K, device='mps')
            
            x = solve(A.clone(), b.clone())
            
            # Verify with residual check
            if batch is None:
                if K == 1:
                    residual = (A @ x - b).abs().max().item()
                else:
                    residual = (A @ x - b).abs().max().item()
            else:
                if K == 1:
                    residual = (torch.bmm(A, x.unsqueeze(-1)).squeeze(-1) - b).abs().max().item()
                else:
                    residual = (torch.bmm(A, x) - b).abs().max().item()
            
            if residual < 1e-2:  # Relaxed for numerical tolerance
                print(f"  ✓ {desc} (residual={residual:.2e})")
                passed += 1
            else:
                print(f"  ✗ {desc}: high residual {residual:.2e}")
                failed += 1
        except Exception as e:
            print(f"  ✗ {desc}: {e}")
            failed += 1
    
    return passed, failed


def main():
    parser = argparse.ArgumentParser(description="Quick dtype test harness")
    parser.add_argument("--activations", action="store_true", help="Test activations only")
    parser.add_argument("--training", action="store_true", help="Test training ops only")
    parser.add_argument("--sdpa", action="store_true", help="Test SDPA only")
    parser.add_argument("--solve", action="store_true", help="Test solve only")
    parser.add_argument("--all", action="store_true", help="Test everything (default)")
    args = parser.parse_args()
    
    # Default to all if nothing specified
    if not (args.activations or args.training or args.sdpa or args.solve):
        args.all = True
    
    print("=" * 50)
    print("METALOPS QUICK DTYPE TEST")
    print("=" * 50)
    
    total_passed, total_failed = 0, 0
    
    if args.activations or args.all:
        p, f = test_activations()
        total_passed += p
        total_failed += f
    
    if args.training or args.all:
        p, f = test_training()
        total_passed += p
        total_failed += f
    
    if args.sdpa or args.all:
        p, f = test_sdpa()
        total_passed += p
        total_failed += f
    
    if args.solve or args.all:
        p, f = test_solve()
        total_passed += p
        total_failed += f
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {total_passed} passed, {total_failed} failed")
    print("=" * 50)
    
    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
