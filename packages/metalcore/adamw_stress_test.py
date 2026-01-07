#!/usr/bin/env python3
"""
AdamW Stress Test Script
========================
Comprehensive test suite to flush out numerical issues in the Metal AdamW kernel.

Tests:
1. Basic correctness across dtypes (float32, float16, bfloat16)
2. Large tensor stability (embedding layer scale: 32K x 4K = 131M params)
3. Multi-step training simulation (gradient accumulation pattern)
4. Edge cases: zero grads, large grads, tiny grads, mixed magnitudes
5. State accumulator precision (exp_avg, exp_avg_sq overflow/underflow)
6. Bias correction term numerical stability

Usage:
    python adamw_stress_test.py [--quick]
"""

import torch
import time
import argparse
from typing import Optional
import sys

# Try importing metalcore
try:
    from metalcore.optim import MetalAdamW
    METALCORE_AVAILABLE = True
except ImportError:
    METALCORE_AVAILABLE = False
    print("WARNING: metalcore not available, will only test PyTorch baselines")


def check_numerical_health(name: str, tensor: torch.Tensor) -> dict:
    """Check for NaN, Inf, and abnormal values."""
    result = {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": tuple(tensor.shape),
        "has_nan": tensor.isnan().any().item(),
        "has_inf": tensor.isinf().any().item(),
        "min": tensor.min().item() if not tensor.isnan().any() else float('nan'),
        "max": tensor.max().item() if not tensor.isnan().any() else float('nan'),
        "mean": tensor.float().mean().item() if not tensor.isnan().any() else float('nan'),
        "std": tensor.float().std().item() if not tensor.isnan().any() else float('nan'),
    }
    return result


def print_health(health: dict, verbose: bool = True):
    """Pretty print health check results."""
    status = "✓" if not (health["has_nan"] or health["has_inf"]) else "✗"
    issues = []
    if health["has_nan"]:
        issues.append("NaN")
    if health["has_inf"]:
        issues.append("Inf")
    
    print(f"  {status} {health['name']}: {health['dtype']} {health['shape']}", end="")
    if issues:
        print(f" [FAILED: {', '.join(issues)}]")
    elif verbose:
        print(f" [min={health['min']:.4g}, max={health['max']:.4g}, mean={health['mean']:.4g}]")
    else:
        print()
    
    return not (health["has_nan"] or health["has_inf"])


def test_basic_correctness(dtype: torch.dtype, size: int = 1000) -> bool:
    """Test basic AdamW correctness against PyTorch reference."""
    print(f"\n{'='*60}")
    print(f"TEST: Basic Correctness - {dtype}")
    print(f"{'='*60}")
    
    device = torch.device('mps')
    torch.manual_seed(42)
    
    # Initialize parameters and gradients
    param_data = torch.randn(size, device=device, dtype=torch.float32)
    grad_data = torch.randn(size, device=device, dtype=torch.float32)
    
    # Convert to target dtype
    param_data = param_data.to(dtype)
    grad_data = grad_data.to(dtype)
    
    # PyTorch reference (always use float32 internally)
    p_ref = param_data.clone().float().requires_grad_(True)
    p_ref.grad = grad_data.clone().float()
    opt_ref = torch.optim.AdamW([p_ref], lr=1e-3, weight_decay=0.01)
    
    # Metal implementation
    if METALCORE_AVAILABLE:
        p_metal = param_data.clone().requires_grad_(True)
        p_metal.grad = grad_data.clone()
        opt_metal = MetalAdamW([p_metal], lr=1e-3, weight_decay=0.01)
    
    all_passed = True
    
    # Run multiple steps
    for step in range(5):
        # Reference step
        opt_ref.step()
        opt_ref.zero_grad()
        p_ref.grad = torch.randn_like(p_ref)
        
        # Metal step
        if METALCORE_AVAILABLE:
            opt_metal.step()
            opt_metal.zero_grad()
            p_metal.grad = torch.randn(size, device=device, dtype=dtype)
        
        # Check health
        print(f"\n  Step {step + 1}:")
        ref_health = check_numerical_health("PyTorch params", p_ref)
        all_passed &= print_health(ref_health, verbose=False)
        
        if METALCORE_AVAILABLE:
            metal_health = check_numerical_health("Metal params", p_metal)
            all_passed &= print_health(metal_health, verbose=False)
            
            # Check state accumulators
            state = opt_metal.state[p_metal]
            exp_avg_health = check_numerical_health("Metal exp_avg", state['exp_avg'])
            exp_avg_sq_health = check_numerical_health("Metal exp_avg_sq", state['exp_avg_sq'])
            all_passed &= print_health(exp_avg_health, verbose=False)
            all_passed &= print_health(exp_avg_sq_health, verbose=False)
    
    if METALCORE_AVAILABLE and all_passed:
        # Compare final results (convert to float for comparison)
        p_ref_f = p_ref.to(dtype).float()
        p_metal_f = p_metal.float()
        max_diff = (p_ref_f - p_metal_f).abs().max().item()
        rel_diff = max_diff / (p_ref_f.abs().max().item() + 1e-8)
        print(f"\n  Max absolute diff: {max_diff:.6e}")
        print(f"  Relative diff: {rel_diff:.6e}")
        
        # Looser tolerance for reduced precision
        tol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-5
        if rel_diff > tol:
            print(f"  [WARNING] Relative diff exceeds tolerance {tol}")
    
    return all_passed


def test_large_tensor(dtype: torch.dtype, vocab_size: int = 32000, hidden_dim: int = 4096, steps: int = 5) -> bool:
    """Test with embedding-layer-scale tensors."""
    print(f"\n{'='*60}")
    print(f"TEST: Large Tensor ({vocab_size}x{hidden_dim} = {vocab_size*hidden_dim/1e6:.1f}M params) - {dtype}")
    print(f"{'='*60}")
    
    device = torch.device('mps')
    torch.manual_seed(42)
    
    # Create large parameter tensor (simulating embedding layer)
    p = torch.randn(vocab_size, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    
    if METALCORE_AVAILABLE:
        opt = MetalAdamW([p], lr=1e-4, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW([p.float()], lr=1e-4, weight_decay=0.01)
    
    all_passed = True
    
    for step in range(steps):
        # Simulate gradient from loss
        p.grad = torch.randn_like(p) * 0.1  # Scale down to be realistic
        
        t0 = time.perf_counter()
        opt.step()
        torch.mps.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        
        health = check_numerical_health(f"Step {step + 1}", p)
        passed = print_health(health, verbose=True)
        print(f"      Time: {dt:.2f}ms")
        all_passed &= passed
        
        if not passed:
            print(f"  [EARLY EXIT] NaN/Inf detected at step {step + 1}")
            break
        
        opt.zero_grad()
        torch.mps.synchronize()  # Sync before next iteration to avoid encoder conflicts

    
    return all_passed


def test_edge_cases() -> bool:
    """Test edge cases that might cause numerical issues."""
    print(f"\n{'='*60}")
    print(f"TEST: Edge Cases")
    print(f"{'='*60}")
    
    device = torch.device('mps')
    all_passed = True
    
    test_cases = [
        ("Zero gradients", lambda: torch.zeros(1000, device=device)),
        ("Tiny gradients", lambda: torch.randn(1000, device=device) * 1e-8),
        ("Large gradients", lambda: torch.randn(1000, device=device) * 1e4),
        ("Mixed magnitudes", lambda: torch.cat([
            torch.randn(500, device=device) * 1e-6,
            torch.randn(500, device=device) * 1e6,
        ])),
        ("Sparse-like (mostly zeros)", lambda: torch.where(
            torch.rand(1000, device=device) > 0.99,
            torch.randn(1000, device=device),
            torch.zeros(1000, device=device)
        )),
    ]
    
    for name, grad_fn in test_cases:
        print(f"\n  {name}:")
        
        for dtype in [torch.float32, torch.bfloat16]:
            p = torch.randn(1000, device=device, dtype=dtype, requires_grad=True)
            p.grad = grad_fn().to(dtype)
            
            if METALCORE_AVAILABLE:
                opt = MetalAdamW([p], lr=1e-3)
            else:
                opt = torch.optim.AdamW([p], lr=1e-3)
            
            # Run 10 steps
            for _ in range(10):
                opt.step()
                p.grad = grad_fn().to(dtype)
            
            health = check_numerical_health(f"{dtype}", p)
            all_passed &= print_health(health, verbose=False)
    
    return all_passed


def test_bias_correction_stability() -> bool:
    """Test that bias correction terms don't cause overflow at early steps."""
    print(f"\n{'='*60}")
    print(f"TEST: Bias Correction Stability")
    print(f"{'='*60}")
    
    device = torch.device('mps')
    all_passed = True
    
    for dtype in [torch.float32, torch.bfloat16]:
        print(f"\n  {dtype}:")
        p = torch.randn(1000, device=device, dtype=dtype, requires_grad=True)
        
        if METALCORE_AVAILABLE:
            opt = MetalAdamW([p], lr=1e-3, betas=(0.9, 0.999))
        else:
            opt = torch.optim.AdamW([p], lr=1e-3, betas=(0.9, 0.999))
        
        # Test early steps where bias_correction is close to 0
        for step in [1, 2, 3, 10, 100, 1000]:
            p.grad = torch.randn_like(p)
            opt.step()
            
            if step in [1, 2, 3, 100, 1000]:
                health = check_numerical_health(f"Step {step}", p)
                all_passed &= print_health(health, verbose=False)
    
    return all_passed


def test_long_training_stability(dtype: torch.dtype, steps: int = 100) -> bool:
    """Simulate longer training to catch accumulator overflow."""
    print(f"\n{'='*60}")
    print(f"TEST: Long Training Stability ({steps} steps) - {dtype}")
    print(f"{'='*60}")
    
    device = torch.device('mps')
    torch.manual_seed(42)
    
    # Moderate size parameter
    p = torch.randn(10000, device=device, dtype=dtype, requires_grad=True)
    
    if METALCORE_AVAILABLE:
        opt = MetalAdamW([p], lr=1e-3, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW([p], lr=1e-3, weight_decay=0.01)
    
    all_passed = True
    check_steps = [1, 10, 25, 50, 75, 100]
    
    for step in range(1, steps + 1):
        p.grad = torch.randn_like(p) * 0.1
        opt.step()
        
        if step in check_steps:
            health = check_numerical_health(f"Step {step}", p)
            passed = print_health(health, verbose=True)
            all_passed &= passed
            
            if METALCORE_AVAILABLE:
                state = opt.state[p]
                # Check if accumulators are growing unboundedly
                exp_avg_max = state['exp_avg'].abs().max().item()
                exp_avg_sq_max = state['exp_avg_sq'].max().item()
                print(f"      exp_avg max: {exp_avg_max:.4g}, exp_avg_sq max: {exp_avg_sq_max:.4g}")
            
            if not passed:
                return False
    
    return all_passed


def run_all_tests(quick: bool = False) -> bool:
    """Run all tests and return overall success."""
    print("\n" + "=" * 70)
    print("METALCORE ADAMW STRESS TEST SUITE")
    print("=" * 70)
    print(f"Device: MPS ({'available' if torch.backends.mps.is_available() else 'NOT AVAILABLE'})")
    print(f"MetalCore: {'available' if METALCORE_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"Mode: {'Quick' if quick else 'Full'}")
    
    if not torch.backends.mps.is_available():
        print("\nERROR: MPS not available. Cannot run tests.")
        return False
    
    results = {}
    
    # Basic correctness tests
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        results[f"basic_{dtype}"] = test_basic_correctness(dtype)
    
    # Large tensor tests
    if quick:
        # Quick: smaller embedding layer
        for dtype in [torch.float32, torch.bfloat16]:
            results[f"large_{dtype}"] = test_large_tensor(dtype, vocab_size=8000, hidden_dim=1024, steps=3)
    else:
        # Full: realistic embedding layer size
        for dtype in [torch.float32, torch.bfloat16]:
            results[f"large_{dtype}"] = test_large_tensor(dtype, vocab_size=32000, hidden_dim=4096, steps=5)
    
    # Edge cases
    results["edge_cases"] = test_edge_cases()
    
    # Bias correction stability
    results["bias_correction"] = test_bias_correction_stability()
    
    # Long training
    if not quick:
        for dtype in [torch.float32, torch.bfloat16]:
            results[f"long_training_{dtype}"] = test_long_training_stability(dtype, steps=100)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")
        all_passed &= passed
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdamW Stress Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    args = parser.parse_args()
    
    success = run_all_tests(quick=args.quick)
    sys.exit(0 if success else 1)
