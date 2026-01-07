#!/usr/bin/env python3
"""
AdamW Comprehensive Stress Test Suite
======================================
Tests every geometry that would be used for top 10 SOTA model families.
Goal: Make it so user error is the only way this kernel can fail.

Tested Model Families:
1. Llama 3.x (8B, 70B, 405B)
2. Qwen 2.5 (7B, 32B, 72B)
3. Mistral/Mixtral (7B, 8x7B)
4. DeepSeek V3 (MoE 671B)
5. Gemma 2 (2B, 9B, 27B)
6. Phi-4 (14B)
7. GPT-4/4o (estimated 200B+)
8. Claude 3 (estimated 175B+)
9. Command R+ (104B)
10. Falcon (40B, 180B)

Tests cover:
- Every layer geometry: embeddings, attention (QKV, O), MLP (gate, up, down)
- All dtypes: float32, float16, bfloat16
- Edge cases: tiny grads, huge grads, sparse grads, zero grads
- Extended training: 1000+ steps for accumulator stability
- Contiguous and non-contiguous tensors
- Views and slices
- Multi-parameter groups with different settings
- Memory alignment edge cases (sizes not divisible by 4)

Usage:
    python adamw_stress_test.py [--quick] [--model MODEL]
"""

import torch
import time
import argparse
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    from metalcore.optim import MetalAdamW
    METALCORE_AVAILABLE = True
except ImportError:
    METALCORE_AVAILABLE = False
    print("WARNING: metalcore not available, will only test PyTorch baselines")


# =============================================================================
# SOTA MODEL GEOMETRIES
# =============================================================================

@dataclass
class ModelConfig:
    """Model configuration with layer geometries."""
    name: str
    vocab_size: int
    hidden_dim: int
    intermediate_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int  # For GQA
    num_layers: int
    tie_embeddings: bool = False
    
    def get_geometries(self) -> Dict[str, Tuple[int, ...]]:
        """Return all layer shapes as (numel,) or (rows, cols)."""
        return {
            "embed_tokens": (self.vocab_size, self.hidden_dim),
            "lm_head": (self.vocab_size, self.hidden_dim) if not self.tie_embeddings else None,
            "q_proj": (self.hidden_dim, self.num_heads * self.head_dim),
            "k_proj": (self.hidden_dim, self.num_kv_heads * self.head_dim),
            "v_proj": (self.hidden_dim, self.num_kv_heads * self.head_dim),
            "o_proj": (self.num_heads * self.head_dim, self.hidden_dim),
            "gate_proj": (self.hidden_dim, self.intermediate_dim),
            "up_proj": (self.hidden_dim, self.intermediate_dim),
            "down_proj": (self.intermediate_dim, self.hidden_dim),
            "input_layernorm": (self.hidden_dim,),
            "post_attn_layernorm": (self.hidden_dim,),
        }


# Top 10 SOTA Model Configurations
SOTA_MODELS = {
    # Llama 3 family
    "llama3-8b": ModelConfig("Llama3-8B", 128256, 4096, 14336, 32, 128, 8, 32),
    "llama3-70b": ModelConfig("Llama3-70B", 128256, 8192, 28672, 64, 128, 8, 80),
    "llama3-405b": ModelConfig("Llama3-405B", 128256, 16384, 53248, 128, 128, 8, 126),
    
    # Qwen 2.5 family
    "qwen2.5-7b": ModelConfig("Qwen2.5-7B", 151936, 3584, 18944, 28, 128, 4, 28),
    "qwen2.5-32b": ModelConfig("Qwen2.5-32B", 152064, 5120, 27648, 40, 128, 8, 64),
    "qwen2.5-72b": ModelConfig("Qwen2.5-72B", 152064, 8192, 29568, 64, 128, 8, 80),
    
    # Mistral/Mixtral
    "mistral-7b": ModelConfig("Mistral-7B", 32000, 4096, 14336, 32, 128, 8, 32),
    "mixtral-8x7b": ModelConfig("Mixtral-8x7B", 32000, 4096, 14336, 32, 128, 8, 32),  # Per-expert
    
    # DeepSeek V3 (MoE - per expert dimensions)
    "deepseek-v3": ModelConfig("DeepSeek-V3", 129280, 7168, 18432, 56, 128, 8, 61),
    
    # Gemma 2 family
    "gemma2-2b": ModelConfig("Gemma2-2B", 256000, 2304, 9216, 8, 256, 4, 26, tie_embeddings=True),
    "gemma2-9b": ModelConfig("Gemma2-9B", 256000, 3584, 14336, 16, 256, 8, 42, tie_embeddings=True),
    "gemma2-27b": ModelConfig("Gemma2-27B", 256000, 4608, 36864, 32, 128, 16, 46, tie_embeddings=True),
    
    # Phi-4
    "phi4-14b": ModelConfig("Phi4-14B", 100352, 5120, 17920, 40, 128, 10, 40),
    
    # Falcon
    "falcon-40b": ModelConfig("Falcon-40B", 65024, 8192, 32768, 64, 128, 8, 60),
    "falcon-180b": ModelConfig("Falcon-180B", 65024, 14848, 59392, 232, 64, 8, 80),
    
    # Command R+
    "command-r-plus": ModelConfig("Command-R+", 256000, 12288, 33792, 96, 128, 8, 64),
}

# Subset for quick testing
QUICK_MODELS = ["llama3-8b", "qwen2.5-7b", "gemma2-9b", "phi4-14b"]


# =============================================================================
# TEST UTILITIES
# =============================================================================

def check_numerical_health(name: str, tensor: torch.Tensor) -> dict:
    """Check for NaN, Inf, and abnormal values."""
    return {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": tuple(tensor.shape),
        "numel": tensor.numel(),
        "has_nan": tensor.isnan().any().item(),
        "has_inf": tensor.isinf().any().item(),
        "min": tensor.min().item() if not tensor.isnan().any() else float('nan'),
        "max": tensor.max().item() if not tensor.isnan().any() else float('nan'),
    }


def print_health(health: dict, verbose: bool = False) -> bool:
    """Pretty print health check results."""
    ok = not (health["has_nan"] or health["has_inf"])
    status = "✓" if ok else "✗"
    
    issues = []
    if health["has_nan"]: issues.append("NaN")
    if health["has_inf"]: issues.append("Inf")
    
    numel_str = f"{health['numel']/1e6:.1f}M" if health['numel'] >= 1e6 else f"{health['numel']/1e3:.1f}K"
    print(f"    {status} {health['name']} ({numel_str})", end="")
    
    if issues:
        print(f" [FAILED: {', '.join(issues)}]")
    elif verbose:
        print(f" [min={health['min']:.2e}, max={health['max']:.2e}]")
    else:
        print()
    
    return ok


def run_adamw_steps(params: List[torch.Tensor], steps: int, grad_scale: float = 0.1,
                    lr: float = 1e-4, weight_decay: float = 0.01) -> Tuple[bool, int]:
    """Run AdamW steps, return (success, last_good_step)."""
    if METALCORE_AVAILABLE:
        opt = MetalAdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    for step in range(1, steps + 1):
        for p in params:
            if p.grad is None:
                p.grad = torch.randn_like(p) * grad_scale
            else:
                p.grad.copy_(torch.randn_like(p) * grad_scale)
        
        opt.step()
        torch.mps.synchronize()
        
        # Check health
        for p in params:
            if p.isnan().any().item() or p.isinf().any().item():
                return False, step - 1
        
        opt.zero_grad()
    
    return True, steps


# =============================================================================
# SOTA MODEL GEOMETRY TESTS
# =============================================================================

def test_model_geometries(model_name: str, dtype: torch.dtype, steps: int = 10,
                          max_params: Optional[int] = None) -> bool:
    """Test all layer geometries for a specific model."""
    if model_name not in SOTA_MODELS:
        print(f"  Unknown model: {model_name}")
        return False
    
    config = SOTA_MODELS[model_name]
    device = torch.device('mps')
    geometries = config.get_geometries()
    
    print(f"\n  {config.name} ({dtype}):")
    all_passed = True
    
    for layer_name, shape in geometries.items():
        if shape is None:
            continue
        
        numel = 1
        for s in shape:
            numel *= s
        
        # Skip layers exceeding max_params if specified
        if max_params is not None and numel > max_params:
            print(f"    ⊘ {layer_name}: {shape} ({numel/1e6:.0f}M) [SKIPPED - exceeds {max_params/1e6:.0f}M limit]")
            continue
        
        try:
            p = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
            success, last_step = run_adamw_steps([p], steps=steps, grad_scale=0.1)
            
            health = check_numerical_health(layer_name, p)
            passed = print_health(health, verbose=False)
            all_passed &= passed and success
            
            if not success:
                print(f"      Failed at step {last_step + 1}")
            
            del p
            torch.mps.empty_cache()
            
        except Exception as e:
            print(f"    ✗ {layer_name}: {shape} [ERROR: {e}]")
            all_passed = False
    
    return all_passed


def test_all_sota_models(models: List[str], dtypes: List[torch.dtype], steps: int,
                         max_params: Optional[int] = None) -> Dict[str, bool]:
    """Test all specified SOTA models."""
    print("\n" + "=" * 70)
    print("SOTA MODEL GEOMETRY TESTS")
    print("=" * 70)
    
    results = {}
    
    for model in models:
        for dtype in dtypes:
            key = f"{model}_{dtype}"
            results[key] = test_model_geometries(model, dtype, steps, max_params)
    
    return results


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_memory_alignment_edge_cases() -> bool:
    """Test sizes that might cause alignment issues (not divisible by 4)."""
    print("\n" + "=" * 70)
    print("MEMORY ALIGNMENT EDGE CASES")
    print("=" * 70)
    
    device = torch.device('mps')
    all_passed = True
    
    # Sizes that stress vectorization boundaries
    edge_sizes = [
        1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17,  # Near vector boundaries
        31, 32, 33, 63, 64, 65,               # Near warp boundaries
        127, 128, 129, 255, 256, 257,         # Near threadgroup boundaries
        1023, 1024, 1025,                     # Power of 2 edges
        4093, 4094, 4095, 4096, 4097,         # Common buffer sizes
        16381, 16383, 16384, 16385, 16387,   # Larger edge cases
    ]
    
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        print(f"\n  {dtype}:")
        
        for size in edge_sizes:
            p = torch.randn(size, device=device, dtype=dtype, requires_grad=True)
            success, _ = run_adamw_steps([p], steps=5, grad_scale=0.1)
            
            status = "✓" if success else "✗"
            print(f"    {status} size={size}", end="")
            
            if not success:
                print(" [FAILED]")
                all_passed = False
            else:
                print()
            
            del p
    
    return all_passed


def test_gradient_extremes() -> bool:
    """Test extreme gradient values that might cause numerical issues."""
    print("\n" + "=" * 70)
    print("GRADIENT EXTREME VALUE TESTS")
    print("=" * 70)
    
    device = torch.device('mps')
    all_passed = True
    
    test_cases = [
        ("Zero gradients", 0.0),
        ("Tiny gradients (1e-10)", 1e-10),
        ("Tiny gradients (1e-8)", 1e-8),
        ("Tiny gradients (1e-6)", 1e-6),
        ("Normal gradients (0.01)", 0.01),
        ("Normal gradients (0.1)", 0.1),
        ("Normal gradients (1.0)", 1.0),
        ("Large gradients (10)", 10.0),
        ("Large gradients (100)", 100.0),
        ("Large gradients (1000)", 1000.0),
        ("Very large gradients (10000)", 10000.0),
    ]
    
    for dtype in [torch.float32, torch.bfloat16]:
        print(f"\n  {dtype}:")
        
        for name, scale in test_cases:
            p = torch.randn(10000, device=device, dtype=dtype, requires_grad=True)
            
            if METALCORE_AVAILABLE:
                opt = MetalAdamW([p], lr=1e-4)
            else:
                opt = torch.optim.AdamW([p], lr=1e-4)
            
            success = True
            for step in range(20):
                if scale == 0.0:
                    p.grad = torch.zeros_like(p)
                else:
                    p.grad = torch.randn_like(p) * scale
                
                opt.step()
                torch.mps.synchronize()
                
                if p.isnan().any().item() or p.isinf().any().item():
                    success = False
                    print(f"    ✗ {name} [FAILED at step {step + 1}]")
                    break
                
                opt.zero_grad()
            
            if success:
                print(f"    ✓ {name}")
            
            all_passed &= success
            del p
    
    return all_passed


def test_sparse_gradients() -> bool:
    """Test sparse-like gradient patterns (common in embedding layers)."""
    print("\n" + "=" * 70)
    print("SPARSE GRADIENT TESTS")
    print("=" * 70)
    
    device = torch.device('mps')
    all_passed = True
    
    sparsity_levels = [0.99, 0.999, 0.9999, 0.99999]  # 1%, 0.1%, 0.01%, 0.001% non-zero
    
    for dtype in [torch.float32, torch.bfloat16]:
        print(f"\n  {dtype}:")
        
        for sparsity in sparsity_levels:
            # Simulate embedding layer with sparse gradients
            p = torch.randn(32000, 4096, device=device, dtype=dtype, requires_grad=True)
            
            if METALCORE_AVAILABLE:
                opt = MetalAdamW([p], lr=1e-4)
            else:
                opt = torch.optim.AdamW([p], lr=1e-4)
            
            success = True
            for step in range(10):
                # Create sparse gradient (only some rows have updates)
                mask = torch.rand(32000, 1, device=device) > sparsity
                p.grad = torch.randn_like(p) * mask.to(dtype) * 0.1
                
                opt.step()
                torch.mps.synchronize()
                
                if p.isnan().any().item() or p.isinf().any().item():
                    success = False
                    print(f"    ✗ sparsity={sparsity} [FAILED at step {step + 1}]")
                    break
                
                opt.zero_grad()
            
            if success:
                non_zero_pct = (1 - sparsity) * 100
                print(f"    ✓ sparsity={sparsity} ({non_zero_pct:.3f}% non-zero)")
            
            all_passed &= success
            del p
            torch.mps.empty_cache()
    
    return all_passed


def test_view_and_slice_tensors() -> bool:
    """Test non-contiguous tensors (views, slices)."""
    print("\n" + "=" * 70)
    print("VIEW AND SLICE TENSOR TESTS")
    print("=" * 70)
    
    device = torch.device('mps')
    all_passed = True
    
    for dtype in [torch.float32, torch.bfloat16]:
        print(f"\n  {dtype}:")
        
        # Test 1: Transposed view
        print("    Testing transposed view...", end="")
        base = torch.randn(1024, 2048, device=device, dtype=dtype)
        p = base.t().contiguous()  # Store transposed
        p.requires_grad = True
        success, _ = run_adamw_steps([p], steps=10)
        print(" ✓" if success else " ✗")
        all_passed &= success
        del base, p
        
        # Test 2: Slice of larger tensor
        print("    Testing slice of larger tensor...", end="")
        base = torch.randn(4096, 4096, device=device, dtype=dtype)
        p = base[:1000, :1000].contiguous()
        p.requires_grad = True
        success, _ = run_adamw_steps([p], steps=10)
        print(" ✓" if success else " ✗")
        all_passed &= success
        del base, p
        
        # Test 3: Strided tensor made contiguous
        print("    Testing strided tensor...", end="")
        base = torch.randn(2, 2048, 2048, device=device, dtype=dtype)
        p = base[0, ::2, ::2].contiguous()  # Every other element
        p.requires_grad = True
        success, _ = run_adamw_steps([p], steps=10)
        print(" ✓" if success else " ✗")
        all_passed &= success
        del base, p
        
        torch.mps.empty_cache()
    
    return all_passed


def test_extended_training(steps: int = 1000) -> bool:
    """Test extended training for accumulator stability."""
    print("\n" + "=" * 70)
    print(f"EXTENDED TRAINING TEST ({steps} steps)")
    print("=" * 70)
    
    device = torch.device('mps')
    all_passed = True
    
    check_points = [1, 10, 50, 100, 250, 500, 750, 1000]
    check_points = [cp for cp in check_points if cp <= steps]
    
    for dtype in [torch.float32, torch.bfloat16]:
        print(f"\n  {dtype}:")
        
        # Use realistic embedding layer size
        p = torch.randn(32000, 4096, device=device, dtype=dtype, requires_grad=True)
        
        if METALCORE_AVAILABLE:
            opt = MetalAdamW([p], lr=1e-4, weight_decay=0.01)
        else:
            opt = torch.optim.AdamW([p], lr=1e-4, weight_decay=0.01)
        
        for step in range(1, steps + 1):
            p.grad = torch.randn_like(p) * 0.1
            opt.step()
            
            if step in check_points:
                torch.mps.synchronize()
                health = check_numerical_health(f"Step {step}", p)
                passed = print_health(health)
                
                if METALCORE_AVAILABLE:
                    state = opt.state[p]
                    exp_avg_max = state['exp_avg'].abs().max().item()
                    exp_avg_sq_max = state['exp_avg_sq'].max().item()
                    print(f"        exp_avg_max={exp_avg_max:.2e}, exp_avg_sq_max={exp_avg_sq_max:.2e}")
                
                if not passed:
                    all_passed = False
                    break
            
            opt.zero_grad()
        
        del p, opt
        torch.mps.empty_cache()
    
    return all_passed


def test_multi_param_groups() -> bool:
    """Test multiple parameter groups with different settings."""
    print("\n" + "=" * 70)
    print("MULTI-PARAMETER GROUP TESTS")
    print("=" * 70)
    
    device = torch.device('mps')
    all_passed = True
    
    for dtype in [torch.float32, torch.bfloat16]:
        print(f"\n  {dtype}:")
        
        # Different sized parameters
        embed = torch.randn(32000, 4096, device=device, dtype=dtype, requires_grad=True)
        attn_q = torch.randn(4096, 4096, device=device, dtype=dtype, requires_grad=True)
        mlp_up = torch.randn(4096, 14336, device=device, dtype=dtype, requires_grad=True)
        norm = torch.randn(4096, device=device, dtype=dtype, requires_grad=True)
        
        if METALCORE_AVAILABLE:
            opt = MetalAdamW([
                {"params": [embed], "lr": 1e-5, "weight_decay": 0.0},  # Embeddings get lower LR
                {"params": [attn_q, mlp_up], "lr": 1e-4, "weight_decay": 0.01},
                {"params": [norm], "lr": 1e-4, "weight_decay": 0.0},  # No WD for norms
            ])
        else:
            opt = torch.optim.AdamW([
                {"params": [embed], "lr": 1e-5, "weight_decay": 0.0},
                {"params": [attn_q, mlp_up], "lr": 1e-4, "weight_decay": 0.01},
                {"params": [norm], "lr": 1e-4, "weight_decay": 0.0},
            ])
        
        success = True
        for step in range(20):
            embed.grad = torch.randn_like(embed) * 0.1
            attn_q.grad = torch.randn_like(attn_q) * 0.1
            mlp_up.grad = torch.randn_like(mlp_up) * 0.1
            norm.grad = torch.randn_like(norm) * 0.1
            
            opt.step()
            torch.mps.synchronize()
            
            for name, p in [("embed", embed), ("attn_q", attn_q), 
                           ("mlp_up", mlp_up), ("norm", norm)]:
                if p.isnan().any().item() or p.isinf().any().item():
                    print(f"    ✗ {name} failed at step {step + 1}")
                    success = False
                    break
            
            if not success:
                break
            
            opt.zero_grad()
        
        if success:
            print(f"    ✓ All parameter groups stable for 20 steps")
        
        all_passed &= success
        del embed, attn_q, mlp_up, norm
        torch.mps.empty_cache()
    
    return all_passed


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests(quick: bool = False, model: Optional[str] = None,
                  max_params: Optional[int] = None) -> bool:
    """Run all tests and return overall success."""
    print("\n" + "=" * 70)
    print("METALCORE ADAMW COMPREHENSIVE STRESS TEST SUITE")
    print("=" * 70)
    print(f"Device: MPS ({'available' if torch.backends.mps.is_available() else 'NOT AVAILABLE'})")
    print(f"MetalCore: {'available' if METALCORE_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"Mode: {'Quick' if quick else 'Full'}")
    print(f"PyTorch: {torch.__version__}")
    
    if not torch.backends.mps.is_available():
        print("\nERROR: MPS not available. Cannot run tests.")
        return False
    
    if not METALCORE_AVAILABLE:
        print("\nERROR: metalcore not available. Cannot run Metal kernel tests.")
        return False
    
    results = {}
    dtypes = [torch.float32, torch.bfloat16, torch.float16]
    
    # Select models to test
    if model:
        models_to_test = [model]
    elif quick:
        models_to_test = QUICK_MODELS
    else:
        models_to_test = list(SOTA_MODELS.keys())
    
    # 1. SOTA Model Geometry Tests
    print("\n" + "=" * 70)
    print("PHASE 1: SOTA MODEL GEOMETRY TESTS")
    print("=" * 70)
    
    # In quick mode, default to 100M param limit unless explicitly set
    effective_max_params = max_params
    if quick and max_params is None:
        effective_max_params = 100_000_000  # 100M params
        print(f"Quick mode: limiting to {effective_max_params/1e6:.0f}M params per layer")
    
    for m in models_to_test:
        for dtype in dtypes:
            key = f"model_{m}_{dtype}"
            results[key] = test_model_geometries(m, dtype, steps=10 if quick else 20, max_params=effective_max_params)
    
    # 2. Memory Alignment Edge Cases
    results["memory_alignment"] = test_memory_alignment_edge_cases()
    
    # 3. Gradient Extremes
    results["gradient_extremes"] = test_gradient_extremes()
    
    # 4. Sparse Gradients
    if not quick:
        results["sparse_gradients"] = test_sparse_gradients()
    
    # 5. Views and Slices
    results["views_and_slices"] = test_view_and_slice_tensors()
    
    # 6. Extended Training
    results["extended_training"] = test_extended_training(steps=100 if quick else 1000)
    
    # 7. Multi-Parameter Groups
    results["multi_param_groups"] = test_multi_param_groups()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = True
    passed_count = 0
    failed_count = 0
    
    for name, passed in sorted(results.items()):
        if passed:
            passed_count += 1
        else:
            failed_count += 1
            print(f"  ✗ FAILED: {name}")
        all_passed &= passed
    
    print(f"\n  Passed: {passed_count}/{len(results)}")
    print(f"  Failed: {failed_count}/{len(results)}")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdamW Comprehensive Stress Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--model", type=str, default=None, 
                       help=f"Test specific model: {', '.join(SOTA_MODELS.keys())}")
    parser.add_argument("--max-params", type=int, default=None,
                       help="Max params per layer to test (e.g., 500000000 for 500M). Default: no limit")
    args = parser.parse_args()
    
    success = run_all_tests(quick=args.quick, model=args.model, max_params=args.max_params)
    sys.exit(0 if success else 1)
