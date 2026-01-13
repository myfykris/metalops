import torch
import metalcore_backend as mc
import time
import torch.nn.functional as F

print("=" * 70)
print("Benchmarking fused_lora_attention_fwd (PyTorch SDPA Fallback)")
print("=" * 70)

B, L, D = 1, 256, 2048
num_heads, num_kv_heads = 32, 32
head_dim = D // num_heads
D_kv = head_dim * num_kv_heads
rank = 64
lora_scale = 0.5

print(f"Config: B={B}, L={L}, D={D}, rank={rank}")

if not torch.backends.mps.is_available():
    print("MPS not available!")
    exit(1)

hidden = torch.randn(B, L, D, device='mps') * 0.1  # Typical hidden state magnitude
res = torch.randn(B, L, D, device='mps') * 0.1
ln_weight = torch.ones(D, device='mps')  # Typical initialized to 1
ln_eps = 1e-6

# Xavier/He-style initialization for weights: scale = sqrt(2/fan_in)
# For D=2048, scale ≈ 0.03
import math
w_scale = math.sqrt(2 / D)  # ~0.031 for D=2048
lora_scale_init = math.sqrt(2 / rank)  # ~0.18 for rank=64

W_q = torch.randn(D, D, device='mps') * w_scale
W_k = torch.randn(D_kv, D, device='mps') * w_scale
W_v = torch.randn(D_kv, D, device='mps') * w_scale
W_o = torch.randn(D, D, device='mps') * w_scale

# LoRA uses smaller init (typically 0 for B, small for A)
A_q = torch.randn(rank, D, device='mps') * lora_scale_init
B_q = torch.zeros(D, rank, device='mps')  # B typically init to 0
A_k = torch.randn(rank, D, device='mps') * lora_scale_init
B_k = torch.zeros(D_kv, rank, device='mps')
A_v = torch.randn(rank, D, device='mps') * lora_scale_init
B_v = torch.zeros(D_kv, rank, device='mps')
A_o = torch.randn(rank, D, device='mps') * lora_scale_init
B_o = torch.zeros(D, rank, device='mps')

# RoPE cos/sin are bounded [-1, 1]
cos = torch.cos(torch.randn(L, 32, device='mps'))  # Bounded to [-1, 1]
sin = torch.sin(torch.randn(L, 32, device='mps'))  # Bounded to [-1, 1]

def pytorch_ref():
    """Reference implementation matching the Metal fused function (no RoPE)"""
    normed = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + ln_eps) * ln_weight
    
    q = torch.mm(normed.view(-1, D), W_q.t())
    k = torch.mm(normed.view(-1, D), W_k.t())
    v = torch.mm(normed.view(-1, D), W_v.t())
    
    q += lora_scale * torch.mm(torch.mm(normed.view(-1, D), A_q.t()), B_q.t())
    k += lora_scale * torch.mm(torch.mm(normed.view(-1, D), A_k.t()), B_k.t())
    v += lora_scale * torch.mm(torch.mm(normed.view(-1, D), A_v.t()), B_v.t())
    
    q = q.view(B, L, num_heads, head_dim).transpose(1, 2)
    k = k.view(B, L, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(B, L, num_kv_heads, head_dim).transpose(1, 2)
    
    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn = attn.transpose(1, 2).contiguous().view(B*L, D)
    
    out = torch.mm(attn, W_o.t())
    out += lora_scale * torch.mm(torch.mm(attn, A_o.t()), B_o.t())
    
    return res.view(B*L, D) + out

try:
    print("Checking correctness...")
    cos_identity = torch.ones_like(cos)
    sin_identity = torch.zeros_like(sin)
    
    out_metal, _ = mc.fused_lora_attention_fwd(
        hidden, res, ln_weight, ln_eps,
        W_q, W_k, W_v,
        A_q, B_q, A_k, B_k, A_v, B_v, lora_scale,
        W_o, A_o, B_o,
        cos_identity, sin_identity, 
        num_heads, num_kv_heads, 0.0, True, False
    )
    torch.mps.synchronize()
    
    out_ref = pytorch_ref()
    torch.mps.synchronize()
    
    # CPU baseline (float64) for reference
    def cpu_ref_f64():
        h_cpu = hidden.cpu().double()
        res_cpu = res.cpu().double()
        ln_w_cpu = ln_weight.cpu().double()
        W_q_cpu, W_k_cpu, W_v_cpu, W_o_cpu = W_q.cpu().double(), W_k.cpu().double(), W_v.cpu().double(), W_o.cpu().double()
        A_q_cpu, B_q_cpu = A_q.cpu().double(), B_q.cpu().double()
        A_k_cpu, B_k_cpu = A_k.cpu().double(), B_k.cpu().double()
        A_v_cpu, B_v_cpu = A_v.cpu().double(), B_v.cpu().double()
        A_o_cpu, B_o_cpu = A_o.cpu().double(), B_o.cpu().double()
        
        normed = h_cpu * torch.rsqrt(h_cpu.pow(2).mean(-1, keepdim=True) + ln_eps) * ln_w_cpu
        q = torch.mm(normed.view(-1, D), W_q_cpu.t()) + lora_scale * torch.mm(torch.mm(normed.view(-1, D), A_q_cpu.t()), B_q_cpu.t())
        k = torch.mm(normed.view(-1, D), W_k_cpu.t()) + lora_scale * torch.mm(torch.mm(normed.view(-1, D), A_k_cpu.t()), B_k_cpu.t())
        v = torch.mm(normed.view(-1, D), W_v_cpu.t()) + lora_scale * torch.mm(torch.mm(normed.view(-1, D), A_v_cpu.t()), B_v_cpu.t())
        
        q = q.view(B, L, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(B, L, num_kv_heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(B, L, num_kv_heads, head_dim).permute(0, 2, 1, 3)
        
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.permute(0, 2, 1, 3).contiguous().view(B*L, D)
        out = torch.mm(attn, W_o_cpu.t()) + lora_scale * torch.mm(torch.mm(attn, A_o_cpu.t()), B_o_cpu.t())
        return res_cpu.view(B*L, D) + out
    
    # CPU float32 baseline - same ops, same order as Metal
    def cpu_ref_f32():
        h_cpu = hidden.cpu().float()
        res_cpu = res.cpu().float()
        ln_w_cpu = ln_weight.cpu().float()
        W_q_cpu, W_k_cpu, W_v_cpu, W_o_cpu = W_q.cpu().float(), W_k.cpu().float(), W_v.cpu().float(), W_o.cpu().float()
        A_q_cpu, B_q_cpu = A_q.cpu().float(), B_q.cpu().float()
        A_k_cpu, B_k_cpu = A_k.cpu().float(), B_k.cpu().float()
        A_v_cpu, B_v_cpu = A_v.cpu().float(), B_v.cpu().float()
        A_o_cpu, B_o_cpu = A_o.cpu().float(), B_o.cpu().float()
        
        normed = h_cpu * torch.rsqrt(h_cpu.pow(2).mean(-1, keepdim=True) + ln_eps) * ln_w_cpu
        q = torch.mm(normed.view(-1, D), W_q_cpu.t()) + lora_scale * torch.mm(torch.mm(normed.view(-1, D), A_q_cpu.t()), B_q_cpu.t())
        k = torch.mm(normed.view(-1, D), W_k_cpu.t()) + lora_scale * torch.mm(torch.mm(normed.view(-1, D), A_k_cpu.t()), B_k_cpu.t())
        v = torch.mm(normed.view(-1, D), W_v_cpu.t()) + lora_scale * torch.mm(torch.mm(normed.view(-1, D), A_v_cpu.t()), B_v_cpu.t())
        
        q = q.view(B, L, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(B, L, num_kv_heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(B, L, num_kv_heads, head_dim).permute(0, 2, 1, 3)
        
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.permute(0, 2, 1, 3).contiguous().view(B*L, D)
        out = torch.mm(attn, W_o_cpu.t()) + lora_scale * torch.mm(torch.mm(attn, A_o_cpu.t()), B_o_cpu.t())
        return res_cpu.view(B*L, D) + out
    
    out_cpu_f64 = cpu_ref_f64()
    out_cpu_f32 = cpu_ref_f32()
    
    diff = (out_metal.view(-1) - out_ref.view(-1)).abs().max().item()
    cpu_f32_vs_f64 = (out_cpu_f32.double() - out_cpu_f64).abs().max().item()
    metal_vs_cpu_f64 = (out_metal.view(-1).cpu().double() - out_cpu_f64.view(-1)).abs().max().item()
    metal_vs_cpu_f32 = (out_metal.view(-1).cpu() - out_cpu_f32.view(-1)).abs()
    
    print(f"✓ Metal vs MPS Ref Max Diff: {diff:.6f}")
    print(f"✓ CPU(f32) vs CPU(f64) Max Diff: {cpu_f32_vs_f64:.6f} (float32 baseline)")
    print(f"✓ Metal vs CPU(f64) Max Diff: {metal_vs_cpu_f64:.6f}")
    print(f"✓ Metal vs CPU(f32): max={metal_vs_cpu_f32.max().item():.4f}, mean={metal_vs_cpu_f32.mean().item():.6f}, 99%={metal_vs_cpu_f32.quantile(0.99).item():.6f}")
    
    # Benchmark
    print("\nBenchmarking...")
    pk_iters = 50
    
    # Warmup
    for _ in range(5): 
        mc.fused_lora_attention_fwd(hidden, res, ln_weight, ln_eps, W_q, W_k, W_v, 
            A_q, B_q, A_k, B_k, A_v, B_v, lora_scale, W_o, A_o, B_o, 
            cos, sin, num_heads, num_kv_heads, 0.0, True, False)
    torch.mps.synchronize()
    
    start = time.perf_counter()
    for _ in range(pk_iters):
        mc.fused_lora_attention_fwd(hidden, res, ln_weight, ln_eps, W_q, W_k, W_v,
            A_q, B_q, A_k, B_k, A_v, B_v, lora_scale, W_o, A_o, B_o, 
            cos, sin, num_heads, num_kv_heads, 0.0, True, False)
    torch.mps.synchronize()
    ms_metal = (time.perf_counter() - start) / pk_iters * 1000
    
    # PyTorch Warmup
    for _ in range(5): pytorch_ref()
    torch.mps.synchronize()
    
    start = time.perf_counter()
    for _ in range(pk_iters):
        pytorch_ref()
    torch.mps.synchronize()
    ms_torch = (time.perf_counter() - start) / pk_iters * 1000
    
    print(f"\nResults (L={L}):")
    print(f"  PyTorch Ref: {ms_torch:.3f} ms")
    print(f"  Metal Fused: {ms_metal:.3f} ms")
    print(f"  Speedup: {ms_torch / ms_metal:.2f}x")
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
