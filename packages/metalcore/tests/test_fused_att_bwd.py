
import torch
import sys
import os
from pathlib import Path

# Add package source to path
pkg_path = Path(__file__).parent.parent / "src"
sys.path.append(str(pkg_path))

import metalcore

def test_fused_attention_bwd():
    device = torch.device("mps")
    dtype = torch.float32 # or bfloat16 if M3
    
    # Dimensions
    B, S, H, D = 2, 128, 8, 64
    Hidden = H * D
    Rank = 16
    
    # Inputs
    hidden_states = torch.randn(B, S, Hidden, device=device, dtype=dtype, requires_grad=True)
    
    # Weights
    Wq = torch.randn(H*D, Hidden, device=device, dtype=dtype, requires_grad=True)
    Wk = torch.randn(H*D, Hidden, device=device, dtype=dtype, requires_grad=True)
    Wv = torch.randn(H*D, Hidden, device=device, dtype=dtype, requires_grad=True)
    
    # LoRA
    Aq = torch.randn(Rank, Hidden, device=device, dtype=dtype, requires_grad=True)
    Bq = torch.randn(H*D, Rank, device=device, dtype=dtype, requires_grad=True)
    
    Ak = torch.randn(Rank, Hidden, device=device, dtype=dtype, requires_grad=True)
    Bk = torch.randn(H*D, Rank, device=device, dtype=dtype, requires_grad=True)
    
    Av = torch.randn(Rank, Hidden, device=device, dtype=dtype, requires_grad=True)
    Bv = torch.randn(H*D, Rank, device=device, dtype=dtype, requires_grad=True)
    
    scale = 0.5
    
    # RMSNorm
    rms_w = torch.randn(Hidden, device=device, dtype=dtype, requires_grad=True)
    eps = 1e-5
    
    # RoPE
    cos = torch.randn(S, D//2, device=device, dtype=dtype)
    sin = torch.randn(S, D//2, device=device, dtype=dtype)
    
    # --- Reference Forward & Backward ---
    # 1. RMSNorm
    # PyTorch RMSNorm
    x_f32 = hidden_states.float()
    w_f32 = rms_w.float()
    ms = x_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(ms + eps)
    x_norm = x_f32 * rstd * w_f32
    x_norm_ref = x_norm.to(dtype)
    
    # Capture rstd for custom op (core_mps expectation: rstd is [B, S] or [B*S])
    # rmsnorm_bwd takes rstd tensor.
    rstd_t = rstd.squeeze(-1).to(dtype).detach()
    
    # 2. QKV Linear + LoRA
    # Q = X @ W.T + scale * X @ A.T @ B.T
    def lora_linear(x, W, A, B, s):
        base = torch.mm(x.view(-1, Hidden), W.t())
        lora = torch.mm(torch.mm(x.view(-1, Hidden), A.t()), B.t())
        return base + s * lora
        
    q_ref = lora_linear(x_norm_ref, Wq, Aq, Bq, scale).view(B, S, H, D)
    k_ref = lora_linear(x_norm_ref, Wk, Ak, Bk, scale).view(B, S, H, D)
    v_ref = lora_linear(x_norm_ref, Wv, Av, Bv, scale).view(B, S, H, D)
    
    # 3. RoPE
    # Naive RoPE for verification
    def apply_rope(x, c, s):
        # split half
        x1 = x[..., :D//2]
        x2 = x[..., D//2:]
        # c, s shape [S, D//2] -> broadcast to [B, S, H, D//2]
        c = c.view(1, S, 1, D//2)
        s = s.view(1, S, 1, D//2)
        out1 = x1 * c - x2 * s
        out2 = x2 * c + x1 * s
        return torch.cat([out1, out2], -1)

    q_rot = apply_rope(q_ref, cos, sin)
    k_rot = apply_rope(k_ref, cos, sin)
    v_rot = v_ref # V not rotated
    
    # Mock Gradients
    d_q_rot = torch.randn_like(q_rot)
    d_k_rot = torch.randn_like(k_rot)
    d_v_rot = torch.randn_like(v_rot)
    
    # Run Ref Backward
    (q_rot * d_q_rot + k_rot * d_k_rot + v_rot * d_v_rot).sum().backward()
    
    ref_grads = {
        "hidden": hidden_states.grad.clone(),
        "Wq": Wq.grad.clone(), "Wk": Wk.grad.clone(), "Wv": Wv.grad.clone(),
        "Aq": Aq.grad.clone(), "Bq": Bq.grad.clone(),
        "rms_w": rms_w.grad.clone()
    }
    
    # --- Custom Backward ---
    # fused_attention_bwd_metal(d_q, d_k, d_v, cos, sin, x_norm, Wq, Wk, Wv, Aq, Bq..., scale, hidden_states, rms_w, rstd)
    
    # Note: custom op takes 4D d_q/d_k?
    # Signature: (d_q, d_k, d_v, cos, sin, ...) 
    # d_q comes from SDPA logic, so it is [B, S, H, D] or [Batch*H, S, D]? 
    # rope_bwd expects 4D? Yes.
    # But core_mps.mm impl reshapes them to 2D for matmul.
    
    # Call backend
    # Note: We need x_norm from forward. x_norm_ref.
    
    results = metalcore.fused_attention_bwd(
        d_q_rot, d_k_rot, d_v_rot,
        cos, sin,
        x_norm_ref,
        Wq, Wk, Wv,
        Aq, Bq, Ak, Bk, Av, Bv,
        scale,
        hidden_states,
        rms_w,
        rstd_t
    )
    
    (d_hidden, dWq, dWk, dWv, dA_q, dB_q, dA_k, dB_k, dA_v, dB_v, d_rms_w) = results
    
    # --- Verify ---
    print("Verifying Backward Gradients:")
    
    def check(name, ref, start, tol=1e-2):
        if ref is None or start is None:
            print(f"{name}: SKIP (None)")
            return
        # Flatten
        r = ref.view(-1)
        s = start.view(-1)
        # Cosine sim
        cosim = torch.nn.functional.cosine_similarity(r, s, dim=0, eps=1e-6)
        diff = (r - s).abs().max()
        print(f"{name}: Cosim={cosim.item():.4f}, MaxDiff={diff.item():.4f}")
        if cosim < 0.99 and diff > tol:
             print(f"FAILED {name}")
             # print first few
             # print(f"Ref: {r[:5]}")
             # print(f"Got: {s[:5]}")
    
    check("Hidden", ref_grads["hidden"], d_hidden)
    check("Wq", ref_grads["Wq"], dWq)
    check("Aq", ref_grads["Aq"], dA_q)
    check("Bq", ref_grads["Bq"], dB_q)
    check("RMS Weight", ref_grads["rms_w"], d_rms_w)

if __name__ == "__main__":
    test_fused_attention_bwd()
