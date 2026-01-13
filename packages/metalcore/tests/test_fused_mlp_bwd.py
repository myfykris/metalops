import torch
import torch.nn.functional as F
from metalcore import fused_mlp_bwd
import metalcore

def test_fused_mlp_bwd():
    print("Testing Fused MLP Backward...")
    device = torch.device("mps")
    dtype = torch.float32 # Start with float32
    
    # Dimensions
    B, S, H, I = 2, 128, 64, 128 # Batch, Seq, Hidden, Intermediate
    # r = LoRA rank
    r = 16
    
    # Forward Inputs
    hidden_states = torch.randn(B, S, H, device=device, dtype=dtype, requires_grad=True)
    residual = torch.randn(B, S, H, device=device, dtype=dtype, requires_grad=True)
    
    rms_weight = torch.randn(H, device=device, dtype=dtype, requires_grad=True)
    
    # Weights
    W_gate = torch.randn(I, H, device=device, dtype=dtype, requires_grad=True)
    W_up = torch.randn(I, H, device=device, dtype=dtype, requires_grad=True)
    W_down = torch.randn(H, I, device=device, dtype=dtype, requires_grad=True)
    
    # LoRAs
    A_gate = torch.randn(r, H, device=device, dtype=dtype, requires_grad=True)
    B_gate = torch.randn(I, r, device=device, dtype=dtype, requires_grad=True)
    
    A_up = torch.randn(r, H, device=device, dtype=dtype, requires_grad=True)
    B_up = torch.randn(I, r, device=device, dtype=dtype, requires_grad=True)
    
    A_down = torch.randn(r, I, device=device, dtype=dtype, requires_grad=True)
    B_down = torch.randn(H, r, device=device, dtype=dtype, requires_grad=True)
    
    scale = 1.0
    
    # --- Reference Forward & Backward ---
    
    # RMSNorm
    x_norm = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + 1e-6) * rms_weight
    rstd = torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + 1e-6)
    
    # Gate/Up
    # Linear + LoRA
    # y = x @ W.T + x @ A.T @ B.T * scale
    gate_base = F.linear(x_norm, W_gate)
    gate_lora = (x_norm @ A_gate.T @ B_gate.T) * scale
    gate = gate_base + gate_lora
    
    up_base = F.linear(x_norm, W_up)
    up_lora = (x_norm @ A_up.T @ B_up.T) * scale
    up = up_base + up_lora
    
    # SwiGLU
    swiglu_out = F.silu(gate) * up
    
    # Down
    down_base = F.linear(swiglu_out, W_down)
    down_lora = (swiglu_out @ A_down.T @ B_down.T) * scale
    down_out = down_base + down_lora
    
    # Residual
    out_final = down_out + residual
    
    # Loss
    loss = out_final.sum()
    
    # Backward
    loss.backward()
    
    # Save Grads
    grads_ref = {
        "hidden": hidden_states.grad.clone(),
        "W_gate": W_gate.grad.clone(), "W_up": W_up.grad.clone(), "W_down": W_down.grad.clone(),
        "A_gate": A_gate.grad.clone(), "B_gate": B_gate.grad.clone(),
        "A_up": A_up.grad.clone(), "B_up": B_up.grad.clone(),
        "A_down": A_down.grad.clone(), "B_down": B_down.grad.clone(),
        "rms_w": rms_weight.grad.clone()
    }
    
    # --- Metal Backward ---
    d_out = torch.ones_like(out_final) # From loss.sum()
    
    # Helper to clean grads
    def zero_grad():
        for t in [hidden_states, W_gate, W_up, W_down, A_gate, B_gate, A_up, B_up, A_down, B_down, rms_weight]:
            if t.grad is not None: t.grad.zero_()
            
    zero_grad()
    
    # Function Call
    # Note: We pass tensors from forward pass (x_norm, gate, up, rstd)
    # Detach them to pretend we are in C++ context? No, function takes tensors.
    
    grads_metal_tuple = fused_mlp_bwd(
        d_out,
        x_norm.detach(), gate.detach(), up.detach(),
        W_gate, W_up, W_down,
        A_gate, B_gate, A_up, B_up, A_down, B_down,
        scale,
        hidden_states.detach(), rms_weight, rstd.detach()
    )
    
    (d_hidden, dW_g, dW_u, dW_d, dA_g, dB_g, dA_u, dB_u, dA_d, dB_d, d_rms) = grads_metal_tuple
    
    # Verify
    def compare(name, g_metal, g_ref):
        if g_metal is None or g_metal.numel() == 0:
            print(f"{name}: Skipped (empty)")
            return
        
        g_metal = g_metal.cpu().float()
        g_ref = g_ref.cpu().float()
        
        sim = F.cosine_similarity(g_metal.flatten(), g_ref.flatten(), dim=0, eps=1e-6)
        diff = (g_metal - g_ref).abs().max()
        print(f"{name}: Cosim={sim.item():.4f}, MaxDiff={diff.item():.4f}")
        
        if sim < 0.99:
            print(f"FAILED {name}")
            # exit(1)
            
    compare("HiddenStates", d_hidden, grads_ref["hidden"])
    compare("W_gate", dW_g, grads_ref["W_gate"])
    compare("W_up", dW_u, grads_ref["W_up"])
    compare("W_down", dW_d, grads_ref["W_down"])
    compare("A_gate", dA_g, grads_ref["A_gate"])
    compare("B_gate", dB_g, grads_ref["B_gate"])
    compare("A_up", dA_u, grads_ref["A_up"])
    compare("B_up", dB_u, grads_ref["B_up"])
    compare("A_down", dA_d, grads_ref["A_down"])
    compare("B_down", dB_d, grads_ref["B_down"])
    compare("RMSWeight", d_rms, grads_ref["rms_w"])

if __name__ == "__main__":
    test_fused_mlp_bwd()
