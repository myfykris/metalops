import torch
import torch.nn.functional as F
from metalcore import swiglu_bwd
import metalcore

def test_swiglu_bwd_correctness():
    device = torch.device("mps")
    dtype = torch.float32 # Test float first
    
    B, D = 32, 1024
    
    # Inputs
    gate = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    up = torch.randn(B, D, device=device, dtype=dtype, requires_grad=True)
    
    # Forward
    y = F.silu(gate) * up
    
    # Fake grad output
    d_h = torch.randn_like(y)
    
    # PyTorch Autograd Backward
    y.backward(d_h, retain_graph=True)
    d_gate_ref = gate.grad.clone()
    d_up_ref = up.grad.clone()
    
    # Metal Backward
    gate.grad.zero_()
    up.grad.zero_()
    
    d_gate_metal, d_up_metal = swiglu_bwd(d_h, gate, up)
    
    # Verify
    print(f"Verifying SwiGLU Backward ({dtype}):")
    
    # Cosine Sim
    cos_gate = F.cosine_similarity(d_gate_metal.flatten(), d_gate_ref.flatten(), dim=0, eps=1e-6)
    cos_up = F.cosine_similarity(d_up_metal.flatten(), d_up_ref.flatten(), dim=0, eps=1e-6)
    
    # Max Diff
    diff_gate = (d_gate_metal - d_gate_ref).abs().max().item()
    diff_up = (d_up_metal - d_up_ref).abs().max().item()
    
    print(f"d_gate: Cosim={cos_gate:.4f}, MaxDiff={diff_gate:.4f}")
    print(f"d_up:   Cosim={cos_up:.4f}, MaxDiff={diff_up:.4f}")
    
    if cos_gate < 0.99 or cos_up < 0.99:
        print("FAILED SwiGLU Correctness")
        exit(1)
        
def test_swiglu_bwd_strided():
    device = torch.device("mps")
    dtype = torch.float32
    
    # Make a larger tensor and slice it to create strides
    # [32, 2048] -> slice output
    
    B, D_full = 32, 2048
    D = 1024
    
    gate_full = torch.randn(B, D_full, device=device, dtype=dtype, requires_grad=True)
    up_full = torch.randn(B, D_full, device=device, dtype=dtype, requires_grad=True)
    dh_full = torch.randn(B, D_full, device=device, dtype=dtype)
    
    # Strided slices (every other element, or slice)
    # Let's try simple slicing first [:, :D]
    gate = gate_full[:, :D]
    up = up_full[:, :D]
    d_h = dh_full[:, :D]
    
    # Force non-contiguous if needed, though slicing usually does that
    # But MPS handles contiguous slices well.
    # Let's try Transpose! This is a real stride test.
    # [D, B].t() -> [B, D] with stride (1, D)
    
    gate_t = torch.randn(D, B, device=device, dtype=dtype, requires_grad=True).t() # [B, D], stride [1, B] ? No, stride [1, D] if standard.
    # To get non-standard stride:
    gate_t = torch.randn(D, B, device=device, dtype=dtype, requires_grad=True).transpose(0, 1) # [B, D]
    up_t = torch.randn(D, B, device=device, dtype=dtype, requires_grad=True).transpose(0, 1)
    dh_t = torch.randn(D, B, device=device, dtype=dtype).transpose(0, 1)
    
    gate_t.retain_grad()
    up_t.retain_grad()
    
    # PyTorch Ref
    y = F.silu(gate_t) * up_t
    y.backward(dh_t, retain_graph=True)
    d_gate_ref = gate_t.grad.clone()
    d_up_ref = up_t.grad.clone()
    
    # Metal
    d_gate_metal, d_up_metal = swiglu_bwd(dh_t, gate_t, up_t)
    
    print(f"\nVerifying SwiGLU Backward STRIDED (Transpose):")
    # Output should be contiguous? metalcore returns contiguous tensors
    print(f"Metal Output Stride: {d_gate_metal.stride()}")
    
    cos_gate = F.cosine_similarity(d_gate_metal.flatten(), d_gate_ref.flatten(), dim=0, eps=1e-6)
    cos_up = F.cosine_similarity(d_up_metal.flatten(), d_up_ref.flatten(), dim=0, eps=1e-6)
    
    print(f"d_gate: Cosim={cos_gate:.4f}")
    print(f"d_up:   Cosim={cos_up:.4f}")
    
    if cos_gate < 0.99 or cos_up < 0.99:
        print("FAILED Strided")
        exit(1)

if __name__ == "__main__":
    test_swiglu_bwd_correctness()
    test_swiglu_bwd_strided()
