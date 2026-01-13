import torch
import torch.nn as nn
import metalcore
import time

# Enable overrides to bring in MetalRMSNorm, etc.
metalcore.enable_pytorch_overrides(verbose=True)

class TestModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.RMSNorm(dim) # Should be MetalRMSNorm
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = nn.SiLU() # Should be metal_silu via override if input is MPS

    def forward(self, x):
        # rmsnorm
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def run_test():
    if not torch.backends.mps.is_available():
        print("MPS not available, skipping")
        return

    device = torch.device("mps")
    dim = 4096 
    batch_size = 32 # Decent size to make kernels run for a bit

    print(f"Creating model (dim={dim})...")
    model = TestModel(dim).to(device)
    
    # Check if overrides worked
    print(f"Norm type: {type(model.norm)}")
    if "MetalRMSNorm" not in str(type(model.norm)):
        print("WARNING: RMSNorm NOT overridden!")

    # Use MetalAdamW (automatically swapped if overrides enabled? 
    # overrides patches torch.optim.AdamW class)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f"Optimizer type: {type(optimizer)}")
    if "MetalAdamW" not in str(type(optimizer)):
        print("WARNING: AdamW NOT overridden!")

    x = torch.randn(batch_size, dim, device=device, requires_grad=True)
    target = torch.randn(batch_size, dim, device=device)
    
    print("Starting training loop...")
    
    for i in range(10):
        print(f"Step {i+1}")
        optimizer.zero_grad()
        
        y = model(x)
        loss = nn.functional.mse_loss(y, target)
        
        loss.backward()
        
        # Crash reportedly happens here
        optimizer.step()
        
        # Sync to force execution and catch errors
        torch.mps.synchronize()
        
    print("Test passed without crash!")

if __name__ == "__main__":
    run_test()
