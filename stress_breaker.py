import torch
import metalsvd
import math

def run_fuzz():
    print("Beginning Stress Breaker Fuzzing...")
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    
    failures = []
    
    def check(name, A_gen_func):
        try:
            A = A_gen_func().to(device)
            # Forward
            U, S, V = metalsvd.svd(A)
            # Reconstruct check (skip if input was NaN/Inf as output will be too)
            if not torch.isnan(A).any() and not torch.isinf(A).any():
                 A_recon = U @ torch.diag_embed(S) @ V.transpose(-2, -1)
                 err = (A - A_recon).norm()
                 if torch.isnan(err):
                     failures.append(f"{name}: Result contained NaNs")
                 elif err > 1e-3: # loose tolerance for stress
                     failures.append(f"{name}: Reconstruction Error {err.item():.2e}")
            else:
                # If input has NaNs, just ensure no crash
                pass
            print(f"  [PASS] {name}")
        except Exception as e:
            failures.append(f"{name}: CRASHED - {str(e)}")
            print(f"  [FAIL] {name} - {str(e)}")

    # 1. Zero Matrix
    check("Zero Matrix (64x64)", lambda: torch.zeros(64, 64))
    
    # 2. NaN Input
    def gen_nan():
        A = torch.randn(32, 32)
        A[0, 0] = float('nan')
        return A
    check("NaN Input (32x32)", gen_nan)
    
    # 3. Inf Input
    def gen_inf():
        A = torch.randn(32, 32)
        A[0, 0] = float('inf')
        return A
    check("Inf Input (32x32)", gen_inf)
    
    # 4. Tiny Matrix
    check("Tiny (2x2)", lambda: torch.randn(2, 2))
    
    # 5. Skinny (100x2)
    check("Skinny (100x2)", lambda: torch.randn(100, 2))
    
    # 6. Wide (2x100)
    check("Wide (2x100)", lambda: torch.randn(2, 100))
    
    # 7. Rank Deficient (Cols are scale of each other)
    def gen_rank_def():
        A = torch.randn(32, 1)
        return A.expand(32, 32).clone()
    check("Rank 1 Matrix (32x32)", gen_rank_def)
    
    # 8. Non-contiguous
    def gen_non_cont():
        A = torch.randn(64, 64)
        return A.t() # Transpose makes it non-contiguous storage
    check("Non-contiguous Input", gen_non_cont)
    
    # 9. Duplicate Singular Values (Identity)
    # Hard case for Jacobi convergence?
    check("Identity (32x32)", lambda: torch.eye(32))
    
    # 10. High Condition Number
    def gen_ill_cond():
        U, _, V = torch.svd(torch.randn(32, 32))
        S = torch.logspace(0, -5, 32) # 1e0 to 1e-5
        return U @ torch.diag(S) @ V.t()
    check("Ill-Conditioned (1e5)", gen_ill_cond)
    
    print("\n" + "="*30)
    if failures:
        print(f"FAILED {len(failures)} TESTS:")
        for f in failures:
            print(f" - {f}")
    else:
        print("ALL STRESS TESTS PASSED.")

if __name__ == "__main__":
    run_fuzz()
