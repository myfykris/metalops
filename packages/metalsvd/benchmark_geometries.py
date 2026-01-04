import torch
import metalsvd
import time
import math

def run_case(M, N, label):
    print(f"\n--- Benchmarking {label} ({M}x{N}) ---")
    device = torch.device('mps')
    
    # 1. Setup
    try:
        A = torch.randn(M, N, device=device).to(torch.float32)
        A_cpu = A.float().cpu()
    except Exception as e:
        print(f"Skipping {label}: OOM or Error creating initialization tensor: {e}")
        return

    # Strategies to test
    strategies = [
        ("CPU Baseline", "cpu"),
        ("Metal SVD", "metal"),
    ]
    
    results = []
    
    for strategy_name, mode in strategies:
        print(f"Running {strategy_name}...")
        torch.mps.synchronize()
        start = time.time()
        
        try:
            if mode == "cpu":
                torch.linalg.svd(A_cpu, full_matrices=False)
                dur = time.time() - start
                
                # Validation (CPU is gold standard, skip check or check against itself?)
                # We assume CPU is valid.
                recon_err = 0.0
                is_valid = True
                
            else:
                # Metal Modes - use default SVD
                U, S, V = metalsvd.svd(A)
                torch.mps.synchronize()
                dur = time.time() - start
                
                # Validation
                U_f = U.float().cpu()
                S_f = S.float().cpu()
                V_f = V.float().cpu()
                
                # metalsvd returns V, not Vt. So A ≈ U @ diag(S) @ V.T
                A_rec = U_f @ torch.diag(S_f) @ V_f.T
                start_check = time.time()
                recon_err = torch.norm(A_cpu - A_rec) / torch.norm(A_cpu)
                ortho_err = torch.norm(U_f.T @ U_f - torch.eye(U_f.shape[1])) / math.sqrt(U_f.shape[1])
                
                is_valid = recon_err < 0.05 # 5% tolerance
                if not is_valid:
                    print(f"  WARNING: High Error {recon_err:.6f}")

            print(f"  Time: {dur:.4f}s | Valid: {is_valid} | Err: {recon_err:.6f}")
            results.append((strategy_name, dur, is_valid, recon_err))
            
        except Exception as e:
            print(f"  Failed: {e}")
            results.append((strategy_name, float('inf'), False, None))

    # Return results for this geometry
    return {
        "geometry": label,
        "shape": f"{M}x{N}",
        "results": results
    }

def benchmark_geometries():
    cases = [
        (4096, 14336, "Huge Fat"),
        (14336, 4096, "Huge Tall"),
        (1024, 4096, "Medium Fat"),
        (4096, 1024, "Medium Tall"),
        (768, 3072, "Small Fat"),
        (3072, 768, "Small Tall"),
        # Edge cases for heuristic (1.5x)
        (2048, 3000, "Borderline (1.46x)"),
        (2048, 3200, "Borderline (1.56x)"),
        # Square Matrices (Should hit Standard Kernel)
        (1024, 1024, "Square Small"),
        (2048, 2048, "Square Medium"),
        (4096, 4096, "Square Large")
    ]
    
    all_data = []
    
    for M, N, label in cases:
        data = run_case(M, N, label)
        if data:
            all_data.append(data)

    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT")
    print("="*80)
    
    # Define columns
    headers = f"| {'Geometry':<20} | {'Shape':<15} | {'CPU (s / Err)':<20} | {'Metal (s / Err)':<20} | {'Speedup':<10} |"
    print(headers)
    print("|" + "-"*22 + "|" + "-"*17 + "|" + "-"*22 + "|" + "-"*22 + "|" + "-"*12 + "|")
    
    for item in all_data:
        geo = item['geometry']
        shape = item['shape']
        
        # Extract metrics
        res_map = {r[0]: (r[1], r[2], r[3]) for r in item['results']} 
        
        def fmt(name):
            if name not in res_map: return "N/A", None
            dur, valid, err = res_map[name]
            if dur == float('inf'): return "FAIL", None
            err_str = f"{err:.4f}" if err is not None else "?"
            return f"{dur:>6.2f}s ({err_str})", dur

        cpu_str, cpu_dur = fmt("CPU Baseline")
        metal_str, metal_dur = fmt("Metal SVD")
        
        if cpu_dur and metal_dur and metal_dur > 0:
            speedup = f"{cpu_dur / metal_dur:.2f}x"
        else:
            speedup = "N/A"
        
        print(f"| {geo:<20} | {shape:<15} | {cpu_str:<20} | {metal_str:<20} | {speedup:<10} |")
    print("="*80)

    print("="*80)

def warmup_kernels():
    print("Warming up Metal kernels (compiling PSOs)...")
    device = torch.device('mps')
    
    # 1. Scalar Kernel (Small Matrix)
    try:
        A = torch.randn(256, 256, device=device).to(torch.float32)
        metalsvd.svd(A, strategy='standard')
    except: pass
    
    # 2. Vectorized Kernel (Aligned 4096)
    try:
        A = torch.randn(4096, 4096, device=device).to(torch.float32)
        # Run just 1 sweep or use internal API? svd() will run full convergence. 
        # It's fine, 4096 square takes ~18s in Standard. That's too long for warmup.
        # But user wants "absolutely sure". 
        # Maybe use a smaller vectorized size: 512x512 (divisible by 4)
        A_small = torch.randn(512, 512, device=device).to(torch.float32)
        metalsvd.svd(A_small, strategy='standard')
    except: pass
    
    # 3. Hybrid Kernel (Gram Path)
    try:
        A = torch.randn(1024, 2048, device=device).to(torch.float32)
        metalsvd.svd(A, strategy='gram')
    except: pass
    
    torch.mps.synchronize()
    print("Warmup complete.")

if __name__ == "__main__":
    warmup_kernels()
    benchmark_geometries()
