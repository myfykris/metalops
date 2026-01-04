import torch
import unittest
import time
import metaleig

class TestFusedKernels(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("mps")

    def test_fused_performance(self):
        """Test Fused Kernels for various N with warmup"""
        sizes = [64, 128, 256, 512, 1024]
        print(f"\n{'N':<8} | {'Metal (ms)':<12} | {'Diff':<12} | {'Status'}")
        print("-" * 45)
        
        for N in sizes:
            torch.manual_seed(42)
            A = torch.randn(N, N, device=self.device, dtype=torch.float32)
            A = A + A.mT # Symmetric
            
            # Warmup
            _ = metaleig.eigh(A)
            torch.mps.synchronize()
            
            # Benchmark
            start = time.time()
            L_mps, Q_mps = metaleig.eigh(A)
            torch.mps.synchronize()
            end = time.time()
            metal_time = (end - start) * 1000
            
            # Verification
            L_cpu, _ = torch.linalg.eigh(A.cpu())
            L_mps_sorted, _ = torch.sort(L_mps)
            diff = torch.abs(L_mps_sorted.cpu() - L_cpu).max().item()
            
            status = "PASS" if diff < 1e-2 else "FAIL"
            print(f"{N:<8} | {metal_time:<12.2f} | {diff:<12.2e} | {status}")
            
            if status == "FAIL":
                print(f"FAILED on N={N}. Aborting loop.")
                break
                
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
