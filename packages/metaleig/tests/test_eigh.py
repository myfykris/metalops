import torch
import unittest
import metaleig

class TestEigh(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("mps")
        
    def test_eigh_random_symmetric(self):
        N = 128
        torch.manual_seed(42)
        A = torch.randn(N, N, device=self.device)
        A = A + A.mT # Make symmetric
        
        L_metal, Q_metal = metaleig.eigh(A)
        
        # CPU Ground Truth
        L_cpu, Q_cpu = torch.linalg.eigh(A.cpu())
        
        # Verify Eigenvalues
        # Sort both just in case (eigh returns sorted, but metal might have slight drift? No, sort is required)
        # Metal implementation returns L from dot product. Is it sorted?
        # The Jacobi method converges to diagonal. The diagonal elements are not necessarily sorted.
        # So we must sort L_metal and Q_metal.
        
        L_metal_sorted, indices = torch.sort(L_metal)
        Q_metal_sorted = Q_metal[:, indices]
        
        # Compare L
        diff_L = (L_metal_sorted.cpu() - L_cpu).abs().max()
        print(f"Max Eigenvalue Diff: {diff_L.item()}")
        self.assertTrue(diff_L < 1e-4)
        
        # Verify Reconstruction: A = Q L Q.T
        A_recon = Q_metal_sorted @ torch.diag(L_metal_sorted) @ Q_metal_sorted.mT
        diff_recon = (A - A_recon).abs().max()
        print(f"Reconstruction Diff: {diff_recon.item()}")
        self.assertTrue(diff_recon < 1e-3)

    def test_eigvalsh(self):
        N = 64
        torch.manual_seed(123)
        A = torch.randn(N, N, device=self.device)
        A = A + A.mT
        
        L = metaleig.eigvalsh(A)
        L_cpu = torch.linalg.eigvalsh(A.cpu())
        
        L_sorted, _ = torch.sort(L)
        diff = (L_sorted.cpu() - L_cpu).abs().max()
        print(f"Eigvalsh Diff: {diff.item()}")
        self.assertTrue(diff < 1e-4)

if __name__ == '__main__':
    unittest.main()
