"""
Comprehensive tests for metalcore package.
Tests correctness, edge cases, and error handling.
"""

import torch
import sys
sys.path.insert(0, 'packages/metalcore/src')
import metalcore


def test_trsm_lower_basic():
    """Test basic lower triangular solve."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    L = torch.tensor([
        [2.0, 0.0, 0.0],
        [1.0, 3.0, 0.0],
        [2.0, 1.0, 4.0]
    ], device=device)
    b = torch.tensor([4.0, 7.0, 18.0], device=device)
    
    x = metalcore.trsm(L, b, lower=True)
    residual = torch.max(torch.abs(L @ x - b)).item()
    
    assert residual < 1e-6, f"Residual too high: {residual}"


def test_trsm_upper_basic():
    """Test basic upper triangular solve."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    U = torch.tensor([
        [2.0, 1.0, 3.0],
        [0.0, 3.0, 2.0],
        [0.0, 0.0, 4.0]
    ], device=device)
    b = torch.tensor([10.0, 11.0, 8.0], device=device)
    
    x = metalcore.trsm(U, b, lower=False)
    residual = torch.max(torch.abs(U @ x - b)).item()
    
    assert residual < 1e-6, f"Residual too high: {residual}"


def test_trsm_well_conditioned():
    """Test trsm with well-conditioned matrix."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    N = 100
    L = torch.tril(torch.randn(N, N, device=device))
    L += torch.eye(N, device=device) * N  # Strong diagonal
    b = torch.randn(N, device=device)
    
    x = metalcore.trsm(L, b, lower=True)
    residual = torch.max(torch.abs(L @ x - b)).item()
    
    assert residual < 1e-4, f"Residual too high: {residual}"


def test_trsm_batched():
    """Test batched triangular solve."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    batch = 8
    N = 20
    L = torch.tril(torch.randn(batch, N, N, device=device))
    L += torch.eye(N, device=device).unsqueeze(0) * N
    b = torch.randn(batch, N, device=device)
    
    # This should work (batched case)
    x = metalcore.trsm(L, b.unsqueeze(-1), lower=True)
    
    # Check each batch
    for i in range(batch):
        residual = torch.max(torch.abs(L[i] @ x[i].squeeze() - b[i])).item()
        assert residual < 1e-4, f"Batch {i} residual too high: {residual}"


def test_qr_basic():
    """Test basic QR decomposition."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    M, N = 10, 5
    A = torch.randn(M, N, device=device)
    
    Q, R = metalcore.qr(A, mode='reduced')
    
    # Check shapes
    assert Q.shape == (M, N), f"Q shape wrong: {Q.shape}"
    assert R.shape == (N, N), f"R shape wrong: {R.shape}"
    
    # Check reconstruction
    recon_err = torch.max(torch.abs(Q @ R - A)).item()
    assert recon_err < 1e-5, f"Reconstruction error too high: {recon_err}"
    
    # Check orthogonality
    orth_err = torch.max(torch.abs(Q.T @ Q - torch.eye(N, device=device))).item()
    assert orth_err < 1e-5, f"Orthogonality error too high: {orth_err}"
    
    # Check R is upper triangular
    tril_err = torch.max(torch.abs(torch.tril(R, -1))).item()
    assert tril_err < 1e-10, f"R is not upper triangular: {tril_err}"


def test_qr_square():
    """Test QR on square matrix."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    N = 20
    A = torch.randn(N, N, device=device)
    
    Q, R = metalcore.qr(A, mode='reduced')
    
    recon_err = torch.max(torch.abs(Q @ R - A)).item()
    assert recon_err < 1e-5, f"Reconstruction error: {recon_err}"


def test_qr_complete_mode():
    """Test QR with complete mode."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    M, N = 10, 5
    A = torch.randn(M, N, device=device)
    
    Q, R = metalcore.qr(A, mode='complete')
    
    # Check shapes
    assert Q.shape == (M, M), f"Q shape wrong: {Q.shape}"
    assert R.shape == (M, N), f"R shape wrong: {R.shape}"


def test_qr_r_only():
    """Test QR with R-only mode."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    M, N = 10, 5
    A = torch.randn(M, N, device=device)
    
    R = metalcore.qr(A, mode='r')
    
    # Check shape
    assert R.shape == (N, N), f"R shape wrong: {R.shape}"
    
    # Check R is upper triangular
    tril_err = torch.max(torch.abs(torch.tril(R, -1))).item()
    assert tril_err < 1e-10, f"R is not upper triangular"


def test_qr_near_singular():
    """Test QR on near-singular matrix."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    M, N = 10, 5
    # Create rank-deficient matrix
    A = torch.randn(M, 3, device=device) @ torch.randn(3, N, device=device)
    
    Q, R = metalcore.qr(A, mode='reduced')
    
    # Should still complete without error
    recon_err = torch.max(torch.abs(Q @ R - A)).item()
    # Error may be higher for ill-conditioned matrices
    assert recon_err < 1e-3, f"Reconstruction error: {recon_err}"


def test_householder_basic():
    """Test Householder vector computation."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    x = torch.randn(10, device=device)
    v, tau = metalcore.householder_vector(x)
    
    # v[0] should be 1 (normalized form)
    assert abs(v[0].item() - 1.0) < 1e-6, f"v[0] should be 1, got {v[0]}"
    
    # Apply H to x, should get [-norm(x), 0, 0, ...]
    # H @ x = x - tau * v * (v.T @ x)
    norm_x = torch.norm(x).item()
    H_x = x - tau * v * torch.dot(v, x)
    
    # Check H_x ≈ [±norm(x), 0, 0, ...]
    assert abs(abs(H_x[0].item()) - norm_x) < 1e-5, f"H_x[0] should be ±norm(x)"
    assert torch.max(torch.abs(H_x[1:])).item() < 1e-5, "H_x[1:] should be 0"


def test_householder_zero_vector():
    """Test Householder with zero vector."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    x = torch.zeros(10, device=device)
    v, tau = metalcore.householder_vector(x)
    
    # tau should be 0 for zero vector
    assert tau.item() < 1e-6, f"tau should be 0 for zero vector"


def test_householder_already_e1():
    """Test Householder when x is already along e_1."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    x = torch.zeros(10, device=device)
    x[0] = 5.0
    
    v, tau = metalcore.householder_vector(x)
    
    # tau should be 0 or very small (no reflection needed if H @ x = +norm(x)*e1)
    # Actually, if sign(x[0])=+1, we get H@x = -norm(x)*e1, so tau != 0
    # But the result should still be correct
    H_x = x - tau * v * torch.dot(v, x)
    
    assert abs(abs(H_x[0].item()) - 5.0) < 1e-5


def test_qr_solve_basic():
    """Test QR-based least squares solve."""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    M, N = 20, 10
    A = torch.randn(M, N, device=device)
    # Make it well-conditioned
    A = A + 0.1 * torch.eye(M, N, device=device)
    
    b = torch.randn(M, device=device)
    
    x = metalcore.qr_solve(A, b)
    
    # Check against torch.linalg.lstsq
    x_ref = torch.linalg.lstsq(A.cpu(), b.cpu()).solution
    
    error = torch.max(torch.abs(x.cpu() - x_ref)).item()
    assert error < 1e-4, f"Solution error: {error}"


def run_all_tests():
    """Run all tests and report results."""
    import traceback
    
    tests = [
        test_trsm_lower_basic,
        test_trsm_upper_basic,
        test_trsm_well_conditioned,
        # test_trsm_batched,  # Skip batched for now
        test_qr_basic,
        test_qr_square,
        test_qr_complete_mode,
        test_qr_r_only,
        test_qr_near_singular,
        test_householder_basic,
        test_householder_zero_vector,
        test_householder_already_e1,
        test_qr_solve_basic,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
