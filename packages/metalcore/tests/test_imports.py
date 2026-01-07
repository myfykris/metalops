"""
=============================================================================
METALCORE TEST SUITE - Import Tests Only (for CI)
=============================================================================

IMPORTANT NOTES FOR FUTURE DEVELOPMENT:
---------------------------------------

1. GitHub Actions CI runs on macos-latest which does NOT have MPS (Metal 
   Performance Shaders) available. MPS requires Apple Silicon or a 
   compatible GPU.

2. Therefore, ALL tests in this directory should be IMPORT-ONLY tests.
   They verify that Python files parse correctly and classes/functions
   are exported, but do NOT execute any Metal kernel code.

3. For actual functionality testing (kernels, optimizers, etc.), run
   tests LOCALLY on Apple Silicon Mac with:
   
       pytest tests/  # Local with MPS
       python adamw_stress_test.py --quick  # Stress test AdamW
       python benchmark.py --quick  # Benchmark all ops

4. If you need to add MPS-dependent tests in the future, they should:
   - Be in a separate file (e.g., test_mps_*.py)
   - Use @pytest.mark.skipif(not torch.backends.mps.is_available(), ...)
   - NOT be run in CI (add to pytest.ini or conftest.py exclusions)

5. The CI workflow (.github/workflows/ci.yml) runs:
       pytest  # Runs only tests in this directory
   
   It does NOT have access to Metal kernels, so any test calling
   Metal operations will fail with errors like:
   - "kernel not available for dtype Float"
   - "Could not find .metallib"

=============================================================================
"""

import pytest


def test_metalcore_import():
    """Test that metalcore package imports without error."""
    import metalcore
    assert hasattr(metalcore, '__version__')


def test_submodule_imports():
    """Test that all submodules import without error."""
    from metalcore import rmsnorm
    from metalcore import optim
    from metalcore import qr
    from metalcore import activations
    from metalcore import ops
    

def test_classes_exist():
    """Test that main classes are accessible."""
    from metalcore import MetalRMSNorm
    from metalcore import MetalAdamW
    from metalcore import MetalSoftmax
    from metalcore import MetalLayerNorm
    

def test_function_exports():
    """Test that main functions are exported."""
    from metalcore import fused_softmax
    from metalcore import layer_norm
    from metalcore import embedding_bag
    from metalcore import gather
    from metalcore import scatter_add
    from metalcore import index_select


def test_version_format():
    """Test that version string is properly formatted."""
    import metalcore
    version = metalcore.__version__
    assert isinstance(version, str)
    parts = version.split('.')
    assert len(parts) >= 2, "Version should be X.Y or X.Y.Z format"


def test_has_metal_kernels_check():
    """Test that has_metal_kernels() helper exists and returns bool."""
    from metalcore import has_metal_kernels
    result = has_metal_kernels()
    assert isinstance(result, bool)
    # Note: This will return False in CI (no Metal), True locally on Apple Silicon
