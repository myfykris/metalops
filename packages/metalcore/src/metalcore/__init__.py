"""
metalcore: Foundational Metal Linear Algebra Primitives

This package provides GPU-accelerated building blocks for linear algebra:
- trsm: Triangular solve
- householder: Householder reflections
- qr: QR decomposition
- lstsq: Least squares solver
- pinv: Pseudo-inverse
- solve: Linear system solver
- cholesky: Cholesky decomposition
- cholesky_solve: Cholesky-based solve
- svd: Singular Value Decomposition
- eigh: Eigendecomposition (symmetric)
- High-impact: lu_batched, syrk_batched, frobenius_norm_batched, softmax_batched, trace_batched
"""

from .trsm import trsm, trsm_batched
from .householder import householder_vector, apply_householder
from .qr import qr, qr_solve
from .lstsq import lstsq
from .pinv import pinv
from .solve import solve
from .cholesky import cholesky, cholesky_solve
from .svd import svd
from .eigh import eigh

# High-impact ops (direct backend exports)
try:
    import metalcore_backend as _mc
    lu_batched = _mc.lu_batched
    syrk_batched = _mc.syrk_batched
    frobenius_norm_batched = _mc.frobenius_norm_batched
    softmax_batched = _mc.softmax_batched
    trace_batched = _mc.trace_batched
except ImportError:
    pass

__version__ = "0.0.1"
__all__ = [
    "trsm",
    "trsm_batched", 
    "householder_vector",
    "apply_householder",
    "qr",
    "qr_solve",
    "lstsq",
    "pinv",
    "solve",
    "cholesky",
    "cholesky_solve",
    "svd",
    "eigh",
    # High-impact ops
    "lu_batched",
    "syrk_batched",
    "frobenius_norm_batched",
    "softmax_batched",
    "trace_batched",
]
