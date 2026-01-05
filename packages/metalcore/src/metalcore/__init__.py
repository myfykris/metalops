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
"""

from .trsm import trsm, trsm_batched
from .householder import householder_vector, apply_householder
from .qr import qr
from .lstsq import lstsq
from .pinv import pinv
from .solve import solve
from .cholesky import cholesky, cholesky_solve
from .svd import svd
from .eigh import eigh

__version__ = "0.0.1"
__all__ = [
    "trsm",
    "trsm_batched", 
    "householder_vector",
    "apply_householder",
    "qr",
    "lstsq",
    "pinv",
    "solve",
    "cholesky",
    "cholesky_solve",
    "svd",
    "eigh",
]
