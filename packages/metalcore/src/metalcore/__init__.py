"""
metalcore: Foundational Metal Linear Algebra Primitives

This package provides GPU-accelerated building blocks for linear algebra:
- trsm: Triangular solve
- householder: Householder reflections
- qr: QR decomposition
"""

from .trsm import trsm, trsm_batched
from .householder import householder_vector, apply_householder
from .qr import qr

__version__ = "0.0.1"
__all__ = [
    "trsm",
    "trsm_batched", 
    "householder_vector",
    "apply_householder",
    "qr",
]
