# Configuration for metalsvd

# If True, single matrix operations (non-batched) will fall back to CPU execution
# to avoid Metal driver overhead and leverage efficient CPU algorithms (D&C).
# Batched operations will still run on Metal as they benefit from GPU throughput.
ENABLE_CPU_FALLBACK = True

# If True, enables De Rijk strategy: pre-sorting columns by norm descending.
# This concentrates "energy" in the top-left, improving Jacobi convergence speed/accuracy.
ENABLE_DE_RIJK_OPT = True

# Minimum N to use Gram strategy (A^T @ A + eigh) instead of Jacobi.
# Gram is faster for large matrices. Default 512 gives ~1.5x speedup.
# Set to 1024 for marginally better precision (~1e-4 vs ~5e-5 error).
GRAM_THRESHOLD = 512
