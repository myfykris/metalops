# Configuration for metalcore

# If True, single matrix QR will fall back to CPU execution
# (CPU LAPACK is 3-50x faster for single matrix QR due to sequential dependencies).
# Batched QR operations will still run on Metal.
# Set to False to force all operations to use GPU kernels.
ENABLE_CPU_FALLBACK = True

# SVD Configuration
# Threshold for switching to Gram matrix strategy (use for N >= this value)
GRAM_THRESHOLD = 512

# Enable De Rijk column sorting optimization for SVD
ENABLE_DE_RIJK_OPT = True
