
# Configuration for metaleig

# If True, single matrix operations (non-batched) will fall back to CPU execution
# to avoid Metal driver overhead and leverage efficient CPU D&C algorithms.
# Batched operations will still run on Metal as they benefit from GPU throughput.
ENABLE_CPU_FALLBACK = True
