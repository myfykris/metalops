# MetalSVD Specialized Kernel Architecture

## Strategy Overview
To achieve maximum performance on Apple Silicon GPUs (M1/M2/M3), MetalSVD employs a strategy of **Specialized Kernels**. Instead of a single "one-size-fits-all" kernel with complex runtime branching, we generate highly tuned kernels for specific matrix sizes ($N$). This allows us to:
1.  **Maximize Parallelism**: Hardcode `ThreadsPerPair` (TPP) to fully utilize the 1024-thread limit per threadgroup.
2.  **Minimize Registers**: Remove dynamic loops and branching, reducing register pressure and increasing occupancy.
3.  **Optimize Latency**: Use local threadgroup barriers (`mem_threadgroup`) instead of global device barriers where possible.

## Kernel Variants
We aim to cover **90%+ of workload cases** with the following specialized kernels. All kernels operate on a single threadgroup per matrix (Batch=1 per group), ensuring data locality.

### 1. Small / Batched Matrices ($N \le 128$) (Current Production)
*   **Variant**: `svd_fused_block_kernel_generic`
*   **Target**: Attention Heads, LoRA adapters, small batches.
*   **Configuration**:
    *   `ThreadsPerPair`: 32 (Full Warp) if N=64; TPP=16 if N=128.
    *   `Pairs Processed`: N/2
    *   `Total Threads`: Pairs * TPP <= 1024.

### 2. Medium Matrices ($N = 256$) (Implemented)
*   **Variant**: `svd_fused_block_kernel_256`
*   **Target**: Feature projections, small embeddings.
*   **Configuration**:
    *   `ThreadsPerPair`: 8
    *   `Pairs Processed`: 128
    *   `Total Threads`: 128 * 8 = 1024 (Exact Hardware Limit)
    *   **Status**: Working. ~37ms latency (Single), high throughput scale.

### 3. Large Matrices ($N = 512$) (Proposed)
*   **Variant**: `svd_fused_block_kernel_512`
*   **Target**: Standard embeddings (BERT-base output), medium layers.
*   **Configuration**:
    *   `ThreadsPerPair`: 4
    *   `Pairs Processed`: 256
    *   `Total Threads`: 256 * 4 = 1024 (Exact Hardware Limit)

### 4. X-Large Matrices ($N = 1024$) (Proposed)
*   **Variant**: `svd_fused_block_kernel_1024`
*   **Target**: Large embeddings (roberta-large), hidden layers.
*   **Configuration**:
    *   `ThreadsPerPair`: 2
    *   `Pairs Processed`: 512
    *   `Total Threads`: 512 * 2 = 1024 (Exact Hardware Limit)

### 5. Massive Matrices ($N = 2048$) (Proposed)
*   **Variant**: `svd_fused_block_kernel_2048`
*   **Target**: LLM hidden states (7B param).
*   **Configuration**:
    *   `ThreadsPerPair`: 1
    *   `Pairs Processed`: 1024
    *   `Total Threads`: 1024 * 1 = 1024 (Exact Hardware Limit)
    *   **Note**: This is the limit for the single-block "Fused" arch. N > 2048 requires Block Jacobi (Sub-blocks).

## Implementation Checklist
- [x] Refactor text-based macros to clean `DEFINE_*` macros.
- [x] Implement `svd_fused_256` (TPP=8).
- [ ] Implement `svd_fused_512` (TPP=4).
- [ ] Implement `svd_fused_1024` (TPP=2).
- [ ] Implement `svd_fused_2048` (TPP=1).
- [ ] Update `svd_mps.mm` dispatch logic map `N` to specific PSO.
