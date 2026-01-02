#include <metal_stdlib>
using namespace metal;

constant float EPSILON = 1e-6;

// -----------------------------------------------------------------------------
// Helper: SIMD Reduction
// -----------------------------------------------------------------------------
inline float simd_reduction(float val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return val;
}

// -----------------------------------------------------------------------------
// Kernel: Optimized Jacobi Rotation
// -----------------------------------------------------------------------------
// Design: 
// - Each Threadgroup handles ONE Pair (i, j).
// - Threads in the group parallelize the dot product and update.
// - Assumption: M is large enough to warrant this.
//
// Shared Memory:
// - Used to reduce partial dot products across SIMD groups.

kernel void jacobi_rotate_kernel_optimized(
    device float* A_T [[buffer(0)]],
    device float* V_T [[buffer(1)]],
    device const int2* Pairs [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& BatchStrideA [[buffer(5)]],
    constant uint& BatchStrideV [[buffer(6)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]], // pair_idx
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // 1. Identify Pair and Batch
    // Need to decode grid. 
    // Dispatch: (NumPairs, Batch, 1) -> No, Metal limits grid dims.
    // Let's assume Dispatch is 1D or 2D.
    // Let's preserve the Logic:
    // Grid: (NumPairs, 1, Batch)
    
    int pair_idx = group_id; 
    int batch_idx = gid.z; // If dispatch is 3D (NumPairs, 1, Batch)
    
    int2 pair = Pairs[pair_idx];
    int i = pair.x;
    int j = pair.y;
    
    uint batch_offset_A = batch_idx * BatchStrideA;
    uint batch_offset_V = batch_idx * BatchStrideV;
    
    device float* col_i = A_T + batch_offset_A + i * M;
    device float* col_j = A_T + batch_offset_A + j * M;
    
    // 2. Parallel Dot Product
    float part_ii = 0.0f;
    float part_jj = 0.0f;
    float part_ij = 0.0f;
    
    // Grid stride loop
    for (uint k = tid; k < M; k += threads_per_group) {
        float val_i = col_i[k];
        float val_j = col_j[k];
        part_ii += val_i * val_i;
        part_jj += val_j * val_j;
        part_ij += val_i * val_j;
    }
    
    // SIMD reduction
    part_ii = simd_reduction(part_ii);
    part_jj = simd_reduction(part_jj);
    part_ij = simd_reduction(part_ij);
    
    // First thread of each SIMD group writes to shared
    if (simd_lane_id == 0) {
        shared_mem[simd_group_id * 3 + 0] = part_ii;
        shared_mem[simd_group_id * 3 + 1] = part_jj;
        shared_mem[simd_group_id * 3 + 2] = part_ij;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // First thread of group sums up shared mem
    // Assuming < 32 SIMD groups (1024 threads / 32 = 32). 
    // Standard threadgroup size is usually < 1024.
    
    if (tid == 0) {
        float sum_ii = 0.0f;
        float sum_jj = 0.0f;
        float sum_ij = 0.0f;
        
        uint num_simd_groups = (threads_per_group + 31) / 32;
        for (uint s = 0; s < num_simd_groups; ++s) {
            sum_ii += shared_mem[s * 3 + 0];
            sum_jj += shared_mem[s * 3 + 1];
            sum_ij += shared_mem[s * 3 + 2];
        }
        
        // 3. Compute Rotation (c, s)
        float c = 1.0f, s = 0.0f;
        if (abs(sum_ij) > EPSILON) {
            float tau = (sum_jj - sum_ii) / (2.0f * sum_ij);
            float t;
            float sqrt_term = sqrt(1.0f + tau * tau);
            if (tau >= 0.0f) {
                t = 1.0f / (tau + sqrt_term);
            } else {
                t = -1.0f / (-tau + sqrt_term);
            }
            c = 1.0f / sqrt(1.0f + t * t);
            s = t * c;
        }
        
        // Store c, s in shared mem for broadcast
        shared_mem[0] = c;
        shared_mem[1] = s;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float c = shared_mem[0];
    float s = shared_mem[1];
    
    // Optimization: If small rotation, skip update?
    // Jacobi usually updates always.
    
    // 4. Parallel Update A
    for (uint k = tid; k < M; k += threads_per_group) {
        float val_i = col_i[k];
        float val_j = col_j[k];
        col_i[k] = c * val_i - s * val_j;
        col_j[k] = s * val_i + c * val_j;
    }
    
    // 5. Parallel Update V (N length)
    // Careful: Threadgroup size might be larger than N (e.g. N=110, threads=256)
    // Or smaller. Loop handles it.
    device float* v_col_i = V_T + batch_offset_V + i * N;
    device float* v_col_j = V_T + batch_offset_V + j * N;
    
    // Independent loop for V
    for (uint k = tid; k < N; k += threads_per_group) {
        float val_vi = v_col_i[k];
        float val_vj = v_col_j[k];
        v_col_i[k] = c * val_vi - s * val_vj;
        v_col_j[k] = s * val_vi + c * val_vj;
    }
}
