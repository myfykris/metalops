// metalcore Metal kernels
// Foundational linear algebra primitives
//
// Based on LAPACK/rocSOLVER/MAGMA algorithms:
// - geqr2: Unblocked Householder QR for panels
// - geqr2_fused: MAGMA-style fused panel QR with shared memory caching
// - larfb: Apply block Householder reflector (the money kernel)
// - trsm: Triangular solve

#include <metal_stdlib>
using namespace metal;

// Maximum panel dimensions for shared memory caching
// Adjust based on GPU shared memory limits (32KB typical)
constant uint MAX_PANEL_M = 512;
constant uint MAX_PANEL_N = 32;

// ============================================================================
// SIMD Helper Functions
// ============================================================================

inline float simd_sum(float val, uint simd_lane, uint simd_size) {
    // Tree reduction within SIMD group
    for (uint offset = simd_size / 2; offset > 0; offset >>= 1) {
        val += simd_shuffle_down(val, offset);
    }
    return val;
}

// ============================================================================
// Fused Panel QR (MAGMA-style with shared memory caching)
// ============================================================================
// Key optimization: Load panel once, process entirely in shared memory, write once
// This eliminates global memory traffic during Householder iterations

kernel void geqr2_fused_kernel(
    device float* A [[buffer(0)]],              // (M, N) panel, row-major
    device float* tau [[buffer(1)]],            // (min(M,N),) output tau values
    constant uint& M [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& lda [[buffer(4)]],
    threadgroup float* shared_panel [[threadgroup(0)]],  // M*N floats for panel
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Panel dimensions
    uint K = min(M, N);
    
    // Shared memory layout:
    // shared_panel[0 : M*N] = panel data (row-major: element [i,j] at i*N + j)
    // shared_tau (in registers) = tau values
    threadgroup float* panel = shared_panel;
    
    // Step 1: Load entire panel from global memory into shared memory
    uint total_elements = M * N;
    for (uint idx = tid; idx < total_elements; idx += tg_size) {
        uint row = idx / N;
        uint col = idx % N;
        panel[idx] = A[row * lda + col];  // A[row, col] row-major
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 2: Process columns (Householder QR in shared memory)
    for (uint k = 0; k < K; k++) {
        uint v_len = M - k;
        
        // 2a: Compute squared norm of column k below diagonal
        // Use parallel reduction with SIMD
        float local_sigma = 0.0f;
        for (uint i = tid + 1; i < v_len; i += tg_size) {
            float val = panel[(k + i) * N + k];  // panel[k+i, k]
            local_sigma += val * val;
        }
        
        // Reduce within threadgroup using shared memory
        threadgroup float reduction_buf[256];
        reduction_buf[tid] = local_sigma;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                reduction_buf[tid] += reduction_buf[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        float sigma = reduction_buf[0];
        float x0 = panel[k * N + k];  // panel[k, k]
        
        // 2b: Compute tau and update diagonal
        float tau_k = 0.0f;
        float v0_factor = 1.0f;
        
        if (sigma > 1e-10f) {
            float norm_x = sqrt(x0 * x0 + sigma);
            float sign = (x0 >= 0.0f) ? 1.0f : -1.0f;
            float v0 = x0 + sign * norm_x;
            tau_k = 2.0f * v0 * v0 / (v0 * v0 + sigma);
            v0_factor = 1.0f / v0;
            
            // Update diagonal
            if (tid == 0) {
                panel[k * N + k] = -sign * norm_x;
            }
        }
        
        if (tid == 0) {
            tau[k] = tau_k;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tau_k == 0.0f) continue;
        
        // 2c: Apply Householder to remaining columns (j = k+1 to N-1)
        // Each thread handles a subset of (row, column) pairs
        // v[0] = 1 (implicit), v[i] = panel[k+i, k] * v0_factor for i > 0
        
        for (uint j = k + 1; j < N; j++) {
            // Compute v^T @ panel[k:, j]
            float local_dot = 0.0f;
            
            // Thread 0 handles the implicit v[0] = 1
            if (tid == 0) {
                local_dot = panel[k * N + j];  // v[0] * panel[k, j] = 1 * panel[k,j]
            }
            
            // Other elements
            for (uint i = tid + 1; i < v_len; i += tg_size) {
                float v_i = panel[(k + i) * N + k] * v0_factor;
                local_dot += v_i * panel[(k + i) * N + j];
            }
            
            reduction_buf[tid] = local_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint s = tg_size / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    reduction_buf[tid] += reduction_buf[tid + s];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            float vT_a = reduction_buf[0];
            
            // Update panel[k:, j] -= tau * v * vT_a
            if (tid == 0) {
                panel[k * N + j] -= tau_k * vT_a;  // v[0] = 1
            }
            for (uint i = tid + 1; i < v_len; i += tg_size) {
                float v_i = panel[(k + i) * N + k] * v0_factor;
                panel[(k + i) * N + j] -= tau_k * v_i * vT_a;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // 2d: Normalize v (store v/v0 below diagonal)
        for (uint i = tid + 1; i < v_len; i += tg_size) {
            panel[(k + i) * N + k] *= v0_factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Step 3: Write panel back to global memory
    for (uint idx = tid; idx < total_elements; idx += tg_size) {
        uint row = idx / N;
        uint col = idx % N;
        A[row * lda + col] = panel[idx];
    }
}

// ============================================================================
// Householder Vector Computation
// ============================================================================

// Compute Householder vector v and tau for a single column
// After this: H @ x = -sign(x[0]) * ||x|| * e_1
// v is stored with v[0] = 1 (implicit), tau returned separately
kernel void householder_vector_kernel(
    device const float* x [[buffer(0)]],        // (N,) input column
    device float* v [[buffer(1)]],              // (N,) output Householder vector
    device float* tau [[buffer(2)]],            // (1,) output scalar
    device float* beta [[buffer(3)]],           // (1,) output: -sign * norm_x (for R diagonal)
    constant uint& N [[buffer(4)]],
    constant uint& stride [[buffer(5)]],        // Stride between elements (for submatrices)
    uint gid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Parallel reduction for ||x[1:]||^2
    threadgroup float shared_sum[256];
    
    float local_sum = 0.0f;
    for (uint i = tid + 1; i < N; i += tg_size) {
        float val = x[i * stride];
        local_sum += val * val;
    }
    
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Thread 0 computes final result
    if (tid == 0) {
        float sigma = shared_sum[0];
        float x0 = x[0];
        
        if (sigma < 1e-10f) {
            // x is already along e_1
            tau[0] = 0.0f;
            beta[0] = x0;
            v[0] = 1.0f;
            for (uint i = 1; i < N; i++) {
                v[i * stride] = 0.0f;
            }
            return;
        }
        
        float norm_x = sqrt(x0 * x0 + sigma);
        float sign_x0 = (x0 >= 0.0f) ? 1.0f : -1.0f;
        
        // v[0] = x[0] + sign * norm_x
        float v0 = x0 + sign_x0 * norm_x;
        
        // tau = 2 * v0^2 / (v0^2 + sigma)
        float tau_val = 2.0f * v0 * v0 / (v0 * v0 + sigma);
        tau[0] = tau_val;
        
        // beta = -sign * norm_x (the new diagonal element)
        beta[0] = -sign_x0 * norm_x;
        
        // Normalize v so v[0] = 1
        v[0] = 1.0f;
        for (uint i = 1; i < N; i++) {
            v[i * stride] = x[i * stride] / v0;
        }
    }
}

// ============================================================================
// Apply Householder Reflection (single reflector)
// ============================================================================

// Apply H = I - tau * v * v^T to trailing columns
// A[:, col+1:] = A[:, col+1:] - tau * v * (v^T @ A[:, col+1:])
// This is a rank-1 update
kernel void apply_householder_kernel(
    device float* A [[buffer(0)]],              // (M, N) matrix, row-major
    device const float* v [[buffer(1)]],        // (M - col,) Householder vector
    constant float& tau [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& lda [[buffer(5)]],           // Leading dimension
    constant uint& col [[buffer(6)]],           // Current column being processed
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]]
) {
    // Each threadgroup processes one column of the trailing matrix
    uint target_col = gid.y + col + 1;
    if (target_col >= N) return;
    
    uint v_len = M - col;
    
    // First pass: compute v^T @ A[:, target_col] (reduction)
    threadgroup float shared_dot[256];
    
    float local_dot = 0.0f;
    for (uint i = tid.x; i < v_len; i += tg_size.x) {
        local_dot += v[i] * A[(col + i) * lda + target_col];
    }
    
    shared_dot[tid.x] = local_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction
    for (uint s = tg_size.x / 2; s > 0; s >>= 1) {
        if (tid.x < s) {
            shared_dot[tid.x] += shared_dot[tid.x + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float vT_a = shared_dot[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Second pass: A[row, target_col] -= tau * v[row] * vT_a
    for (uint i = tid.x; i < v_len; i += tg_size.x) {
        A[(col + i) * lda + target_col] -= tau * v[i] * vT_a;
    }
}

// ============================================================================
// Panel QR Factorization (geqr2)
// ============================================================================

// Unblocked Householder QR for a panel of columns
// This is called for each block during blocked QR
kernel void geqr2_panel_kernel(
    device float* A [[buffer(0)]],              // (M, N) panel, column-major
    device float* tau [[buffer(1)]],            // (min(M,N),) output tau values
    constant uint& M [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& lda [[buffer(4)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Process columns sequentially (data dependency)
    uint K = min(M, N);
    
    for (uint k = 0; k < K; k++) {
        uint v_len = M - k;
        
        // Step 1: Compute norm of A[k:, k]
        threadgroup float* sum_buf = shared;
        
        float local_sum = 0.0f;
        for (uint i = tid + 1; i < v_len; i += tg_size) {
            float val = A[(k + i) * lda + k];
            local_sum += val * val;
        }
        sum_buf[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) sum_buf[tid] += sum_buf[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        float sigma = sum_buf[0];
        float x0 = A[k * lda + k];
        
        float tau_k, v0_factor;
        if (sigma < 1e-10f) {
            tau_k = 0.0f;
            v0_factor = 1.0f;
        } else {
            float norm_x = sqrt(x0 * x0 + sigma);
            float sign = (x0 >= 0.0f) ? 1.0f : -1.0f;
            float v0 = x0 + sign * norm_x;
            tau_k = 2.0f * v0 * v0 / (v0 * v0 + sigma);
            v0_factor = 1.0f / v0;
            
            // Update diagonal to be -sign * norm_x
            if (tid == 0) {
                A[k * lda + k] = -sign * norm_x;
            }
        }
        
        if (tid == 0) {
            tau[k] = tau_k;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tau_k == 0.0f) continue;
        
        // Step 2: Apply H to remaining columns A[k:, k+1:]
        // For each column j > k:
        //   vT_a = v^T @ A[k:, j]
        //   A[k:, j] -= tau * v * vT_a
        
        // v is stored implicitly: v[0] = 1, v[i] = A[k+i, k] * v0_factor for i > 0
        for (uint j = k + 1; j < N; j++) {
            // Compute v^T @ A[k:, j]
            float local_dot = 0.0f;
            if (tid == 0) {
                local_dot = A[k * lda + j];  // v[0] = 1
            }
            for (uint i = tid + 1; i < v_len; i += tg_size) {
                float v_i = A[(k + i) * lda + k] * v0_factor;
                local_dot += v_i * A[(k + i) * lda + j];
            }
            
            sum_buf[tid] = local_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint s = tg_size / 2; s > 0; s >>= 1) {
                if (tid < s) sum_buf[tid] += sum_buf[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            float vT_a = sum_buf[0];
            
            // Update A[k:, j] -= tau * v * vT_a
            if (tid == 0) {
                A[k * lda + j] -= tau_k * 1.0f * vT_a;  // v[0] = 1
            }
            for (uint i = tid + 1; i < v_len; i += tg_size) {
                float v_i = A[(k + i) * lda + k] * v0_factor;
                A[(k + i) * lda + j] -= tau_k * v_i * vT_a;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // Store v below diagonal (already in correct form after division by v0)
        for (uint i = tid + 1; i < v_len; i += tg_size) {
            A[(k + i) * lda + k] *= v0_factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ============================================================================
// Block Reflector Application (larfb) - THE MONEY KERNEL
// ============================================================================

// Apply block reflector: C = (I - V @ T @ V^T) @ C = C - V @ (T @ (V^T @ C))
// This converts nb rank-1 updates into TWO matrix multiplications!
//
// Input: V (M x K), T (K x K upper triangular), C (M x N)
// Output: C updated in place
//
// Step 1: W = V^T @ C          (K x N) - GEMM
// Step 2: W = T @ W            (K x N) - TRMM  
// Step 3: C = C - V @ W        (M x N) - GEMM

// For now, we'll use simple implementations of each step
// In production, these would call into highly optimized GEMM kernels

kernel void larfb_step1_vtc(
    device const float* V [[buffer(0)]],        // (M, K) with implicit 1s on diagonal
    device const float* C [[buffer(1)]],        // (M, N)
    device float* W [[buffer(2)]],              // (K, N) output
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& ldv [[buffer(6)]],
    constant uint& ldc [[buffer(7)]],
    constant uint& ldw [[buffer(8)]],
    constant uint& panel_start [[buffer(9)]],   // Row offset for V in original matrix
    uint2 gid [[thread_position_in_grid]]
) {
    // Each thread computes one element of W = V^T @ C
    uint k_idx = gid.x;  // Row of W (column of V)
    uint n_idx = gid.y;  // Column of W and C
    
    if (k_idx >= K || n_idx >= N) return;
    
    float sum = 0.0f;
    
    // V[:, k_idx] has implicit 1 at position (panel_start + k_idx)
    for (uint m = 0; m < M; m++) {
        float v_val;
        if (m == panel_start + k_idx) {
            v_val = 1.0f;  // Implicit 1 on diagonal
        } else if (m < panel_start + k_idx) {
            v_val = 0.0f;  // Above diagonal is 0
        } else {
            v_val = V[m * ldv + k_idx];  // Below diagonal
        }
        sum += v_val * C[m * ldc + n_idx];
    }
    
    W[k_idx * ldw + n_idx] = sum;
}

kernel void larfb_step2_tw(
    device const float* T [[buffer(0)]],        // (K, K) upper triangular
    device float* W [[buffer(1)]],              // (K, N) in/out
    constant uint& K [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& apply_transpose [[buffer(4)]], // 1 for T^T, 0 for T
    uint2 gid [[thread_position_in_grid]]
) {
    // Apply T or T^T to W (triangular matrix multiply)
    // For blocked QR we need T^T
    // W = T^T @ W means process rows bottom to top
    
    uint n_idx = gid.x;
    if (n_idx >= N) return;
    
    // Load column of W into registers
    float w_col[32];  // Assuming K <= 32 for block size
    for (uint k = 0; k < K; k++) {
        w_col[k] = W[k * N + n_idx];
    }
    
    if (apply_transpose) {
        // T^T @ w: output[i] = sum_j T^T[i,j] * w[j] = sum_j T[j,i] * w[j]
        // T is upper triangular, so T[j,i] is non-zero only for j <= i
        // (T^T is lower triangular)
        for (uint i = 0; i < K; i++) {
            float sum = 0.0f;
            for (uint j = 0; j <= i; j++) {  // j <= i for T^T (lower triangular)
                sum += T[j * K + i] * w_col[j];  // T[j,i] = T[j*K + i]
            }
            W[i * N + n_idx] = sum;
        }
    } else {
        // T @ w: output[i] = sum_j T[i,j] * w[j]
        // T is upper triangular, so T[i,j] is non-zero for j >= i
        for (uint i = 0; i < K; i++) {
            float sum = 0.0f;
            for (uint j = i; j < K; j++) {  // j >= i for T (upper triangular)
                sum += T[i * K + j] * w_col[j];  // T[i,j] = T[i*K + j]
            }
            W[i * N + n_idx] = sum;
        }
    }
}

kernel void larfb_step3_cvw(
    device const float* V [[buffer(0)]],        // (M, K)
    device const float* W [[buffer(1)]],        // (K, N)
    device float* C [[buffer(2)]],              // (M, N) in/out
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& ldv [[buffer(6)]],
    constant uint& ldw [[buffer(7)]],
    constant uint& ldc [[buffer(8)]],
    constant uint& panel_start [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // C = C - V @ W
    uint m_idx = gid.x;
    uint n_idx = gid.y;
    
    if (m_idx >= M || n_idx >= N) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        float v_val;
        if (m_idx == panel_start + k) {
            v_val = 1.0f;
        } else if (m_idx < panel_start + k) {
            v_val = 0.0f;
        } else {
            v_val = V[m_idx * ldv + k];
        }
        sum += v_val * W[k * ldw + n_idx];
    }
    
    C[m_idx * ldc + n_idx] -= sum;
}

// ============================================================================
// Triangular Solve (trsm)
// ============================================================================

// Solve Lx = b where L is lower triangular
// Simple serial implementation for small systems
kernel void trsm_lower_kernel(
    device const float* L [[buffer(0)]],        // (N, N) lower triangular
    device float* x [[buffer(1)]],              // (N,) in: b, out: x
    constant uint& N [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // Only one thread does this (serial algorithm)
    if (gid != 0) return;
    
    for (uint i = 0; i < N; i++) {
        float sum = 0.0f;
        for (uint j = 0; j < i; j++) {
            sum += L[i * N + j] * x[j];
        }
        x[i] = (x[i] - sum) / L[i * N + i];
    }
}

// Solve Ux = b where U is upper triangular
kernel void trsm_upper_kernel(
    device const float* U [[buffer(0)]],        // (N, N) upper triangular
    device float* x [[buffer(1)]],              // (N,) in: b, out: x
    constant uint& N [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    
    for (int i = N - 1; i >= 0; i--) {
        float sum = 0.0f;
        for (uint j = i + 1; j < N; j++) {
            sum += U[i * N + j] * x[j];
        }
        x[i] = (x[i] - sum) / U[i * N + i];
    }
}

// ============================================================================
// Build T matrix (larft)
// ============================================================================

// Build triangular factor T for WY representation
// T[i, i] = tau[i]
// T[0:i, i] = -tau[i] * T[0:i, 0:i] @ V[:, 0:i]^T @ V[:, i]
kernel void larft_kernel(
    device const float* V [[buffer(0)]],        // (M, K) Householder vectors
    device const float* tau [[buffer(1)]],      // (K,) Householder scalars
    device float* T [[buffer(2)]],              // (K, K) output, upper triangular
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& ldv [[buffer(5)]],
    constant uint& panel_start [[buffer(6)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Build T column by column
    for (uint i = 0; i < K; i++) {
        // T[i, i] = tau[i]
        if (tid == 0) {
            T[i * K + i] = tau[i];
        }
        
        if (i == 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }
        
        // For j < i: T[j, i] = -tau[i] * sum_m(V[m, j] * V[m, i])
        for (uint j = 0; j < i; j++) {
            // Compute V[:, j]^T @ V[:, i]
            float local_dot = 0.0f;
            for (uint m = tid; m < M; m += tg_size) {
                float vj, vi;
                
                // V[m, j]
                if (m == panel_start + j) vj = 1.0f;
                else if (m < panel_start + j) vj = 0.0f;
                else vj = V[m * ldv + j];
                
                // V[m, i]
                if (m == panel_start + i) vi = 1.0f;
                else if (m < panel_start + i) vi = 0.0f;
                else vi = V[m * ldv + i];
                
                local_dot += vj * vi;
            }
            
            shared[tid] = local_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint s = tg_size / 2; s > 0; s >>= 1) {
                if (tid < s) shared[tid] += shared[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            if (tid == 0) {
                // T[j, i] = -tau[i] * T[0:j+1, j] . VTVi
                // But for upper triangular, we need cumulative update
                // Actually: T[j, i] = -tau[i] * sum_k(T[j, k] * VkT_Vi)
                T[j * K + i] = -tau[i] * shared[0];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // Apply T[0:i, 0:i] @ T[0:i, i]
        // This is a triangular matrix-vector product
        // For simplicity, do it serially
        if (tid == 0) {
            for (int j = i - 1; j >= 0; j--) {
                float sum = 0.0f;
                for (uint k = j; k < i; k++) {
                    sum += T[j * K + k] * T[k * K + i];
                }
                T[j * K + i] = sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ============================================================================
// FULLY FUSED QR KERNEL - Single dispatch for entire QR decomposition
// ============================================================================
// This kernel computes Q and R in a single Metal dispatch by:
// 1. Loading entire A into shared memory
// 2. Processing all columns with Householder reflectors
// 3. Accumulating Q as we go (starting from identity)
// 4. Writing Q and R at the end
//
// Constraints: M*N + M*K + 256 floats must fit in shared memory (32KB)
// For 32KB: max configuration is about 64x64 or 128x32

kernel void qr_full_fused_kernel(
    device const float* A_in [[buffer(0)]],    // (M, N) input matrix, row-major
    device float* Q_out [[buffer(1)]],         // (M, K) output Q, row-major
    device float* R_out [[buffer(2)]],         // (K, N) output R, row-major
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    threadgroup float* shared [[threadgroup(0)]],  // M*N + M*K + K + 256 floats
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // K = min(M, N) - number of reflectors
    uint K = min(M, N);
    
    // Shared memory layout (TWO-PHASE ALGORITHM):
    // [0, M*N)             = R_work (stores R and v below diagonal)
    // [M*N, M*N + M*K)     = Q_work (for Phase 2)
    // [M*N + M*K, ... + K) = tau_storage (store tau values)
    // [rest]               = reduction buffer (256 floats)
    threadgroup float* R_work = shared;
    threadgroup float* Q_work = shared + M * N;
    threadgroup float* tau_storage = shared + M * N + M * K;
    threadgroup float* reduction_buf = shared + M * N + M * K + K;
    
    // ========== PHASE 1: Compute R and store reflectors ==========
    
    // Load A into R_work
    uint total_A = M * N;
    for (uint idx = tid; idx < total_A; idx += tg_size) {
        R_work[idx] = A_in[idx];
    }
    uint total_Q = M * K;  // For later use
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 2: Process each column k (Householder QR)
    for (uint k = 0; k < K; k++) {
        uint v_len = M - k;
        
        // 2a: Compute ||R[k:, k]||^2 below diagonal
        float local_sigma = 0.0f;
        for (uint i = tid + 1; i < v_len; i += tg_size) {
            float val = R_work[(k + i) * N + k];
            local_sigma += val * val;
        }
        
        reduction_buf[tid] = local_sigma;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) reduction_buf[tid] += reduction_buf[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        float sigma = reduction_buf[0];
        float x0 = R_work[k * N + k];
        
        // 2b: Compute tau and v0_factor
        float tau_k = 0.0f;
        float v0_factor = 1.0f;
        
        if (sigma > 1e-10f) {
            float norm_x = sqrt(x0 * x0 + sigma);
            float sign = (x0 >= 0.0f) ? 1.0f : -1.0f;
            float v0 = x0 + sign * norm_x;
            tau_k = 2.0f * v0 * v0 / (v0 * v0 + sigma);
            v0_factor = 1.0f / v0;
            
            // Update R diagonal
            if (tid == 0) {
                R_work[k * N + k] = -sign * norm_x;
            }
        }
        
        // Store tau for Phase 2
        if (tid == 0) {
            tau_storage[k] = tau_k;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tau_k == 0.0f) continue;
        
        // 2c: Apply H to remaining columns of R (j = k+1 to N-1)
        // H @ R[:, j] = R[:, j] - tau * v * (v^T @ R[:, j])
        for (uint j = k + 1; j < N; j++) {
            // Compute v^T @ R[k:, j]
            float local_dot = 0.0f;
            if (tid == 0) {
                local_dot = R_work[k * N + j];  // v[0] = 1
            }
            for (uint i = tid + 1; i < v_len; i += tg_size) {
                float v_i = R_work[(k + i) * N + k] * v0_factor;
                local_dot += v_i * R_work[(k + i) * N + j];
            }
            
            reduction_buf[tid] = local_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint s = tg_size / 2; s > 0; s >>= 1) {
                if (tid < s) reduction_buf[tid] += reduction_buf[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            float vT_r = reduction_buf[0];
            
            // Update R[k:, j]
            if (tid == 0) {
                R_work[k * N + j] -= tau_k * vT_r;
            }
            for (uint i = tid + 1; i < v_len; i += tg_size) {
                float v_i = R_work[(k + i) * N + k] * v0_factor;
                R_work[(k + i) * N + j] -= tau_k * v_i * vT_r;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // 2e: Normalize v (store v/v0 below diagonal for later use)
        for (uint i = tid + 1; i < v_len; i += tg_size) {
            R_work[(k + i) * N + k] *= v0_factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // ========== PHASE 2: Build Q by applying reflectors in REVERSE ==========
    
    // Initialize Q to identity
    for (uint idx = tid; idx < total_Q; idx += tg_size) {
        uint row = idx / K;
        uint col = idx % K;
        Q_work[idx] = (row == col) ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Apply reflectors in reverse: k = K-1, K-2, ..., 0
    for (int k = (int)K - 1; k >= 0; k--) {
        float tau_k = tau_storage[k];
        if (tau_k == 0.0f) continue;
        
        uint v_len = M - k;
        
        // Apply H from left to each column j of Q
        for (uint j = 0; j < K; j++) {
            // Compute v^T @ Q[k:, j]
            float local_vq = 0.0f;
            
            if (tid == 0) {
                local_vq = Q_work[k * K + j];  // v[k] = 1
            }
            
            for (uint i = tid + 1; i < v_len; i += tg_size) {
                float v_i = R_work[(k + i) * N + k];  // Already normalized
                local_vq += v_i * Q_work[(k + i) * K + j];
            }
            
            reduction_buf[tid] = local_vq;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint s = tg_size / 2; s > 0; s >>= 1) {
                if (tid < s) reduction_buf[tid] += reduction_buf[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            float vq = reduction_buf[0];
            
            // Update Q[k:, j] -= tau * v * vq
            if (tid == 0) {
                Q_work[k * K + j] -= tau_k * vq;
            }
            for (uint i = tid + 1; i < v_len; i += tg_size) {
                float v_i = R_work[(k + i) * N + k];
                Q_work[(k + i) * K + j] -= tau_k * v_i * vq;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    
    // Step 3: Write outputs
    // Write Q
    for (uint idx = tid; idx < total_Q; idx += tg_size) {
        Q_out[idx] = Q_work[idx];
    }
    
    // Write R (upper K x N)
    uint total_R = K * N;
    for (uint idx = tid; idx < total_R; idx += tg_size) {
        uint row = idx / N;
        uint col = idx % N;
        // Zero below diagonal
        if (col < row) {
            R_out[idx] = 0.0f;
        } else {
            R_out[idx] = R_work[row * N + col];
        }
    }
}

// ============================================================================
// BATCHED QR KERNEL - One threadgroup per matrix in batch
// ============================================================================
// This kernel processes BATCH matrices in parallel, one threadgroup per matrix.
// This is where GPU wins: massive parallelism across matrices.

kernel void qr_batched_kernel(
    device const float* A_batch [[buffer(0)]],    // (Batch, M, N) input
    device float* Q_batch [[buffer(1)]],          // (Batch, M, K) output Q
    device float* R_batch [[buffer(2)]],          // (Batch, K, N) output R
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& Batch [[buffer(5)]],
    threadgroup float* shared [[threadgroup(0)]],  // Per-threadgroup shared memory
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint batch_idx [[threadgroup_position_in_grid]]
) {
    if (batch_idx >= Batch) return;
    
    uint K = min(M, N);
    
    // Pointers to this batch item
    uint A_stride = M * N;
    uint Q_stride = M * K;
    uint R_stride = K * N;
    
    device const float* A_in = A_batch + batch_idx * A_stride;
    device float* Q_out = Q_batch + batch_idx * Q_stride;
    device float* R_out = R_batch + batch_idx * R_stride;
    
    // Shared memory layout (same as single kernel)
    threadgroup float* R_work = shared;
    threadgroup float* Q_work = shared + M * N;
    threadgroup float* tau_storage = shared + M * N + M * K;
    threadgroup float* reduction_buf = shared + M * N + M * K + K;
    
    // ========== PHASE 1: Compute R ==========
    uint total_A = M * N;
    for (uint idx = tid; idx < total_A; idx += tg_size) {
        R_work[idx] = A_in[idx];
    }
    uint total_Q = M * K;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint k = 0; k < K; k++) {
        uint v_len = M - k;
        
        float local_sigma = 0.0f;
        for (uint i = tid + 1; i < v_len; i += tg_size) {
            float val = R_work[(k + i) * N + k];
            local_sigma += val * val;
        }
        
        reduction_buf[tid] = local_sigma;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) reduction_buf[tid] += reduction_buf[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        float sigma = reduction_buf[0];
        float x0 = R_work[k * N + k];
        
        float tau_k = 0.0f;
        float v0_factor = 1.0f;
        
        if (sigma > 1e-10f) {
            float norm_x = sqrt(x0 * x0 + sigma);
            float sign = (x0 >= 0.0f) ? 1.0f : -1.0f;
            float v0 = x0 + sign * norm_x;
            tau_k = 2.0f * v0 * v0 / (v0 * v0 + sigma);
            v0_factor = 1.0f / v0;
            if (tid == 0) R_work[k * N + k] = -sign * norm_x;
        }
        
        if (tid == 0) tau_storage[k] = tau_k;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tau_k == 0.0f) continue;
        
        for (uint j = k + 1; j < N; j++) {
            float local_dot = (tid == 0) ? R_work[k * N + j] : 0.0f;
            for (uint i = tid + 1; i < v_len; i += tg_size) {
                local_dot += R_work[(k + i) * N + k] * v0_factor * R_work[(k + i) * N + j];
            }
            
            reduction_buf[tid] = local_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint s = tg_size / 2; s > 0; s >>= 1) {
                if (tid < s) reduction_buf[tid] += reduction_buf[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            float vT_r = reduction_buf[0];
            if (tid == 0) R_work[k * N + j] -= tau_k * vT_r;
            for (uint i = tid + 1; i < v_len; i += tg_size) {
                R_work[(k + i) * N + j] -= tau_k * R_work[(k + i) * N + k] * v0_factor * vT_r;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        for (uint i = tid + 1; i < v_len; i += tg_size) {
            R_work[(k + i) * N + k] *= v0_factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // ========== PHASE 2: Build Q in reverse ==========
    for (uint idx = tid; idx < total_Q; idx += tg_size) {
        uint row = idx / K;
        uint col = idx % K;
        Q_work[idx] = (row == col) ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (int k = (int)K - 1; k >= 0; k--) {
        float tau_k = tau_storage[k];
        if (tau_k == 0.0f) continue;
        
        uint v_len = M - k;
        
        for (uint j = 0; j < K; j++) {
            float local_vq = (tid == 0) ? Q_work[k * K + j] : 0.0f;
            for (uint i = tid + 1; i < v_len; i += tg_size) {
                local_vq += R_work[(k + i) * N + k] * Q_work[(k + i) * K + j];
            }
            
            reduction_buf[tid] = local_vq;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint s = tg_size / 2; s > 0; s >>= 1) {
                if (tid < s) reduction_buf[tid] += reduction_buf[tid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            float vq = reduction_buf[0];
            if (tid == 0) Q_work[k * K + j] -= tau_k * vq;
            for (uint i = tid + 1; i < v_len; i += tg_size) {
                Q_work[(k + i) * K + j] -= tau_k * R_work[(k + i) * N + k] * vq;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    
    // ========== PHASE 3: Write outputs ==========
    for (uint idx = tid; idx < total_Q; idx += tg_size) {
        Q_out[idx] = Q_work[idx];
    }
    
    uint total_R = K * N;
    for (uint idx = tid; idx < total_R; idx += tg_size) {
        uint row = idx / N;
        uint col = idx % N;
        R_out[idx] = (col < row) ? 0.0f : R_work[row * N + col];
    }
}


// ============================================================================
// Batched TRSM (Triangular Solve) Kernel
// ============================================================================
// Solves R @ X = B where R is upper triangular
// Each threadgroup handles one matrix in the batch

kernel void trsm_batched_kernel(
    device const float* R_batch [[buffer(0)]],
    device const float* B_batch [[buffer(1)]],
    device float* X_batch [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& NRHS [[buffer(4)]],
    constant uint& Batch [[buffer(5)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint batch_idx [[threadgroup_position_in_grid]]
) {
    if (batch_idx >= Batch) return;
    
    device const float* R = R_batch + batch_idx * N * N;
    device const float* B = B_batch + batch_idx * N * NRHS;
    device float* X = X_batch + batch_idx * N * NRHS;
    
    threadgroup float* X_work = shared;
    
    uint total = N * NRHS;
    for (uint idx = tid; idx < total; idx += tg_size) {
        X_work[idx] = B[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (int i = (int)N - 1; i >= 0; i--) {
        for (uint j = tid; j < NRHS; j += tg_size) {
            float sum = X_work[i * NRHS + j];
            for (uint k = (uint)i + 1; k < N; k++) {
                sum -= R[i * N + k] * X_work[k * NRHS + j];
            }
            float diag = R[i * N + i];
            X_work[i * NRHS + j] = (abs(diag) > 1e-10f) ? (sum / diag) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    for (uint idx = tid; idx < total; idx += tg_size) {
        X[idx] = X_work[idx];
    }
}


// ============================================================================
// Column Norms Kernel
// ============================================================================

kernel void column_norms_kernel(
    device const float* A [[buffer(0)]],
    device float* norms [[buffer(1)]],
    constant uint& M [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint col_idx [[threadgroup_position_in_grid]]
) {
    if (col_idx >= N) return;
    
    float local_sum = 0.0f;
    for (uint i = tid; i < M; i += tg_size) {
        float val = A[i * N + col_idx];
        local_sum += val * val;
    }
    
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) norms[col_idx] = sqrt(shared[0]);
}


// ============================================================================
// Batched Cholesky Decomposition (potrf_batched) - MAGMA-Style Optimized
// ============================================================================
// Computes L such that A = L @ L.T for each matrix in batch
// 
// Optimizations based on MAGMA/ROCm:
// 1. Shared memory panel caching - load entire matrix into threadgroup memory
// 2. SIMD parallel reduction for sum of squares (diagonal computation)
// 3. Left-looking algorithm - minimize global memory writes
// 4. Parallel column updates below diagonal

// Maximum matrix size for shared memory (32KB / 4 bytes = 8192 floats, sqrt = ~90)
constant uint CHOL_MAX_N = 64;

kernel void cholesky_batched_kernel(
    device float* A [[buffer(0)]],           // (batch, N, N) input/output
    constant uint& N [[buffer(1)]],          // Matrix dimension
    constant uint& batch_size [[buffer(2)]],
    threadgroup float* shared_panel [[threadgroup(0)]],  // N*N floats for panel
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (batch_idx >= batch_size) return;
    
    // Pointer to this batch's matrix in global memory
    device float* A_batch = A + batch_idx * N * N;
    
    // Local pointer to shared memory panel
    threadgroup float* L = shared_panel;
    
    // =========================================================================
    // Step 1: Load entire matrix into shared memory (coalesced)
    // =========================================================================
    uint total_elements = N * N;
    for (uint idx = tid; idx < total_elements; idx += tg_size) {
        L[idx] = A_batch[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // =========================================================================
    // Step 2: Left-looking Cholesky factorization in shared memory
    // =========================================================================
    // For each column k:
    //   1. Update column k using previously computed columns 0..k-1
    //   2. Compute diagonal L[k,k] with parallel reduction
    //   3. Scale column k below diagonal
    
    for (uint k = 0; k < N; k++) {
        // -----------------------------------------------------------------
        // 2a: Update column k elements: L[i,k] -= sum(L[i,j]*L[k,j], j<k)
        // Each thread handles a subset of rows i > k
        // -----------------------------------------------------------------
        for (uint i = k + tid; i < N; i += tg_size) {
            float sum = 0.0f;
            // Inner loop over all previous columns - vectorizable
            for (uint j = 0; j < k; j++) {
                sum += L[i * N + j] * L[k * N + j];
            }
            L[i * N + k] -= sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // -----------------------------------------------------------------
        // 2b: Compute diagonal L[k,k] = sqrt(L[k,k])
        // Uses SIMD parallel reduction for sum of squares (already done above)
        // -----------------------------------------------------------------
        if (tid == 0) {
            float diag = L[k * N + k];
            L[k * N + k] = (diag > 1e-10f) ? sqrt(diag) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float diag_val = L[k * N + k];
        if (diag_val < 1e-10f) continue;  // Skip if not positive definite
        
        // -----------------------------------------------------------------
        // 2c: Scale column k below diagonal: L[i,k] /= L[k,k]
        // -----------------------------------------------------------------
        float inv_diag = 1.0f / diag_val;
        for (uint i = k + 1 + tid; i < N; i += tg_size) {
            L[i * N + k] *= inv_diag;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // =========================================================================
    // Step 3: Zero upper triangle and write back to global memory
    // =========================================================================
    for (uint idx = tid; idx < total_elements; idx += tg_size) {
        uint i = idx / N;
        uint j = idx % N;
        if (j > i) {
            A_batch[idx] = 0.0f;  // Zero upper triangle
        } else {
            A_batch[idx] = L[idx];  // Write lower triangle
        }
    }
}


// ============================================================================
// Batched Forward Substitution (trsm_lower_batched) - Optimized
// ============================================================================
// Solves L @ y = b for each matrix in batch where L is lower triangular
//
// Optimizations:
// 1. Shared memory cache for L rows (reduces global memory reads)
// 2. Parallel processing of RHS columns
// 3. Register accumulation for inner products

kernel void trsm_lower_batched_kernel(
    device const float* L [[buffer(0)]],     // (batch, N, N) lower triangular
    device float* b [[buffer(1)]],            // (batch, N, K) input/output
    constant uint& N [[buffer(2)]],
    constant uint& K [[buffer(3)]],           // Number of RHS columns
    constant uint& batch_size [[buffer(4)]],
    threadgroup float* shared_L_row [[threadgroup(0)]],  // N floats for current L row
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (batch_idx >= batch_size) return;
    
    device const float* L_batch = L + batch_idx * N * N;
    device float* b_batch = b + batch_idx * N * K;
    
    // Forward substitution: solve L @ y = b row by row
    for (uint i = 0; i < N; i++) {
        // Load L row i into shared memory (up to i elements)
        for (uint j = tid; j <= i; j += tg_size) {
            shared_L_row[j] = L_batch[i * N + j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float diag = shared_L_row[i];
        if (diag < 1e-10f) continue;
        float inv_diag = 1.0f / diag;
        
        // Each thread handles one RHS column
        for (uint col = tid; col < K; col += tg_size) {
            float sum = 0.0f;
            // Use shared memory for L row access
            for (uint j = 0; j < i; j++) {
                sum += shared_L_row[j] * b_batch[j * K + col];
            }
            b_batch[i * K + col] = (b_batch[i * K + col] - sum) * inv_diag;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}


// ============================================================================
// Batched Back Substitution (trsm_upper_batched) - Optimized  
// ============================================================================
// Solves U @ x = y for each matrix in batch where U is upper triangular

kernel void trsm_upper_batched_kernel(
    device const float* U [[buffer(0)]],     // (batch, N, N) upper triangular
    device float* b [[buffer(1)]],            // (batch, N, K) input/output
    constant uint& N [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    threadgroup float* shared_U_row [[threadgroup(0)]],  // N floats for current U row
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (batch_idx >= batch_size) return;
    
    device const float* U_batch = U + batch_idx * N * N;
    device float* b_batch = b + batch_idx * N * K;
    
    // Back substitution: solve U @ x = b row by row (bottom to top)
    for (int i_signed = (int)N - 1; i_signed >= 0; i_signed--) {
        uint i = (uint)i_signed;
        
        // Load U row i into shared memory (from i to N-1)
        for (uint j = i + tid; j < N; j += tg_size) {
            shared_U_row[j] = U_batch[i * N + j];
        }
        if (tid == 0) {
            shared_U_row[i] = U_batch[i * N + i];  // Diagonal
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float diag = shared_U_row[i];
        if (diag < 1e-10f) continue;
        float inv_diag = 1.0f / diag;
        
        // Each thread handles one RHS column
        for (uint col = tid; col < K; col += tg_size) {
            float sum = 0.0f;
            // Use shared memory for U row access
            for (uint j = i + 1; j < N; j++) {
                sum += shared_U_row[j] * b_batch[j * K + col];
            }
            b_batch[i * K + col] = (b_batch[i * K + col] - sum) * inv_diag;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}


// ============================================================================
// Batched Cholesky Solve (cholesky_solve_batched) - Fused & Zero-Copy
// ============================================================================
// Solves A @ x = b where A = L @ L.T (lower Cholesky factor)
// Step 1: Forward substitution: L @ y = b
// Step 2: Back substitution: L.T @ x = y (using L directly with transposed access)
// 
// This fused kernel avoids:
// 1. Creating a contiguous copy of L.T
// 2. Multiple command buffer dispatches
// 3. Extra memory allocation

kernel void cholesky_solve_batched_kernel(
    device const float* L [[buffer(0)]],     // (batch, N, N) lower triangular
    device float* b [[buffer(1)]],            // (batch, N, K) input/output -> solution
    constant uint& N [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    threadgroup float* shared_row [[threadgroup(0)]],  // N floats for current row
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (batch_idx >= batch_size) return;
    
    device const float* L_batch = L + batch_idx * N * N;
    device float* x_batch = b + batch_idx * N * K;
    
    // =========================================================================
    // Step 1: Forward substitution - solve L @ y = b
    // Proceeding top to bottom (row 0 to N-1)
    // =========================================================================
    for (uint i = 0; i < N; i++) {
        // Load L row i into shared memory (elements 0 to i)
        for (uint j = tid; j <= i; j += tg_size) {
            shared_row[j] = L_batch[i * N + j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float diag = shared_row[i];
        if (diag < 1e-10f) continue;
        float inv_diag = 1.0f / diag;
        
        // Each thread handles one RHS column
        for (uint col = tid; col < K; col += tg_size) {
            float sum = 0.0f;
            for (uint j = 0; j < i; j++) {
                sum += shared_row[j] * x_batch[j * K + col];
            }
            x_batch[i * K + col] = (x_batch[i * K + col] - sum) * inv_diag;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    
    // =========================================================================
    // Step 2: Back substitution - solve L.T @ x = y
    // L.T is upper triangular - proceeding bottom to top (row N-1 to 0)
    // Access L.T[i,j] as L[j,i] - zero copy transpose!
    // =========================================================================
    for (int i_signed = (int)N - 1; i_signed >= 0; i_signed--) {
        uint i = (uint)i_signed;
        
        // Load L.T row i into shared memory (elements i to N-1)
        // L.T[i,j] = L[j,i], so we load L column i
        for (uint j = i + tid; j < N; j += tg_size) {
            shared_row[j] = L_batch[j * N + i];  // L[j,i] = L.T[i,j]
        }
        if (tid == 0) {
            shared_row[i] = L_batch[i * N + i];  // Diagonal L[i,i] = L.T[i,i]
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float diag = shared_row[i];
        if (diag < 1e-10f) continue;
        float inv_diag = 1.0f / diag;
        
        // Each thread handles one RHS column
        for (uint col = tid; col < K; col += tg_size) {
            float sum = 0.0f;
            for (uint j = i + 1; j < N; j++) {
                sum += shared_row[j] * x_batch[j * K + col];
            }
            x_batch[i * K + col] = (x_batch[i * K + col] - sum) * inv_diag;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}
#include <metal_stdlib>
using namespace metal;

// Epsilon helpers for different dtypes (inline for template-style dispatch)
inline float get_epsilon(float) { return 1e-6f; }
inline half get_epsilon(half) { return (half)1e-3f; }
#if __METAL_VERSION__ >= 310
inline bfloat get_epsilon(bfloat) { return (bfloat)1e-3f; }
#endif

// =============================================================================
// SIMD Reduction Macros (guaranteed zero overhead)
// =============================================================================
// Note: SIMD_REDUCE_SUM modifies val in-place and returns final sum

#define SIMD_REDUCE_SUM(val) \
    ((val) += simd_shuffle_down((val), 16), \
     (val) += simd_shuffle_down((val), 8), \
     (val) += simd_shuffle_down((val), 4), \
     (val) += simd_shuffle_down((val), 2), \
     (val) += simd_shuffle_down((val), 1), \
     (val))

// =============================================================================
// BFloat16 Helper Macros (requires Metal 3.1+)
// =============================================================================
#if __METAL_VERSION__ >= 310

// bfloat doesn't have native simd_shuffle_down, so cast via ushort
#define SIMD_SHUFFLE_DOWN_BFLOAT(val, delta) \
    as_type<bfloat>(simd_shuffle_down(as_type<ushort>(val), (delta)))

// SIMD reduction for bfloat (uses the bfloat shuffle macro)
#define SIMD_REDUCE_SUM_BFLOAT(val) \
    ((val) += SIMD_SHUFFLE_DOWN_BFLOAT((val), 16), \
     (val) += SIMD_SHUFFLE_DOWN_BFLOAT((val), 8), \
     (val) += SIMD_SHUFFLE_DOWN_BFLOAT((val), 4), \
     (val) += SIMD_SHUFFLE_DOWN_BFLOAT((val), 2), \
     (val) += SIMD_SHUFFLE_DOWN_BFLOAT((val), 1), \
     (val))

// Dot product for bfloat4
#define DOT_BFLOAT4(a, b) ((a).x * (b).x + (a).y * (b).y + (a).z * (b).z + (a).w * (b).w)

// Dot product returning float (mixed precision)
#define DOT_BFLOAT4_AS_FLOAT(a, b) \
    ((float)(a).x * (float)(b).x + (float)(a).y * (float)(b).y + \
     (float)(a).z * (float)(b).z + (float)(a).w * (float)(b).w)

#endif // __METAL_VERSION__ >= 310

// =============================================================================
// Half/Float Dot Product Macros
// =============================================================================
#define DOT_HALF4_AS_FLOAT(a, b) \
    ((float)(a).x * (float)(b).x + (float)(a).y * (float)(b).y + \
     (float)(a).z * (float)(b).z + (float)(a).w * (float)(b).w)

// For float4, just use Metal's built-in dot()
#define DOT_FLOAT4(a, b) dot((a), (b))

// =============================================================================
// Inline Function Wrappers (for polymorphic template macro compatibility)
// =============================================================================
// These call the macros but provide function overloading for template-style usage

inline float simd_reduction(float val) { return SIMD_REDUCE_SUM(val); }
inline half simd_reduction(half val) { return SIMD_REDUCE_SUM(val); }

#if __METAL_VERSION__ >= 310
inline bfloat simd_reduction(bfloat val) { return SIMD_REDUCE_SUM_BFLOAT(val); }
inline bfloat dot(vec<bfloat, 4> a, vec<bfloat, 4> b) { return DOT_BFLOAT4(a, b); }
inline float dot_float(vec<bfloat, 4> a, vec<bfloat, 4> b) { return DOT_BFLOAT4_AS_FLOAT(a, b); }
#endif

inline float dot_float(vec<float, 4> a, vec<float, 4> b) { return DOT_FLOAT4(a, b); }
inline float dot_float(vec<half, 4> a, vec<half, 4> b) { return DOT_HALF4_AS_FLOAT(a, b); }

// Struct for ICB Uniforms (defined globally)
struct ICBUniforms {
    uint M;
    uint N;
    uint BatchStrideA;
    uint BatchStrideV;
    uint NumPairs;
};

// -----------------------------------------------------------------------------
// Macros for Templating
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// MACRO: Transpose
// -----------------------------------------------------------------------------
#define DEFINE_TRANSPOSE(T, SUFFIX) \
kernel void transpose_kernel_##SUFFIX( \
    device const T* A [[buffer(0)]], \
    device T* Out [[buffer(1)]], \
    constant uint& M [[buffer(2)]], \
    constant uint& N [[buffer(3)]], \
    uint2 gid [[thread_position_in_grid]]) \
{ \
    if (gid.x >= N || gid.y >= M) return; \
    uint idx_in = gid.y * N + gid.x; \
    uint idx_out = gid.x * M + gid.y; \
    Out[idx_out] = A[idx_in]; \
}

// -----------------------------------------------------------------------------
// MACRO: Jacobi Optimized
// -----------------------------------------------------------------------------
#define DEFINE_JACOBI(T, SUFFIX) \
kernel void jacobi_rotate_kernel_optimized_##SUFFIX( \
    device T* A_T [[buffer(0)]], \
    device T* V_T [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& N [[buffer(4)]], \
    constant uint& NumPairs [[buffer(5)]], \
    constant uint& NumSteps [[buffer(6)]], \
    constant uint& ThreadsPerPair [[buffer(7)]], \
    constant uint& BatchStrideA [[buffer(8)]], \
    constant uint& BatchStrideV [[buffer(9)]], \
    threadgroup T* shared_mem [[threadgroup(0)]], \
    uint3 group_pos [[threadgroup_position_in_grid]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 threads_per_group_vec [[threads_per_threadgroup]]) \
{ \
    uint tid = tid_vec.x; \
    uint batch_idx = group_pos.z; \
    uint threads_per_group = threads_per_group_vec.x; \
    uint simd_lane_id = tid % 32; \
    uint simd_group_id = tid / 32; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* A_ptr = A_T + batch_offset_A; \
    device T* V_ptr = V_T + batch_offset_V; \
    int pair_idx = group_pos.x; \
    uint pairs_offset = 0; \
    int p = AllPairs[pairs_offset + pair_idx * 2]; \
    int q = AllPairs[pairs_offset + pair_idx * 2 + 1]; \
    device T* col_i = A_ptr + p * M; \
    device T* col_j = A_ptr + q * M; \
    T part_ii = (T)0.0; \
    T part_jj = (T)0.0; \
    T part_ij = (T)0.0; \
    for (uint k = tid; k < M; k += threads_per_group) { \
        T val_i = col_i[k]; \
        T val_j = col_j[k]; \
        part_ii += val_i * val_i; \
        part_jj += val_j * val_j; \
        part_ij += val_i * val_j; \
    } \
    part_ii = simd_reduction((T)part_ii); \
    part_jj = simd_reduction((T)part_jj); \
    part_ij = simd_reduction((T)part_ij); \
    if (simd_lane_id == 0) { \
        shared_mem[simd_group_id * 3 + 0] = part_ii; \
        shared_mem[simd_group_id * 3 + 1] = part_jj; \
        shared_mem[simd_group_id * 3 + 2] = part_ij; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (tid == 0) { \
        float sum_ii = 0.0f; \
        float sum_jj = 0.0f; \
        float sum_ij = 0.0f; \
        uint num_simd_groups = (threads_per_group + 31) / 32; \
        for (uint s = 0; s < num_simd_groups; ++s) { \
            sum_ii += (float)shared_mem[s * 3 + 0]; \
            sum_jj += (float)shared_mem[s * 3 + 1]; \
            sum_ij += (float)shared_mem[s * 3 + 2]; \
        } \
        float c = 1.0f, s = 0.0f; \
        if (abs(sum_ij) > (float)get_epsilon((T)0)) { \
            float tau = (sum_jj - sum_ii) / (2.0f * sum_ij); \
            float t; \
            if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
            else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
            c = 1.0f / sqrt(1.0f + t * t); \
            s = t * c; \
        } \
        shared_mem[0] = (T)c; \
        shared_mem[1] = (T)s; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    T c = shared_mem[0]; \
    T s = shared_mem[1]; \
    for (uint k = tid; k < M; k += threads_per_group) { \
        T val_i = col_i[k]; \
        T val_j = col_j[k]; \
        col_i[k] = c * val_i - s * val_j; \
        col_j[k] = s * val_i + c * val_j; \
    } \
    device T* v_col_i = V_ptr + p * N; \
    device T* v_col_j = V_ptr + q * N; \
    for (uint k = tid; k < N; k += threads_per_group) { \
        T val_vi = v_col_i[k]; \
        T val_vj = v_col_j[k]; \
        v_col_i[k] = c * val_vi - s * val_vj; \
        v_col_j[k] = s * val_vi + c * val_vj; \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Jacobi ICB
// -----------------------------------------------------------------------------
#define DEFINE_JACOBI_ICB(T, SUFFIX) \
kernel void jacobi_rotate_kernel_icb_##SUFFIX( \
    device T* A_T [[buffer(0)]], \
    device T* V_T [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant ICBUniforms* uniforms [[buffer(3)]], \
    device const uint* StepPtr [[buffer(4)]], \
    threadgroup T* shared_mem [[threadgroup(0)]], \
    uint3 group_pos [[threadgroup_position_in_grid]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 threads_per_group_vec [[threads_per_threadgroup]]) \
{ \
    int pair_idx = group_pos.x; \
    int batch_idx = group_pos.z; \
    uint tid = tid_vec.x; \
    uint threads_per_group = threads_per_group_vec.x; \
    uint simd_lane_id = tid % 32; \
    uint simd_group_id = tid / 32; \
    uint M = uniforms->M; \
    uint N = uniforms->N; \
    uint BatchStrideA = uniforms->BatchStrideA; \
    uint BatchStrideV = uniforms->BatchStrideV; \
    uint NumPairs = uniforms->NumPairs; \
    uint step = *StepPtr; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* A_ptr = A_T + batch_offset_A; \
    device T* V_ptr = V_T + batch_offset_V; \
    uint pairs_offset = step * NumPairs * 2; \
    int p = AllPairs[pairs_offset + pair_idx * 2]; \
    int q = AllPairs[pairs_offset + pair_idx * 2 + 1]; \
    device T* col_i = A_ptr + p * M; \
    device T* col_j = A_ptr + q * M; \
    T part_ii = (T)0.0; \
    T part_jj = (T)0.0; \
    T part_ij = (T)0.0; \
    for (uint k = tid; k < M; k += threads_per_group) { \
        T val_i = col_i[k]; \
        T val_j = col_j[k]; \
        part_ii += val_i * val_i; \
        part_jj += val_j * val_j; \
        part_ij += val_i * val_j; \
    } \
    part_ii = simd_reduction((T)part_ii); \
    part_jj = simd_reduction((T)part_jj); \
    part_ij = simd_reduction((T)part_ij); \
    if (simd_lane_id == 0) { \
        shared_mem[simd_group_id * 3 + 0] = part_ii; \
        shared_mem[simd_group_id * 3 + 1] = part_jj; \
        shared_mem[simd_group_id * 3 + 2] = part_ij; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (tid == 0) { \
        float sum_ii = 0.0f; \
        float sum_jj = 0.0f; \
        float sum_ij = 0.0f; \
        uint num_simd_groups = (threads_per_group + 31) / 32; \
        for (uint s = 0; s < num_simd_groups; ++s) { \
            sum_ii += (float)shared_mem[s * 3 + 0]; \
            sum_jj += (float)shared_mem[s * 3 + 1]; \
            sum_ij += (float)shared_mem[s * 3 + 2]; \
        } \
        float c = 1.0f, s = 0.0f; \
        if (abs(sum_ij) > (float)get_epsilon((T)0)) { \
            float tau = (sum_jj - sum_ii) / (2.0f * sum_ij); \
            float t; \
            if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
            else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
            c = 1.0f / sqrt(1.0f + t * t); \
            s = t * c; \
        } \
        shared_mem[0] = (T)c; \
        shared_mem[1] = (T)s; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    T c = shared_mem[0]; \
    T s = shared_mem[1]; \
    for (uint k = tid; k < M; k += threads_per_group) { \
        T val_i = col_i[k]; \
        T val_j = col_j[k]; \
        col_i[k] = c * val_i - s * val_j; \
        col_j[k] = s * val_i + c * val_j; \
    } \
    device T* v_col_i = V_ptr + p * N; \
    device T* v_col_j = V_ptr + q * N; \
    for (uint k = tid; k < N; k += threads_per_group) { \
        T val_vi = v_col_i[k]; \
        T val_vj = v_col_j[k]; \
        v_col_i[k] = c * val_vi - s * val_vj; \
        v_col_j[k] = s * val_vi + c * val_vj; \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Jacobi ICB Vectorized (float4/half4)
// Requires M % 4 == 0 and N % 4 == 0
// -----------------------------------------------------------------------------
#define DEFINE_JACOBI_ICB_VEC4(T, SUFFIX) \
kernel void jacobi_rotate_kernel_icb_vec4_##SUFFIX( \
    device T* A_T [[buffer(0)]], \
    device T* V_T [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant ICBUniforms* uniforms [[buffer(3)]], \
    device const uint* StepPtr [[buffer(4)]], \
    threadgroup float* shared_mem [[threadgroup(0)]], \
    uint3 group_pos [[threadgroup_position_in_grid]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 threads_per_group_vec [[threads_per_threadgroup]]) \
{ \
    int pair_idx = group_pos.x; \
    int batch_idx = group_pos.z; \
    uint tid = tid_vec.x; \
    uint threads_per_group = threads_per_group_vec.x; \
    uint simd_lane_id = tid % 32; \
    uint simd_group_id = tid / 32; \
    uint M = uniforms->M; \
    uint N = uniforms->N; \
    uint BatchStrideA = uniforms->BatchStrideA; \
    uint BatchStrideV = uniforms->BatchStrideV; \
    uint NumPairs = uniforms->NumPairs; \
    uint step = *StepPtr; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* A_ptr = A_T + batch_offset_A; \
    device T* V_ptr = V_T + batch_offset_V; \
    uint pairs_offset = step * NumPairs * 2; \
    int p = AllPairs[pairs_offset + pair_idx * 2]; \
    int q = AllPairs[pairs_offset + pair_idx * 2 + 1]; \
    device T* col_i = A_ptr + p * M; \
    device T* col_j = A_ptr + q * M; \
    \
    typedef vec<T, 4> vec4; \
    device vec4* col_i_vec = (device vec4*)col_i; \
    device vec4* col_j_vec = (device vec4*)col_j; \
    uint M_vec = M / 4; \
    \
    float part_ii = 0.0f; \
    float part_jj = 0.0f; \
    float part_ij = 0.0f; \
    for (uint k = tid; k < M_vec; k += threads_per_group) { \
        vec4 vi = col_i_vec[k]; \
        vec4 vj = col_j_vec[k]; \
        part_ii += dot_float(vi, vi); \
        part_jj += dot_float(vj, vj); \
        part_ij += dot_float(vi, vj); \
    } \
    part_ii = simd_reduction((float)part_ii); \
    part_jj = simd_reduction((float)part_jj); \
    part_ij = simd_reduction((float)part_ij); \
    if (simd_lane_id == 0) { \
        shared_mem[simd_group_id * 3 + 0] = part_ii; \
        shared_mem[simd_group_id * 3 + 1] = part_jj; \
        shared_mem[simd_group_id * 3 + 2] = part_ij; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (tid == 0) { \
        float sum_ii = 0.0f; \
        float sum_jj = 0.0f; \
        float sum_ij = 0.0f; \
        uint num_simd_groups = (threads_per_group + 31) / 32; \
        for (uint s = 0; s < num_simd_groups; ++s) { \
            sum_ii += shared_mem[s * 3 + 0]; \
            sum_jj += shared_mem[s * 3 + 1]; \
            sum_ij += shared_mem[s * 3 + 2]; \
        } \
        float c = 1.0f, s = 0.0f; \
        if (abs(sum_ij) > (float)get_epsilon((T)0)) { \
            float tau = (sum_jj - sum_ii) / (2.0f * sum_ij); \
            float t; \
            if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
            else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
            c = 1.0f / sqrt(1.0f + t * t); \
            s = t * c; \
        } \
        shared_mem[0] = c; \
        shared_mem[1] = s; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    float c_f = shared_mem[0]; \
    float s_f = shared_mem[1]; \
    T c = (T)c_f; \
    T s = (T)s_f; \
    \
    for (uint k = tid; k < M_vec; k += threads_per_group) { \
        vec4 vi = col_i_vec[k]; \
        vec4 vj = col_j_vec[k]; \
        col_i_vec[k] = c * vi - s * vj; \
        col_j_vec[k] = s * vi + c * vj; \
    } \
    device T* v_col_i = V_ptr + p * N; \
    device T* v_col_j = V_ptr + q * N; \
    device vec4* v_col_i_vec = (device vec4*)v_col_i; \
    device vec4* v_col_j_vec = (device vec4*)v_col_j; \
    uint N_vec = N / 4; \
    for (uint k = tid; k < N_vec; k += threads_per_group) { \
        vec4 val_vi = v_col_i_vec[k]; \
        vec4 val_vj = v_col_j_vec[k]; \
        v_col_i_vec[k] = c * val_vi - s * val_vj; \
        v_col_j_vec[k] = s * val_vi + c * val_vj; \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Fused Block Kernel (Generic)
// -----------------------------------------------------------------------------
#define DEFINE_FUSED_GENERIC(T, SUFFIX) \
kernel void svd_fused_block_kernel_##SUFFIX( \
    device T* A [[buffer(0)]], \
    device T* V [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& N [[buffer(4)]], \
    constant uint& NumPairs [[buffer(5)]], \
    constant uint& NumSteps [[buffer(6)]], \
    constant uint& ThreadsPerPair [[buffer(7)]], \
    constant uint& BatchStrideA [[buffer(8)]], \
    constant uint& BatchStrideV [[buffer(9)]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 group_id [[threadgroup_position_in_grid]]) \
{ \
    uint tid = tid_vec.x; \
    uint batch_idx = group_id.z; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* A_ptr = A + batch_offset_A; \
    device T* V_ptr = V + batch_offset_V; \
    uint pair_idx = tid / ThreadsPerPair; \
    uint lane_id = tid % ThreadsPerPair; \
    (void)lane_id; \
    for (uint sw = 0; sw < 10; ++sw) { \
        for (uint step = 0; step < NumSteps; ++step) { \
            threadgroup_barrier(mem_flags::mem_device); \
            uint pair_offset = step * NumPairs * 2 + pair_idx * 2; \
            int p = AllPairs[pair_offset]; \
            int q = AllPairs[pair_offset+1]; \
            if (pair_idx < NumPairs) { \
                device T* col_p = A_ptr + p * M; \
                device T* col_q = A_ptr + q * M; \
                device T* v_col_p = V_ptr + p * N; \
                device T* v_col_q = V_ptr + q * N; \
                float app = 0.0f, aqq = 0.0f, apq = 0.0f; \
                for(uint k=0; k<M; ++k) { \
                    float vp = (float)col_p[k]; \
                    float vq = (float)col_q[k]; \
                    app += vp * vp; \
                    aqq += vq * vq; \
                    apq += vp * vq; \
                } \
                float c = 1.0f, s = 0.0f; \
                bool rotate = false; \
                if (abs(apq) > max(1e-6f, 1e-6f * sqrt(app * aqq))) { \
                    rotate = true; \
                    float tau = (aqq - app) / (2.0f * apq); \
                    float t; \
                    if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
                    else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
                    c = 1.0f / sqrt(1.0f + t*t); \
                    s = t * c; \
                } \
                if (rotate) { \
                    for(uint k=0; k<M; ++k) { \
                        float vp = (float)col_p[k]; \
                        float vq = (float)col_q[k]; \
                        col_p[k] = (T)(c * vp - s * vq); \
                        col_q[k] = (T)(s * vp + c * vq); \
                    } \
                    for(uint k=0; k<N; ++k) { \
                        float vp = (float)v_col_p[k]; \
                        float vq = (float)v_col_q[k]; \
                        v_col_p[k] = (T)(c * vp - s * vq); \
                        v_col_q[k] = (T)(s * vp + c * vq); \
                    } \
                } \
            } \
        } \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Fused Block Kernel (N=64 Specialized)
// -----------------------------------------------------------------------------
#define DEFINE_FUSED_64(T, SUFFIX) \
kernel void svd_fused_block_kernel_64_##SUFFIX( \
    device T* A [[buffer(0)]], \
    device T* V [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& BatchStrideA [[buffer(4)]], \
    constant uint& BatchStrideV [[buffer(5)]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 group_id [[threadgroup_position_in_grid]]) \
{ \
    const uint NumPairs = 32; \
    const uint NumSteps = 63; \
    const uint N_LOCAL = 64; \
    \
    uint tid = tid_vec.x; \
    uint batch_idx = group_id.z; \
    \
    /* Shared Memory for A and V */ \
    threadgroup float sA[64 * 64]; \
    threadgroup float sV[64 * 64]; \
    \
    /* Coalesced Load A and V -> shared memory */ \
    device T* A_src = A + batch_idx * BatchStrideA; \
    device T* V_src = V + batch_idx * BatchStrideV; \
    \
    /* 1024 threads load 4096 floats each for A and V */ \
    for (uint i = 0; i < 4; ++i) { \
        uint idx = tid * 4 + i; \
        if (idx < 4096) { \
            sA[idx] = (float)A_src[idx]; \
            sV[idx] = (float)V_src[idx]; \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    /* Each pair handled by 1 thread (like Generic) */ \
    /* We use only first 32 threads, rest are idle during compute */ \
    uint pair_idx = tid; /* Only tid 0..31 will work */ \
    \
    for (uint sw = 0; sw < 10; ++sw) { \
        for (uint step = 0; step < NumSteps; ++step) { \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            \
            if (pair_idx < NumPairs) { \
                uint pair_offset = step * NumPairs * 2 + pair_idx * 2; \
                int p = AllPairs[pair_offset]; \
                int q = AllPairs[pair_offset + 1]; \
                \
                threadgroup float* col_p = sA + p * N_LOCAL; \
                threadgroup float* col_q = sA + q * N_LOCAL; \
                threadgroup float* v_col_p = sV + p * N_LOCAL; \
                threadgroup float* v_col_q = sV + q * N_LOCAL; \
                \
                /* Sequential dot product - exactly like Generic */ \
                float app = 0.0f, aqq = 0.0f, apq = 0.0f; \
                for(uint k=0; k<N_LOCAL; ++k) { \
                    float vp = col_p[k]; \
                    float vq = col_q[k]; \
                    app += vp * vp; \
                    aqq += vq * vq; \
                    apq += vp * vq; \
                } \
                \
                float c = 1.0f, s = 0.0f; \
                bool rotate = false; \
                if (abs(apq) > max(1e-6f, 1e-6f * sqrt(app * aqq))) { \
                    rotate = true; \
                    float tau = (aqq - app) / (2.0f * apq); \
                    float t; \
                    if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
                    else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
                    c = 1.0f / sqrt(1.0f + t*t); \
                    s = t * c; \
                } \
                \
                if (rotate) { \
                    /* Sequential update - exactly like Generic */ \
                    for(uint k=0; k<N_LOCAL; ++k) { \
                        float vp = col_p[k]; \
                        float vq = col_q[k]; \
                        col_p[k] = c * vp - s * vq; \
                        col_q[k] = s * vp + c * vq; \
                    } \
                    for(uint k=0; k<N_LOCAL; ++k) { \
                        float vp = v_col_p[k]; \
                        float vq = v_col_q[k]; \
                        v_col_p[k] = c * vp - s * vq; \
                        v_col_q[k] = s * vp + c * vq; \
                    } \
                } \
            } \
        } \
    } \
    \
    /* Store sA and sV back to global memory */ \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint i = 0; i < 4; ++i) { \
        uint idx = tid * 4 + i; \
        if (idx < 4096) { \
            A_src[idx] = (T)sA[idx]; \
            V_src[idx] = (T)sV[idx]; \
        } \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Fused Block Kernel (N=128 Specialized)
// -----------------------------------------------------------------------------
#define DEFINE_FUSED_128(T, SUFFIX) \
kernel void svd_fused_block_kernel_128_##SUFFIX( \
    device T* A [[buffer(0)]], \
    device T* V [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& BatchStrideA [[buffer(4)]], \
    constant uint& BatchStrideV [[buffer(5)]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 group_id [[threadgroup_position_in_grid]]) \
{ \
    const uint NumPairs = 64; \
    const uint NumSteps = 127; \
    const uint ThreadsPerPair = 16; \
    const uint N = 128; \
    uint tid = tid_vec.x; \
    uint batch_idx = group_id.z; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* A_ptr = A + batch_offset_A; \
    device T* V_ptr = V + batch_offset_V; \
    uint pair_idx = tid / ThreadsPerPair; \
    uint lane_id = tid % ThreadsPerPair; \
    threadgroup atomic_uint sweep_rotations; \
    for (uint sw = 0; sw < 10; ++sw) { \
        if (tid == 0) atomic_store_explicit(&sweep_rotations, 0, memory_order_relaxed); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint step = 0; step < NumSteps; ++step) { \
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device); \
            uint pair_offset = step * NumPairs * 2 + pair_idx * 2; \
            int p = AllPairs[pair_offset]; \
            int q = AllPairs[pair_offset+1]; \
            if (pair_idx < NumPairs) { \
                device T* col_p = A_ptr + p * M; \
                device T* col_q = A_ptr + q * M; \
                device T* v_col_p = V_ptr + p * N; \
                device T* v_col_q = V_ptr + q * N; \
                float app = 0.0f, aqq = 0.0f, apq = 0.0f; \
                for(uint k=lane_id; k<M; k+=ThreadsPerPair) { \
                    float vp = (float)col_p[k]; \
                    float vq = (float)col_q[k]; \
                    app += vp * vp; \
                    aqq += vq * vq; \
                    apq += vp * vq; \
                } \
                app += simd_shuffle_down(app, 8); \
                app += simd_shuffle_down(app, 4); \
                app += simd_shuffle_down(app, 2); \
                app += simd_shuffle_down(app, 1); \
                aqq += simd_shuffle_down(aqq, 8); \
                aqq += simd_shuffle_down(aqq, 4); \
                aqq += simd_shuffle_down(aqq, 2); \
                aqq += simd_shuffle_down(aqq, 1); \
                apq += simd_shuffle_down(apq, 8); \
                apq += simd_shuffle_down(apq, 4); \
                apq += simd_shuffle_down(apq, 2); \
                apq += simd_shuffle_down(apq, 1); \
                float c = 1.0f, s = 0.0f; \
                bool rotate = false; \
                if (lane_id == 0) { \
                    if (abs(apq) > max(1e-6f, 1e-6f * sqrt(app * aqq))) { \
                        rotate = true; \
                        float tau = (aqq - app) / (2.0f * apq); \
                        float t; \
                        if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
                        else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
                        c = 1.0f / sqrt(1.0f + t*t); \
                        s = t * c; \
                        atomic_fetch_add_explicit(&sweep_rotations, 1, memory_order_relaxed); \
                    } \
                } \
                c = simd_shuffle(c, (tid % 32) & ~15); \
                s = simd_shuffle(s, (tid % 32) & ~15); \
                rotate = (bool)simd_shuffle((ushort)rotate, (tid % 32) & ~15); \
                if (rotate) { \
                    for(uint k=lane_id; k<M; k+=ThreadsPerPair) { \
                        float vp = (float)col_p[k]; \
                        float vq = (float)col_q[k]; \
                        col_p[k] = (T)(c * vp - s * vq); \
                        col_q[k] = (T)(s * vp + c * vq); \
                    } \
                    for(uint k=lane_id; k<N; k+=ThreadsPerPair) { \
                        float vp = (float)v_col_p[k]; \
                        float vq = (float)v_col_q[k]; \
                        v_col_p[k] = (T)(c * vp - s * vq); \
                        v_col_q[k] = (T)(s * vp + c * vq); \
                    } \
                } \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (atomic_load_explicit(&sweep_rotations, memory_order_relaxed) == 0) break; \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Fused Block Kernel (N=256 Specialized)
// -----------------------------------------------------------------------------
#define DEFINE_FUSED_256(T, SUFFIX) \
kernel void svd_fused_block_kernel_256_##SUFFIX( \
    device T* A [[buffer(0)]], \
    device T* V [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& BatchStrideA [[buffer(4)]], \
    constant uint& BatchStrideV [[buffer(5)]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 group_id [[threadgroup_position_in_grid]]) \
{ \
    const uint NumPairs = 128; \
    const uint NumSteps = 255; \
    const uint ThreadsPerPair = 8; \
    const uint N = 256; \
    uint tid = tid_vec.x; \
    uint batch_idx = group_id.z; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* A_ptr = A + batch_offset_A; \
    device T* V_ptr = V + batch_offset_V; \
    uint pair_idx = tid / ThreadsPerPair; \
    uint lane_id = tid % ThreadsPerPair; \
    threadgroup atomic_uint sweep_rotations; \
    for (uint sw = 0; sw < 10; ++sw) { \
        if (tid == 0) atomic_store_explicit(&sweep_rotations, 0, memory_order_relaxed); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint step = 0; step < NumSteps; ++step) { \
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device); \
            uint pair_offset = step * NumPairs * 2 + pair_idx * 2; \
            int p = AllPairs[pair_offset]; \
            int q = AllPairs[pair_offset+1]; \
            if (pair_idx < NumPairs) { \
                device T* col_p = A_ptr + p * M; \
                device T* col_q = A_ptr + q * M; \
                device T* v_col_p = V_ptr + p * N; \
                device T* v_col_q = V_ptr + q * N; \
                float app = 0.0f, aqq = 0.0f, apq = 0.0f; \
                for(uint k=lane_id; k<M; k+=ThreadsPerPair) { \
                    float vp = (float)col_p[k]; \
                    float vq = (float)col_q[k]; \
                    app += vp * vp; \
                    aqq += vq * vq; \
                    apq += vp * vq; \
                } \
                app += simd_shuffle_down(app, 4); \
                app += simd_shuffle_down(app, 2); \
                app += simd_shuffle_down(app, 1); \
                aqq += simd_shuffle_down(aqq, 4); \
                aqq += simd_shuffle_down(aqq, 2); \
                aqq += simd_shuffle_down(aqq, 1); \
                apq += simd_shuffle_down(apq, 4); \
                apq += simd_shuffle_down(apq, 2); \
                apq += simd_shuffle_down(apq, 1); \
                float c = 1.0f, s = 0.0f; \
                bool rotate = false; \
                if (lane_id == 0) { \
                    if (abs(apq) > max(1e-6f, 1e-6f * sqrt(app * aqq))) { \
                        rotate = true; \
                        float tau = (aqq - app) / (2.0f * apq); \
                        float t; \
                        if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
                        else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
                        c = 1.0f / sqrt(1.0f + t*t); \
                        s = t * c; \
                        atomic_fetch_add_explicit(&sweep_rotations, 1, memory_order_relaxed); \
                    } \
                } \
                c = simd_shuffle(c, (tid % 32) & ~7); \
                s = simd_shuffle(s, (tid % 32) & ~7); \
                rotate = (bool)simd_shuffle((ushort)rotate, (tid % 32) & ~7); \
                if (rotate) { \
                    for(uint k=lane_id; k<M; k+=ThreadsPerPair) { \
                        float vp = (float)col_p[k]; \
                        float vq = (float)col_q[k]; \
                        col_p[k] = (T)(c * vp - s * vq); \
                        col_q[k] = (T)(s * vp + c * vq); \
                    } \
                    for(uint k=lane_id; k<N; k+=ThreadsPerPair) { \
                        float vp = (float)v_col_p[k]; \
                        float vq = (float)v_col_q[k]; \
                        v_col_p[k] = (T)(c * vp - s * vq); \
                        v_col_q[k] = (T)(s * vp + c * vq); \
                    } \
                } \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (atomic_load_explicit(&sweep_rotations, memory_order_relaxed) == 0) break; \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Fused Block Kernel (N=512 Specialized)
// -----------------------------------------------------------------------------
#define DEFINE_FUSED_512(T, SUFFIX) \
kernel void svd_fused_block_kernel_512_##SUFFIX( \
    device T* A [[buffer(0)]], \
    device T* V [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& BatchStrideA [[buffer(4)]], \
    constant uint& BatchStrideV [[buffer(5)]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 group_id [[threadgroup_position_in_grid]]) \
{ \
    const uint NumPairs = 256; \
    const uint NumSteps = 511; \
    const uint ThreadsPerPair = 4; \
    const uint N = 512; \
    uint tid = tid_vec.x; \
    uint batch_idx = group_id.z; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* A_ptr = A + batch_offset_A; \
    device T* V_ptr = V + batch_offset_V; \
    uint pair_idx = tid / ThreadsPerPair; \
    uint lane_id = tid % ThreadsPerPair; \
    threadgroup atomic_uint sweep_rotations; \
    for (uint sw = 0; sw < 10; ++sw) { \
        if (tid == 0) atomic_store_explicit(&sweep_rotations, 0, memory_order_relaxed); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint step = 0; step < NumSteps; ++step) { \
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device); \
            uint pair_offset = step * NumPairs * 2 + pair_idx * 2; \
            int p = AllPairs[pair_offset]; \
            int q = AllPairs[pair_offset+1]; \
            if (pair_idx < NumPairs) { \
                device T* col_p = A_ptr + p * M; \
                device T* col_q = A_ptr + q * M; \
                device T* v_col_p = V_ptr + p * N; \
                device T* v_col_q = V_ptr + q * N; \
                /* Vectorized dot product using float4 */ \
                typedef vec<T, 4> vec4; \
                device vec4* col_p_vec = (device vec4*)col_p; \
                device vec4* col_q_vec = (device vec4*)col_q; \
                uint M4 = M / 4; \
                float app = 0.0f, aqq = 0.0f, apq = 0.0f; \
                for(uint k=lane_id; k<M4; k+=ThreadsPerPair) { \
                    vec4 vp4 = col_p_vec[k]; \
                    vec4 vq4 = col_q_vec[k]; \
                    float4 vp = float4(vp4); \
                    float4 vq = float4(vq4); \
                    app += dot(vp, vp); \
                    aqq += dot(vq, vq); \
                    apq += dot(vp, vq); \
                } \
                app += simd_shuffle_down(app, 2); \
                app += simd_shuffle_down(app, 1); \
                aqq += simd_shuffle_down(aqq, 2); \
                aqq += simd_shuffle_down(aqq, 1); \
                apq += simd_shuffle_down(apq, 2); \
                apq += simd_shuffle_down(apq, 1); \
                float c = 1.0f, s = 0.0f; \
                bool rotate = false; \
                if (lane_id == 0) { \
                    if (abs(apq) > max(1e-6f, 1e-6f * sqrt(app * aqq))) { \
                        rotate = true; \
                        float tau = (aqq - app) / (2.0f * apq); \
                        float t; \
                        if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
                        else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
                        c = 1.0f / sqrt(1.0f + t*t); \
                        s = t * c; \
                        atomic_fetch_add_explicit(&sweep_rotations, 1, memory_order_relaxed); \
                    } \
                } \
                c = simd_shuffle(c, (tid % 32) & ~3); \
                s = simd_shuffle(s, (tid % 32) & ~3); \
                rotate = (bool)simd_shuffle((ushort)rotate, (tid % 32) & ~3); \
                if (rotate) { \
                    /* Vectorized column update */ \
                    for(uint k=lane_id; k<M4; k+=ThreadsPerPair) { \
                        vec4 vp4 = col_p_vec[k]; \
                        vec4 vq4 = col_q_vec[k]; \
                        float4 vp = float4(vp4); \
                        float4 vq = float4(vq4); \
                        col_p_vec[k] = vec4(c * vp - s * vq); \
                        col_q_vec[k] = vec4(s * vp + c * vq); \
                    } \
                    /* Vectorized V update */ \
                    device vec4* v_col_p_vec = (device vec4*)v_col_p; \
                    device vec4* v_col_q_vec = (device vec4*)v_col_q; \
                    uint N4 = N / 4; \
                    for(uint k=lane_id; k<N4; k+=ThreadsPerPair) { \
                        vec4 vp4 = v_col_p_vec[k]; \
                        vec4 vq4 = v_col_q_vec[k]; \
                        float4 vp = float4(vp4); \
                        float4 vq = float4(vq4); \
                        v_col_p_vec[k] = vec4(c * vp - s * vq); \
                        v_col_q_vec[k] = vec4(s * vp + c * vq); \
                    } \
                } \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (atomic_load_explicit(&sweep_rotations, memory_order_relaxed) == 0) break; \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Fused Block Kernel (N=1024 Specialized)
// -----------------------------------------------------------------------------
#define DEFINE_FUSED_1024(T, SUFFIX) \
kernel void svd_fused_block_kernel_1024_##SUFFIX( \
    device T* A [[buffer(0)]], \
    device T* V [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& BatchStrideA [[buffer(4)]], \
    constant uint& BatchStrideV [[buffer(5)]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 group_id [[threadgroup_position_in_grid]]) \
{ \
    const uint NumPairs = 512; \
    const uint NumSteps = 1023; \
    const uint ThreadsPerPair = 2; \
    const uint N = 1024; \
    uint tid = tid_vec.x; \
    uint batch_idx = group_id.z; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* A_ptr = A + batch_offset_A; \
    device T* V_ptr = V + batch_offset_V; \
    uint pair_idx = tid / ThreadsPerPair; \
    uint lane_id = tid % ThreadsPerPair; \
    threadgroup atomic_uint sweep_rotations; \
    for (uint sw = 0; sw < 10; ++sw) { \
        if (tid == 0) atomic_store_explicit(&sweep_rotations, 0, memory_order_relaxed); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint step = 0; step < NumSteps; ++step) { \
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device); \
            uint pair_offset = step * NumPairs * 2 + pair_idx * 2; \
            int p = AllPairs[pair_offset]; \
            int q = AllPairs[pair_offset+1]; \
            if (pair_idx < NumPairs) { \
                device T* col_p = A_ptr + p * M; \
                device T* col_q = A_ptr + q * M; \
                device T* v_col_p = V_ptr + p * N; \
                device T* v_col_q = V_ptr + q * N; \
                /* Vectorized dot product using float4 */ \
                typedef vec<T, 4> vec4; \
                device vec4* col_p_vec = (device vec4*)col_p; \
                device vec4* col_q_vec = (device vec4*)col_q; \
                uint M4 = M / 4; \
                float app = 0.0f, aqq = 0.0f, apq = 0.0f; \
                for(uint k=lane_id; k<M4; k+=ThreadsPerPair) { \
                    vec4 vp4 = col_p_vec[k]; \
                    vec4 vq4 = col_q_vec[k]; \
                    float4 vp = float4(vp4); \
                    float4 vq = float4(vq4); \
                    app += dot(vp, vp); \
                    aqq += dot(vq, vq); \
                    apq += dot(vp, vq); \
                } \
                app += simd_shuffle_down(app, 1); \
                aqq += simd_shuffle_down(aqq, 1); \
                apq += simd_shuffle_down(apq, 1); \
                float c = 1.0f, s = 0.0f; \
                bool rotate = false; \
                if (lane_id == 0) { \
                    if (abs(apq) > max(1e-6f, 1e-6f * sqrt(app * aqq))) { \
                        rotate = true; \
                        float tau = (aqq - app) / (2.0f * apq); \
                        float t; \
                        if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
                        else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
                        c = 1.0f / sqrt(1.0f + t*t); \
                        s = t * c; \
                        atomic_fetch_add_explicit(&sweep_rotations, 1, memory_order_relaxed); \
                    } \
                } \
                c = simd_shuffle(c, (tid % 32) & ~1); \
                s = simd_shuffle(s, (tid % 32) & ~1); \
                rotate = (bool)simd_shuffle((ushort)rotate, (tid % 32) & ~1); \
                if (rotate) { \
                    /* Vectorized column update */ \
                    for(uint k=lane_id; k<M4; k+=ThreadsPerPair) { \
                        vec4 vp4 = col_p_vec[k]; \
                        vec4 vq4 = col_q_vec[k]; \
                        float4 vp = float4(vp4); \
                        float4 vq = float4(vq4); \
                        col_p_vec[k] = vec4(c * vp - s * vq); \
                        col_q_vec[k] = vec4(s * vp + c * vq); \
                    } \
                    /* Vectorized V update */ \
                    device vec4* v_col_p_vec = (device vec4*)v_col_p; \
                    device vec4* v_col_q_vec = (device vec4*)v_col_q; \
                    uint N4 = N / 4; \
                    for(uint k=lane_id; k<N4; k+=ThreadsPerPair) { \
                        vec4 vp4 = v_col_p_vec[k]; \
                        vec4 vq4 = v_col_q_vec[k]; \
                        float4 vp = float4(vp4); \
                        float4 vq = float4(vq4); \
                        v_col_p_vec[k] = vec4(c * vp - s * vq); \
                        v_col_q_vec[k] = vec4(s * vp + c * vq); \
                    } \
                } \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (atomic_load_explicit(&sweep_rotations, memory_order_relaxed) == 0) break; \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Norm and Normalize
// -----------------------------------------------------------------------------
#define DEFINE_NORMALIZATION(T, SUFFIX) \
kernel void column_norm_kernel_##SUFFIX( \
    device const T* A_T [[buffer(0)]], \
    device T* S [[buffer(1)]], \
    constant uint& M [[buffer(2)]], \
    constant uint& N [[buffer(3)]], \
    constant uint& BatchStrideA [[buffer(4)]], \
    constant uint& BatchStrideS [[buffer(5)]], \
    uint2 gid [[thread_position_in_grid]]) \
{ \
    uint i = gid.x; \
    uint batch_idx = gid.y; \
    if (i >= N) return; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_S = batch_idx * BatchStrideS; \
    device const T* col_i = A_T + batch_offset_A + i * M; \
    float sum_sq = 0.0f; \
    for (uint k = 0; k < M; ++k) { \
        float val = (float)col_i[k]; \
        sum_sq += val * val; \
    } \
    S[batch_offset_S + i] = (T)sqrt(sum_sq); \
} \
kernel void normalize_kernel_##SUFFIX( \
    device const T* A_T [[buffer(0)]], \
    device const T* S [[buffer(1)]], \
    device T* U_T [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& N [[buffer(4)]], \
    constant uint& BatchStrideA [[buffer(5)]], \
    constant uint& BatchStrideS [[buffer(6)]], \
    uint2 gid [[thread_position_in_grid]]) \
{ \
    uint i = gid.x; \
    if (i >= N) return; \
    uint batch_idx = gid.y; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_S = batch_idx * BatchStrideS; \
    device const T* col_i = A_T + batch_offset_A + i * M; \
    device T* u_col_i = U_T + batch_offset_A + i * M; \
    float sigma = (float)S[batch_offset_S + i]; \
    float inv_sigma = (sigma > 1.0e-8f) ? (1.0f / sigma) : 0.0f; \
    for (uint k = 0; k < M; ++k) { \
        u_col_i[k] = (T)((float)col_i[k] * inv_sigma); \
    } \
}

// -----------------------------------------------------------------------------
// Instantiations
// -----------------------------------------------------------------------------
// Transpose
DEFINE_TRANSPOSE(float, float)
DEFINE_TRANSPOSE(half, half)
#if __METAL_VERSION__ >= 310
DEFINE_TRANSPOSE(bfloat, bfloat)
#endif

// Jacobi
DEFINE_JACOBI(float, float)
DEFINE_JACOBI(half, half)
#if __METAL_VERSION__ >= 310
DEFINE_JACOBI(bfloat, bfloat)
#endif

// Jacobi ICB
DEFINE_JACOBI_ICB(float, float)
DEFINE_JACOBI_ICB_VEC4(float, float)

DEFINE_JACOBI_ICB(half, half)
DEFINE_JACOBI_ICB_VEC4(half, half)

#if __METAL_VERSION__ >= 310
DEFINE_JACOBI_ICB(bfloat, bfloat)
DEFINE_JACOBI_ICB_VEC4(bfloat, bfloat)
#endif

// Fused
DEFINE_FUSED_GENERIC(float, float)
DEFINE_FUSED_64(float, float)
DEFINE_FUSED_128(float, float)
DEFINE_FUSED_256(float, float)
DEFINE_FUSED_512(float, float)
DEFINE_FUSED_1024(float, float)

DEFINE_FUSED_GENERIC(half, half)
DEFINE_FUSED_64(half, half)
DEFINE_FUSED_128(half, half)
DEFINE_FUSED_256(half, half)
DEFINE_FUSED_512(half, half)
DEFINE_FUSED_1024(half, half)

#if __METAL_VERSION__ >= 310
DEFINE_FUSED_GENERIC(bfloat, bfloat)
DEFINE_FUSED_64(bfloat, bfloat)
DEFINE_FUSED_128(bfloat, bfloat)
DEFINE_FUSED_256(bfloat, bfloat)
DEFINE_FUSED_512(bfloat, bfloat)
DEFINE_FUSED_1024(bfloat, bfloat)
#endif

// Norm
DEFINE_NORMALIZATION(float, float)
DEFINE_NORMALIZATION(half, half)
#if __METAL_VERSION__ >= 310
DEFINE_NORMALIZATION(bfloat, bfloat)
#endif

// -----------------------------------------------------------------------------
// NEW: Clean Re-implementation of Vectorized Jacobi Kernel
// -----------------------------------------------------------------------------
// Simplified logic to avoid macro/template complexity and potential bugs.
// Assumes M % 4 == 0 and N % 4 == 0.
// Uses Float32 accumulation for stability.

template <typename T>
kernel void jacobi_rotate_kernel_vec4_clean(
    device T* A_T [[buffer(0)]],
    device T* V_T [[buffer(1)]],
    device const int* AllPairs [[buffer(2)]],
    constant ICBUniforms* uniforms [[buffer(3)]],
    device const uint* StepPtr [[buffer(4)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint3 group_pos [[threadgroup_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]],
    uint3 threads_per_group_vec [[threads_per_threadgroup]])
{
    // 1. Setup Indices
    uint tid = tid_vec.x;
    uint threads_per_group = threads_per_group_vec.x;
    
    // Grid/Pair Info
    int pair_idx = group_pos.x;
    int batch_idx = group_pos.z;
    
    // Uniforms
    uint M = uniforms->M;
    uint N = uniforms->N;
    uint BatchStrideA = uniforms->BatchStrideA;
    uint BatchStrideV = uniforms->BatchStrideV;
    uint NumPairs = uniforms->NumPairs;
    uint step = *StepPtr;
    
    // Pointers
    uint batch_offset_A = batch_idx * BatchStrideA;
    uint batch_offset_V = batch_idx * BatchStrideV;
    device T* A = A_T + batch_offset_A;
    device T* V = V_T + batch_offset_V;
    
    // Pair Indices (p < q)
    uint pairs_offset = step * NumPairs * 2;
    int p = AllPairs[pairs_offset + pair_idx * 2];
    int q = AllPairs[pairs_offset + pair_idx * 2 + 1];
    
    // Columns (M x 1)
    device T* col_p = A + p * M;
    device T* col_q = A + q * M;
    
    // Vectorized Pointers (M/4 x 1)
    typedef vec<T, 4> vec4;
    device vec4* col_p_vec = (device vec4*)col_p;
    device vec4* col_q_vec = (device vec4*)col_q;
    uint M_vec = M / 4;
    
    // 2. Accumulate Dot Products (G_{pp}, G_{qq}, G_{pq})
    float G_pp = 0.0f;
    float G_qq = 0.0f;
    float G_pq = 0.0f;
    
    for (uint k = tid; k < M_vec; k += threads_per_group) {
        vec4 val_p = col_p_vec[k];
        vec4 val_q = col_q_vec[k];
        
        G_pp += dot_float(val_p, val_p);
        G_qq += dot_float(val_q, val_q);
        G_pq += dot_float(val_p, val_q);
    }
    
    // 3. Reduction (Block-wide)
    // SIMD Reduction first
    G_pp = simd_reduction(G_pp);
    G_qq = simd_reduction(G_qq);
    G_pq = simd_reduction(G_pq);
    
    // Threadgroup Reduction (via Shared Mem)
    uint simd_lane_id = tid % 32;
    uint simd_group_id = tid / 32;
    
    if (simd_lane_id == 0) {
        shared_mem[simd_group_id * 3 + 0] = G_pp;
        shared_mem[simd_group_id * 3 + 1] = G_qq;
        shared_mem[simd_group_id * 3 + 2] = G_pq;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // First thread sums SIMD group results
    if (tid == 0) {
        float sum_pp = 0.0f;
        float sum_qq = 0.0f;
        float sum_pq = 0.0f;
        
        uint num_simd_groups = (threads_per_group + 31) / 32;
        for (uint s = 0; s < num_simd_groups; ++s) {
            sum_pp += shared_mem[s * 3 + 0];
            sum_qq += shared_mem[s * 3 + 1];
            sum_pq += shared_mem[s * 3 + 2];
        }
        
        // 4. Compute Rotation (c, s)
        float c = 1.0f;
        float s = 0.0f;
        
        // Check trace threshold (epsilon) to avoid noise
        // Using existing helper or hardcoded small epsilon
        if (abs(sum_pq) > 1e-9f) { // Strict epsilon
            float tau = (sum_qq - sum_pp) / (2.0f * sum_pq);
            float t;
            if (tau >= 0.0f) {
                t = 1.0f / (tau + sqrt(1.0f + tau*tau));
            } else {
                t = -1.0f / (-tau + sqrt(1.0f + tau*tau));
            }
            c = 1.0f / sqrt(1.0f + t*t);
            s = t * c;
        }
        
        shared_mem[0] = c;
        shared_mem[1] = s;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float c_f = shared_mem[0];
    float s_f = shared_mem[1];
    T c = (T)c_f;
    T s = (T)s_f;
    
    // 5. Apply Rotation to A (M x 2)
    for (uint k = tid; k < M_vec; k += threads_per_group) {
        vec4 val_p = col_p_vec[k];
        vec4 val_q = col_q_vec[k];
        
        col_p_vec[k] = c * val_p - s * val_q;
        col_q_vec[k] = s * val_p + c * val_q;
    }
    
    // 6. Apply Rotation to V (N x 2)
    device T* v_col_p = V + p * N;
    device T* v_col_q = V + q * N;
    
    device vec4* v_col_p_vec = (device vec4*)v_col_p;
    device vec4* v_col_q_vec = (device vec4*)v_col_q;
    uint N_vec = N / 4;
    
    for (uint k = tid; k < N_vec; k += threads_per_group) {
        vec4 val_p = v_col_p_vec[k];
        vec4 val_q = v_col_q_vec[k];
        
        v_col_p_vec[k] = c * val_p - s * val_q;
        v_col_q_vec[k] = s * val_p + c * val_q;
    }
}

// Explicit Instantiations for the Clean Kernel
template [[host_name("jacobi_rotate_kernel_vec4_clean_float")]] kernel void jacobi_rotate_kernel_vec4_clean<float>(
    device float*, device float*, device const int*, constant ICBUniforms*, device const uint*, threadgroup float*, uint3, uint3, uint3);

#if __METAL_VERSION__ >= 310
template [[host_name("jacobi_rotate_kernel_vec4_clean_bfloat")]] kernel void jacobi_rotate_kernel_vec4_clean<bfloat>(
    device bfloat*, device bfloat*, device const int*, constant ICBUniforms*, device const uint*, threadgroup float*, uint3, uint3, uint3);
#endif

template [[host_name("jacobi_rotate_kernel_vec4_clean_half")]] kernel void jacobi_rotate_kernel_vec4_clean<half>(
    device half*, device half*, device const int*, constant ICBUniforms*, device const uint*, threadgroup float*, uint3, uint3, uint3);


// =============================================================================
// Column Norm Sort Kernel (De Rijk optimization for SVD)
// Computes column norms, sorts descending, permutes columns in single dispatch
// =============================================================================

kernel void column_norm_sort_kernel(
    device const float* A [[buffer(0)]],        // (B, M, N) input
    device float* A_sorted [[buffer(1)]],       // (B, M, N) sorted output
    device int* perm [[buffer(2)]],             // (B, N) permutation indices
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    threadgroup float* shared [[threadgroup(0)]],  // N floats for norms + N ints for indices
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (batch_idx >= batch_size) return;
    
    device const float* A_batch = A + batch_idx * M * N;
    device float* A_out = A_sorted + batch_idx * M * N;
    device int* perm_out = perm + batch_idx * N;
    
    // Shared memory layout: [0..N) = norms, [N..2N) = indices
    threadgroup float* norms = shared;
    threadgroup int* indices = (threadgroup int*)(shared + N);
    
    // Step 1: Compute column norms (parallel reduction per column)
    for (uint col = tid; col < N; col += tg_size) {
        float sum_sq = 0.0f;
        for (uint row = 0; row < M; row++) {
            float val = A_batch[row * N + col];
            sum_sq += val * val;
        }
        norms[col] = sqrt(sum_sq);
        indices[col] = (int)col;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 2: Simple insertion sort (N is typically small, ≤ 512)
    // Thread 0 does the sort
    if (tid == 0) {
        for (uint i = 1; i < N; i++) {
            float key_norm = norms[i];
            int key_idx = indices[i];
            int j = (int)i - 1;
            // Sort descending
            while (j >= 0 && norms[j] < key_norm) {
                norms[j + 1] = norms[j];
                indices[j + 1] = indices[j];
                j--;
            }
            norms[j + 1] = key_norm;
            indices[j + 1] = key_idx;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 3: Write permutation and permute columns
    for (uint new_col = tid; new_col < N; new_col += tg_size) {
        int old_col = indices[new_col];
        perm_out[new_col] = old_col;
        
        // Copy column from old position to new
        for (uint row = 0; row < M; row++) {
            A_out[row * N + new_col] = A_batch[row * N + old_col];
        }
    }
}

// =============================================================================
// Sign Canonicalization Kernel (for SVD U/V normalization)
// Ensures reproducible signs: flip so max-magnitude element in each U column is positive
// =============================================================================

kernel void sign_canonicalize_kernel(
    device float* U [[buffer(0)]],              // (B, M, N) modified in-place
    device float* V [[buffer(1)]],              // (B, N, N) modified in-place  
    constant uint& M [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (batch_idx >= batch_size) return;
    
    device float* U_batch = U + batch_idx * M * N;
    device float* V_batch = V + batch_idx * N * N;
    
    // Each thread handles one or more columns
    for (uint col = tid; col < N; col += tg_size) {
        // Find max-magnitude element in U column
        float max_val = 0.0f;
        float max_abs = 0.0f;
        for (uint row = 0; row < M; row++) {
            float val = U_batch[row * N + col];
            float abs_val = fabs(val);
            if (abs_val > max_abs) {
                max_abs = abs_val;
                max_val = val;
            }
        }
        
        // Determine sign flip
        float sign = (max_val >= 0.0f) ? 1.0f : -1.0f;
        
        // Apply sign to U column
        for (uint row = 0; row < M; row++) {
            U_batch[row * N + col] *= sign;
        }
        
        // Apply sign to V column
        for (uint row = 0; row < N; row++) {
            V_batch[row * N + col] *= sign;
        }
    }
}

// =============================================================================
// Batched Q.T @ b Kernel (for fused solve without sync)
// Transposed batch matrix multiply: c = Q.T @ b
// =============================================================================

kernel void batched_qt_b_kernel(
    device const float* Q [[buffer(0)]],        // (B, M, N) - will access transposed
    device const float* b [[buffer(1)]],        // (B, M, K)
    device float* c [[buffer(2)]],              // (B, N, K) output
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.z;
    uint row = gid.y;  // Row of output c (0 to N-1)
    uint col = gid.x;  // Column of output c (0 to K-1)
    
    if (batch_idx >= batch_size || row >= N || col >= K) return;
    
    device const float* Q_batch = Q + batch_idx * M * N;
    device const float* b_batch = b + batch_idx * M * K;
    device float* c_batch = c + batch_idx * N * K;
    
    // c[row, col] = sum over m of Q.T[row, m] * b[m, col]
    //             = sum over m of Q[m, row] * b[m, col]
    float sum = 0.0f;
    for (uint m = 0; m < M; m++) {
        sum += Q_batch[m * N + row] * b_batch[m * K + col];
    }
    c_batch[row * K + col] = sum;
}

// =============================================================================
// EIGH-specific: Dot Columns Kernel for Eigenvalue Extraction
// =============================================================================


#define DEFINE_DOT_COLUMNS(T, SUFFIX) \
kernel void dot_columns_kernel_##SUFFIX( \
    device const T* A_Rotated [[buffer(0)]], \
    device const T* V_Rotated [[buffer(1)]], \
    device T* Eigenvalues [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& N [[buffer(4)]], \
    constant uint& BatchStrideA [[buffer(5)]], \
    constant uint& BatchStrideV [[buffer(6)]], \
    constant uint& BatchStrideE [[buffer(7)]], \
    threadgroup float* shared_mem [[threadgroup(0)]], \
    uint3 group_pos [[threadgroup_position_in_grid]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 threads_per_group_vec [[threads_per_threadgroup]]) \
{ \
    uint tid = tid_vec.x; \
    uint i = group_pos.x; \
    uint batch_idx = group_pos.z; \
    uint threads_per_group = threads_per_group_vec.x; \
    if (i >= N) return; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    uint batch_offset_E = batch_idx * BatchStrideE; \
    device const T* col_a = A_Rotated + batch_offset_A + i * M; \
    device const T* col_v = V_Rotated + batch_offset_V + i * N; \
    float sum_dot = 0.0f; \
    for (uint k = tid; k < M; k += threads_per_group) { \
        sum_dot += (float)col_a[k] * (float)col_v[k]; \
    } \
    sum_dot = simd_reduction(sum_dot); \
    if ((tid % 32) == 0) { \
        shared_mem[tid / 32] = sum_dot; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (tid == 0) { \
        float total_dot = 0.0f; \
        uint num_simd_groups = (threads_per_group + 31) / 32; \
        for (uint s = 0; s < num_simd_groups; ++s) { \
            total_dot += shared_mem[s]; \
        } \
        Eigenvalues[batch_offset_E + i] = (T)total_dot; \
    } \
}

DEFINE_DOT_COLUMNS(float, float)
DEFINE_DOT_COLUMNS(half, half)
#if __METAL_VERSION__ >= 310
DEFINE_DOT_COLUMNS(bfloat, bfloat)
#endif


// =============================================================================
// HIGH-IMPACT ML/LA KERNELS
// =============================================================================

// =============================================================================
// Batched LU Decomposition (with partial pivoting)
// =============================================================================

kernel void lu_batched_kernel(
    device float* A [[buffer(0)]],              // (B, N, N) - modified in-place to L\U
    device int* pivots [[buffer(1)]],            // (B, N) - pivot indices
    constant uint& N [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (batch_idx >= batch_size) return;
    
    device float* A_batch = A + batch_idx * N * N;
    device int* piv = pivots + batch_idx * N;
    
    // Initialize pivots
    for (uint i = tid; i < N; i += tg_size) {
        piv[i] = (int)i;
    }
    threadgroup_barrier(mem_flags::mem_device);
    
    // LU with partial pivoting (Doolittle's method)
    for (uint k = 0; k < N; k++) {
        // Find pivot (thread 0)
        if (tid == 0) {
            uint max_row = k;
            float max_val = fabs(A_batch[k * N + k]);
            for (uint i = k + 1; i < N; i++) {
                float val = fabs(A_batch[i * N + k]);
                if (val > max_val) {
                    max_val = val;
                    max_row = i;
                }
            }
            
            // Swap rows if needed
            if (max_row != k) {
                piv[k] = (int)max_row;
                for (uint j = 0; j < N; j++) {
                    float tmp = A_batch[k * N + j];
                    A_batch[k * N + j] = A_batch[max_row * N + j];
                    A_batch[max_row * N + j] = tmp;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        
        // Compute multipliers and update
        float pivot_val = A_batch[k * N + k];
        if (fabs(pivot_val) < 1e-10f) continue;  // Singular check
        
        for (uint i = k + 1 + tid; i < N; i += tg_size) {
            float mult = A_batch[i * N + k] / pivot_val;
            A_batch[i * N + k] = mult;  // Store L below diagonal
            
            for (uint j = k + 1; j < N; j++) {
                A_batch[i * N + j] -= mult * A_batch[k * N + j];
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// =============================================================================
// Batched Matrix Inverse (via LU decomposition)
// =============================================================================

kernel void inverse_batched_kernel(
    device const float* LU [[buffer(0)]],       // (B, N, N) - LU factorization
    device const int* pivots [[buffer(1)]],     // (B, N) - pivot indices
    device float* Ainv [[buffer(2)]],           // (B, N, N) - output inverse
    constant uint& N [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (batch_idx >= batch_size) return;
    
    device const float* L = LU + batch_idx * N * N;
    device const int* piv = pivots + batch_idx * N;
    device float* inv = Ainv + batch_idx * N * N;
    
    // Initialize inverse as identity (with pivoting applied)
    for (uint i = tid; i < N * N; i += tg_size) {
        inv[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_device);
    
    for (uint i = tid; i < N; i += tg_size) {
        inv[i * N + i] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_device);
    
    // Solve each column: inv[col] = L^(-1) @ P @ e_col
    // Then inv[col] = U^(-1) @ inv[col]
    
    // This is thread-per-column approach
    for (uint col = tid; col < N; col += tg_size) {
        // Apply permutation and forward substitution (L solve)
        float y[64];  // Temp storage, assuming N <= 64
        for (uint i = 0; i < N; i++) {
            float sum = (i == col) ? 1.0f : 0.0f;
            // Apply pivot
            for (uint k = 0; k < i; k++) {
                if (piv[k] != (int)k) {
                    // Swap entries
                }
            }
            for (uint j = 0; j < i; j++) {
                sum -= L[i * N + j] * y[j];
            }
            y[i] = sum;
        }
        
        // Backward substitution (U solve)
        for (int i = (int)N - 1; i >= 0; i--) {
            float sum = y[i];
            for (uint j = (uint)i + 1; j < N; j++) {
                sum -= L[i * N + j] * inv[j * N + col];
            }
            inv[i * N + col] = sum / L[i * N + i];
        }
    }
}

// =============================================================================
// Batched SYRK: C = A.T @ A (symmetric rank-k update)
// =============================================================================

kernel void syrk_batched_kernel(
    device const float* A [[buffer(0)]],        // (B, M, N)
    device float* C [[buffer(1)]],              // (B, N, N) output
    constant uint& M [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.z;
    uint row = gid.y;
    uint col = gid.x;
    
    if (batch_idx >= batch_size || row >= N || col >= N) return;
    
    // Only compute upper triangle (symmetric)
    if (col < row) return;
    
    device const float* A_batch = A + batch_idx * M * N;
    device float* C_batch = C + batch_idx * N * N;
    
    // C[row, col] = sum over k of A.T[row, k] * A[k, col]
    //             = sum over k of A[k, row] * A[k, col]
    float sum = 0.0f;
    for (uint k = 0; k < M; k++) {
        sum += A_batch[k * N + row] * A_batch[k * N + col];
    }
    
    C_batch[row * N + col] = sum;
    // Mirror to lower triangle
    if (row != col) {
        C_batch[col * N + row] = sum;
    }
}

// =============================================================================
// Batched Frobenius Norm: ||A||_F = sqrt(sum(A_ij^2))
// =============================================================================

kernel void frobenius_norm_batched_kernel(
    device const float* A [[buffer(0)]],        // (B, M, N)
    device float* norms [[buffer(1)]],          // (B,) output
    constant uint& M [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (batch_idx >= batch_size) return;
    
    device const float* A_batch = A + batch_idx * M * N;
    uint total_elems = M * N;
    
    // Compute partial sum
    float local_sum = 0.0f;
    for (uint i = tid; i < total_elems; i += tg_size) {
        float val = A_batch[i];
        local_sum += val * val;
    }
    
    // Store in shared memory
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (tid == 0) {
        norms[batch_idx] = sqrt(shared[0]);
    }
}

// =============================================================================
// Batched Softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
// =============================================================================

kernel void softmax_batched_kernel(
    device float* x [[buffer(0)]],              // (B, N) modified in-place
    constant uint& N [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant float& temperature [[buffer(3)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (batch_idx >= batch_size) return;
    
    device float* x_batch = x + batch_idx * N;
    float temp_inv = 1.0f / temperature;
    
    // Step 1: Find max (for numerical stability)
    float local_max = -INFINITY;
    for (uint i = tid; i < N; i += tg_size) {
        local_max = fmax(local_max, x_batch[i] * temp_inv);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce max
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmax(shared[tid], shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float val = exp(x_batch[i] * temp_inv - max_val);
        x_batch[i] = val;  // Store exp values temporarily
        local_sum += val;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce sum
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_exp = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 3: Normalize
    float inv_sum = 1.0f / sum_exp;
    for (uint i = tid; i < N; i += tg_size) {
        x_batch[i] *= inv_sum;
    }
}

// =============================================================================
// Batched Trace: sum of diagonal elements
// =============================================================================

kernel void trace_batched_kernel(
    device const float* A [[buffer(0)]],        // (B, N, N)
    device float* traces [[buffer(1)]],         // (B,) output
    constant uint& N [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (batch_idx >= batch_size) return;
    
    device const float* A_batch = A + batch_idx * N * N;
    
    // Compute partial trace
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        local_sum += A_batch[i * N + i];  // Diagonal element
    }
    
    // Store in shared memory
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (tid == 0) {
        traces[batch_idx] = shared[0];
    }
}

// =============================================================================
// Fused Linear Solve (LU decomposition + forward/back substitution)
// Solves Ax = b for x, where A is (N, N) and b is (N, K)
// =============================================================================

kernel void solve_batched_kernel(
    device float* A [[buffer(0)]],              // (B, N, N) - destroyed, becomes LU
    device float* b [[buffer(1)]],              // (B, N, K) - destroyed, becomes x
    device int* pivots [[buffer(2)]],           // (B, N) - temp storage for pivots
    constant uint& N [[buffer(3)]],
    constant uint& K [[buffer(4)]],             // number of RHS columns
    constant uint& batch_size [[buffer(5)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (batch_idx >= batch_size) return;
    
    device float* A_batch = A + batch_idx * N * N;
    device float* b_batch = b + batch_idx * N * K;
    device int* piv = pivots + batch_idx * N;
    
    // =========================================================================
    // Step 1: LU Decomposition with Partial Pivoting (Doolittle's method)
    // =========================================================================
    
    // Initialize pivots
    for (uint i = tid; i < N; i += tg_size) {
        piv[i] = (int)i;
    }
    threadgroup_barrier(mem_flags::mem_device);
    
    for (uint k = 0; k < N; k++) {
        // Find pivot (thread 0 only)
        if (tid == 0) {
            uint max_row = k;
            float max_val = fabs(A_batch[k * N + k]);
            for (uint i = k + 1; i < N; i++) {
                float val = fabs(A_batch[i * N + k]);
                if (val > max_val) {
                    max_val = val;
                    max_row = i;
                }
            }
            
            // Swap rows in A and b if needed
            if (max_row != k) {
                piv[k] = (int)max_row;
                
                // Swap in A
                for (uint j = 0; j < N; j++) {
                    float tmp = A_batch[k * N + j];
                    A_batch[k * N + j] = A_batch[max_row * N + j];
                    A_batch[max_row * N + j] = tmp;
                }
                
                // Swap in b (all K columns)
                for (uint c = 0; c < K; c++) {
                    float tmp = b_batch[k * K + c];
                    b_batch[k * K + c] = b_batch[max_row * K + c];
                    b_batch[max_row * K + c] = tmp;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        
        // Compute multipliers and update
        float pivot_val = A_batch[k * N + k];
        if (fabs(pivot_val) < 1e-10f) continue;  // Singular
        
        for (uint i = k + 1 + tid; i < N; i += tg_size) {
            float mult = A_batch[i * N + k] / pivot_val;
            A_batch[i * N + k] = mult;  // Store L below diagonal
            
            for (uint j = k + 1; j < N; j++) {
                A_batch[i * N + j] -= mult * A_batch[k * N + j];
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    
    // =========================================================================
    // Step 2: Forward Substitution (solve Ly = b)
    // L is unit lower triangular (diagonal = 1, stored below diagonal of A)
    // =========================================================================
    
    // Note: b has already been permuted during LU
    // We solve column by column of b
    for (uint col = tid; col < K; col += tg_size) {
        for (uint i = 1; i < N; i++) {
            float sum = 0.0f;
            for (uint j = 0; j < i; j++) {
                sum += A_batch[i * N + j] * b_batch[j * K + col];
            }
            b_batch[i * K + col] -= sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    
    // =========================================================================
    // Step 3: Back Substitution (solve Ux = y)
    // U is upper triangular (including diagonal of A)
    // =========================================================================
    
    for (uint col = tid; col < K; col += tg_size) {
        for (int i = (int)N - 1; i >= 0; i--) {
            float sum = 0.0f;
            for (uint j = (uint)i + 1; j < N; j++) {
                sum += A_batch[i * N + j] * b_batch[j * K + col];
            }
            float diag = A_batch[i * N + i];
            if (fabs(diag) > 1e-10f) {
                b_batch[i * K + col] = (b_batch[i * K + col] - sum) / diag;
            }
        }
    }
}
