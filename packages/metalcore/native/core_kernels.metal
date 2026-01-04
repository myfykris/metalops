// metalcore Metal kernels
// Foundational linear algebra primitives

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Triangular Solve (trsm)
// ============================================================================

// Forward substitution for lower triangular systems
// Solves Lx = b where L is lower triangular
kernel void trsm_lower_forward(
    device const float* L [[buffer(0)]],        // (N, N) lower triangular
    device float* X [[buffer(1)]],              // (N, K) - input b, output x
    constant uint& N [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    // Each threadgroup handles one column of X
    // Rows are processed sequentially due to data dependency
    if (gid >= K) return;
    
    uint col = gid;
    
    for (uint row = 0; row < N; row++) {
        // x[row, col] = (b[row, col] - sum(L[row, 0:row] * x[0:row, col])) / L[row, row]
        float sum = 0.0f;
        for (uint j = 0; j < row; j++) {
            sum += L[row * N + j] * X[j * K + col];
        }
        X[row * K + col] = (X[row * K + col] - sum) / L[row * N + row];
    }
}

// Back substitution for upper triangular systems
// Solves Ux = b where U is upper triangular
kernel void trsm_upper_backward(
    device const float* U [[buffer(0)]],        // (N, N) upper triangular
    device float* X [[buffer(1)]],              // (N, K) - input b, output x
    constant uint& N [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= K) return;
    
    uint col = gid;
    
    for (int row = N - 1; row >= 0; row--) {
        float sum = 0.0f;
        for (uint j = row + 1; j < N; j++) {
            sum += U[row * N + j] * X[j * K + col];
        }
        X[row * K + col] = (X[row * K + col] - sum) / U[row * N + row];
    }
}

// ============================================================================
// Householder Reflections
// ============================================================================

// Apply Householder reflection H = I - tau * v * v^T to matrix A
// Result: A = A - tau * v @ (v^T @ A)
kernel void apply_householder_left(
    device float* A [[buffer(0)]],              // (M, N) matrix to transform
    device const float* v [[buffer(1)]],        // (M,) Householder vector
    constant float& tau [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& start_row [[buffer(5)]],     // Apply to A[start_row:, :]
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgsize [[threads_per_threadgroup]]
) {
    // Each thread handles one element of the output
    uint row = gid.x + start_row;
    uint col = gid.y;
    
    if (row >= M || col >= N) return;
    
    // Compute v^T @ A[:, col] (need reduction)
    // For now, each thread computes full dot product (inefficient but correct)
    float vT_A_col = 0.0f;
    uint v_len = M - start_row;
    for (uint i = 0; i < v_len; i++) {
        vT_A_col += v[i] * A[(start_row + i) * N + col];
    }
    
    // A[row, col] -= tau * v[row - start_row] * vT_A_col
    A[row * N + col] -= tau * v[row - start_row] * vT_A_col;
}

// ============================================================================
// QR helpers
// ============================================================================

// Compute Householder vector v and tau for a column
// v, tau such that H @ x = ||x|| * e_1
kernel void householder_vector_kernel(
    device const float* x [[buffer(0)]],        // (N,) input vector
    device float* v [[buffer(1)]],              // (N,) output Householder vector
    device float* tau [[buffer(2)]],            // (1,) output scalar
    constant uint& N [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // This is a sequential operation - only thread 0 does the work
    if (gid != 0) return;
    
    // Compute norm of x[1:]
    float sigma = 0.0f;
    for (uint i = 1; i < N; i++) {
        sigma += x[i] * x[i];
    }
    
    // Copy x to v
    for (uint i = 0; i < N; i++) {
        v[i] = x[i];
    }
    
    if (sigma < 1e-10f) {
        // x is already along e_1
        tau[0] = 0.0f;
        return;
    }
    
    float norm_x = sqrt(x[0] * x[0] + sigma);
    float sign = (x[0] >= 0.0f) ? 1.0f : -1.0f;
    
    v[0] = x[0] + sign * norm_x;
    
    // tau = 2 * v[0]^2 / (v[0]^2 + sigma)
    tau[0] = 2.0f * v[0] * v[0] / (v[0] * v[0] + sigma);
    
    // Normalize v so v[0] = 1
    float v0 = v[0];
    for (uint i = 0; i < N; i++) {
        v[i] /= v0;
    }
}
