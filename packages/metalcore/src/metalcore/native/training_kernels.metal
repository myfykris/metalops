#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------------
// RMSNorm Kernels
// -----------------------------------------------------------------------------

constant float EPSILON = 1e-5;

// Forward Pass:
// y = x * w * rsqrt(mean(x^2) + eps)
// One threadgroup per row (B rows).
// N is hidden dimension.
kernel void rmsnorm_fwd(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    device float* Rstd [[buffer(3)]], // Save for backward
    constant uint& N [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 threadsPerThreadgroup [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles one row
    uint row = tgid.x;
    uint tid_x = tid.x;
    
    // Offset for this row
    uint offset = row * N;
    
    // 1. Compute Sum of Squares (Reduction)
    float sum_sq = 0.0f;
    
    // Stride loop to cover N elements with limited threads
    for (uint i = tid_x; i < N; i += threadsPerThreadgroup.x) {
        float val = X[offset + i];
        sum_sq += val * val;
    }
    
    // Threadgroup Reduction
    // We use simd_sum and then shared memory for inter-simd reduction
    float simd_sum_sq = simd_sum(sum_sq);
    
    // Allocate shared memory for partial sums
    // Max 32 simdgroups in a 1024 thread block
    threadgroup float shared_sums[32];
    
    // Derived from args
    // uint simd_lane_id = simd_lane_id_in_group();
    // uint simd_group_id = simd_group_id_in_group();
    
    if (simd_lane_id == 0) {
        shared_sums[simd_group_id] = simd_sum_sq;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction by first warp
    float total_sum_sq = 0.0f;
    if (simd_group_id == 0) {
        // Only first warp/simdgroup works now
        // Determine number of active simdgroups
        uint num_simd_groups = (threadsPerThreadgroup.x + 31) / 32;
        
        float partial = 0.0f;
        if (simd_lane_id < num_simd_groups) {
            partial = shared_sums[simd_lane_id];
        }
        total_sum_sq = simd_sum(partial);
    }
    
    // Broadcast total_sum_sq to all threads via shared memory
    if (simd_group_id == 0 && simd_lane_id == 0) {
        shared_sums[0] = total_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total_sum_sq = shared_sums[0];
    
    // 2. Compute Rstd
    float rstd = rsqrt(total_sum_sq / float(N) + eps);
    
    // Save rstd for backward
    if (tid_x == 0) {
        Rstd[row] = rstd;
    }
    
    // 3. Write Output
    for (uint i = tid_x; i < N; i += threadsPerThreadgroup.x) {
        float val = X[offset + i];
        float w = W[i];
        Y[offset + i] = val * rstd * w;
    }
}


// Backward Pass Inputs:
// grad_y (B, N)
// x (B, N)
// rstd (B) - from forward
// w (N)
// Outputs:
// grad_x (B, N)
// grad_w (N) (Requires global reduction across B, handled separately or naively here)
// For now, we implement grad_x. grad_w is usually handled by a separate kernel or naive accumulation.
kernel void rmsnorm_bwd_dx(
    device const float* dY [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device const float* Rstd [[buffer(2)]],
    device const float* W [[buffer(3)]],
    device float* dX [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadsPerThreadgroup [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.x;
    uint offset = row * N;
    uint tid_x = tid.x;
    
    float rstd = Rstd[row];
    
    // Calculate dot(dX, X) part for the gradient formula
    // dL/dx = rstd * w * dY - rstd^3 * x * sum(dY * w * x) / N (Simplified derivation for RMSNorm)
    // Actually derivation:
    // y_i = x_i * w_i * rstd
    // Let inv_n = 1/N
    // dL/dx_i = (dL/dy_i * w_i * rstd) - (x_i * rstd^3 * inv_n * sum_j(dL/dy_j * w_j * x_j))
    
    float sum_dy_w_x = 0.0f;
    
    for (uint i = tid_x; i < N; i += threadsPerThreadgroup.x) {
        float dy = dY[offset + i];
        float x_val = X[offset + i];
        float w = W[i];
        sum_dy_w_x += dy * w * x_val;
    }
    
    // Reduction for sum_dy_w_x
    float simd_sum_val = simd_sum(sum_dy_w_x);
    
    threadgroup float shared_sums[32];
    // uint simd_lane_id = simd_lane_id_in_group();
    // uint simd_group_id = simd_group_id_in_group();
    
    if (simd_lane_id == 0) {
        shared_sums[simd_group_id] = simd_sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float total_sum = 0.0f;
    if (simd_group_id == 0) {
        uint num_simd_groups = (threadsPerThreadgroup.x + 31) / 32;
        float partial = 0.0f;
        if (simd_lane_id < num_simd_groups) {
            partial = shared_sums[simd_lane_id];
        }
        total_sum = simd_sum(partial);
    }
    
    if (simd_group_id == 0 && simd_lane_id == 0) {
        shared_sums[0] = total_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total_sum = shared_sums[0];
    
    // Final dX calc
    float term2_coeff = total_sum * rstd * rstd * rstd / float(N);
    
    for (uint i = tid_x; i < N; i += threadsPerThreadgroup.x) {
        float dy = dY[offset + i];
        float x_val = X[offset + i];
        float w = W[i];
        
        float term1 = dy * w * rstd;
        float term2 = x_val * term2_coeff;
        
        dX[offset + i] = term1 - term2;
    }
}

// Compute dW = sum(dY * X * rstd, dim=0)
// Grid: N threads (1 per column). Loop over B rows.
// Optimized for coalesced reads of dY and X.
kernel void rmsnorm_bwd_dw(
    device const float* dY [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device const float* Rstd [[buffer(2)]],
    device float* dW [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& B [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;
    
    float sum_val = 0.0f;
    
    // Loop over batch
    for (uint row = 0; row < B; ++row) {
        // Broadcast load of Rstd (same for all threads in warp)
        float r = Rstd[row];
        
        // Coalesced loads (id is consecutive)
        uint offset = row * N + id;
        
        sum_val += dY[offset] * X[offset] * r;
    }
    
    dW[id] = sum_val;
}

// -----------------------------------------------------------------------------
// AdamW Kernel
// -----------------------------------------------------------------------------

// Fused step:
// p = p - lr * (beta1 * m + (1-beta1)*g) / (sqrt(beta2*v + (1-beta2)*g*g) + eps) - lr * wd * p
// Vectorized for float4
kernel void adamw_step(
    device float4* params [[buffer(0)]],
    device const float4* grads [[buffer(1)]],
    device float4* exp_avg [[buffer(2)]],
    device float4* exp_avg_sq [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant float& bias_correction1 [[buffer(9)]],
    constant float& bias_correction2 [[buffer(10)]],
    uint id [[thread_position_in_grid]]
) {
    float4 p = params[id];
    float4 g = grads[id];
    float4 m = exp_avg[id];
    float4 v = exp_avg_sq[id];
    
    // Usually AdamW applies Weight Decay to param FIRST (decoupled)
    // p = p - lr * wd * p
    p = p - lr * weight_decay * p;
    
    // Update moments
    // m = beta1 * m + (1 - beta1) * g
    m = beta1 * m + (1.0f - beta1) * g;
    
    // v = beta2 * v + (1 - beta2) * g * g
    v = beta2 * v + (1.0f - beta2) * (g * g);
    
    // Bias correction
    // m_hat = m / (1 - beta1^t)
    // v_hat = v / (1 - beta2^t)
    // We pass bias_correction terms as scalars: bc1 = 1 - beta1^t, bc2 = 1 - beta2^t
    float4 m_hat = m / bias_correction1;
    float4 v_hat = v / bias_correction2;
    
    // Update param
    // p = p - lr * m_hat / (sqrt(v_hat) + eps)
    float4 denom = sqrt(v_hat) + eps;
    p = p - lr * (m_hat / denom);
    
    // Write back
    params[id] = p;
    exp_avg[id] = m;
    exp_avg_sq[id] = v;
}
