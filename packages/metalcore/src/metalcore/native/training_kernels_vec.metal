#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------------
// RMSNorm Vectorized Kernels (float4)
// -----------------------------------------------------------------------------

// Vectorized Forward Pass
kernel void rmsnorm_fwd_vec4(
    device const float4* X [[buffer(0)]],
    device const float4* W [[buffer(1)]],
    device float4* Y [[buffer(2)]],
    device float* Rstd [[buffer(3)]],
    constant uint& N [[buffer(4)]],   // Original scalar N
    constant float& eps [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 threadsPerThreadgroup [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.x;
    uint tid_x = tid.x;
    uint N_vec = N / 4;
    uint offset = row * N_vec;
    
    float sum_sq = 0.0f;
    
    // Vectorized load and reduction
    for (uint i = tid_x; i < N_vec; i += threadsPerThreadgroup.x) {
        float4 val = X[offset + i];
        sum_sq += dot(val, val);
    }
    
    // Threadgroup Reduction (same logic as scalar)
    float simd_sum_sq = simd_sum(sum_sq);
    
    threadgroup float shared_sums[32];
    
    if (simd_lane_id == 0) {
        shared_sums[simd_group_id] = simd_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float total_sum_sq = 0.0f;
    if (simd_group_id == 0) {
        uint num_simd_groups = (threadsPerThreadgroup.x + 31) / 32;
        float partial = 0.0f;
        if (simd_lane_id < num_simd_groups) {
            partial = shared_sums[simd_lane_id];
        }
        total_sum_sq = simd_sum(partial);
    }
    
    if (simd_group_id == 0 && simd_lane_id == 0) {
        shared_sums[0] = total_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total_sum_sq = shared_sums[0];
    
    // Compute Rstd
    float rstd = rsqrt(total_sum_sq / float(N) + eps);
    
    if (tid_x == 0) {
        Rstd[row] = rstd;
    }
    
    // Vectorized Write
    for (uint i = tid_x; i < N_vec; i += threadsPerThreadgroup.x) {
        float4 val = X[offset + i];
        float4 w = W[i];
        Y[offset + i] = val * rstd * w;
    }
}

// Vectorized Backward dx
kernel void rmsnorm_bwd_dx_vec4(
    device const float4* dY [[buffer(0)]],
    device const float4* X [[buffer(1)]],
    device const float* Rstd [[buffer(2)]],
    device const float4* W [[buffer(3)]],
    device float4* dX [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 threadsPerThreadgroup [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.x;
    uint N_vec = N / 4;
    uint offset = row * N_vec;
    uint tid_x = tid.x;
    
    float rstd = Rstd[row];
    
    float sum_dy_w_x = 0.0f;
    
    for (uint i = tid_x; i < N_vec; i += threadsPerThreadgroup.x) {
        float4 dy = dY[offset + i];
        float4 x_val = X[offset + i];
        float4 w = W[i];
        // Dot product sum
        sum_dy_w_x += dot(dy * w, x_val);
    }
    
    // Limit reduction
    float simd_sum_val = simd_sum(sum_dy_w_x);
    
    threadgroup float shared_sums[32];
    
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
    
    float term2_coeff = total_sum * rstd * rstd * rstd / float(N);
    
    for (uint i = tid_x; i < N_vec; i += threadsPerThreadgroup.x) {
        float4 dy = dY[offset + i];
        float4 x_val = X[offset + i];
        float4 w = W[i];
        
        float4 term1 = dy * w * rstd;
        float4 term2 = x_val * term2_coeff;
        
        dX[offset + i] = term1 - term2;
    }
}

// Vectorized Backward dw
kernel void rmsnorm_bwd_dw_vec4(
    device const float4* dY [[buffer(0)]],
    device const float4* X [[buffer(1)]],
    device const float* Rstd [[buffer(2)]],
    device float4* dW [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& B [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    uint N_vec = N / 4;
    if (id >= N_vec) return;
    
    float4 sum_val = float4(0.0f);
    
    for (uint row = 0; row < B; ++row) {
        float r = Rstd[row];
        uint offset = row * N_vec + id;
        sum_val += dY[offset] * X[offset] * r;
    }
    
    dW[id] = sum_val;
}
