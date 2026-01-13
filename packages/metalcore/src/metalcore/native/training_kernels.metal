// Author: Kris Bailey
// Copyright 2026
// Email: kris@krisbailey.com
#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------------
// RMSNorm Kernels
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Fused Add + RMSNorm (vLLM-style optimization)
// Combines: residual = input + residual; output = rmsnorm(residual)
// Saves one memory round-trip compared to separate ops
// -----------------------------------------------------------------------------
kernel void fused_add_rmsnorm(
    device float* input [[buffer(0)]],       // [..., hidden_size] - overwritten with output
    device float* residual [[buffer(1)]],    // [..., hidden_size] - updated in-place
    device const float* W [[buffer(2)]],     // [hidden_size]
    device float* Rstd [[buffer(3)]],        // [B] - save for backward
    constant uint& N [[buffer(4)]],          // hidden_size
    constant float& eps [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 threadsPerThreadgroup [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.x;
    uint tid_x = tid.x;
    uint offset = row * N;
    
    // Pass 1: Add residual and compute sum of squares
    float sum_sq = 0.0f;
    
    for (uint i = tid_x; i < N; i += threadsPerThreadgroup.x) {
        // Fused add: input[i] + residual[i]
        float val = input[offset + i] + residual[offset + i];
        // Store fused value back to residual
        residual[offset + i] = val;
        sum_sq += val * val;
    }
    
    // Reduction (same as rmsnorm_fwd)
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
    
    float rstd = rsqrt(total_sum_sq / float(N) + eps);
    
    if (tid_x == 0) {
        Rstd[row] = rstd;
    }
    
    // Pass 2: Apply RMSNorm and write to input
    for (uint i = tid_x; i < N; i += threadsPerThreadgroup.x) {
        float val = residual[offset + i];
        float w = W[i];
        input[offset + i] = val * rstd * w;
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

// -----------------------------------------------------------------------------
// AdamW ILP=4 Kernel (DeepSpeed-style optimization)
// Process 4 float4 vectors per thread to hide memory latency
// -----------------------------------------------------------------------------
kernel void adamw_step_ilp4(
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
    constant uint& numel [[buffer(11)]],  // Number of float4 elements
    uint id [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    // ILP = 4: Each thread processes 4 float4 vectors
    constexpr int ILP = 4;
    
    // Stride through data with ILP
    for (uint base = id * ILP; base < numel; base += threads * ILP) {
        // Load all values into registers first (hide latency)
        float4 r_p[ILP];
        float4 r_g[ILP];
        float4 r_m[ILP];
        float4 r_v[ILP];
        
        // Coalesced loads with bounds checking
        for (int ii = 0; ii < ILP; ii++) {
            uint idx = base + ii;
            if (idx < numel) {
                r_p[ii] = params[idx];
                r_g[ii] = grads[idx];
                r_m[ii] = exp_avg[idx];
                r_v[ii] = exp_avg_sq[idx];
            }
        }
        
        // Compute updates (all in registers)
        for (int ii = 0; ii < ILP; ii++) {
            uint idx = base + ii;
            if (idx < numel) {
                // Weight decay (decoupled AdamW)
                r_p[ii] = r_p[ii] - lr * weight_decay * r_p[ii];
                
                // Update moments
                r_m[ii] = beta1 * r_m[ii] + (1.0f - beta1) * r_g[ii];
                r_v[ii] = beta2 * r_v[ii] + (1.0f - beta2) * (r_g[ii] * r_g[ii]);
                
                // Bias correction and parameter update
                float4 m_hat = r_m[ii] / bias_correction1;
                float4 v_hat = r_v[ii] / bias_correction2;
                float4 denom = sqrt(v_hat) + eps;
                r_p[ii] = r_p[ii] - lr * (m_hat / denom);
            }
        }
        
        // Write back (coalesced stores)
        for (int ii = 0; ii < ILP; ii++) {
            uint idx = base + ii;
            if (idx < numel) {
                params[idx] = r_p[ii];
                exp_avg[idx] = r_m[ii];
                exp_avg_sq[idx] = r_v[ii];
            }
        }
    }
}

// =============================================================================
// HALF PRECISION (fp16) VARIANTS
// =============================================================================

// -----------------------------------------------------------------------------
// RMSNorm Half Precision
// Note: Accumulation is done in float for numerical stability
// -----------------------------------------------------------------------------

kernel void rmsnorm_fwd_half(
    device const half* X [[buffer(0)]],
    device const half* W [[buffer(1)]],
    device half* Y [[buffer(2)]],
    device float* Rstd [[buffer(3)]], // Keep rstd in float for backward
    constant uint& N [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 threadsPerThreadgroup [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.x;
    uint tid_x = tid.x;
    uint offset = row * N;
    
    // Accumulate in float for stability
    float sum_sq = 0.0f;
    
    for (uint i = tid_x; i < N; i += threadsPerThreadgroup.x) {
        float val = float(X[offset + i]);
        sum_sq += val * val;
    }
    
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
    
    float rstd = rsqrt(total_sum_sq / float(N) + eps);
    
    if (tid_x == 0) {
        Rstd[row] = rstd;
    }
    
    for (uint i = tid_x; i < N; i += threadsPerThreadgroup.x) {
        float val = float(X[offset + i]);
        float w = float(W[i]);
        Y[offset + i] = half(val * rstd * w);
    }
}

// -----------------------------------------------------------------------------
// RMSNorm Half4 ILP=2 - 8-wide effective vectorization using 2 half4 loads
// Requires N to be divisible by 8
// -----------------------------------------------------------------------------
kernel void rmsnorm_fwd_half_vec(
    device const half4* X [[buffer(0)]],
    device const half4* W [[buffer(1)]],
    device half4* Y [[buffer(2)]],
    device float* Rstd [[buffer(3)]],
    constant uint& N4 [[buffer(4)]],  // N / 4
    constant float& eps [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 threadsPerThreadgroup [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.x;
    uint tid_x = tid.x;
    uint offset = row * N4;
    
    // Accumulate in float for stability
    float sum_sq = 0.0f;
    
    // ILP=2: Process 2 half4 vectors per iteration (8 elements total)
    for (uint i = tid_x * 2; i < N4; i += threadsPerThreadgroup.x * 2) {
        // Load 2 half4 vectors (hide latency)
        half4 vec0 = X[offset + i];
        half4 vec1 = (i + 1 < N4) ? X[offset + i + 1] : half4(0.0h);
        
        // Accumulate first vector
        sum_sq += float(vec0.x) * float(vec0.x);
        sum_sq += float(vec0.y) * float(vec0.y);
        sum_sq += float(vec0.z) * float(vec0.z);
        sum_sq += float(vec0.w) * float(vec0.w);
        
        // Accumulate second vector
        if (i + 1 < N4) {
            sum_sq += float(vec1.x) * float(vec1.x);
            sum_sq += float(vec1.y) * float(vec1.y);
            sum_sq += float(vec1.z) * float(vec1.z);
            sum_sq += float(vec1.w) * float(vec1.w);
        }
    }
    
    // Reduction
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
    
    uint N = N4 * 4;
    float rstd = rsqrt(total_sum_sq / float(N) + eps);
    
    if (tid_x == 0) {
        Rstd[row] = rstd;
    }
    
    // ILP=2: Process 2 half4 vectors per iteration
    for (uint i = tid_x * 2; i < N4; i += threadsPerThreadgroup.x * 2) {
        half4 x0 = X[offset + i];
        half4 w0 = W[i];
        half4 y0;
        y0.x = half(float(x0.x) * rstd * float(w0.x));
        y0.y = half(float(x0.y) * rstd * float(w0.y));
        y0.z = half(float(x0.z) * rstd * float(w0.z));
        y0.w = half(float(x0.w) * rstd * float(w0.w));
        Y[offset + i] = y0;
        
        if (i + 1 < N4) {
            half4 x1 = X[offset + i + 1];
            half4 w1 = W[i + 1];
            half4 y1;
            y1.x = half(float(x1.x) * rstd * float(w1.x));
            y1.y = half(float(x1.y) * rstd * float(w1.y));
            y1.z = half(float(x1.z) * rstd * float(w1.z));
            y1.w = half(float(x1.w) * rstd * float(w1.w));
            Y[offset + i + 1] = y1;
        }
    }
}


// -----------------------------------------------------------------------------
// RMSNorm BFloat16 Forward
// -----------------------------------------------------------------------------
#if __METAL_VERSION__ >= 310

kernel void rmsnorm_fwd_bfloat(
    device const bfloat* X [[buffer(0)]],
    device const bfloat* W [[buffer(1)]],
    device bfloat* Y [[buffer(2)]],
    device float* Rstd [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 threadsPerThreadgroup [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.x;
    uint tid_x = tid.x;
    uint offset = row * N;
    
    // Accumulate in float for stability
    float sum_sq = 0.0f;
    
    for (uint i = tid_x; i < N; i += threadsPerThreadgroup.x) {
        float val = float(X[offset + i]);
        sum_sq += val * val;
    }
    
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
    
    float rstd = rsqrt(total_sum_sq / float(N) + eps);
    
    if (tid_x == 0) {
        Rstd[row] = rstd;
    }
    
    for (uint i = tid_x; i < N; i += threadsPerThreadgroup.x) {
        float val = float(X[offset + i]);
        float w = float(W[i]);
        Y[offset + i] = bfloat(val * rstd * w);
    }
}

#endif // __METAL_VERSION__ >= 310


kernel void rmsnorm_bwd_dx_half(
    device const half* dY [[buffer(0)]],
    device const half* X [[buffer(1)]],
    device const float* Rstd [[buffer(2)]],
    device const half* W [[buffer(3)]],
    device half* dX [[buffer(4)]],
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
    float sum_dy_w_x = 0.0f;
    
    for (uint i = tid_x; i < N; i += threadsPerThreadgroup.x) {
        float dy = float(dY[offset + i]);
        float x_val = float(X[offset + i]);
        float w = float(W[i]);
        sum_dy_w_x += dy * w * x_val;
    }
    
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
    
    for (uint i = tid_x; i < N; i += threadsPerThreadgroup.x) {
        float dy = float(dY[offset + i]);
        float x_val = float(X[offset + i]);
        float w = float(W[i]);
        
        float term1 = dy * w * rstd;
        float term2 = x_val * term2_coeff;
        
        dX[offset + i] = half(term1 - term2);
    }
}

kernel void rmsnorm_bwd_dw_half(
    device const half* dY [[buffer(0)]],
    device const half* X [[buffer(1)]],
    device const float* Rstd [[buffer(2)]],
    device half* dW [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& B [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= N) return;
    
    float sum_val = 0.0f;
    
    for (uint row = 0; row < B; ++row) {
        float r = Rstd[row];
        uint offset = row * N + id;
        sum_val += float(dY[offset]) * float(X[offset]) * r;
    }
    
    dW[id] = half(sum_val);
}

// -----------------------------------------------------------------------------
// AdamW Half Precision
// Note: Optimizer state (exp_avg, exp_avg_sq) kept in float for stability
// -----------------------------------------------------------------------------

kernel void adamw_step_half(
    device half4* params [[buffer(0)]],
    device const half4* grads [[buffer(1)]],
    device float4* exp_avg [[buffer(2)]],   // Keep in float32
    device float4* exp_avg_sq [[buffer(3)]],// Keep in float32
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant float& bias_correction1 [[buffer(9)]],
    constant float& bias_correction2 [[buffer(10)]],
    uint id [[thread_position_in_grid]]
) {
    // Load params and grads, convert to float for computation
    float4 p = float4(params[id]);
    float4 g = float4(grads[id]);
    float4 m = exp_avg[id];
    float4 v = exp_avg_sq[id];
    
    // Weight decay (decoupled)
    p = p - lr * weight_decay * p;
    
    // Update moments (in float)
    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * (g * g);
    
    // Bias correction
    float4 m_hat = m / bias_correction1;
    float4 v_hat = v / bias_correction2;
    
    // Update param
    float4 denom = sqrt(v_hat) + eps;
    p = p - lr * (m_hat / denom);
    
    // Write back (params in half, state in float)
    params[id] = half4(p);
    exp_avg[id] = m;
    exp_avg_sq[id] = v;
}

// -----------------------------------------------------------------------------
// AdamW Half Precision - Scalar (for tail handling)
// -----------------------------------------------------------------------------

kernel void adamw_step_half_scalar(
    device half* params [[buffer(0)]],
    device const half* grads [[buffer(1)]],
    device float* exp_avg [[buffer(2)]],
    device float* exp_avg_sq [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant float& bias_correction1 [[buffer(9)]],
    constant float& bias_correction2 [[buffer(10)]],
    uint id [[thread_position_in_grid]]
) {
    float p = float(params[id]);
    float g = float(grads[id]);
    float m = exp_avg[id];
    float v = exp_avg_sq[id];
    
    // Weight decay (decoupled)
    p = p - lr * weight_decay * p;
    
    // Update moments
    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * (g * g);
    
    // Bias correction
    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;
    
    // Update param
    float denom = sqrt(v_hat) + eps;
    p = p - lr * (m_hat / denom);
    
    // Write back
    params[id] = half(p);
    exp_avg[id] = m;
    exp_avg_sq[id] = v;
}

// -----------------------------------------------------------------------------
// AdamW Half Precision - ILP=4 (DeepSpeed-style, for large tensors)
// Process 4 half4 vectors per thread to hide memory latency
// -----------------------------------------------------------------------------

kernel void adamw_step_half_ilp4(
    device half4* params [[buffer(0)]],
    device const half4* grads [[buffer(1)]],
    device float4* exp_avg [[buffer(2)]],
    device float4* exp_avg_sq [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant float& bias_correction1 [[buffer(9)]],
    constant float& bias_correction2 [[buffer(10)]],
    constant uint& numel [[buffer(11)]],
    uint id [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    constexpr int ILP = 4;
    float one_minus_beta1 = 1.0f - beta1;
    float one_minus_beta2 = 1.0f - beta2;
    float eps_sq = eps * eps;  // For rsqrt optimization
    
    for (uint base = id * ILP; base < numel; base += threads * ILP) {
        // Load all values into registers
        float4 r_p[ILP], r_g[ILP], r_m[ILP], r_v[ILP];
        
        for (int ii = 0; ii < ILP; ii++) {
            uint idx = base + ii;
            if (idx < numel) {
                r_p[ii] = float4(params[idx]);
                r_g[ii] = float4(grads[idx]);
                r_m[ii] = exp_avg[idx];
                r_v[ii] = exp_avg_sq[idx];
            }
        }
        
        // Compute updates in registers
        for (int ii = 0; ii < ILP; ii++) {
            uint idx = base + ii;
            if (idx < numel) {
                // Weight decay
                r_p[ii] = r_p[ii] - lr * weight_decay * r_p[ii];
                
                // Update moments
                r_m[ii] = beta1 * r_m[ii] + one_minus_beta1 * r_g[ii];
                r_v[ii] = beta2 * r_v[ii] + one_minus_beta2 * (r_g[ii] * r_g[ii]);
                
                // Bias correction + update (rsqrt optimization)
                float4 m_hat = r_m[ii] / bias_correction1;
                float4 v_hat = r_v[ii] / bias_correction2;
                r_p[ii] = r_p[ii] - lr * m_hat * rsqrt(v_hat + eps_sq);
            }
        }
        
        // Write back
        for (int ii = 0; ii < ILP; ii++) {
            uint idx = base + ii;
            if (idx < numel) {
                params[idx] = half4(r_p[ii]);
                exp_avg[idx] = r_m[ii];
                exp_avg_sq[idx] = r_v[ii];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// AdamW BFloat16 Precision

// Note: Optimizer state (exp_avg, exp_avg_sq) kept in float for stability
// bfloat16 requires Metal 3.1+ (macOS 14+)
// -----------------------------------------------------------------------------

#if __METAL_VERSION__ >= 310

kernel void adamw_step_bfloat(
    device bfloat4* params [[buffer(0)]],
    device const bfloat4* grads [[buffer(1)]],
    device float4* exp_avg [[buffer(2)]],   // Keep in float32
    device float4* exp_avg_sq [[buffer(3)]],// Keep in float32
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant float& bias_correction1 [[buffer(9)]],
    constant float& bias_correction2 [[buffer(10)]],
    uint id [[thread_position_in_grid]]
) {
    // Load params and grads, convert to float for computation
    float4 p = float4(params[id]);
    float4 g = float4(grads[id]);
    float4 m = exp_avg[id];
    float4 v = exp_avg_sq[id];
    
    // Weight decay (decoupled)
    p = p - lr * weight_decay * p;
    
    // Update moments (in float)
    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * (g * g);
    
    // Bias correction
    float4 m_hat = m / bias_correction1;
    float4 v_hat = v / bias_correction2;
    
    // Update param
    float4 denom = sqrt(v_hat) + eps;
    p = p - lr * (m_hat / denom);
    
    // Write back (params in bfloat, state in float)
    params[id] = bfloat4(p);
    exp_avg[id] = m;
    exp_avg_sq[id] = v;
}

// BFloat16 Scalar (for tail handling)
kernel void adamw_step_bfloat_scalar(
    device bfloat* params [[buffer(0)]],
    device const bfloat* grads [[buffer(1)]],
    device float* exp_avg [[buffer(2)]],
    device float* exp_avg_sq [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant float& bias_correction1 [[buffer(9)]],
    constant float& bias_correction2 [[buffer(10)]],
    uint id [[thread_position_in_grid]]
) {
    float p = float(params[id]);
    float g = float(grads[id]);
    float m = exp_avg[id];
    float v = exp_avg_sq[id];
    
    // Weight decay (decoupled)
    p = p - lr * weight_decay * p;
    
    // Update moments
    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * (g * g);
    
    // Bias correction
    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;
    
    // Update param
    float denom = sqrt(v_hat) + eps;
    p = p - lr * (m_hat / denom);
    
    // Write back
    params[id] = bfloat(p);
    exp_avg[id] = m;
    exp_avg_sq[id] = v;
}

// -----------------------------------------------------------------------------
// AdamW BFloat16 - ILP=4 (DeepSpeed-style, for large tensors)
// Process 4 bfloat4 vectors per thread to hide memory latency
// -----------------------------------------------------------------------------

kernel void adamw_step_bfloat_ilp4(
    device bfloat4* params [[buffer(0)]],
    device const bfloat4* grads [[buffer(1)]],
    device float4* exp_avg [[buffer(2)]],
    device float4* exp_avg_sq [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant float& bias_correction1 [[buffer(9)]],
    constant float& bias_correction2 [[buffer(10)]],
    constant uint& numel [[buffer(11)]],
    uint id [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    constexpr int ILP = 4;
    float one_minus_beta1 = 1.0f - beta1;
    float one_minus_beta2 = 1.0f - beta2;
    float eps_sq = eps * eps;  // For rsqrt optimization
    
    for (uint base = id * ILP; base < numel; base += threads * ILP) {
        // Load all values into registers
        float4 r_p[ILP], r_g[ILP], r_m[ILP], r_v[ILP];
        
        for (int ii = 0; ii < ILP; ii++) {
            uint idx = base + ii;
            if (idx < numel) {
                r_p[ii] = float4(params[idx]);
                r_g[ii] = float4(grads[idx]);
                r_m[ii] = exp_avg[idx];
                r_v[ii] = exp_avg_sq[idx];
            }
        }
        
        // Compute updates in registers
        for (int ii = 0; ii < ILP; ii++) {
            uint idx = base + ii;
            if (idx < numel) {
                // Weight decay
                r_p[ii] = r_p[ii] - lr * weight_decay * r_p[ii];
                
                // Update moments
                r_m[ii] = beta1 * r_m[ii] + one_minus_beta1 * r_g[ii];
                r_v[ii] = beta2 * r_v[ii] + one_minus_beta2 * (r_g[ii] * r_g[ii]);
                
                // Bias correction + update (rsqrt optimization)
                float4 m_hat = r_m[ii] / bias_correction1;
                float4 v_hat = r_v[ii] / bias_correction2;
                r_p[ii] = r_p[ii] - lr * m_hat * rsqrt(v_hat + eps_sq);
            }
        }
        
        // Write back
        for (int ii = 0; ii < ILP; ii++) {
            uint idx = base + ii;
            if (idx < numel) {
                params[idx] = bfloat4(r_p[ii]);
                exp_avg[idx] = r_m[ii];
                exp_avg_sq[idx] = r_v[ii];
            }
        }
    }
}

#endif // __METAL_VERSION__ >= 310

// =============================================================================
// Scalar Tail Kernels (for elements not divisible by 4)
// =============================================================================

// AdamW Scalar (float32) - handles tail elements
kernel void adamw_step_scalar(
    device float* params [[buffer(0)]],
    device const float* grads [[buffer(1)]],
    device float* exp_avg [[buffer(2)]],
    device float* exp_avg_sq [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant float& bias_correction1 [[buffer(9)]],
    constant float& bias_correction2 [[buffer(10)]],
    uint id [[thread_position_in_grid]]
) {
    float p = params[id];
    float g = grads[id];
    float m = exp_avg[id];
    float v = exp_avg_sq[id];
    
    // Weight Decay
    p = p - lr * weight_decay * p;
    
    // Update moments
    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * (g * g);
    
    // Bias correction
    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;
    
    // Update param (rsqrt optimization)
    float eps_sq = eps * eps;
    p = p - lr * m_hat * rsqrt(v_hat + eps_sq);
    
    // Write back
    params[id] = p;
    exp_avg[id] = m;
    exp_avg_sq[id] = v;
}

// =============================================================================
// RMSNorm Vectorized Kernels (float4 for bandwidth optimization)
// =============================================================================

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
    
    for (uint i = tid_x; i < N_vec; i += threadsPerThreadgroup.x) {
        float4 val = X[offset + i];
        sum_sq += dot(val, val);
    }
    
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
    
    float rstd = rsqrt(total_sum_sq / float(N) + eps);
    
    if (tid_x == 0) {
        Rstd[row] = rstd;
    }
    
    for (uint i = tid_x; i < N_vec; i += threadsPerThreadgroup.x) {
        float4 val = X[offset + i];
        float4 w = W[i];
        Y[offset + i] = val * rstd * w;
    }
}

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
        sum_dy_w_x += dot(dy * w, x_val);
    }
    
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

// =============================================================================
// FUSED SOFTMAX - Optimized Single-Pass with Threadgroup Memory
// =============================================================================
// Key optimization: Load entire row into threadgroup memory ONCE.
// All max/sum/normalize operations happen in fast on-chip SRAM.
// Based on Triton's fused softmax pattern.

// For rows that fit in threadgroup memory (dim <= 8192)
kernel void fused_softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& dim [[buffer(2)]],        // dimension to softmax over
    constant uint& outer_size [[buffer(3)]], // product of dims before softmax dim
    constant uint& inner_size [[buffer(4)]], // product of dims after softmax dim (usually 1)
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles one softmax row
    uint row = tgid.x;
    if (row >= outer_size) return;
    
    uint row_offset = row * dim * inner_size;
    uint tid_x = tid.x;
    
    // Threadgroup memory to cache the row - avoid repeated global reads
    // Max 8192 elements (32KB at fp32) - sufficient for most vocab sizes
    threadgroup float row_cache[8192];
    
    // PHASE 1: Load row into threadgroup memory + find max in one pass
    float local_max = -INFINITY;
    for (uint i = tid_x; i < dim; i += tg_size.x) {
        float val = input[row_offset + i * inner_size];
        row_cache[i] = val;  // Cache in threadgroup memory
        local_max = max(local_max, val);
    }
    
    // SIMD reduction for max
    float simd_max_val = simd_max(local_max);
    
    threadgroup float shared_reduce[32];
    if (simd_lane == 0) {
        shared_reduce[simd_group] = simd_max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float global_max = -INFINITY;
    if (simd_group == 0) {
        uint num_groups = (tg_size.x + 31) / 32;
        float partial = (simd_lane < num_groups) ? shared_reduce[simd_lane] : -INFINITY;
        global_max = simd_max(partial);
    }
    if (simd_group == 0 && simd_lane == 0) {
        shared_reduce[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = shared_reduce[0];
    
    // PHASE 2: Compute exp(x - max) in-place in threadgroup memory + sum
    float local_sum = 0.0f;
    for (uint i = tid_x; i < dim; i += tg_size.x) {
        float exp_val = exp(row_cache[i] - global_max);
        row_cache[i] = exp_val;  // Store exp result back to cache
        local_sum += exp_val;
    }
    
    // SIMD reduction for sum
    float simd_sum_val = simd_sum(local_sum);
    if (simd_lane == 0) {
        shared_reduce[simd_group] = simd_sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float global_sum = 0.0f;
    if (simd_group == 0) {
        uint num_groups = (tg_size.x + 31) / 32;
        float partial = (simd_lane < num_groups) ? shared_reduce[simd_lane] : 0.0f;
        global_sum = simd_sum(partial);
    }
    if (simd_group == 0 && simd_lane == 0) {
        shared_reduce[0] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = shared_reduce[0];
    
    float inv_sum = 1.0f / global_sum;
    
    // PHASE 3: Normalize from threadgroup cache and write to global memory
    // This is the ONLY global memory write
    for (uint i = tid_x; i < dim; i += tg_size.x) {
        output[row_offset + i * inner_size] = row_cache[i] * inv_sum;
    }
}

// Vectorized softmax for dim % 4 == 0
kernel void fused_softmax_vec4(
    device const float4* input [[buffer(0)]],
    device float4* output [[buffer(1)]],
    constant uint& dim [[buffer(2)]],        // original scalar dim
    constant uint& outer_size [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.x;
    if (row >= outer_size) return;
    
    uint dim_vec = dim / 4;
    uint row_offset = row * dim_vec;
    uint tid_x = tid.x;
    
    // Phase 1: Max with vectorized loads
    float local_max = -INFINITY;
    for (uint i = tid_x; i < dim_vec; i += tg_size.x) {
        float4 val = input[row_offset + i];
        local_max = max(local_max, max(max(val.x, val.y), max(val.z, val.w)));
    }
    
    float simd_max_val = simd_max(local_max);
    threadgroup float shared_max[32];
    if (simd_lane == 0) shared_max[simd_group] = simd_max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float global_max = -INFINITY;
    if (simd_group == 0) {
        uint ng = (tg_size.x + 31) / 32;
        float p = (simd_lane < ng) ? shared_max[simd_lane] : -INFINITY;
        global_max = simd_max(p);
    }
    if (simd_group == 0 && simd_lane == 0) shared_max[0] = global_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = shared_max[0];
    
    // Phase 2: Sum
    float local_sum = 0.0f;
    for (uint i = tid_x; i < dim_vec; i += tg_size.x) {
        float4 val = input[row_offset + i];
        float4 e = exp(val - global_max);
        local_sum += e.x + e.y + e.z + e.w;
    }
    
    float simd_sum_val = simd_sum(local_sum);
    threadgroup float shared_sum[32];
    if (simd_lane == 0) shared_sum[simd_group] = simd_sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float global_sum = 0.0f;
    if (simd_group == 0) {
        uint ng = (tg_size.x + 31) / 32;
        float p = (simd_lane < ng) ? shared_sum[simd_lane] : 0.0f;
        global_sum = simd_sum(p);
    }
    if (simd_group == 0 && simd_lane == 0) shared_sum[0] = global_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = shared_sum[0];
    
    float inv_sum = 1.0f / global_sum;
    
    // Phase 3: Write
    for (uint i = tid_x; i < dim_vec; i += tg_size.x) {
        float4 val = input[row_offset + i];
        output[row_offset + i] = exp(val - global_max) * inv_sum;
    }
}

// Half-precision softmax with native half types for 2x bandwidth
// Computes in float for numerical stability, reads/writes in half
kernel void fused_softmax_half(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    constant uint& outer_size [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.x;
    if (row >= outer_size) return;
    
    uint row_offset = row * dim;
    uint tid_x = tid.x;
    
    // Use float threadgroup cache for numerical stability
    threadgroup float row_cache[8192];
    
    // PHASE 1: Load half → float into cache + find max
    float local_max = -INFINITY;
    for (uint i = tid_x; i < dim; i += tg_size.x) {
        float val = float(input[row_offset + i]);  // half → float
        row_cache[i] = val;
        local_max = max(local_max, val);
    }
    
    // SIMD reduction for max
    float simd_max_val = simd_max(local_max);
    threadgroup float shared_reduce[32];
    if (simd_lane == 0) shared_reduce[simd_group] = simd_max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float global_max = -INFINITY;
    if (simd_group == 0) {
        uint ng = (tg_size.x + 31) / 32;
        float p = (simd_lane < ng) ? shared_reduce[simd_lane] : -INFINITY;
        global_max = simd_max(p);
    }
    if (simd_group == 0 && simd_lane == 0) shared_reduce[0] = global_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = shared_reduce[0];
    
    // PHASE 2: Compute exp in-place + sum
    float local_sum = 0.0f;
    for (uint i = tid_x; i < dim; i += tg_size.x) {
        float exp_val = exp(row_cache[i] - global_max);
        row_cache[i] = exp_val;
        local_sum += exp_val;
    }
    
    // SIMD reduction for sum
    float simd_sum_val = simd_sum(local_sum);
    if (simd_lane == 0) shared_reduce[simd_group] = simd_sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float global_sum = 0.0f;
    if (simd_group == 0) {
        uint ng = (tg_size.x + 31) / 32;
        float p = (simd_lane < ng) ? shared_reduce[simd_lane] : 0.0f;
        global_sum = simd_sum(p);
    }
    if (simd_group == 0 && simd_lane == 0) shared_reduce[0] = global_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = shared_reduce[0];
    
    float inv_sum = 1.0f / global_sum;
    
    // PHASE 3: Normalize and write as half
    for (uint i = tid_x; i < dim; i += tg_size.x) {
        output[row_offset + i] = half(row_cache[i] * inv_sum);  // float → half
    }
}

// =============================================================================
// BFloat16 Softmax with Direct Bit Truncation
// =============================================================================
// bf16 = upper 16 bits of fp32 (same exponent, truncated mantissa)
// Direct bit shift is faster than formal conversion

#if __METAL_VERSION__ >= 310

// Macros for guaranteed zero-overhead bf16↔fp32 conversion
// bf16 is literally upper 16 bits of fp32 - just bit manipulation
#define FLOAT_TO_BFLOAT_FAST(f) as_type<bfloat>(ushort(as_type<uint>(f) >> 16))
#define BFLOAT_TO_FLOAT_FAST(b) as_type<float>(uint(as_type<ushort>(b)) << 16)

kernel void fused_softmax_bfloat(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    constant uint& outer_size [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.x;
    if (row >= outer_size) return;
    
    uint row_offset = row * dim;
    uint tid_x = tid.x;
    
    // Float cache for numerical stability
    threadgroup float row_cache[8192];
    
    // PHASE 1: Load bf16 → fp32 into cache + find max
    float local_max = -INFINITY;
    for (uint i = tid_x; i < dim; i += tg_size.x) {
        float val = BFLOAT_TO_FLOAT_FAST(input[row_offset + i]);  // bf16 → fp32
        row_cache[i] = val;
        local_max = max(local_max, val);
    }
    
    // SIMD reduction for max
    float simd_max_val = simd_max(local_max);
    threadgroup float shared_reduce[32];
    if (simd_lane == 0) shared_reduce[simd_group] = simd_max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float global_max = -INFINITY;
    if (simd_group == 0) {
        uint ng = (tg_size.x + 31) / 32;
        float p = (simd_lane < ng) ? shared_reduce[simd_lane] : -INFINITY;
        global_max = simd_max(p);
    }
    if (simd_group == 0 && simd_lane == 0) shared_reduce[0] = global_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = shared_reduce[0];
    
    // PHASE 2: Compute exp in-place + sum
    float local_sum = 0.0f;
    for (uint i = tid_x; i < dim; i += tg_size.x) {
        float exp_val = exp(row_cache[i] - global_max);
        row_cache[i] = exp_val;
        local_sum += exp_val;
    }
    
    // SIMD reduction for sum
    float simd_sum_val = simd_sum(local_sum);
    if (simd_lane == 0) shared_reduce[simd_group] = simd_sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float global_sum = 0.0f;
    if (simd_group == 0) {
        uint ng = (tg_size.x + 31) / 32;
        float p = (simd_lane < ng) ? shared_reduce[simd_lane] : 0.0f;
        global_sum = simd_sum(p);
    }
    if (simd_group == 0 && simd_lane == 0) shared_reduce[0] = global_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = shared_reduce[0];
    
    float inv_sum = 1.0f / global_sum;
    
    // PHASE 3: Normalize and write as bf16 (direct bit truncation)
    for (uint i = tid_x; i < dim; i += tg_size.x) {
        output[row_offset + i] = FLOAT_TO_BFLOAT_FAST(row_cache[i] * inv_sum);
    }
}

#endif // __METAL_VERSION__ >= 310

// Half-precision layernorm with half4 vectorization
// Computes in float for stability, half4 for bandwidth
kernel void layernorm_fwd_half(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    device float* mean_out [[buffer(4)]],
    device float* rstd_out [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.x;
    uint offset = row * N;
    uint tid_x = tid.x;
    
    // PASS 1: Compute stats with half4 vectorized loads
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    uint N_vec = N / 4;
    device const half4* input_vec = (device const half4*)(input + offset);
    
    for (uint i = tid_x; i < N_vec; i += tg_size.x) {
        half4 h = input_vec[i];
        float4 v = float4(h);  // half4 → float4
        local_sum += v.x + v.y + v.z + v.w;
        local_sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    // Remainder
    for (uint i = N_vec * 4 + tid_x; i < N; i += tg_size.x) {
        float v = float(input[offset + i]);
        local_sum += v;
        local_sum_sq += v * v;
    }
    
    // SIMD reduction
    float simd_s = simd_sum(local_sum);
    float simd_sq = simd_sum(local_sum_sq);
    
    threadgroup float shared_s[32], shared_sq[32];
    if (simd_lane == 0) { shared_s[simd_group] = simd_s; shared_sq[simd_group] = simd_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float tot_s = 0, tot_sq = 0;
    if (simd_group == 0) {
        uint ng = (tg_size.x + 31) / 32;
        tot_s = simd_sum((simd_lane < ng) ? shared_s[simd_lane] : 0.0f);
        tot_sq = simd_sum((simd_lane < ng) ? shared_sq[simd_lane] : 0.0f);
    }
    if (simd_group == 0 && simd_lane == 0) { shared_s[0] = tot_s; shared_sq[0] = tot_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tot_s = shared_s[0]; tot_sq = shared_sq[0];
    
    float mean = tot_s / float(N);
    float rstd = rsqrt(tot_sq / float(N) - mean * mean + eps);
    
    if (tid_x == 0) { mean_out[row] = mean; rstd_out[row] = rstd; }
    
    // PASS 2: Normalize with half4 vectorized load/store
    device const half4* weight_vec = (device const half4*)weight;
    device const half4* bias_vec = (device const half4*)bias;
    device half4* output_vec = (device half4*)(output + offset);
    
    for (uint i = tid_x; i < N_vec; i += tg_size.x) {
        float4 v = float4(input_vec[i]);
        float4 w = float4(weight_vec[i]);
        float4 b = float4(bias_vec[i]);
        float4 normalized = (v - mean) * rstd;
        output_vec[i] = half4(normalized * w + b);
    }
    // Remainder
    for (uint i = N_vec * 4 + tid_x; i < N; i += tg_size.x) {
        float v = float(input[offset + i]);
        float normalized = (v - mean) * rstd;
        output[offset + i] = half(normalized * float(weight[i]) + float(bias[i]));
    }
}

// =============================================================================
// BFloat16 LayerNorm with Direct Bit Truncation
// =============================================================================
#if __METAL_VERSION__ >= 310

kernel void layernorm_fwd_bfloat(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const bfloat* bias [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    device float* mean_out [[buffer(4)]],
    device float* rstd_out [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.x;
    uint offset = row * N;
    uint tid_x = tid.x;
    
    // PASS 1: Compute stats (all in fp32)
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (uint i = tid_x; i < N; i += tg_size.x) {
        float v = BFLOAT_TO_FLOAT_FAST(input[offset + i]);  // bf16 → fp32
        local_sum += v;
        local_sum_sq += v * v;
    }
    
    // SIMD reduction
    float simd_s = simd_sum(local_sum);
    float simd_sq = simd_sum(local_sum_sq);
    
    threadgroup float shared_s[32], shared_sq[32];
    if (simd_lane == 0) { shared_s[simd_group] = simd_s; shared_sq[simd_group] = simd_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float tot_s = 0, tot_sq = 0;
    if (simd_group == 0) {
        uint ng = (tg_size.x + 31) / 32;
        tot_s = simd_sum((simd_lane < ng) ? shared_s[simd_lane] : 0.0f);
        tot_sq = simd_sum((simd_lane < ng) ? shared_sq[simd_lane] : 0.0f);
    }
    if (simd_group == 0 && simd_lane == 0) { shared_s[0] = tot_s; shared_sq[0] = tot_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tot_s = shared_s[0]; tot_sq = shared_sq[0];
    
    float mean = tot_s / float(N);
    float rstd = rsqrt(tot_sq / float(N) - mean * mean + eps);
    
    if (tid_x == 0) { mean_out[row] = mean; rstd_out[row] = rstd; }
    
    // PASS 2: Normalize and write as bf16 (direct bit truncation)
    for (uint i = tid_x; i < N; i += tg_size.x) {
        float v = BFLOAT_TO_FLOAT_FAST(input[offset + i]);
        float w = BFLOAT_TO_FLOAT_FAST(weight[i]);
        float b = BFLOAT_TO_FLOAT_FAST(bias[i]);
        float normalized = (v - mean) * rstd;
        output[offset + i] = FLOAT_TO_BFLOAT_FAST(normalized * w + b);
    }
}

#endif // __METAL_VERSION__ >= 310

// =============================================================================
// LAYERNORM - Optimized Vectorized with Minimal TG Memory
// =============================================================================
// Key optimizations:
// 1. float4 vectorized loads for 4x bandwidth
// 2. Minimal threadgroup memory (just reduction buffers)
// 3. SIMD shuffle reductions (no TG barriers in hot loop)
// y = (x - mean) / sqrt(var + eps) * weight + bias

kernel void layernorm_fwd(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* mean_out [[buffer(4)]],   // for backward
    device float* rstd_out [[buffer(5)]],   // for backward  
    constant uint& N [[buffer(6)]],         // normalized dimension
    constant float& eps [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.x;
    uint offset = row * N;
    uint tid_x = tid.x;
    
    // PASS 1: Compute mean and variance with vectorized loads
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Vectorized path for N divisible by 4
    uint N_vec = N / 4;
    device const float4* input_vec = (device const float4*)(input + offset);
    
    for (uint i = tid_x; i < N_vec; i += tg_size.x) {
        float4 v = input_vec[i];
        local_sum += v.x + v.y + v.z + v.w;
        local_sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    // Handle remainder
    for (uint i = N_vec * 4 + tid_x; i < N; i += tg_size.x) {
        float v = input[offset + i];
        local_sum += v;
        local_sum_sq += v * v;
    }
    
    // SIMD reduction (no TG barrier needed)
    float simd_s = simd_sum(local_sum);
    float simd_sq = simd_sum(local_sum_sq);
    
    // Cross-simdgroup reduction with minimal TG memory
    threadgroup float shared_s[32], shared_sq[32];
    if (simd_lane == 0) {
        shared_s[simd_group] = simd_s;
        shared_sq[simd_group] = simd_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float total_s = 0, total_sq = 0;
    if (simd_group == 0) {
        uint ng = (tg_size.x + 31) / 32;
        total_s = simd_sum((simd_lane < ng) ? shared_s[simd_lane] : 0.0f);
        total_sq = simd_sum((simd_lane < ng) ? shared_sq[simd_lane] : 0.0f);
    }
    if (simd_group == 0 && simd_lane == 0) {
        shared_s[0] = total_s;
        shared_sq[0] = total_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total_s = shared_s[0];
    total_sq = shared_sq[0];
    
    float mean = total_s / float(N);
    float var = total_sq / float(N) - mean * mean;
    float rstd = rsqrt(var + eps);
    
    // Save for backward
    if (tid_x == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }
    
    // PASS 2: Normalize with vectorized load/store
    device const float4* weight_vec = (device const float4*)weight;
    device const float4* bias_vec = (device const float4*)bias;
    device float4* output_vec = (device float4*)(output + offset);
    
    for (uint i = tid_x; i < N_vec; i += tg_size.x) {
        float4 v = input_vec[i];
        float4 w = weight_vec[i];
        float4 b = bias_vec[i];
        float4 normalized = (v - mean) * rstd;
        output_vec[i] = normalized * w + b;
    }
    // Handle remainder
    for (uint i = N_vec * 4 + tid_x; i < N; i += tg_size.x) {
        float v = input[offset + i];
        float normalized = (v - mean) * rstd;
        output[offset + i] = normalized * weight[i] + bias[i];
    }
}

// Fused Add + LayerNorm: y = layernorm(x + residual)
kernel void fused_add_layernorm(
    device const float* input [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device float* output [[buffer(4)]],
    device float* mean_out [[buffer(5)]],
    device float* rstd_out [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.x;
    uint offset = row * N;
    uint tid_x = tid.x;
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Fused add and accumulate stats
    for (uint i = tid_x; i < N; i += tg_size.x) {
        float val = input[offset + i] + residual[offset + i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Standard SIMD reduction
    float simd_s = simd_sum(local_sum);
    float simd_sq = simd_sum(local_sum_sq);
    
    threadgroup float sh_s[32], sh_sq[32];
    if (simd_lane == 0) { sh_s[simd_group] = simd_s; sh_sq[simd_group] = simd_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float tot_s = 0, tot_sq = 0;
    if (simd_group == 0) {
        uint ng = (tg_size.x + 31) / 32;
        tot_s = simd_sum((simd_lane < ng) ? sh_s[simd_lane] : 0.0f);
        tot_sq = simd_sum((simd_lane < ng) ? sh_sq[simd_lane] : 0.0f);
    }
    if (simd_group == 0 && simd_lane == 0) { sh_s[0] = tot_s; sh_sq[0] = tot_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tot_s = sh_s[0]; tot_sq = sh_sq[0];
    
    float mean = tot_s / float(N);
    float rstd = rsqrt(tot_sq / float(N) - mean * mean + eps);
    
    if (tid_x == 0) { mean_out[row] = mean; rstd_out[row] = rstd; }
    
    for (uint i = tid_x; i < N; i += tg_size.x) {
        float val = input[offset + i] + residual[offset + i];
        output[offset + i] = (val - mean) * rstd * weight[i] + bias[i];
    }
}

kernel void fused_add_layernorm_half(
    device const half* input [[buffer(0)]],
    device const half* residual [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    device half* output [[buffer(4)]],
    device float* mean_out [[buffer(5)]],
    device float* rstd_out [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.x;
    uint offset = row * N;
    uint tid_x = tid.x;
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Fused add and accumulate stats
    for (uint i = tid_x; i < N; i += tg_size.x) {
        float val = float(input[offset + i]) + float(residual[offset + i]);
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    float simd_s = simd_sum(local_sum);
    float simd_sq = simd_sum(local_sum_sq);
    
    threadgroup float sh_s[32], sh_sq[32];
    if (simd_lane == 0) { sh_s[simd_group] = simd_s; sh_sq[simd_group] = simd_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float tot_s = 0, tot_sq = 0;
    if (simd_group == 0) {
        uint ng = (tg_size.x + 31) / 32;
        tot_s = simd_sum((simd_lane < ng) ? sh_s[simd_lane] : 0.0f);
        tot_sq = simd_sum((simd_lane < ng) ? sh_sq[simd_lane] : 0.0f);
    }
    if (simd_group == 0 && simd_lane == 0) { sh_s[0] = tot_s; sh_sq[0] = tot_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tot_s = sh_s[0]; tot_sq = sh_sq[0];
    
    float mean = tot_s / float(N);
    float rstd = rsqrt(tot_sq / float(N) - mean * mean + eps);
    
    if (tid_x == 0) { mean_out[row] = mean; rstd_out[row] = rstd; }
    
    for (uint i = tid_x; i < N; i += tg_size.x) {
        float val = float(input[offset + i]) + float(residual[offset + i]);
        float nval = (val - mean) * rstd;
        output[offset + i] = half(nval * float(weight[i]) + float(bias[i]));
    }
}

kernel void fused_add_layernorm_bfloat(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* residual [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    device const bfloat* bias [[buffer(3)]],
    device bfloat* output [[buffer(4)]],
    device float* mean_out [[buffer(5)]],
    device float* rstd_out [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.x;
    uint offset = row * N;
    uint tid_x = tid.x;
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (uint i = tid_x; i < N; i += tg_size.x) {
        float val = float(input[offset + i]) + float(residual[offset + i]);
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    float simd_s = simd_sum(local_sum);
    float simd_sq = simd_sum(local_sum_sq);
    
    threadgroup float sh_s[32], sh_sq[32];
    if (simd_lane == 0) { sh_s[simd_group] = simd_s; sh_sq[simd_group] = simd_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float tot_s = 0, tot_sq = 0;
    if (simd_group == 0) {
        uint ng = (tg_size.x + 31) / 32;
        tot_s = simd_sum((simd_lane < ng) ? sh_s[simd_lane] : 0.0f);
        tot_sq = simd_sum((simd_lane < ng) ? sh_sq[simd_lane] : 0.0f);
    }
    if (simd_group == 0 && simd_lane == 0) { sh_s[0] = tot_s; sh_sq[0] = tot_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tot_s = sh_s[0]; tot_sq = sh_sq[0];
    
    float mean = tot_s / float(N);
    float rstd = rsqrt(tot_sq / float(N) - mean * mean + eps);
    
    if (tid_x == 0) { mean_out[row] = mean; rstd_out[row] = rstd; }
    
    for (uint i = tid_x; i < N; i += tg_size.x) {
        float val = float(input[offset + i]) + float(residual[offset + i]);
        float nval = (val - mean) * rstd;
        output[offset + i] = bfloat(nval * float(weight[i]) + float(bias[i]));
    }
}

// =============================================================================
// EMBEDDING BAG - Coalesced Reads + Parallel Reduction
// =============================================================================
// Supports sum, mean, max modes with per-sample weights

kernel void embedding_bag_sum(
    device const float* embeddings [[buffer(0)]],  // [num_embeddings, dim]
    device const uint* indices [[buffer(1)]],      // [total_indices]
    device const uint* offsets [[buffer(2)]],      // [batch_size + 1]
    device const float* weights [[buffer(3)]],     // [total_indices] or null
    device float* output [[buffer(4)]],            // [batch_size, dim]
    constant uint& dim [[buffer(5)]],
    constant uint& has_weights [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    uint batch_idx = gid.y;
    uint d = gid.x;
    
    if (d >= dim) return;
    
    uint start = offsets[batch_idx];
    uint end = offsets[batch_idx + 1];
    
    float sum = 0.0f;
    for (uint i = start; i < end; i++) {
        uint idx = indices[i];
        float val = embeddings[idx * dim + d];
        if (has_weights) {
            val *= weights[i];
        }
        sum += val;
    }
    
    output[batch_idx * dim + d] = sum;
}

// Simple 1D embedding bag - one thread per output element
kernel void embedding_bag_simple(
    device const float* embeddings [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],  
    device float* output [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    constant uint& mode [[buffer(6)]],  // 0=sum, 1=mean, 2=max
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.y;
    uint d = gid.x;
    
    if (batch_idx >= batch_size || d >= dim) return;
    
    uint start = offsets[batch_idx];
    uint end = offsets[batch_idx + 1];
    uint count = end - start;
    
    if (count == 0) {
        output[batch_idx * dim + d] = 0.0f;
        return;
    }
    
    float result;
    if (mode == 2) {  // max
        result = -INFINITY;
        for (uint i = start; i < end; i++) {
            uint idx = indices[i];
            result = max(result, embeddings[idx * dim + d]);
        }
    } else {  // sum or mean
        result = 0.0f;
        for (uint i = start; i < end; i++) {
            uint idx = indices[i];
            result += embeddings[idx * dim + d];
        }
        if (mode == 1) {  // mean
            result /= float(count);
        }
    }
    
    output[batch_idx * dim + d] = result;
}

// =============================================================================
// SCATTER / GATHER Operations
// =============================================================================

// Gather: out[i] = src[index[i]] - vectorized when possible
kernel void gather_1d(
    device const float* src [[buffer(0)]],
    device const uint* index [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& num_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_elements) return;
    out[id] = src[index[id]];
}

// Gather 2D: out[i, :] = src[index[i], :]
kernel void gather_2d(
    device const float* src [[buffer(0)]],
    device const uint* index [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& num_indices [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint idx = gid.y;
    uint d = gid.x;
    
    if (idx >= num_indices || d >= dim) return;
    
    uint src_row = index[idx];
    out[idx * dim + d] = src[src_row * dim + d];
}

// Scatter Add: dst[index[i]] += src[i] (uses atomic for thread safety)
kernel void scatter_add_1d(
    device atomic_float* dst [[buffer(0)]],
    device const uint* index [[buffer(1)]],
    device const float* src [[buffer(2)]],
    constant uint& num_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= num_elements) return;
    atomic_fetch_add_explicit(&dst[index[id]], src[id], memory_order_relaxed);
}

// Scatter Add 2D: dst[index[i], :] += src[i, :]
kernel void scatter_add_2d(
    device atomic_float* dst [[buffer(0)]],
    device const uint* index [[buffer(1)]],
    device const float* src [[buffer(2)]],
    constant uint& num_indices [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint idx = gid.y;
    uint d = gid.x;
    
    if (idx >= num_indices || d >= dim) return;
    
    uint dst_row = index[idx];
    atomic_fetch_add_explicit(&dst[dst_row * dim + d], src[idx * dim + d], memory_order_relaxed);
}

// Index Select: more general gather with dimension support
kernel void index_select(
    device const float* src [[buffer(0)]],
    device const uint* index [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& num_indices [[buffer(3)]],
    constant uint& src_dim_size [[buffer(4)]],  // size of indexed dimension in src
    constant uint& slice_size [[buffer(5)]],     // product of dims after indexed dim
    uint2 gid [[thread_position_in_grid]]
) {
    uint idx = gid.y;
    uint slice_pos = gid.x;
    
    if (idx >= num_indices || slice_pos >= slice_size) return;
    
    uint src_idx = index[idx];
    out[idx * slice_size + slice_pos] = src[src_idx * slice_size + slice_pos];
}

// -----------------------------------------------------------------------------
// RoPE (Rotary Position Embedding) Kernels
// -----------------------------------------------------------------------------
// Used by Llama, Mistral, Qwen, etc.
// Applies rotation to query/key vectors based on position.
//
// Math:
//   q_rot[..., 0::2] = q[..., 0::2] * cos - q[..., 1::2] * sin
//   q_rot[..., 1::2] = q[..., 1::2] * cos + q[..., 0::2] * sin
//
// Dimensions:
//   q/k: [batch, seq_len, num_heads, head_dim]
//   cos/sin: [seq_len, head_dim/2] or [1, seq_len, 1, head_dim/2] (broadcastable)

// Interleaved RoPE: pairs of consecutive elements get rotated together
// This is the standard HuggingFace format
kernel void rope_fwd_interleaved(
    device const float* qk [[buffer(0)]],      // Query or Key [B, S, H, D]
    device const float* cos [[buffer(1)]],      // [S, D/2]
    device const float* sin [[buffer(2)]],      // [S, D/2]
    device float* out [[buffer(3)]],            // Output [B, S, H, D]
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]       // (d, h, b*s)
) {
    uint d = gid.x;          // Dimension index (0 to head_dim-1)
    uint h = gid.y;          // Head index
    uint bs = gid.z;         // Batch * seq index
    
    if (d >= head_dim || h >= num_heads || bs >= batch_size * seq_len) return;
    
    uint b = bs / seq_len;   // Batch index
    uint s = bs % seq_len;   // Sequence position
    
    // Input offset: [b, s, h, d]
    uint offset = ((b * seq_len + s) * num_heads + h) * head_dim + d;
    
    // cos/sin offset: [s, d/2]
    uint half_d = d / 2;
    uint cs_offset = s * (head_dim / 2) + half_d;
    
    float c = cos[cs_offset];
    float sn = sin[cs_offset];
    
    // Interleaved rotation: even indices use (x0, x1), odd indices use (-x1, x0)
    float x = qk[offset];
    float x_pair;
    
    if (d % 2 == 0) {
        // Even index: pair with next element
        x_pair = qk[offset + 1];
        out[offset] = x * c - x_pair * sn;
    } else {
        // Odd index: pair with previous element
        x_pair = qk[offset - 1];
        out[offset] = x * c + x_pair * sn;
    }
}

// Half-precision version
kernel void rope_fwd_interleaved_half(
    device const half* qk [[buffer(0)]],
    device const half* cos [[buffer(1)]],
    device const half* sin [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint h = gid.y;
    uint bs = gid.z;
    
    if (d >= head_dim || h >= num_heads || bs >= batch_size * seq_len) return;
    
    uint b = bs / seq_len;
    uint s = bs % seq_len;
    
    uint offset = ((b * seq_len + s) * num_heads + h) * head_dim + d;
    uint half_d = d / 2;
    uint cs_offset = s * (head_dim / 2) + half_d;
    
    float c = float(cos[cs_offset]);
    float sn = float(sin[cs_offset]);
    
    float x = float(qk[offset]);
    float x_pair;
    
    if (d % 2 == 0) {
        x_pair = float(qk[offset + 1]);
        out[offset] = half(x * c - x_pair * sn);
    } else {
        x_pair = float(qk[offset - 1]);
        out[offset] = half(x * c + x_pair * sn);
    }
}

// Backward pass for RoPE (same rotation formula but different sign arrangement)
// d_qk = d_out rotated by -theta (same as forward but swap sin sign)
kernel void rope_bwd_interleaved(
    device const float* d_out [[buffer(0)]],    // Gradient from upstream [B, S, H, D]
    device const float* cos [[buffer(1)]],
    device const float* sin [[buffer(2)]],
    device float* d_qk [[buffer(3)]],           // Gradient to q/k [B, S, H, D]
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint h = gid.y;
    uint bs = gid.z;
    
    if (d >= head_dim || h >= num_heads || bs >= batch_size * seq_len) return;
    
    uint b = bs / seq_len;
    uint s = bs % seq_len;
    
    uint offset = ((b * seq_len + s) * num_heads + h) * head_dim + d;
    uint half_d = d / 2;
    uint cs_offset = s * (head_dim / 2) + half_d;
    
    float c = cos[cs_offset];
    float sn = sin[cs_offset];
    
    float dout = d_out[offset];
    float dout_pair;
    
    // Backward: rotate by -theta (swap sin sign)
    if (d % 2 == 0) {
        dout_pair = d_out[offset + 1];
        d_qk[offset] = dout * c + dout_pair * sn;  // Note: +sn instead of -sn
    } else {
        dout_pair = d_out[offset - 1];
        d_qk[offset] = dout * c - dout_pair * sn;  // Note: -sn instead of +sn
    }
}

// Fused RoPE for both Q and K at once (more efficient)
kernel void rope_fwd_qk_fused(
    device const float* Q [[buffer(0)]],        // [B, S, H, D]
    device const float* K [[buffer(1)]],        // [B, S, H_kv, D]
    device const float* cos [[buffer(2)]],      // [S, D/2]
    device const float* sin [[buffer(3)]],      // [S, D/2]
    device float* Q_out [[buffer(4)]],
    device float* K_out [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& num_heads_q [[buffer(8)]],
    constant uint& num_heads_kv [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint h = gid.y;
    uint bs = gid.z;
    
    uint b = bs / seq_len;
    uint s = bs % seq_len;
    
    if (d >= head_dim || bs >= batch_size * seq_len) return;
    
    uint half_d = d / 2;
    uint cs_offset = s * (head_dim / 2) + half_d;
    float c = cos[cs_offset];
    float sn = sin[cs_offset];
    
    // Process Q if this head is valid
    if (h < num_heads_q) {
        uint q_offset = ((b * seq_len + s) * num_heads_q + h) * head_dim + d;
        float q = Q[q_offset];
        float q_pair = (d % 2 == 0) ? Q[q_offset + 1] : Q[q_offset - 1];
        
        if (d % 2 == 0) {
            Q_out[q_offset] = q * c - q_pair * sn;
        } else {
            Q_out[q_offset] = q * c + q_pair * sn;
        }
    }
    
    // Process K if this head is valid for KV (may be fewer heads for GQA)
    if (h < num_heads_kv) {
        uint k_offset = ((b * seq_len + s) * num_heads_kv + h) * head_dim + d;
        float k = K[k_offset];
        float k_pair = (d % 2 == 0) ? K[k_offset + 1] : K[k_offset - 1];
        
        if (d % 2 == 0) {
            K_out[k_offset] = k * c - k_pair * sn;
        } else {
            K_out[k_offset] = k * c + k_pair * sn;
        }
    }
}

// =============================================================================
// Split-Half RoPE (HuggingFace/Liger style)
// =============================================================================
// This is the dominant format in HuggingFace Transformers.
// x1 = x[..., :D/2], x2 = x[..., D/2:]
// out[..., :D/2] = x1 * cos - x2 * sin
// out[..., D/2:] = x2 * cos + x1 * sin
//
// More memory-efficient: each thread handles one element in the first half
// and its corresponding element in the second half.

kernel void rope_fwd_split_half(
    device const float* qk [[buffer(0)]],       // [B, S, H, D]
    device const float* cos [[buffer(1)]],       // [S, D/2]
    device const float* sin [[buffer(2)]],       // [S, D/2]
    device float* out [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]        // (d, h, b*s) where d in [0, D/2)
) {
    uint d = gid.x;              // Index into first half [0, D/2)
    uint h = gid.y;              // Head index
    uint bs = gid.z;             // Batch * seq index
    
    uint half_dim = head_dim / 2;
    if (d >= half_dim || h >= num_heads || bs >= batch_size * seq_len) return;
    
    uint b = bs / seq_len;
    uint s = bs % seq_len;
    
    // Base offset for this (batch, seq, head)
    uint base = ((b * seq_len + s) * num_heads + h) * head_dim;
    
    // Load x1 (first half) and x2 (second half)
    float x1 = qk[base + d];
    float x2 = qk[base + half_dim + d];
    
    // cos/sin offset: [s, d]
    uint cs_offset = s * half_dim + d;
    float c = cos[cs_offset];
    float sn = sin[cs_offset];
    
    // Apply rotation
    out[base + d] = x1 * c - x2 * sn;
    out[base + half_dim + d] = x2 * c + x1 * sn;
}

// Half-precision split-half
kernel void rope_fwd_split_half_half(
    device const half* qk [[buffer(0)]],
    device const half* cos [[buffer(1)]],
    device const half* sin [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint h = gid.y;
    uint bs = gid.z;
    
    uint half_dim = head_dim / 2;
    if (d >= half_dim || h >= num_heads || bs >= batch_size * seq_len) return;
    
    uint b = bs / seq_len;
    uint s = bs % seq_len;
    
    uint base = ((b * seq_len + s) * num_heads + h) * head_dim;
    
    float x1 = float(qk[base + d]);
    float x2 = float(qk[base + half_dim + d]);
    
    uint cs_offset = s * half_dim + d;
    float c = float(cos[cs_offset]);
    float sn = float(sin[cs_offset]);
    
    out[base + d] = half(x1 * c - x2 * sn);
    out[base + half_dim + d] = half(x2 * c + x1 * sn);
}

// Split-half backward
kernel void rope_bwd_split_half(
    device const float* d_out [[buffer(0)]],
    device const float* cos [[buffer(1)]],
    device const float* sin [[buffer(2)]],
    device float* d_qk [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint h = gid.y;
    uint bs = gid.z;
    
    uint half_dim = head_dim / 2;
    if (d >= half_dim || h >= num_heads || bs >= batch_size * seq_len) return;
    
    uint b = bs / seq_len;
    uint s = bs % seq_len;
    
    uint base = ((b * seq_len + s) * num_heads + h) * head_dim;
    
    float dy1 = d_out[base + d];
    float dy2 = d_out[base + half_dim + d];
    
    uint cs_offset = s * half_dim + d;
    float c = cos[cs_offset];
    float sn = sin[cs_offset];
    
    // Backward: y1 = x1*c - x2*s, y2 = x2*c + x1*s
    // dx1 = dy1*c + dy2*s
    // dx2 = -dy1*s + dy2*c
    d_qk[base + d] = dy1 * c + dy2 * sn;
    d_qk[base + half_dim + d] = -dy1 * sn + dy2 * c;
}

// BFloat16 split-half forward
kernel void rope_fwd_split_half_bfloat(
    device const bfloat* qk [[buffer(0)]],
    device const bfloat* cos [[buffer(1)]],
    device const bfloat* sin [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint h = gid.y;
    uint bs = gid.z;
    
    uint half_dim = head_dim / 2;
    if (d >= half_dim || h >= num_heads || bs >= batch_size * seq_len) return;
    
    uint b = bs / seq_len;
    uint s = bs % seq_len;
    
    uint base = ((b * seq_len + s) * num_heads + h) * head_dim;
    
    float x1 = float(qk[base + d]);
    float x2 = float(qk[base + half_dim + d]);
    
    uint cs_offset = s * half_dim + d;
    float c = float(cos[cs_offset]);
    float sn = float(sin[cs_offset]);
    
    out[base + d] = bfloat(x1 * c - x2 * sn);
    out[base + half_dim + d] = bfloat(x2 * c + x1 * sn);
}

// BFloat16 split-half backward
// BFloat16 split-half backward (Vectorized)
kernel void rope_bwd_split_half_bfloat(
    device const bfloat* d_out [[buffer(0)]],
    device const bfloat* cos [[buffer(1)]],
    device const bfloat* sin [[buffer(2)]],
    device bfloat* d_qk [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // We process 4 elements per thread to amortize conversion cost
    uint d = gid.x * 4;
    uint h = gid.y;
    uint bs = gid.z;
    
    uint half_dim = head_dim / 2;
    if (d >= half_dim || h >= num_heads || bs >= batch_size * seq_len) return;
    
    uint b = bs / seq_len;
    uint s = bs % seq_len;
    
    uint base = ((b * seq_len + s) * num_heads + h) * head_dim;
    
    // Vectorized Load High/Low parts
    // Note: We cast to bfloat4* which requires 8-byte alignment? 
    // Usually head_dim is large enough, but indices are base+d.
    // base is (index * head_dim). If head_dim is multiple of 4, base+d is multiple of 4 (bfloat) = 8 bytes.
    // Safe for standard LLM shapes.
    
    bfloat4 dy1_b = *((device const bfloat4*)(d_out + base + d));
    bfloat4 dy2_b = *((device const bfloat4*)(d_out + base + half_dim + d));
    
    float4 dy1 = float4(dy1_b);
    float4 dy2 = float4(dy2_b);
    
    uint cs_offset = s * half_dim + d;
    bfloat4 c_b = *((device const bfloat4*)(cos + cs_offset));
    bfloat4 s_b = *((device const bfloat4*)(sin + cs_offset));
    
    float4 c = float4(c_b);
    float4 sn = float4(s_b);
    
    float4 res1 = dy1 * c + dy2 * sn;
    float4 res2 = -dy1 * sn + dy2 * c;
    
    *((device bfloat4*)(d_qk + base + d)) = bfloat4(res1);
    *((device bfloat4*)(d_qk + base + half_dim + d)) = bfloat4(res2);
}

// Fused Q+K split-half (in-place, matches Liger style)
kernel void rope_fwd_qk_split_half(
    device float* Q [[buffer(0)]],               // [B, S, H_q, D] - MODIFIED IN PLACE
    device float* K [[buffer(1)]],               // [B, S, H_kv, D] - MODIFIED IN PLACE
    device const float* cos [[buffer(2)]],       // [S, D/2]
    device const float* sin [[buffer(3)]],       // [S, D/2]
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& num_heads_q [[buffer(6)]],
    constant uint& num_heads_kv [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;               // [0, D/2)
    uint h = gid.y;               // Max of num_heads_q, num_heads_kv
    uint bs = gid.z;
    
    uint half_dim = head_dim / 2;
    if (d >= half_dim || bs >= batch_size * seq_len) return;
    
    uint b = bs / seq_len;
    uint s = bs % seq_len;
    
    uint cs_offset = s * half_dim + d;
    float c = cos[cs_offset];
    float sn = sin[cs_offset];
    
    // Process Q
    if (h < num_heads_q) {
        uint q_base = ((b * seq_len + s) * num_heads_q + h) * head_dim;
        float q1 = Q[q_base + d];
        float q2 = Q[q_base + half_dim + d];
        Q[q_base + d] = q1 * c - q2 * sn;
        Q[q_base + half_dim + d] = q2 * c + q1 * sn;
    }
    
    // Process K
    if (h < num_heads_kv) {
        uint k_base = ((b * seq_len + s) * num_heads_kv + h) * head_dim;
        float k1 = K[k_base + d];
        float k2 = K[k_base + half_dim + d];
        K[k_base + d] = k1 * c - k2 * sn;
        K[k_base + half_dim + d] = k2 * c + k1 * sn;
    }
}

// =============================================================================
// FUSED SwiGLU ACTIVATION
// =============================================================================
// SwiGLU(gate, up) = silu(gate) * up = (gate * sigmoid(gate)) * up
// This is the elementwise fusion part of the MLP. The matmuls (gate_proj,
// up_proj, down_proj) still use MPS for optimal performance.
//
// Typical MLP flow:
//   gate = x @ W_gate.T  (MPS matmul)
//   up = x @ W_up.T      (MPS matmul)  
//   hidden = swiglu(gate, up)  <-- THIS KERNEL
//   out = hidden @ W_down.T   (MPS matmul)
//
// This kernel fuses: hidden = silu(gate) * up

kernel void swiglu_fwd(
    device const float* gate [[buffer(0)]],    // [M, N]
    device const float* up [[buffer(1)]],      // [M, N]
    device float* out [[buffer(2)]],           // [M, N]
    constant uint& numel [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numel) return;
    
    float g = gate[id];
    float u = up[id];
    
    // silu(gate) * up = gate / (1 + exp(-gate)) * up
    float silu_g = g / (1.0f + exp(-g));
    out[id] = silu_g * u;
}

kernel void swiglu_fwd_half(
    device const half* gate [[buffer(0)]],
    device half* up [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numel) return;
    
    float g = float(gate[id]);
    float u = float(up[id]);
    
    float silu_g = g / (1.0f + exp(-g));
    out[id] = half(silu_g * u);
}

// Vectorized version for better memory bandwidth
kernel void swiglu_fwd_vec4(
    device const float4* gate [[buffer(0)]],
    device const float4* up [[buffer(1)]],
    device float4* out [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_vec = numel / 4;
    if (id >= numel_vec) return;
    
    float4 g = gate[id];
    float4 u = up[id];
    
    // Vectorized silu(gate) * up
    float4 silu_g = g / (1.0f + exp(-g));
    out[id] = silu_g * u;
}

kernel void swiglu_fwd_half_vec4(
    device const half4* gate [[buffer(0)]],
    device const half4* up [[buffer(1)]],
    device half4* out [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_vec = numel / 4;
    if (id >= numel_vec) return;
    
    float4 g = float4(gate[id]);
    float4 u = float4(up[id]);
    
    float4 silu_g = g / (1.0f + exp(-g));
    out[id] = half4(silu_g * u);
}

// -----------------------------------------------------------------------------
// Strided SwiGLU Kernels (2D Dispatch)
// Handles non-contiguous / transposed data correctly
// -----------------------------------------------------------------------------

kernel void swiglu_fwd_strided_half(
    device const half* gate [[buffer(0)]],
    device half* up [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant uint2& shape [[buffer(3)]],     // [rows, cols]
    constant uint2& s_gate [[buffer(4)]],    // [stride_0, stride_1]
    constant uint2& s_up [[buffer(5)]],
    constant uint2& s_out [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= shape.x || col >= shape.y) return;
    
    uint idx_g = row * s_gate.x + col * s_gate.y;
    uint idx_u = row * s_up.x + col * s_up.y;
    uint idx_o = row * s_out.x + col * s_out.y;
    
    float g = float(gate[idx_g]);
    float u = float(up[idx_u]);
    
    float silu_g = g / (1.0f + exp(-g));
    
    // In-place friendly
    out[idx_o] = half(silu_g * u);
}

kernel void swiglu_fwd_strided_bfloat(
    device const bfloat* gate [[buffer(0)]],
    device bfloat* up [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint2& shape [[buffer(3)]],
    constant uint2& s_gate [[buffer(4)]],
    constant uint2& s_up [[buffer(5)]],
    constant uint2& s_out [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= shape.x || col >= shape.y) return;
    
    uint idx_g = row * s_gate.x + col * s_gate.y;
    uint idx_u = row * s_up.x + col * s_up.y;
    uint idx_o = row * s_out.x + col * s_out.y;
    
    float g = float(gate[idx_g]);
    float u = float(up[idx_u]);
    
    float silu_g = g / (1.0f + exp(-g));
    
    out[idx_o] = bfloat(silu_g * u);
}

kernel void swiglu_fwd_strided_float(
    device const float* gate [[buffer(0)]],
    device float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint2& shape [[buffer(3)]],
    constant uint2& s_gate [[buffer(4)]],
    constant uint2& s_up [[buffer(5)]],
    constant uint2& s_out [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= shape.x || col >= shape.y) return;
    
    uint idx_g = row * s_gate.x + col * s_gate.y;
    uint idx_u = row * s_up.x + col * s_up.y;
    uint idx_o = row * s_out.x + col * s_out.y;
    
    float g = gate[idx_g];
    float u = up[idx_u];
    
    float silu_g = g / (1.0f + exp(-g));
    
    out[idx_o] = silu_g * u;
}

// =============================================================================
// FUSED LoRA LINEAR FORWARD
// =============================================================================
// Computes: y = x @ W.T + scale * (x @ A.T @ B.T)
// 
// Traditional approach (3+ kernel calls):
//   temp1 = x @ A.T           // Small: [M, r]
//   temp2 = temp1 @ B.T       // [M, N]
//   base = x @ W.T            // [M, N]
//   y = base + scale * temp2  // Elementwise
//
// This kernel fuses the final addition: y = base + scale * lora
// The matmuls still use MPS, but we save one kernel call + memory roundtrip.
//
// For full fusion (single kernel doing all matmuls), we'd need to beat MPS
// which is extremely difficult. This partial fusion is the pragmatic approach.

kernel void lora_add_fwd(
    device const float* base [[buffer(0)]],    // [M, N] from x @ W.T
    device const float* lora [[buffer(1)]],    // [M, N] from x @ A.T @ B.T
    device float* out [[buffer(2)]],           // [M, N]
    constant float& scale [[buffer(3)]],       // LoRA scaling factor (alpha/r)
    constant uint& numel [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numel) return;
    out[id] = base[id] + scale * lora[id];
}

kernel void lora_add_fwd_half(
    device const half* base [[buffer(0)]],
    device const half* lora [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    constant uint& numel [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numel) return;
    out[id] = half(float(base[id]) + scale * float(lora[id]));
}

kernel void lora_add_fwd_vec4(
    device const float4* base [[buffer(0)]],
    device const float4* lora [[buffer(1)]],
    device float4* out [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    constant uint& numel [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_vec = numel / 4;
    if (id >= numel_vec) return;
    out[id] = base[id] + scale * lora[id];
}

kernel void lora_add_fwd_half_vec4(
    device const half4* base [[buffer(0)]],
    device const half4* lora [[buffer(1)]],
    device half4* out [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    constant uint& numel [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = half4(float4(base[id]) + scale * float4(lora[id]));
}

// BFloat16 versions
kernel void swiglu_fwd_bfloat(
    device const bfloat* gate [[buffer(0)]],
    device const bfloat* up [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numel) return;
    
    float g = float(gate[id]);
    float u = float(up[id]);
    
    float silu_g = g / (1.0f + exp(-g));
    out[id] = bfloat(silu_g * u);
}

kernel void lora_add_fwd_bfloat(
    device const bfloat* base [[buffer(0)]],
    device const bfloat* lora [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    constant uint& numel [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numel) return;
    out[id] = bfloat(float(base[id]) + scale * float(lora[id]));
}




// =============================================================================
// FUSED CROSS-ENTROPY LOSS
// =============================================================================
// Computes: loss = -log(softmax(logits)[target])
// 
// This is critical for LLM training - avoids materializing full vocab softmax.
// For vocab_size=32K, this saves 32K * batch_size * 4 bytes per forward pass.
//
// Algorithm (numerically stable):
// 1. Find max logit for stability
// 2. Compute log-sum-exp: logsumexp = max + log(sum(exp(logits - max)))
// 3. loss = logsumexp - logits[target]
//
// Grid: One thread per sequence position (batch_size threads)

// =============================================================================
// CROSS-ENTROPY - THREADGROUP PARALLEL SIMD REDUCTION
// =============================================================================
// Uses same optimization strategy as softmax_bwd:
// - One threadgroup (256 threads) per batch element
// - Threads cooperate on finding max and computing sum_exp
// - simd_sum for fast SIMD-level reduction
// - ILP-8 unrolling for memory bandwidth
// - Online softmax variant for numerical stability

constant uint CE_THREADS = 256;

kernel void cross_entropy_fwd(
    device const float* logits [[buffer(0)]],    // [batch_size, vocab_size]
    device const int* targets [[buffer(1)]],     // [batch_size]
    device float* losses [[buffer(2)]],          // [batch_size]
    constant uint& batch_size [[buffer(3)]],
    constant uint& vocab_size [[buffer(4)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid_in_group [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]  // [16] for max and sum partials
) {
    if (batch_idx >= batch_size) return;
    
    device const float* row = logits + batch_idx * vocab_size;
    int target = targets[batch_idx];
    
    // Phase 1: Find max with parallel reduction
    float local_max = -INFINITY;
    uint stride = CE_THREADS;
    uint V8 = (vocab_size / 8) * 8;
    
    // ILP-8 unrolled max finding
    for (uint i = tid_in_group * 8; i < V8; i += stride * 8) {
        float v0 = row[i], v1 = row[i+1], v2 = row[i+2], v3 = row[i+3];
        float v4 = row[i+4], v5 = row[i+5], v6 = row[i+6], v7 = row[i+7];
        local_max = max(local_max, max(max(v0, v1), max(v2, v3)));
        local_max = max(local_max, max(max(v4, v5), max(v6, v7)));
    }
    for (uint i = V8 + tid_in_group; i < vocab_size; i += stride) {
        local_max = max(local_max, row[i]);
    }
    
    // SIMD reduction for max
    local_max = simd_max(local_max);
    if (simd_lane == 0) {
        shared[simd_group] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float global_max = -INFINITY;
    if (simd_group == 0) {
        if (simd_lane < 8) {
            global_max = shared[simd_lane];
        }
        global_max = simd_max(global_max);
        if (simd_lane == 0) {
            shared[0] = global_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = shared[0];
    
    // Phase 2: Compute sum_exp with parallel reduction
    float local_sum = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    
    for (uint i = tid_in_group * 8; i < V8; i += stride * 8) {
        acc0 += exp(row[i] - global_max);
        acc1 += exp(row[i+1] - global_max);
        acc2 += exp(row[i+2] - global_max);
        acc3 += exp(row[i+3] - global_max);
        acc0 += exp(row[i+4] - global_max);
        acc1 += exp(row[i+5] - global_max);
        acc2 += exp(row[i+6] - global_max);
        acc3 += exp(row[i+7] - global_max);
    }
    local_sum = acc0 + acc1 + acc2 + acc3;
    for (uint i = V8 + tid_in_group; i < vocab_size; i += stride) {
        local_sum += exp(row[i] - global_max);
    }
    
    // SIMD reduction for sum
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        shared[simd_group + 8] = local_sum;  // Use upper half of shared
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float global_sum = 0.0f;
    if (simd_group == 0) {
        if (simd_lane < 8) {
            global_sum = shared[simd_lane + 8];
        }
        global_sum = simd_sum(global_sum);
        if (simd_lane == 0) {
            // Compute final loss: log(sum_exp) + max - logits[target]
            float log_sum_exp = global_max + log(global_sum);
            losses[batch_idx] = log_sum_exp - row[target];
        }
    }
}

// Half precision version with same threadgroup-parallel strategy
kernel void cross_entropy_fwd_half(
    device const half* logits [[buffer(0)]],
    device const int* targets [[buffer(1)]],
    device float* losses [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& vocab_size [[buffer(4)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid_in_group [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    if (batch_idx >= batch_size) return;
    
    device const half* row = logits + batch_idx * vocab_size;
    int target = targets[batch_idx];
    
    // Phase 1: Find max with parallel reduction (compute in float)
    float local_max = -INFINITY;
    uint stride = CE_THREADS;
    uint V8 = (vocab_size / 8) * 8;
    
    for (uint i = tid_in_group * 8; i < V8; i += stride * 8) {
        // Load half4 pairs for memory coalescing
        half4 h_lo = *reinterpret_cast<device const half4*>(row + i);
        half4 h_hi = *reinterpret_cast<device const half4*>(row + i + 4);
        local_max = max(local_max, max(max(float(h_lo.x), float(h_lo.y)), max(float(h_lo.z), float(h_lo.w))));
        local_max = max(local_max, max(max(float(h_hi.x), float(h_hi.y)), max(float(h_hi.z), float(h_hi.w))));
    }
    for (uint i = V8 + tid_in_group; i < vocab_size; i += stride) {
        local_max = max(local_max, float(row[i]));
    }
    
    local_max = simd_max(local_max);
    if (simd_lane == 0) {
        shared[simd_group] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float global_max = -INFINITY;
    if (simd_group == 0) {
        if (simd_lane < 8) global_max = shared[simd_lane];
        global_max = simd_max(global_max);
        if (simd_lane == 0) shared[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = shared[0];
    
    // Phase 2: Compute sum_exp
    float local_sum = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    
    for (uint i = tid_in_group * 8; i < V8; i += stride * 8) {
        half4 h_lo = *reinterpret_cast<device const half4*>(row + i);
        half4 h_hi = *reinterpret_cast<device const half4*>(row + i + 4);
        acc0 += exp(float(h_lo.x) - global_max) + exp(float(h_lo.y) - global_max);
        acc1 += exp(float(h_lo.z) - global_max) + exp(float(h_lo.w) - global_max);
        acc2 += exp(float(h_hi.x) - global_max) + exp(float(h_hi.y) - global_max);
        acc3 += exp(float(h_hi.z) - global_max) + exp(float(h_hi.w) - global_max);
    }
    local_sum = acc0 + acc1 + acc2 + acc3;
    for (uint i = V8 + tid_in_group; i < vocab_size; i += stride) {
        local_sum += exp(float(row[i]) - global_max);
    }
    
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        shared[simd_group + 8] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float global_sum = 0.0f;
    if (simd_group == 0) {
        if (simd_lane < 8) global_sum = shared[simd_lane + 8];
        global_sum = simd_sum(global_sum);
        if (simd_lane == 0) {
            float log_sum_exp = global_max + log(global_sum);
            losses[batch_idx] = log_sum_exp - float(row[target]);
        }
    }
}

// =============================================================================
// KL DIVERGENCE FOR DISTILLATION
// =============================================================================
// Computes: KL(P || Q) = sum(P * log(P / Q)) = sum(P * (log_P - log_Q))
// Where P = teacher probs, Q = student probs
//
// For efficiency, we work with log-probs directly:
// KL = sum(exp(log_P) * (log_P - log_Q))
//
// Grid: One thread per sequence position

kernel void kl_div_fwd(
    device const float* log_p [[buffer(0)]],     // [batch, vocab]
    device const float* log_q [[buffer(1)]],     // [batch, vocab]
    device float* losses [[buffer(2)]],          // [batch]
    constant uint& batch_size [[buffer(3)]],
    constant uint& vocab_size [[buffer(4)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid_in_group [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    if (batch_idx >= batch_size) return;
    
    device const float* p_row = log_p + batch_idx * vocab_size;
    device const float* q_row = log_q + batch_idx * vocab_size;
    
    // Threadgroup-level reduction
    // ILP-4 unrolling
    float local_sum = 0.0f;
    uint stride = 256; // Assumes fixed threadgroup size of 256 for now, or passed in
    
    for (uint i = tid_in_group; i < vocab_size; i += stride) {
        float p_val = p_row[i];
        float q_val = q_row[i];
        float p_i = exp(p_val);
        local_sum += p_i * (p_val - q_val);
    }
    
    // SIMD reduction
    local_sum = simd_sum(local_sum);
    
    // Write to shared memory (one per SIMD group)
    if (simd_lane == 0) {
        shared[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction by first SIMD group
    if (simd_group == 0) {
        float group_sum = 0.0f;
        // Assuming max 256 threads -> 8 SIMD groups. 
        // We can just sum them all in lane 0 or reduced again.
        // Safer: load all partials (up to 8) and reduce.
        if (simd_lane < 8) { // Max 8 SIMD groups for 256 threads
            group_sum = shared[simd_lane];
        }
        group_sum = simd_sum(group_sum);
        
        if (simd_lane == 0) {
            losses[batch_idx] = group_sum;
        }
    }
}

// Top-K version for efficiency (only compute KL on top-k teacher tokens)
kernel void kl_div_topk_fwd(
    device const float* log_p [[buffer(0)]],     // Teacher log-probs [batch, vocab]
    device const float* log_q [[buffer(1)]],     // Student log-probs [batch, vocab]
    device const int* topk_indices [[buffer(2)]],// [batch, k] top-k token indices
    device float* losses [[buffer(3)]],          // [batch]
    constant uint& batch_size [[buffer(4)]],
    constant uint& vocab_size [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= batch_size) return;
    
    device const float* p_row = log_p + tid * vocab_size;
    device const float* q_row = log_q + tid * vocab_size;
    device const int* indices = topk_indices + tid * k;
    
    float kl = 0.0f;
    for (uint i = 0; i < k; i++) {
        int idx = indices[i];
        float p_i = exp(p_row[idx]);
        kl += p_i * (p_row[idx] - q_row[idx]);
    }
    
    losses[tid] = kl;
}

// =============================================================================
// FUSED LoRA QKV PROJECTION
// =============================================================================
// Computes Q, K, V with LoRA in one pass:
//   Q = W_q @ x + alpha * (B_q @ A_q @ x)
//   K = W_k @ x + alpha * (B_k @ A_k @ x)
//   V = W_v @ x + alpha * (B_v @ A_v @ x)
//
// This is a "scheduling" kernel - the actual matmuls use MPS, but we
// orchestrate them efficiently. For now, we provide the building block:
// a fused LoRA linear that can be called for each of Q, K, V.
//
// The real optimization is at the Python/C++ level where we:
// 1. Batch the x @ A.T computation for all 3 projections if A matrices are same
// 2. Use command buffer fusion to avoid sync overhead

// Single LoRA linear: y = base + scale * lora
// Already implemented as lora_add_fwd above

// LoRA matmul helper: computes x @ A.T @ B.T in sequence
// For actual use, we dispatch two MPS matmuls in the same command buffer
// This kernel is just the final addition if needed separately

kernel void lora_linear_fwd(
    device const float* x [[buffer(0)]],         // [M, in_features]
    device const float* W [[buffer(1)]],         // [out_features, in_features]
    device const float* A [[buffer(2)]],         // [rank, in_features]
    device const float* B [[buffer(3)]],         // [out_features, rank]
    device float* out [[buffer(4)]],             // [M, out_features]
    constant float& scale [[buffer(5)]],         // alpha / rank
    constant uint& M [[buffer(6)]],              // batch
    constant uint& in_features [[buffer(7)]],
    constant uint& out_features [[buffer(8)]],
    constant uint& rank [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = gid.y;
    uint n = gid.x;
    
    if (m >= M || n >= out_features) return;
    
    // Compute base: x @ W.T at position [m, n]
    float base_val = 0.0f;
    for (uint k = 0; k < in_features; k++) {
        base_val += x[m * in_features + k] * W[n * in_features + k];
    }
    
    // Compute LoRA: (x @ A.T) @ B.T at position [m, n]
    // First compute x @ A.T to get [M, rank]
    float lora_val = 0.0f;
    for (uint r = 0; r < rank; r++) {
        float xa = 0.0f;
        for (uint k = 0; k < in_features; k++) {
            xa += x[m * in_features + k] * A[r * in_features + k];
        }
        lora_val += xa * B[n * rank + r];
    }
    
    out[m * out_features + n] = base_val + scale * lora_val;
}

// =============================================================================
// SOFTMAX BACKWARD - THREADGROUP PARALLEL SIMD REDUCTION
// =============================================================================
// Computes: dX = softmax * (dY - sum(softmax * dY))
//
// OPTIMIZATION STRATEGY (matching PyTorch CUDA):
// - One threadgroup per row (batch*head element)
// - Threads within threadgroup cooperate on reduction
// - Use simdgroup_sum for fast SIMD-level reduction
// - Then use threadgroup shared memory for cross-SIMD reduction
// - ILP-8 unrolling for memory bandwidth

// Threadgroup size: 256 threads = 8 SIMDgroups of 32
constant uint SOFTMAX_BWD_THREADS = 256;
constant uint SIMD_SIZE = 32;

kernel void softmax_bwd(
    device const float* probs [[buffer(0)]],      // [N, L] softmax output
    device const float* d_probs [[buffer(1)]],    // [N, L] upstream gradient
    device float* d_logits [[buffer(2)]],         // [N, L] output gradient
    constant uint& N [[buffer(3)]],               // batch * heads  
    constant uint& L [[buffer(4)]],               // sequence length
    uint row_idx [[threadgroup_position_in_grid]],
    uint tid_in_group [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]  // [8] for simd group partial sums
) {
    if (row_idx >= N) return;
    
    device const float* p_row = probs + row_idx * L;
    device const float* dp_row = d_probs + row_idx * L;
    device float* dx_row = d_logits + row_idx * L;
    
    // Phase 1: Parallel reduction for dot = sum(p * dp)
    // Each thread accumulates a partial sum with ILP-8 unrolling
    float local_dot = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    
    uint stride = SOFTMAX_BWD_THREADS;
    uint L8 = (L / 8) * 8;  // Round down to multiple of 8
    
    // Each thread processes elements stride apart, ILP-8 unrolled
    for (uint i = tid_in_group * 8; i < L8; i += stride * 8) {
        // Load 8 elements at a time (ILP-8)
        float p0 = p_row[i], p1 = p_row[i+1], p2 = p_row[i+2], p3 = p_row[i+3];
        float p4 = p_row[i+4], p5 = p_row[i+5], p6 = p_row[i+6], p7 = p_row[i+7];
        float dp0 = dp_row[i], dp1 = dp_row[i+1], dp2 = dp_row[i+2], dp3 = dp_row[i+3];
        float dp4 = dp_row[i+4], dp5 = dp_row[i+5], dp6 = dp_row[i+6], dp7 = dp_row[i+7];
        
        acc0 += p0 * dp0;
        acc1 += p1 * dp1;
        acc2 += p2 * dp2;
        acc3 += p3 * dp3;
        acc4 += p4 * dp4;
        acc5 += p5 * dp5;
        acc6 += p6 * dp6;
        acc7 += p7 * dp7;
    }
    
    local_dot = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
    
    // Handle remaining elements
    for (uint i = L8 + tid_in_group; i < L; i += stride) {
        local_dot += p_row[i] * dp_row[i];
    }
    
    // SIMD-level reduction using simd_sum
    local_dot = simd_sum(local_dot);
    
    // Store SIMD group partial sum to shared memory
    if (simd_lane == 0) {
        shared[simd_group] = local_dot;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction across SIMD groups (only first SIMD group)
    float dot = 0.0f;
    if (simd_group == 0) {
        uint num_simd_groups = SOFTMAX_BWD_THREADS / SIMD_SIZE;  // 8
        if (simd_lane < num_simd_groups) {
            dot = shared[simd_lane];
        }
        dot = simd_sum(dot);
        // Broadcast to all threads in first simd group
        if (simd_lane == 0) {
            shared[0] = dot;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dot = shared[0];  // All threads read the final dot product
    
    // Phase 2: Compute output dx = p * (dp - dot)
    // Each thread writes strided elements, ILP-8 unrolled
    for (uint i = tid_in_group * 8; i < L8; i += stride * 8) {
        float p0 = p_row[i], p1 = p_row[i+1], p2 = p_row[i+2], p3 = p_row[i+3];
        float p4 = p_row[i+4], p5 = p_row[i+5], p6 = p_row[i+6], p7 = p_row[i+7];
        float dp0 = dp_row[i], dp1 = dp_row[i+1], dp2 = dp_row[i+2], dp3 = dp_row[i+3];
        float dp4 = dp_row[i+4], dp5 = dp_row[i+5], dp6 = dp_row[i+6], dp7 = dp_row[i+7];
        
        dx_row[i] = p0 * (dp0 - dot);
        dx_row[i+1] = p1 * (dp1 - dot);
        dx_row[i+2] = p2 * (dp2 - dot);
        dx_row[i+3] = p3 * (dp3 - dot);
        dx_row[i+4] = p4 * (dp4 - dot);
        dx_row[i+5] = p5 * (dp5 - dot);
        dx_row[i+6] = p6 * (dp6 - dot);
        dx_row[i+7] = p7 * (dp7 - dot);
    }
    
    // Handle remaining elements
    for (uint i = L8 + tid_in_group; i < L; i += stride) {
        dx_row[i] = p_row[i] * (dp_row[i] - dot);
    }
}

// Half precision version with half8 vectorization
kernel void softmax_bwd_half(
    device const half* probs [[buffer(0)]],
    device const half* d_probs [[buffer(1)]],
    device half* d_logits [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& L [[buffer(4)]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint tid_in_group [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    if (row_idx >= N) return;
    
    device const half* p_row = probs + row_idx * L;
    device const half* dp_row = d_probs + row_idx * L;
    device half* dx_row = d_logits + row_idx * L;
    
    // Phase 1: Parallel reduction with half8 vectorization
    float local_dot = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    
    uint stride = SOFTMAX_BWD_THREADS;
    uint L8 = (L / 8) * 8;
    
    // Process 8 half elements at a time per thread
    for (uint i = tid_in_group * 8; i < L8; i += stride * 8) {
        // Load as half4 pairs for better memory coalescing
        half4 p_lo = *reinterpret_cast<device const half4*>(p_row + i);
        half4 p_hi = *reinterpret_cast<device const half4*>(p_row + i + 4);
        half4 dp_lo = *reinterpret_cast<device const half4*>(dp_row + i);
        half4 dp_hi = *reinterpret_cast<device const half4*>(dp_row + i + 4);
        
        // Accumulate in float for precision
        acc0 += float(p_lo.x) * float(dp_lo.x) + float(p_lo.y) * float(dp_lo.y);
        acc1 += float(p_lo.z) * float(dp_lo.z) + float(p_lo.w) * float(dp_lo.w);
        acc2 += float(p_hi.x) * float(dp_hi.x) + float(p_hi.y) * float(dp_hi.y);
        acc3 += float(p_hi.z) * float(dp_hi.z) + float(p_hi.w) * float(dp_hi.w);
    }
    
    local_dot = acc0 + acc1 + acc2 + acc3;
    
    for (uint i = L8 + tid_in_group; i < L; i += stride) {
        local_dot += float(p_row[i]) * float(dp_row[i]);
    }
    
    // SIMD reduction
    local_dot = simd_sum(local_dot);
    
    if (simd_lane == 0) {
        shared[simd_group] = local_dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float dot = 0.0f;
    if (simd_group == 0) {
        if (simd_lane < 8) {
            dot = shared[simd_lane];
        }
        dot = simd_sum(dot);
        if (simd_lane == 0) {
            shared[0] = dot;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dot = shared[0];
    
    // Phase 2: Output with half4 vectorized writes
    for (uint i = tid_in_group * 8; i < L8; i += stride * 8) {
        half4 p_lo = *reinterpret_cast<device const half4*>(p_row + i);
        half4 p_hi = *reinterpret_cast<device const half4*>(p_row + i + 4);
        half4 dp_lo = *reinterpret_cast<device const half4*>(dp_row + i);
        half4 dp_hi = *reinterpret_cast<device const half4*>(dp_row + i + 4);
        
        half4 dx_lo = half4(
            half(float(p_lo.x) * (float(dp_lo.x) - dot)),
            half(float(p_lo.y) * (float(dp_lo.y) - dot)),
            half(float(p_lo.z) * (float(dp_lo.z) - dot)),
            half(float(p_lo.w) * (float(dp_lo.w) - dot))
        );
        half4 dx_hi = half4(
            half(float(p_hi.x) * (float(dp_hi.x) - dot)),
            half(float(p_hi.y) * (float(dp_hi.y) - dot)),
            half(float(p_hi.z) * (float(dp_hi.z) - dot)),
            half(float(p_hi.w) * (float(dp_hi.w) - dot))
        );
        
        *reinterpret_cast<device half4*>(dx_row + i) = dx_lo;
        *reinterpret_cast<device half4*>(dx_row + i + 4) = dx_hi;
    }
    
    for (uint i = L8 + tid_in_group; i < L; i += stride) {
        dx_row[i] = half(float(p_row[i]) * (float(dp_row[i]) - dot));
    }
}

// BFloat16 version
kernel void softmax_bwd_bfloat(
    device const bfloat* probs [[buffer(0)]],
    device const bfloat* d_probs [[buffer(1)]],
    device bfloat* d_logits [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& L [[buffer(4)]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint tid_in_group [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    if (row_idx >= N) return;
    
    device const bfloat* p_row = probs + row_idx * L;
    device const bfloat* dp_row = d_probs + row_idx * L;
    device bfloat* dx_row = d_logits + row_idx * L;
    
    float local_dot = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    
    uint stride = SOFTMAX_BWD_THREADS;
    uint L8 = (L / 8) * 8;
    
    // Process 8 bfloat elements (2x bfloat4)
    for (uint i = tid_in_group * 8; i < L8; i += stride * 8) {
        bfloat4 p_lo = *reinterpret_cast<device const bfloat4*>(p_row + i);
        bfloat4 p_hi = *reinterpret_cast<device const bfloat4*>(p_row + i + 4);
        bfloat4 dp_lo = *reinterpret_cast<device const bfloat4*>(dp_row + i);
        bfloat4 dp_hi = *reinterpret_cast<device const bfloat4*>(dp_row + i + 4);
        
        acc0 += float(p_lo.x) * float(dp_lo.x) + float(p_lo.y) * float(dp_lo.y);
        acc1 += float(p_lo.z) * float(dp_lo.z) + float(p_lo.w) * float(dp_lo.w);
        acc2 += float(p_hi.x) * float(dp_hi.x) + float(p_hi.y) * float(dp_hi.y);
        acc3 += float(p_hi.z) * float(dp_hi.z) + float(p_hi.w) * float(dp_hi.w);
    }
    
    local_dot = acc0 + acc1 + acc2 + acc3;
    
    for (uint i = L8 + tid_in_group; i < L; i += stride) {
        local_dot += float(p_row[i]) * float(dp_row[i]);
    }
    
    // SIMD reduction
    local_dot = simd_sum(local_dot);
    
    if (simd_lane == 0) {
        shared[simd_group] = local_dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float dot = 0.0f;
    if (simd_group == 0) {
        if (simd_lane < 8) {
            dot = shared[simd_lane];
        }
        dot = simd_sum(dot);
        if (simd_lane == 0) {
            shared[0] = dot;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dot = shared[0];
    
    // Output
    for (uint i = tid_in_group * 8; i < L8; i += stride * 8) {
        bfloat4 p_lo = *reinterpret_cast<device const bfloat4*>(p_row + i);
        bfloat4 p_hi = *reinterpret_cast<device const bfloat4*>(p_row + i + 4);
        bfloat4 dp_lo = *reinterpret_cast<device const bfloat4*>(dp_row + i);
        bfloat4 dp_hi = *reinterpret_cast<device const bfloat4*>(dp_row + i + 4);
        
        bfloat4 dx_lo = bfloat4(
            bfloat(float(p_lo.x) * (float(dp_lo.x) - dot)),
            bfloat(float(p_lo.y) * (float(dp_lo.y) - dot)),
            bfloat(float(p_lo.z) * (float(dp_lo.z) - dot)),
            bfloat(float(p_lo.w) * (float(dp_lo.w) - dot))
        );
        bfloat4 dx_hi = bfloat4(
            bfloat(float(p_hi.x) * (float(dp_hi.x) - dot)),
            bfloat(float(p_hi.y) * (float(dp_hi.y) - dot)),
            bfloat(float(p_hi.z) * (float(dp_hi.z) - dot)),
            bfloat(float(p_hi.w) * (float(dp_hi.w) - dot))
        );
        
        *reinterpret_cast<device bfloat4*>(dx_row + i) = dx_lo;
        *reinterpret_cast<device bfloat4*>(dx_row + i + 4) = dx_hi;
    }
    
    for (uint i = L8 + tid_in_group; i < L; i += stride) {
        dx_row[i] = bfloat(float(p_row[i]) * (float(dp_row[i]) - dot));
    }
}

// =============================================================================
// FUSED LoRA QKV PROJECTION
// =============================================================================
// Computes Q, K, V with LoRA in a single kernel:
//   Q = x @ W_q.T + scale_q * (x @ A_q.T @ B_q.T)
//   K = x @ W_k.T + scale_k * (x @ A_k.T @ B_k.T)
//   V = x @ W_v.T + scale_v * (x @ A_v.T @ B_v.T)
//
// This is a high-value fusion:
// - 12 separate matmuls → 1 kernel dispatch
// - All intermediate results stay in registers
// - Massive reduction in kernel launch overhead
//
// Grid: 2D, (out_features_total, M) where out_features_total = 3 * head_dim * num_heads

kernel void fused_lora_qkv_fwd(
    device const float* x [[buffer(0)]],           // [M, in_features]
    device const float* W_q [[buffer(1)]],         // [out_q, in_features]
    device const float* W_k [[buffer(2)]],         // [out_k, in_features]
    device const float* W_v [[buffer(3)]],         // [out_v, in_features]
    device const float* A_q [[buffer(4)]],         // [rank, in_features]
    device const float* B_q [[buffer(5)]],         // [out_q, rank]
    device const float* A_k [[buffer(6)]],         // [rank, in_features]
    device const float* B_k [[buffer(7)]],         // [out_k, rank]
    device const float* A_v [[buffer(8)]],         // [rank, in_features]
    device const float* B_v [[buffer(9)]],         // [out_v, rank]
    device float* Q [[buffer(10)]],                // [M, out_q]
    device float* K [[buffer(11)]],                // [M, out_k]
    device float* V [[buffer(12)]],                // [M, out_v]
    constant float& scale [[buffer(13)]],          // alpha / rank
    constant uint& M [[buffer(14)]],               // batch * seq
    constant uint& in_features [[buffer(15)]],     // hidden_dim
    constant uint& out_q [[buffer(16)]],           // num_heads * head_dim
    constant uint& out_k [[buffer(17)]],           // num_kv_heads * head_dim
    constant uint& out_v [[buffer(18)]],           // num_kv_heads * head_dim
    constant uint& rank [[buffer(19)]],            // LoRA rank
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = gid.y;  // batch/seq position
    uint n = gid.x;  // output position
    
    if (m >= M) return;
    
    // Determine which projection (Q, K, or V) and local output index
    uint total_out = out_q + out_k + out_v;
    if (n >= total_out) return;
    
    device const float* x_row = x + m * in_features;
    
    if (n < out_q) {
        // Q projection
        float base_val = 0.0f;
        for (uint k = 0; k < in_features; k++) {
            base_val += x_row[k] * W_q[n * in_features + k];
        }
        
        float lora_val = 0.0f;
        for (uint r = 0; r < rank; r++) {
            float xa = 0.0f;
            for (uint k = 0; k < in_features; k++) {
                xa += x_row[k] * A_q[r * in_features + k];
            }
            lora_val += xa * B_q[n * rank + r];
        }
        
        Q[m * out_q + n] = base_val + scale * lora_val;
        
    } else if (n < out_q + out_k) {
        // K projection
        uint k_idx = n - out_q;
        float base_val = 0.0f;
        for (uint k = 0; k < in_features; k++) {
            base_val += x_row[k] * W_k[k_idx * in_features + k];
        }
        
        float lora_val = 0.0f;
        for (uint r = 0; r < rank; r++) {
            float xa = 0.0f;
            for (uint k = 0; k < in_features; k++) {
                xa += x_row[k] * A_k[r * in_features + k];
            }
            lora_val += xa * B_k[k_idx * rank + r];
        }
        
        K[m * out_k + k_idx] = base_val + scale * lora_val;
        
    } else {
        // V projection
        uint v_idx = n - out_q - out_k;
        float base_val = 0.0f;
        for (uint k = 0; k < in_features; k++) {
            base_val += x_row[k] * W_v[v_idx * in_features + k];
        }
        
        float lora_val = 0.0f;
        for (uint r = 0; r < rank; r++) {
            float xa = 0.0f;
            for (uint k = 0; k < in_features; k++) {
                xa += x_row[k] * A_v[r * in_features + k];
            }
            lora_val += xa * B_v[v_idx * rank + r];
        }
        
        V[m * out_v + v_idx] = base_val + scale * lora_val;
    }
}

// =============================================================================
// FUSED ADAMW ALL PARAMETERS
// =============================================================================
// Updates multiple parameter tensors in a single kernel dispatch.
// Uses indirect buffer access - params packed sequentially.
//
// This reduces kernel launch overhead from N launches to 1.
// For LoRA with 100+ small tensors, this is significant.

kernel void adamw_step_multi(
    device float* params [[buffer(0)]],           // All params concatenated
    device const float* grads [[buffer(1)]],      // All grads concatenated
    device float* exp_avg [[buffer(2)]],          // First moments
    device float* exp_avg_sq [[buffer(3)]],       // Second moments
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant float& bias_correction1 [[buffer(9)]],
    constant float& bias_correction2 [[buffer(10)]],
    constant uint& total_numel [[buffer(11)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_numel) return;
    
    float p = params[tid];
    float g = grads[tid];
    float m = exp_avg[tid];
    float v = exp_avg_sq[tid];
    
    // AdamW update
    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * g * g;
    
    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;
    
    // Weight decay (decoupled)
    p = p - lr * weight_decay * p;
    
    // Adam update
    p = p - lr * m_hat / (sqrt(v_hat) + eps);
    
    params[tid] = p;
    exp_avg[tid] = m;
    exp_avg_sq[tid] = v;
}

// BFloat16 KL Divergence
kernel void kl_div_fwd_bfloat(
    device const bfloat* log_p [[buffer(0)]],
    device const bfloat* log_q [[buffer(1)]],
    device float* loss [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& vocab_size [[buffer(4)]],
    uint thread_idx [[thread_position_in_grid]],
    uint group_idx [[threadgroup_position_in_grid]],
    uint tid_in_group [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
     uint row = group_idx;
     if (row >= batch_size) return;
     
     uint lane = tid_in_group;
     float local_sum = 0.0f; // Renamed to local_sum to match reducer
     
     // Stride loop
     for (uint i = lane; i < vocab_size; i += 256) {
         uint idx = row * vocab_size + i;
         float lp = float(log_p[idx]);
         float lq = float(log_q[idx]);
         float p = exp(lp);
         local_sum += p * (lp - lq);
     }
     
    // SIMD reduction
    local_sum = simd_sum(local_sum);
    
    if (simd_lane == 0) {
        shared[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_group == 0) {
        float group_sum = 0.0f;
        if (simd_lane < 8) {
            group_sum = shared[simd_lane];
        }
        group_sum = simd_sum(group_sum);
        
        if (simd_lane == 0) {
            loss[row] = group_sum;
        }
    }
}

// BFloat16 Top-K KL Divergence
kernel void kl_div_topk_fwd_bfloat(
    device const bfloat* log_p [[buffer(0)]],
    device const bfloat* log_q [[buffer(1)]],
    device const int* topk_indices [[buffer(2)]],
    device float* loss [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& vocab_size [[buffer(5)]],
    constant int& k [[buffer(6)]],
    uint thread_idx [[thread_position_in_grid]],
    uint group_idx [[threadgroup_position_in_grid]],
    uint tid_in_group [[thread_index_in_threadgroup]]
) {
    uint row = group_idx;
    if (row >= batch_size) return;
    
    uint lane = tid_in_group;
    float row_sum = 0.0f;
    
    // Stride loop over K
    for (uint i = lane; i < (uint)k; i += 256) {
        int idx_k = row * k + i;
        int token_idx = topk_indices[idx_k];
        int flat_idx = row * vocab_size + token_idx;
        
        float lp = float(log_p[flat_idx]);
        float lq = float(log_q[flat_idx]);
        
        float p = exp(lp);
        row_sum += p * (lp - lq);
    }
    
    // SIMD reduction
    float sum = simd_sum(row_sum);
    
    threadgroup float shared[8];
    uint simd_id = lane / 32;
    uint lane_id = lane % 32;
    
    if (lane_id == 0) shared[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (lane < 8) {
        sum = shared[lane];
    } else {
        sum = 0.0f;
    }
    
    if (simd_id == 0) {
        sum = simd_sum(sum);
        if (lane == 0) loss[row] = sum;
    }
}

// BFloat16 Cross Entropy
kernel void cross_entropy_fwd_bfloat(
    device const bfloat* logits [[buffer(0)]],
    device const int* targets [[buffer(1)]],
    device float* losses [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& vocab_size [[buffer(4)]],
    uint thread_idx [[thread_position_in_grid]],
    uint group_idx [[threadgroup_position_in_grid]],
    uint tid_in_group [[thread_index_in_threadgroup]]
) {
    // 1 group per batch element
    uint row = group_idx;
    if (row >= batch_size) return;
    
    uint lane = tid_in_group;
    
    // 1. Find max for numerical stability
    float local_max = -INFINITY;
    for (uint i = lane; i < vocab_size; i += 256) {
        float val = float(logits[row * vocab_size + i]);
        local_max = max(local_max, val);
    }
    local_max = simd_max(local_max);
    
    threadgroup float shared_max[8];
    uint simd_id = lane / 32;
    uint lane_id = lane % 32;
    
    if (lane_id == 0) shared_max[simd_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (lane < 8) local_max = shared_max[lane];
    else local_max = -INFINITY;
    
    if (simd_id == 0) {
        local_max = simd_max(local_max);
        if (lane == 0) shared_max[0] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = shared_max[0];
    
    // 2. Sum exp
    float local_sum = 0.0f;
    for (uint i = lane; i < vocab_size; i += 256) {
        float val = float(logits[row * vocab_size + i]);
        local_sum += exp(val - row_max);
    }
    local_sum = simd_sum(local_sum);
    
    threadgroup float shared_sum[8];
    if (lane_id == 0) shared_sum[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (lane < 8) local_sum = shared_sum[lane];
    else local_sum = 0.0f;
    
    if (simd_id == 0) {
        local_sum = simd_sum(local_sum);
        if (lane == 0) shared_sum[0] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_sum_exp = shared_sum[0];
    
    // 3. Final calculation: -log(softmax(target))
    // = -log(exp(logit[target] - max) / sum_exp)
    // = -(logit[target] - max - log(sum_exp))
    // = -logit[target] + max + log(sum_exp)
    if (lane == 0) {
        int target_idx = targets[row];
        float target_logit = float(logits[row * vocab_size + target_idx]);
        losses[row] = -target_logit + row_max + log(row_sum_exp);
    }
}

// FP16 split-half backward
kernel void rope_bwd_split_half_half(
    device const half* d_out [[buffer(0)]],
    device const half* cos [[buffer(1)]],
    device const half* sin [[buffer(2)]],
    device half* d_qk [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint h = gid.y;
    uint bs = gid.z;
    
    uint half_dim = head_dim / 2;
    if (d >= half_dim || h >= num_heads || bs >= batch_size * seq_len) return;
    
    uint b = bs / seq_len;
    uint s = bs % seq_len;
    
    uint base = ((b * seq_len + s) * num_heads + h) * head_dim;
    
    float dy1 = float(d_out[base + d]);
    float dy2 = float(d_out[base + half_dim + d]);
    
    uint cs_offset = s * half_dim + d;
    float c = float(cos[cs_offset]);
    float sn = float(sin[cs_offset]);
    
    d_qk[base + d] = half(dy1 * c + dy2 * sn);
    d_qk[base + half_dim + d] = half(-dy1 * sn + dy2 * c);
}

// =============================================================================
// SWIGLU BACKWARD
// =============================================================================
// d_h = grad_output (from down proj)
// y = silu(gate) * up
// d_up = d_h * silu(gate)
// d_gate = d_h * up * (sigmoid(gate) * (1 + gate * (1 - sigmoid(gate))))

kernel void swiglu_bwd_strided_float(
    device const float* d_h [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const float* up [[buffer(2)]],
    device float* d_gate [[buffer(3)]],
    device float* d_up [[buffer(4)]],
    constant uint2& shape [[buffer(5)]],     // [rows, cols]
    constant uint2& s_d_h [[buffer(6)]],     // strides
    constant uint2& s_gate [[buffer(7)]],
    constant uint2& s_up [[buffer(8)]],
    constant uint2& s_d_gate [[buffer(9)]],
    constant uint2& s_d_up [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= shape.x || col >= shape.y) return;
    
    // Calculate offsets
    uint idx_dh = row * s_d_h.x + col * s_d_h.y;
    uint idx_g  = row * s_gate.x + col * s_gate.y;
    uint idx_u  = row * s_up.x + col * s_up.y;
    uint idx_dg = row * s_d_gate.x + col * s_d_gate.y;
    uint idx_du = row * s_d_up.x + col * s_d_up.y;
    
    float dh_val = d_h[idx_dh];
    float g_val = gate[idx_g];
    float u_val = up[idx_u];
    
    // SiLU and Sigmoid
    float sig_g = 1.0f / (1.0f + exp(-g_val));
    float silu_g = g_val * sig_g;
    
    // d_up
    d_up[idx_du] = dh_val * silu_g;
    
    // d_gate
    // d_silu = sig * (1 + g * (1 - sig))
    float d_silu = sig_g * (1.0f + g_val * (1.0f - sig_g));
    d_gate[idx_dg] = dh_val * u_val * d_silu;
}

kernel void swiglu_bwd_strided_half(
    device const half* d_h [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    device const half* up [[buffer(2)]],
    device half* d_gate [[buffer(3)]],
    device half* d_up [[buffer(4)]],
    constant uint2& shape [[buffer(5)]],
    constant uint2& s_d_h [[buffer(6)]],
    constant uint2& s_gate [[buffer(7)]],
    constant uint2& s_up [[buffer(8)]],
    constant uint2& s_d_gate [[buffer(9)]],
    constant uint2& s_d_up [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= shape.x || col >= shape.y) return;
    
    uint idx_dh = row * s_d_h.x + col * s_d_h.y;
    uint idx_g  = row * s_gate.x + col * s_gate.y;
    uint idx_u  = row * s_up.x + col * s_up.y;
    uint idx_dg = row * s_d_gate.x + col * s_d_gate.y;
    uint idx_du = row * s_d_up.x + col * s_d_up.y;
    
    float dh_val = float(d_h[idx_dh]);
    float g_val = float(gate[idx_g]);
    float u_val = float(up[idx_u]);
    
    float sig_g = 1.0f / (1.0f + exp(-g_val));
    float silu_g = g_val * sig_g;
    
    d_up[idx_du] = half(dh_val * silu_g);
    
    float d_silu = sig_g * (1.0f + g_val * (1.0f - sig_g));
    d_gate[idx_dg] = half(dh_val * u_val * d_silu);
}

kernel void swiglu_bwd_strided_bfloat(
    device const bfloat* d_h [[buffer(0)]],
    device const bfloat* gate [[buffer(1)]],
    device const bfloat* up [[buffer(2)]],
    device bfloat* d_gate [[buffer(3)]],
    device bfloat* d_up [[buffer(4)]],
    constant uint2& shape [[buffer(5)]],
    constant uint2& s_d_h [[buffer(6)]],
    constant uint2& s_gate [[buffer(7)]],
    constant uint2& s_up [[buffer(8)]],
    constant uint2& s_d_gate [[buffer(9)]],
    constant uint2& s_d_up [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= shape.x || col >= shape.y) return;
    
    uint idx_dh = row * s_d_h.x + col * s_d_h.y;
    uint idx_g  = row * s_gate.x + col * s_gate.y;
    uint idx_u  = row * s_up.x + col * s_up.y;
    uint idx_dg = row * s_d_gate.x + col * s_d_gate.y;
    uint idx_du = row * s_d_up.x + col * s_d_up.y;
    
    float dh_val = float(d_h[idx_dh]);
    float g_val = float(gate[idx_g]);
    float u_val = float(up[idx_u]);
    
    float sig_g = 1.0f / (1.0f + exp(-g_val));
    float silu_g = g_val * sig_g;
    
    d_up[idx_du] = bfloat(dh_val * silu_g);
    
    float d_silu = sig_g * (1.0f + g_val * (1.0f - sig_g));
    d_gate[idx_dg] = bfloat(dh_val * u_val * d_silu);
}
