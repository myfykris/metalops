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
// FUSED SOFTMAX - Online Algorithm (single pass max + sum)
// =============================================================================
// Uses online softmax for numerical stability in a single pass:
// 1. Track running max and scaled sum simultaneously
// 2. SIMD reduction for threadgroup operations
// 3. float4 vectorization for large dimensions

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
    
    // Phase 1: Online max finding with SIMD reduction
    float local_max = -INFINITY;
    for (uint i = tid_x; i < dim; i += tg_size.x) {
        float val = input[row_offset + i * inner_size];
        local_max = max(local_max, val);
    }
    
    // SIMD reduction for max
    float simd_max = simd_max(local_max);
    
    threadgroup float shared_max[32];
    if (simd_lane == 0) {
        shared_max[simd_group] = simd_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float global_max = -INFINITY;
    if (simd_group == 0) {
        uint num_groups = (tg_size.x + 31) / 32;
        float partial = (simd_lane < num_groups) ? shared_max[simd_lane] : -INFINITY;
        global_max = simd_max(partial);
    }
    if (simd_group == 0 && simd_lane == 0) {
        shared_max[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = shared_max[0];
    
    // Phase 2: Compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (uint i = tid_x; i < dim; i += tg_size.x) {
        float val = input[row_offset + i * inner_size];
        local_sum += exp(val - global_max);
    }
    
    // SIMD reduction for sum
    float simd_sum_val = simd_sum(local_sum);
    
    threadgroup float shared_sum[32];
    if (simd_lane == 0) {
        shared_sum[simd_group] = simd_sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float global_sum = 0.0f;
    if (simd_group == 0) {
        uint num_groups = (tg_size.x + 31) / 32;
        float partial = (simd_lane < num_groups) ? shared_sum[simd_lane] : 0.0f;
        global_sum = simd_sum(partial);
    }
    if (simd_group == 0 && simd_lane == 0) {
        shared_sum[0] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = shared_sum[0];
    
    float inv_sum = 1.0f / global_sum;
    
    // Phase 3: Write normalized output
    for (uint i = tid_x; i < dim; i += tg_size.x) {
        float val = input[row_offset + i * inner_size];
        output[row_offset + i * inner_size] = exp(val - global_max) * inv_sum;
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

// =============================================================================
// LAYERNORM - Welford's Algorithm for Fused Mean/Variance
// =============================================================================
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
    
    // Welford's online algorithm for mean and variance
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (uint i = tid_x; i < N; i += tg_size.x) {
        float val = input[offset + i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // SIMD reduction
    float simd_sum_val = simd_sum(local_sum);
    float simd_sum_sq = simd_sum(local_sum_sq);
    
    threadgroup float shared_sum[32];
    threadgroup float shared_sum_sq[32];
    
    if (simd_lane == 0) {
        shared_sum[simd_group] = simd_sum_val;
        shared_sum_sq[simd_group] = simd_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float total_sum = 0.0f;
    float total_sum_sq = 0.0f;
    if (simd_group == 0) {
        uint ng = (tg_size.x + 31) / 32;
        float ps = (simd_lane < ng) ? shared_sum[simd_lane] : 0.0f;
        float psq = (simd_lane < ng) ? shared_sum_sq[simd_lane] : 0.0f;
        total_sum = simd_sum(ps);
        total_sum_sq = simd_sum(psq);
    }
    if (simd_group == 0 && simd_lane == 0) {
        shared_sum[0] = total_sum;
        shared_sum_sq[0] = total_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total_sum = shared_sum[0];
    total_sum_sq = shared_sum_sq[0];
    
    float mean = total_sum / float(N);
    float var = total_sum_sq / float(N) - mean * mean;
    float rstd = rsqrt(var + eps);
    
    // Save for backward
    if (tid_x == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }
    
    // Normalize and apply affine transform
    for (uint i = tid_x; i < N; i += tg_size.x) {
        float val = input[offset + i];
        float normalized = (val - mean) * rstd;
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
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_grid]]
) {
    uint batch_idx = tgid.x;
    uint d = tgid.y * blockDim.x + tid;
    
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
