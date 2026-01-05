#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------------
// AdamW Scalar Kernel (Leaf/Tail handling)
// -----------------------------------------------------------------------------

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
    
    // Update param
    float denom = sqrt(v_hat) + eps;
    p = p - lr * (m_hat / denom);
    
    // Write back
    params[id] = p;
    exp_avg[id] = m;
    exp_avg_sq[id] = v;
}
