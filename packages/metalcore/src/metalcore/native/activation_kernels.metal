#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------------
// GELU Activation (Gaussian Error Linear Unit)
// Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// -----------------------------------------------------------------------------

kernel void gelu_fwd(
    device const float4* X [[buffer(0)]],
    device float4* Y [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_vec = numel / 4;
    if (id >= numel_vec) return;
    
    float4 x = X[id];
    
    // Constants
    const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
    const float coeff = 0.044715f;
    
    // Compute tanh argument: sqrt(2/pi) * (x + 0.044715 * x^3)
    float4 x3 = x * x * x;
    float4 arg = sqrt_2_over_pi * (x + coeff * x3);
    
    // tanh approximation (fast)
    float4 tanh_val = tanh(arg);
    
    // GELU: x * 0.5 * (1 + tanh_val)
    Y[id] = x * 0.5f * (1.0f + tanh_val);
}

kernel void gelu_bwd(
    device const float4* dY [[buffer(0)]],
    device const float4* X [[buffer(1)]],
    device float4* dX [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_vec = numel / 4;
    if (id >= numel_vec) return;
    
    float4 x = X[id];
    float4 dy = dY[id];
    
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    float4 x2 = x * x;
    float4 x3 = x2 * x;
    float4 arg = sqrt_2_over_pi * (x + coeff * x3);
    float4 tanh_val = tanh(arg);
    float4 sech2_val = 1.0f - tanh_val * tanh_val;
    
    // d/dx gelu = 0.5 * (1 + tanh) + 0.5 * x * sech^2 * d(arg)/dx
    // d(arg)/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
    float4 darg_dx = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x2);
    float4 grad = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2_val * darg_dx;
    
    dX[id] = dy * grad;
}

// -----------------------------------------------------------------------------
// SiLU Activation (Sigmoid Linear Unit / Swish)
// y = x * sigmoid(x) = x / (1 + exp(-x))
// -----------------------------------------------------------------------------

kernel void silu_fwd(
    device const float4* X [[buffer(0)]],
    device float4* Y [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_vec = numel / 4;
    if (id >= numel_vec) return;
    
    float4 x = X[id];
    float4 sigmoid_x = 1.0f / (1.0f + exp(-x));
    Y[id] = x * sigmoid_x;
}

kernel void silu_bwd(
    device const float4* dY [[buffer(0)]],
    device const float4* X [[buffer(1)]],
    device float4* dX [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_vec = numel / 4;
    if (id >= numel_vec) return;
    
    float4 x = X[id];
    float4 dy = dY[id];
    
    float4 sigmoid_x = 1.0f / (1.0f + exp(-x));
    // d/dx silu = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    //           = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    float4 grad = sigmoid_x * (1.0f + x * (1.0f - sigmoid_x));
    
    dX[id] = dy * grad;
}

// -----------------------------------------------------------------------------
// Bias + GELU Fusion
// y = gelu(x + bias)
// -----------------------------------------------------------------------------

kernel void bias_gelu_fwd(
    device const float4* X [[buffer(0)]],
    device const float4* Bias [[buffer(1)]],  // Broadcasted along batch dim
    device float4* Y [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    constant uint& bias_size [[buffer(4)]],  // Size of bias vector (N/4)
    uint id [[thread_position_in_grid]]
) {
    uint numel_vec = numel / 4;
    if (id >= numel_vec) return;
    
    // Bias is broadcasted: bias[id % bias_size]
    uint bias_idx = id % bias_size;
    float4 x = X[id] + Bias[bias_idx];
    
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    float4 x3 = x * x * x;
    float4 arg = sqrt_2_over_pi * (x + coeff * x3);
    float4 tanh_val = tanh(arg);
    
    Y[id] = x * 0.5f * (1.0f + tanh_val);
}

// -----------------------------------------------------------------------------
// Bias + SiLU Fusion
// y = silu(x + bias)
// -----------------------------------------------------------------------------

kernel void bias_silu_fwd(
    device const float4* X [[buffer(0)]],
    device const float4* Bias [[buffer(1)]],
    device float4* Y [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    constant uint& bias_size [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_vec = numel / 4;
    if (id >= numel_vec) return;
    
    uint bias_idx = id % bias_size;
    float4 x = X[id] + Bias[bias_idx];
    
    float4 sigmoid_x = 1.0f / (1.0f + exp(-x));
    Y[id] = x * sigmoid_x;
}

// -----------------------------------------------------------------------------
// Scalar fallbacks for tail elements
// -----------------------------------------------------------------------------

kernel void gelu_fwd_scalar(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = X[id];
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float arg = sqrt_2_over_pi * (x + coeff * x3);
    float tanh_val = tanh(arg);
    Y[id] = x * 0.5f * (1.0f + tanh_val);
}

kernel void silu_fwd_scalar(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = X[id];
    float sigmoid_x = 1.0f / (1.0f + exp(-x));
    Y[id] = x * sigmoid_x;
}
