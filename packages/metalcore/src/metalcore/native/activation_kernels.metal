#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Activation Macros (guaranteed zero overhead)
// =============================================================================

// GELU constants (used in 10+ kernels)
#define GELU_SQRT_2_OVER_PI 0.7978845608f
#define GELU_SQRT_2_OVER_PI_HALF GELU_SQRT_2_OVER_PI_HALF
#define GELU_COEFF 0.044715f
#define GELU_COEFF_HALF GELU_COEFF_HALF

// Sigmoid: 1 / (1 + exp(-x))
#define SIGMOID_F(x) (1.0f / (1.0f + exp(-(x))))
#define SIGMOID_H(x) (1.0h / (1.0h + exp(-(x))))

// GELU formula: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#define GELU_F(x) ({ \
    float _x = (x); \
    float _x3 = _x * _x * _x; \
    float _arg = GELU_SQRT_2_OVER_PI * (_x + GELU_COEFF * _x3); \
    _x * 0.5f * (1.0f + tanh(_arg)); \
})

#define GELU_H(x) ({ \
    half _x = (x); \
    half _x3 = _x * _x * _x; \
    half _arg = GELU_SQRT_2_OVER_PI_HALF * (_x + GELU_COEFF_HALF * _x3); \
    _x * 0.5h * (1.0h + tanh(_arg)); \
})

// SiLU formula: x * sigmoid(x)
#define SILU_F(x) ({ float _x = (x); _x * SIGMOID_F(_x); })
#define SILU_H(x) ({ half _x = (x); _x * SIGMOID_H(_x); })

// Vectorized versions (for float4/half4)
#define SIGMOID4_F(x) (1.0f / (1.0f + exp(-(x))))
#define SIGMOID4_H(x) (1.0h / (1.0h + exp(-(x))))

// =============================================================================

kernel void gelu_fwd(
    device const float4* X [[buffer(0)]],
    device float4* Y [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_vec = numel / 4;
    if (id >= numel_vec) return;
    
    float4 x = X[id];
    
    // Compute using macro constants
    float4 x3 = x * x * x;
    float4 arg = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    float4 tanh_val = tanh(arg);
    
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
    
    
    
    
    float4 x2 = x * x;
    float4 x3 = x2 * x;
    float4 arg = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    float4 tanh_val = tanh(arg);
    float4 sech2_val = 1.0f - tanh_val * tanh_val;
    
    // d/dx gelu = 0.5 * (1 + tanh) + 0.5 * x * sech^2 * d(arg)/dx
    // d(arg)/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
    float4 darg_dx = GELU_SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEFF * x2);
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
    
    
    
    
    float4 x3 = x * x * x;
    float4 arg = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
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
    
    
    float x3 = x * x * x;
    float arg = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
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

// =============================================================================
// HALF PRECISION (fp16) VARIANTS
// =============================================================================

// -----------------------------------------------------------------------------
// GELU Half Precision
// -----------------------------------------------------------------------------

kernel void gelu_fwd_half(
    device const half4* X [[buffer(0)]],
    device half4* Y [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_vec = numel / 4;
    if (id >= numel_vec) return;
    
    half4 x = X[id];
    
    
    
    
    half4 x3 = x * x * x;
    half4 arg = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    half4 tanh_val = tanh(arg);
    
    Y[id] = x * 0.5h * (1.0h + tanh_val);
}

kernel void gelu_bwd_half(
    device const half4* dY [[buffer(0)]],
    device const half4* X [[buffer(1)]],
    device half4* dX [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_vec = numel / 4;
    if (id >= numel_vec) return;
    
    half4 x = X[id];
    half4 dy = dY[id];
    
    
    
    
    half4 x2 = x * x;
    half4 x3 = x2 * x;
    half4 arg = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    half4 tanh_val = tanh(arg);
    half4 sech2_val = 1.0h - tanh_val * tanh_val;
    
    half4 darg_dx = GELU_SQRT_2_OVER_PI * (1.0h + 3.0h * GELU_COEFF * x2);
    half4 grad = 0.5h * (1.0h + tanh_val) + 0.5h * x * sech2_val * darg_dx;
    
    dX[id] = dy * grad;
}

// -----------------------------------------------------------------------------
// SiLU Half Precision
// -----------------------------------------------------------------------------

kernel void silu_fwd_half(
    device const half4* X [[buffer(0)]],
    device half4* Y [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_vec = numel / 4;
    if (id >= numel_vec) return;
    
    half4 x = X[id];
    half4 sigmoid_x = 1.0h / (1.0h + exp(-x));
    Y[id] = x * sigmoid_x;
}

kernel void silu_bwd_half(
    device const half4* dY [[buffer(0)]],
    device const half4* X [[buffer(1)]],
    device half4* dX [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_vec = numel / 4;
    if (id >= numel_vec) return;
    
    half4 x = X[id];
    half4 dy = dY[id];
    
    half4 sigmoid_x = 1.0h / (1.0h + exp(-x));
    half4 grad = sigmoid_x * (1.0h + x * (1.0h - sigmoid_x));
    
    dX[id] = dy * grad;
}

// -----------------------------------------------------------------------------
// Bias + GELU/SiLU Fusion Half Precision
// -----------------------------------------------------------------------------

kernel void bias_gelu_fwd_half(
    device const half4* X [[buffer(0)]],
    device const half4* Bias [[buffer(1)]],
    device half4* Y [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    constant uint& bias_size [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_vec = numel / 4;
    if (id >= numel_vec) return;
    
    uint bias_idx = id % bias_size;
    half4 x = X[id] + Bias[bias_idx];
    
    
    
    
    half4 x3 = x * x * x;
    half4 arg = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    half4 tanh_val = tanh(arg);
    
    Y[id] = x * 0.5h * (1.0h + tanh_val);
}

kernel void bias_silu_fwd_half(
    device const half4* X [[buffer(0)]],
    device const half4* Bias [[buffer(1)]],
    device half4* Y [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    constant uint& bias_size [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_vec = numel / 4;
    if (id >= numel_vec) return;
    
    uint bias_idx = id % bias_size;
    half4 x = X[id] + Bias[bias_idx];
    
    half4 sigmoid_x = 1.0h / (1.0h + exp(-x));
    Y[id] = x * sigmoid_x;
}

// -----------------------------------------------------------------------------
// Scalar Fallbacks Half Precision
// -----------------------------------------------------------------------------

kernel void gelu_fwd_scalar_half(
    device const half* X [[buffer(0)]],
    device half* Y [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    half x = X[id];
    
    
    half x3 = x * x * x;
    half arg = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    half tanh_val = tanh(arg);
    Y[id] = x * 0.5h * (1.0h + tanh_val);
}

kernel void silu_fwd_scalar_half(
    device const half* X [[buffer(0)]],
    device half* Y [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    half x = X[id];
    half sigmoid_x = 1.0h / (1.0h + exp(-x));
    Y[id] = x * sigmoid_x;
}

// =============================================================================
// BFloat16 Activation Kernels (in-register bf16<->fp32 conversion)
// =============================================================================
#if __METAL_VERSION__ >= 310

// Macro for fast bf16 -> fp32 -> bf16 in registers
#define BFLOAT_TO_FLOAT(b) float(b)
#define FLOAT_TO_BFLOAT(f) bfloat(f)

kernel void gelu_fwd_bfloat(
    device const bfloat* X [[buffer(0)]],
    device bfloat* Y [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_4 = numel / 4;
    if (id >= numel_4) return;
    
    // Load 4 bf16 values, convert to float4 in registers
    uint base = id * 4;
    float4 x = float4(X[base], X[base+1], X[base+2], X[base+3]);
    
    // GELU in fp32
    float4 x3 = x * x * x;
    float4 arg = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    float4 result = x * 0.5f * (1.0f + tanh(arg));
    
    // Convert back to bf16 and store
    Y[base] = bfloat(result.x);
    Y[base+1] = bfloat(result.y);
    Y[base+2] = bfloat(result.z);
    Y[base+3] = bfloat(result.w);
}

kernel void gelu_fwd_bfloat_scalar(
    device const bfloat* X [[buffer(0)]],
    device bfloat* Y [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = float(X[id]);
    float x3 = x * x * x;
    float arg = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    Y[id] = bfloat(x * 0.5f * (1.0f + tanh(arg)));
}

kernel void silu_fwd_bfloat(
    device const bfloat* X [[buffer(0)]],
    device bfloat* Y [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_4 = numel / 4;
    if (id >= numel_4) return;
    
    // Load 4 bf16 values, convert to float4 in registers
    uint base = id * 4;
    float4 x = float4(X[base], X[base+1], X[base+2], X[base+3]);
    
    // SiLU in fp32: x * sigmoid(x)
    float4 sigmoid_x = 1.0f / (1.0f + exp(-x));
    float4 result = x * sigmoid_x;
    
    // Convert back to bf16 and store
    Y[base] = bfloat(result.x);
    Y[base+1] = bfloat(result.y);
    Y[base+2] = bfloat(result.z);
    Y[base+3] = bfloat(result.w);
}

kernel void silu_fwd_bfloat_scalar(
    device const bfloat* X [[buffer(0)]],
    device bfloat* Y [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = float(X[id]);
    float sigmoid_x = 1.0f / (1.0f + exp(-x));
    Y[id] = bfloat(x * sigmoid_x);
}

// GELU backward bf16
kernel void gelu_bwd_bfloat(
    device const bfloat* dY [[buffer(0)]],
    device const bfloat* X [[buffer(1)]],
    device bfloat* dX [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_4 = numel / 4;
    if (id >= numel_4) return;
    
    uint base = id * 4;
    float4 x = float4(X[base], X[base+1], X[base+2], X[base+3]);
    float4 dy = float4(dY[base], dY[base+1], dY[base+2], dY[base+3]);
    
    float4 x2 = x * x;
    float4 x3 = x2 * x;
    float4 arg = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    float4 tanh_val = tanh(arg);
    float4 sech2_val = 1.0f - tanh_val * tanh_val;
    
    float4 darg_dx = GELU_SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEFF * x2);
    float4 grad = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2_val * darg_dx;
    float4 result = dy * grad;
    
    dX[base] = bfloat(result.x);
    dX[base+1] = bfloat(result.y);
    dX[base+2] = bfloat(result.z);
    dX[base+3] = bfloat(result.w);
}

kernel void gelu_bwd_bfloat_scalar(
    device const bfloat* dY [[buffer(0)]],
    device const bfloat* X [[buffer(1)]],
    device bfloat* dX [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float x = float(X[id]);
    float dy = float(dY[id]);
    
    float x2 = x * x;
    float x3 = x2 * x;
    float arg = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    float tanh_val = tanh(arg);
    float sech2_val = 1.0f - tanh_val * tanh_val;
    
    float darg_dx = GELU_SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEFF * x2);
    float grad = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2_val * darg_dx;
    
    dX[id] = bfloat(dy * grad);
}

// SiLU backward bf16
kernel void silu_bwd_bfloat(
    device const bfloat* dY [[buffer(0)]],
    device const bfloat* X [[buffer(1)]],
    device bfloat* dX [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint numel_4 = numel / 4;
    if (id >= numel_4) return;
    
    uint base = id * 4;
    float4 x = float4(X[base], X[base+1], X[base+2], X[base+3]);
    float4 dy = float4(dY[base], dY[base+1], dY[base+2], dY[base+3]);
    
    float4 sigmoid_x = 1.0f / (1.0f + exp(-x));
    float4 grad = sigmoid_x * (1.0f + x * (1.0f - sigmoid_x));
    float4 result = dy * grad;
    
    dX[base] = bfloat(result.x);
    dX[base+1] = bfloat(result.y);
    dX[base+2] = bfloat(result.z);
    dX[base+3] = bfloat(result.w);
}

kernel void silu_bwd_bfloat_scalar(
    device const bfloat* dY [[buffer(0)]],
    device const bfloat* X [[buffer(1)]],
    device bfloat* dX [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float x = float(X[id]);
    float dy = float(dY[id]);
    
    float sigmoid_x = 1.0f / (1.0f + exp(-x));
    float grad = sigmoid_x * (1.0f + x * (1.0f - sigmoid_x));
    
    dX[id] = bfloat(dy * grad);
}

#endif // __METAL_VERSION__ >= 310

