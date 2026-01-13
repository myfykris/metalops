// Author: Kris Bailey
// Copyright 2026
// Email: kris@krisbailey.com
#include <metal_stdlib>

// Tensor core support requires Metal 4+ / M4 hardware
// Disabled for now as metal_tensor header isn't available
// #define METALCORE_HAS_TENSOR 1

#ifdef METALCORE_HAS_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

using namespace metal;

// =============================================================================
// Quantized Matmul Kernels (INT8/INT4 with on-the-fly dequantization)
// =============================================================================
// These kernels enable running large LLM models on Apple Silicon by storing
// weights in low-precision formats (INT8: 2x compression, INT4: 4x compression).
//
// Memory savings:
//   - 7B model: 14GB fp16 -> 7GB INT8 -> 3.5GB INT4
//   - 70B model: 140GB fp16 -> 70GB INT8 -> 35GB INT4
//
// Quantization formula:
//   W_float = (W_quant - zero_point) * scale
//
// Per-group quantization (e.g., group_size=128) preserves accuracy by using
// different scale/zero_point parameters for each group of weights.

// -----------------------------------------------------------------------------
// INT8 Dequantize + Matmul
// -----------------------------------------------------------------------------
// Y = X @ dequant(W_q)
// X: [M, K] fp16/fp32, W_q: [K, N] int8, scale: [K/group_size, N], zero: [K/group_size, N]
// Y: [M, N] fp16/fp32

kernel void matmul_int8_dequant(
    device const float* X [[buffer(0)]],           // [M, K]
    device const char* W_q [[buffer(1)]],          // [K, N] int8 weights
    device const float* scales [[buffer(2)]],      // [num_groups, N]
    device const float* zeros [[buffer(3)]],       // [num_groups, N] (optional, can be nullptr)
    device float* Y [[buffer(4)]],                 // [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]          // (n, m)
) {
    uint n = gid.x;  // Column index in output
    uint m = gid.y;  // Row index in output
    
    if (n >= N || m >= M) return;
    
    float acc = 0.0f;
    
    // Process each element in the K dimension
    for (uint k = 0; k < K; k++) {
        // Get quantized weight
        int8_t w_q = W_q[k * N + n];
        
        // Get scale and zero for this group
        uint group_idx = k / group_size;
        float scale = scales[group_idx * N + n];
        float zero = zeros ? zeros[group_idx * N + n] : 0.0f;
        
        // Dequantize: W_float = (W_q - zero) * scale
        float w = (float(w_q) - zero) * scale;
        
        // Multiply-accumulate
        acc += X[m * K + k] * w;
    }
    
    Y[m * N + n] = acc;
}

// Half-precision version (more common for LLM inference)
kernel void matmul_int8_dequant_half(
    device const half* X [[buffer(0)]],
    device const char* W_q [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const half* zeros [[buffer(3)]],
    device half* Y [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint n = gid.x;
    uint m = gid.y;
    
    if (n >= N || m >= M) return;
    
    float acc = 0.0f;
    
    for (uint k = 0; k < K; k++) {
        int8_t w_q = W_q[k * N + n];
        uint group_idx = k / group_size;
        float scale = float(scales[group_idx * N + n]);
        float zero = zeros ? float(zeros[group_idx * N + n]) : 0.0f;
        float w = (float(w_q) - zero) * scale;
        acc += float(X[m * K + k]) * w;
    }
    
    Y[m * N + n] = half(acc);
}

// -----------------------------------------------------------------------------
// INT4 Dequantize + Matmul (4x compression - 2 weights per byte)
// -----------------------------------------------------------------------------
// INT4 packing: Two 4-bit values packed into one byte
// Lower 4 bits = first value, Upper 4 bits = second value
//
// Optimization: SIMD vectorized with K-dimension loop unrolling

// Basic scalar version for correctness baseline
// NOTE: This kernel has strided W access - use tiled version for performance
kernel void matmul_int4_dequant(
    device const float* X [[buffer(0)]],           // [M, K]
    device const uchar* W_packed [[buffer(1)]],    // [K/2, N] packed int4
    device const float* scales [[buffer(2)]],      // [num_groups, N]
    device const float* zeros [[buffer(3)]],       // [num_groups, N]
    device float* Y [[buffer(4)]],                 // [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint n = gid.x;
    uint m = gid.y;
    
    if (n >= N || m >= M) return;
    
    float acc = 0.0f;
    
    // Process K elements (2 per packed byte)
    for (uint k = 0; k < K; k++) {
        uint byte_idx = k / 2;
        uchar packed = W_packed[byte_idx * N + n];  // Strided access
        
        int w_q = (k % 2 == 0) ? int(packed & 0x0F) : int(packed >> 4);
        
        uint group_idx = k / group_size;
        float scale = scales[group_idx * N + n];
        float zero = zeros[group_idx * N + n];
        float w = (float(w_q) - zero) * scale;
        
        acc += X[m * K + k] * w;
    }
    
    Y[m * N + n] = acc;
}

// Optimized: Process 4 output columns at once using SIMD
kernel void matmul_int4_dequant_vec4(
    device const float* X [[buffer(0)]],           // [M, K]
    device const uchar4* W_packed [[buffer(1)]],   // [K/2, N/4] packed as uchar4
    device const float4* scales [[buffer(2)]],     // [num_groups, N/4]
    device const float4* zeros [[buffer(3)]],      // [num_groups, N/4]
    device float4* Y [[buffer(4)]],                // [M, N/4]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],                // Actual N (will process N/4 columns)
    constant uint& group_size [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint n4 = gid.x;  // Which group of 4 columns
    uint m = gid.y;   // Which row
    
    uint N4 = N / 4;
    if (n4 >= N4 || m >= M) return;
    
    float4 acc = float4(0.0f);
    
    // Process K dimension, 2 elements at a time (one byte unpacks to 2 values)
    for (uint k = 0; k < K; k += 2) {
        uint byte_idx = k / 2;
        
        // Load 4 packed bytes (8 weights for 4 output columns, 2 k-values each)
        uchar4 packed = W_packed[byte_idx * N4 + n4];
        
        // Get group parameters for these k values
        uint group_idx = k / group_size;
        float4 scale = scales[group_idx * N4 + n4];
        float4 zero = zeros[group_idx * N4 + n4];
        
        // Unpack and process first k value (lower 4 bits) - NO -8 offset
        float4 w0 = float4(
            int(packed.x & 0x0F),
            int(packed.y & 0x0F),
            int(packed.z & 0x0F),
            int(packed.w & 0x0F)
        );
        w0 = (w0 - zero) * scale;
        acc += X[m * K + k] * w0;
        
        // Unpack and process second k value (upper 4 bits)
        if (k + 1 < K) {
            float4 w1 = float4(
                int(packed.x >> 4),
                int(packed.y >> 4),
                int(packed.z >> 4),
                int(packed.w >> 4)
            );
            w1 = (w1 - zero) * scale;  // Same group
            acc += X[m * K + k + 1] * w1;
        }
    }
    
    Y[m * N4 + n4] = acc;
}

// Half-precision INT4 (most common for LLM inference)
kernel void matmul_int4_dequant_half(
    device const half* X [[buffer(0)]],
    device const uchar* W_packed [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const half* zeros [[buffer(3)]],
    device half* Y [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint n = gid.x;
    uint m = gid.y;
    
    if (n >= N || m >= M) return;
    
    float acc = 0.0f;
    
    for (uint k = 0; k < K; k++) {
        uint byte_idx = k / 2;
        uchar packed = W_packed[byte_idx * N + n];
        
        // Extract 4-bit value [0, 15] - NO -8 offset, zero handles it
        int w_q;
        if (k % 2 == 0) {
            w_q = int(packed & 0x0F);
        } else {
            w_q = int(packed >> 4);
        }
        
        uint group_idx = k / group_size;
        float scale = float(scales[group_idx * N + n]);
        float zero = float(zeros[group_idx * N + n]);
        float w = (float(w_q) - zero) * scale;
        acc += float(X[m * K + k]) * w;
    }
    
    Y[m * N + n] = half(acc);
}

// Half-precision Vec4 (most common for LLM inference)
kernel void matmul_int4_dequant_half_vec4(
    device const half* X [[buffer(0)]],           // [M, K]
    device const uchar4* W_packed [[buffer(1)]],  // [K/2, N/4]
    device const half4* scales [[buffer(2)]],     // [num_groups, N/4]
    device const half4* zeros [[buffer(3)]],      // [num_groups, N/4]
    device half4* Y [[buffer(4)]],                // [M, N/4]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint n4 = gid.x;
    uint m = gid.y;
    
    uint N4 = N / 4;
    if (n4 >= N4 || m >= M) return;
    
    float4 acc = float4(0.0f);
    
    for (uint k = 0; k < K; k += 2) {
        uint byte_idx = k / 2;
        uchar4 packed = W_packed[byte_idx * N4 + n4];
        
        uint group_idx = k / group_size;
        float4 scale = float4(scales[group_idx * N4 + n4]);
        float4 zero = float4(zeros[group_idx * N4 + n4]);
        
        float4 w0 = float4(
            int(packed.x & 0x0F),
            int(packed.y & 0x0F),
            int(packed.z & 0x0F),
            int(packed.w & 0x0F)
        );
        w0 = (w0 - zero) * scale;
        acc += float(X[m * K + k]) * w0;
        
        if (k + 1 < K) {
            float4 w1 = float4(
                int(packed.x >> 4),
                int(packed.y >> 4),
                int(packed.z >> 4),
                int(packed.w >> 4)
            );
            w1 = (w1 - zero) * scale;
            acc += float(X[m * K + k + 1]) * w1;
        }
    }
    
    Y[m * N4 + n4] = half4(acc);
}

// -----------------------------------------------------------------------------
// TILED INT4 Matmul - Cache X tile in threadgroup memory for massive data reuse
// -----------------------------------------------------------------------------
// Each threadgroup computes a TILE_M x TILE_N output tile.
// X is loaded into shared memory once and reused across TILE_N columns.
// This reduces global memory loads from O(M*K*N) to O(M*K + K*N/2).

#define INT4_TILE_M 8
#define INT4_TILE_N 32
#define INT4_TILE_K 128

kernel void matmul_int4_dequant_tiled(
    device const float* X [[buffer(0)]],
    device const uchar* W_packed [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zeros [[buffer(3)]],
    device float* Y [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]]
) {
    // Output tile this threadgroup computes
    uint tile_m = tgid.y * INT4_TILE_M;
    uint tile_n = tgid.x * INT4_TILE_N;
    
    // Thread's responsibility within tile
    uint local_m = tid.y;  // Which row within tile (0..TILE_M-1)
    uint local_n = tid.x;  // Which column within tile (0..TILE_N-1)
    
    uint global_m = tile_m + local_m;
    uint global_n = tile_n + local_n;
    
    if (global_m >= M || global_n >= N) return;
    
    // Shared memory for X tile and W tile
    threadgroup float X_tile[INT4_TILE_M][INT4_TILE_K];
    // W_tile: cache TILE_K/2 packed bytes x TILE_N columns of W
    // This is read as contiguous rows from [K/2, N] layout!
    threadgroup uchar W_tile[INT4_TILE_K / 2][INT4_TILE_N];
    // Cache scales/zeros for current group (one group covers 128 elements = TILE_K)
    threadgroup float scales_tile[INT4_TILE_N];
    threadgroup float zeros_tile[INT4_TILE_N];
    
    float acc = 0.0f;
    uint thread_idx = tid.y * tg_size.x + tid.x;
    uint num_threads = tg_size.x * tg_size.y;
    
    // Process K in tiles
    for (uint k_base = 0; k_base < K; k_base += INT4_TILE_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Collaborative load of X tile (TILE_M x TILE_K)
        uint x_elements = INT4_TILE_M * INT4_TILE_K;
        for (uint i = thread_idx; i < x_elements; i += num_threads) {
            uint load_m = i / INT4_TILE_K;
            uint load_k = i % INT4_TILE_K;
            uint global_load_m = tile_m + load_m;
            uint global_load_k = k_base + load_k;
            
            if (global_load_m < M && global_load_k < K) {
                X_tile[load_m][load_k] = X[global_load_m * K + global_load_k];
            } else {
                X_tile[load_m][load_k] = 0.0f;
            }
        }
        
        // Collaborative load of W tile (TILE_K/2 x TILE_N)
        // W is [K/2, N] layout, so reading rows [k_base/2, ..] cols [tile_n, ...] is CONTIGUOUS!
        uint w_elements = (INT4_TILE_K / 2) * INT4_TILE_N;
        for (uint i = thread_idx; i < w_elements; i += num_threads) {
            uint load_row = i / INT4_TILE_N;  // which packed byte row
            uint load_col = i % INT4_TILE_N;  // which column within tile
            uint global_byte_row = (k_base / 2) + load_row;
            uint global_col = tile_n + load_col;
            
            if (global_byte_row < (K / 2) && global_col < N) {
                W_tile[load_row][load_col] = W_packed[global_byte_row * N + global_col];
            } else {
                W_tile[load_row][load_col] = 0;
            }
        }
        
        // Load scales/zeros for this group (assuming group_size >= TILE_K)
        uint group_idx = k_base / group_size;
        for (uint i = thread_idx; i < INT4_TILE_N; i += num_threads) {
            uint global_col = tile_n + i;
            if (global_col < N) {
                scales_tile[i] = scales[group_idx * N + global_col];
                zeros_tile[i] = zeros[group_idx * N + global_col];
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Get scale/zero for this thread's column - load once, reuse
        float scale = scales_tile[local_n];
        float zero = zeros_tile[local_n];
        
        // Process K tile in chunks of 32 (16 bytes = 32 nibbles)
        // This uses 128-bit memory transactions for optimal Apple Silicon bandwidth
        uint k_tile_end = min((uint)INT4_TILE_K, K - k_base);
        
        // Process 32 K values at a time (16 packed bytes)
        for (uint kb = 0; kb < k_tile_end; kb += 32) {
            // Read 16 bytes as uint4 from W_tile (already in shared memory)
            // W_tile is [TILE_K/2][TILE_N], we read bytes [kb/2..kb/2+15] for our column
            uint byte_base = kb / 2;
            
            // Unroll: process bytes 0-15, each producing 2 K values
            #pragma unroll
            for (uint b = 0; b < 16 && (kb + b*2) < k_tile_end; b++) {
                uchar packed = W_tile[byte_base + b][local_n];
                
                int w_q0 = int(packed & 0x0F);
                int w_q1 = int(packed >> 4);
                
                float w0 = (float(w_q0) - zero) * scale;
                float w1 = (float(w_q1) - zero) * scale;
                
                uint k0 = kb + b * 2;
                uint k1 = kb + b * 2 + 1;
                
                acc += X_tile[local_m][k0] * w0;
                if (k1 < k_tile_end) {
                    acc += X_tile[local_m][k1] * w1;
                }
            }
        }
    }
    
    Y[global_m * N + global_n] = acc;
}

// -----------------------------------------------------------------------------
// SIMDGROUP-based INT4 Matmul (llama.cpp-style)
// -----------------------------------------------------------------------------
// Key optimizations from llama.cpp Q4_K kernel:
// 1. simd_sum() for K-dimension reduction (32 threads collaborate per row)
// 2. Multi-row processing (NR0 rows per simdgroup for better intensity)
// 3. Register caching of X input
// 4. Pre-computed masks for unpacking

#define NR0 2  // Number of output rows per simdgroup

kernel void matmul_int4_dequant_simd(
    device const float* X [[buffer(0)]],           // [M, K]
    device const uchar* W_packed [[buffer(1)]],    // [K/2, N] packed int4
    device const float* scales [[buffer(2)]],      // [num_groups, N]
    device const float* zeros [[buffer(3)]],       // [num_groups, N]
    device float* Y [[buffer(4)]],                 // [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    // Grid layout: x=N columns, y=M/NR0 row groups
    // Each simdgroup processes NR0 rows
    uint col = tgpig.x;  // Which column
    uint first_row = (tgpig.y * 4 + sgitg) * NR0;  // 4 simdgroups per threadgroup
    
    if (col >= N) return;
    
    // Accumulators for NR0 rows
    float sumf[NR0] = {0.0f};
    
    // Each of 32 threads handles a portion of K
    // Thread i handles K positions: i, i+32, i+64, ...
    for (uint k_base = tiisg; k_base < K; k_base += 32) {
        uint byte_idx = k_base / 2;
        uchar packed = W_packed[byte_idx * N + col];
        
        // Extract 4-bit value
        int w_q;
        if (k_base % 2 == 0) {
            w_q = int(packed & 0x0F);
        } else {
            w_q = int(packed >> 4);
        }
        
        // Get scale and zero for this group
        uint group_idx = k_base / group_size;
        float scale = scales[group_idx * N + col];
        float zero = zeros[group_idx * N + col];
        
        // Dequantize
        float w = (float(w_q) - zero) * scale;
        
        // Multiply-accumulate for each row
        for (int row = 0; row < NR0; row++) {
            uint global_row = first_row + row;
            if (global_row < M) {
                sumf[row] += X[global_row * K + k_base] * w;
            }
        }
    }
    
    // Reduce across simdgroup using simd_sum
    for (int row = 0; row < NR0; row++) {
        float sum_all = simd_sum(sumf[row]);
        
        // Only lane 0 writes output
        if (tiisg == 0) {
            uint global_row = first_row + row;
            if (global_row < M) {
                Y[global_row * N + col] = sum_all;
            }
        }
    }
}

// Half-precision simdgroup version
kernel void matmul_int4_dequant_simd_half(
    device const half* X [[buffer(0)]],
    device const uchar* W_packed [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const half* zeros [[buffer(3)]],
    device half* Y [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    uint col = tgpig.x;
    uint first_row = (tgpig.y * 4 + sgitg) * NR0;
    
    if (col >= N) return;
    
    float sumf[NR0] = {0.0f};
    
    for (uint k_base = tiisg; k_base < K; k_base += 32) {
        uint byte_idx = k_base / 2;
        uchar packed = W_packed[byte_idx * N + col];
        
        int w_q = (k_base % 2 == 0) ? int(packed & 0x0F) : int(packed >> 4);
        
        uint group_idx = k_base / group_size;
        float scale = float(scales[group_idx * N + col]);
        float zero = float(zeros[group_idx * N + col]);
        float w = (float(w_q) - zero) * scale;
        
        for (int row = 0; row < NR0; row++) {
            uint global_row = first_row + row;
            if (global_row < M) {
                sumf[row] += float(X[global_row * K + k_base]) * w;
            }
        }
    }
    
    for (int row = 0; row < NR0; row++) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            uint global_row = first_row + row;
            if (global_row < M) {
                Y[global_row * N + col] = half(sum_all);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Quantization Kernels (convert fp32/fp16 weights to INT8/INT4)
// -----------------------------------------------------------------------------


// Quantize fp32 weights to INT8 with per-group scale/zero
kernel void quantize_to_int8(
    device const float* W [[buffer(0)]],           // [K, N] fp32 weights
    device char* W_q [[buffer(1)]],                // [K, N] int8 output
    device float* scales [[buffer(2)]],            // [num_groups, N]
    device float* zeros [[buffer(3)]],             // [num_groups, N] (min values)
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& group_size [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]          // (n, group_idx)
) {
    uint n = gid.x;
    uint group_idx = gid.y;
    
    uint num_groups = (K + group_size - 1) / group_size;
    if (n >= N || group_idx >= num_groups) return;
    
    uint k_start = group_idx * group_size;
    uint k_end = min(k_start + group_size, K);
    
    // Find min/max in this group
    float min_val = INFINITY;
    float max_val = -INFINITY;
    
    for (uint k = k_start; k < k_end; k++) {
        float val = W[k * N + n];
        min_val = min(min_val, val);
        max_val = max(max_val, val);
    }
    
    // Compute scale and zero
    // INT8 range: -128 to 127
    float scale = (max_val - min_val) / 255.0f;
    scale = max(scale, 1e-8f);  // Avoid division by zero
    float zero = min_val / scale + 128.0f;
    
    scales[group_idx * N + n] = scale;
    zeros[group_idx * N + n] = zero;
    
    // Quantize
    for (uint k = k_start; k < k_end; k++) {
        float val = W[k * N + n];
        int8_t q = int8_t(clamp(round((val / scale) + zero - 128.0f), -128.0f, 127.0f));
        W_q[k * N + n] = q;
    }
}

// Quantize fp32 weights to INT4 (packed, 2 per byte)
kernel void quantize_to_int4(
    device const float* W [[buffer(0)]],           // [K, N] fp32 weights
    device uchar* W_packed [[buffer(1)]],          // [K/2, N] packed int4 output
    device float* scales [[buffer(2)]],            // [num_groups, N]
    device float* zeros [[buffer(3)]],             // [num_groups, N]
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& group_size [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]          // (n, group_idx)
) {
    uint n = gid.x;
    uint group_idx = gid.y;
    
    uint num_groups = (K + group_size - 1) / group_size;
    if (n >= N || group_idx >= num_groups) return;
    
    uint k_start = group_idx * group_size;
    uint k_end = min(k_start + group_size, K);
    
    // Find min/max in this group
    float min_val = INFINITY;
    float max_val = -INFINITY;
    
    for (uint k = k_start; k < k_end; k++) {
        float val = W[k * N + n];
        min_val = min(min_val, val);
        max_val = max(max_val, val);
    }
    
    // Compute scale and zero
    // INT4 range: 0 to 15 (we use unsigned with offset of 8 for signed range)
    float scale = (max_val - min_val) / 15.0f;
    scale = max(scale, 1e-8f);
    float zero = -min_val / scale;  // Offset so min maps to 0
    
    scales[group_idx * N + n] = scale;
    zeros[group_idx * N + n] = zero;
    
    // Quantize and pack (2 values per byte)
    for (uint k = k_start; k < k_end; k += 2) {
        uint byte_idx = k / 2;
        
        // First value (lower 4 bits)
        float val0 = W[k * N + n];
        uint q0 = uint(clamp(round(val0 / scale + zero), 0.0f, 15.0f));
        
        // Second value (upper 4 bits) - check bounds
        uint q1 = 0;
        if (k + 1 < k_end) {
            float val1 = W[(k + 1) * N + n];
            q1 = uint(clamp(round(val1 / scale + zero), 0.0f, 15.0f));
        }
        
        // Pack into byte
        W_packed[byte_idx * N + n] = uchar(q0 | (q1 << 4));
    }
}

// =============================================================================
// MAXIMUM PERFORMANCE INT4 GEMM
// =============================================================================
// 
// Optimizations applied:
// 1. simdgroup operations for fast reductions
// 2. Register blocking: each thread computes 4 output elements
// 3. Vectorized float4 loads for X (contiguous)
// 4. K-loop unrolled by 8 (process 8 K values per iteration)
// 5. Shared memory double buffering (prefetch next tile)
// 6. INT4 read gives 8x bandwidth savings vs float, 4x vs half
// 7. Dequant fused with load - amortized across all M for each W column
// 8. Thread coarsening - fewer threads, more work per thread
//
// Memory reads:
//   X: M*K * 2 bytes (half) - but cached per N-tile
//   W: K*N/8 bytes (INT4 packed) - 8x less than FP32, 4x less than FP16
//   scales/zeros: small, cached per K-tile
//
// For M=32, K=4096, N=4096:
//   X reads: 32 * 4096 * 2 = 256KB
//   W reads: 4096 * 4096 / 8 = 2MB (vs 16MB for FP16)
//   This should approach memory bandwidth limits

#define FAST_TILE_M 32    // Rows per threadgroup
#define FAST_TILE_N 64    // Cols per threadgroup  
#define FAST_TILE_K 32    // K per tile (must be even for INT4)
#define THREADS_M 8       // Threads in M dimension
#define THREADS_N 32      // Threads in N dimension
#define ELEMS_PER_THREAD_N 2  // Each thread handles 2 N columns

kernel void matmul_int4_fast(
    device const half* X [[buffer(0)]],            // [M, K]
    device const uchar* W_packed [[buffer(1)]],    // [K/2, N] packed INT4
    device const half* scales [[buffer(2)]],       // [num_groups, N]
    device const half* zeros [[buffer(3)]],        // [num_groups, N]
    device half* Y [[buffer(4)]],                  // [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint tindex [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Tile positions
    const uint tile_m = tgid.y * FAST_TILE_M;
    const uint tile_n = tgid.x * FAST_TILE_N;
    
    // Thread position: each thread handles 4 M rows, 2 N columns
    const uint local_m = tid.y * 4;  // 0, 4, 8, ... 28
    const uint local_n = tid.x * ELEMS_PER_THREAD_N;  // 0, 2, 4, ... 62
    
    // Shared memory
    threadgroup half X_tile[FAST_TILE_M][FAST_TILE_K + 1];  // +1 to avoid bank conflicts
    threadgroup half W_dequant[FAST_TILE_K][FAST_TILE_N + 1];
    // NEW: Cache scales/zeros for the N-tile
    threadgroup half scales_tile[FAST_TILE_N];
    threadgroup half zeros_tile[FAST_TILE_N];
    
    // Register accumulators - 4 M rows x 2 N cols = 8 elements per thread
    float4 acc0 = float4(0.0f);  // M rows 0-3, N col 0
    float4 acc1 = float4(0.0f);  // M rows 0-3, N col 1
    
    const uint num_threads = THREADS_M * THREADS_N;  // 256
    
    // Process K dimension
    for (uint k_base = 0; k_base < K; k_base += FAST_TILE_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // === Collaborative load of X tile ===
        // X is [M, K], X_tile is [TILE_M, TILE_K]
        // Use float4 loads where possible (4 halfs = 8 bytes)
        const uint x_elements = FAST_TILE_M * FAST_TILE_K;
        for (uint i = tindex; i < x_elements; i += num_threads) {
            uint lm = i / FAST_TILE_K;
            uint lk = i % FAST_TILE_K;
            uint gm = tile_m + lm;
            uint gk = k_base + lk;
            X_tile[lm][lk] = (gm < M && gk < K) ? X[gm * K + gk] : half(0);
        }
        
        // === Pre-load scales/zeros for this K-tile's group ===
        // Only need to load once per group (when k_base changes group)
        uint group_idx = k_base / group_size;
        for (uint i = tindex; i < FAST_TILE_N; i += num_threads) {
            uint global_col = tile_n + i;
            if (global_col < N) {
                scales_tile[i] = scales[group_idx * N + global_col];
                zeros_tile[i] = zeros[group_idx * N + global_col];
            } else {
                scales_tile[i] = half(1);
                zeros_tile[i] = half(0);
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // === Collaborative load and dequant of W tile ===
        // W_packed is [K/2, N], each byte has 2 INT4 values
        // Read INT4, dequant using cached scales/zeros
        const uint w_bytes = (FAST_TILE_K / 2) * FAST_TILE_N;
        for (uint i = tindex; i < w_bytes; i += num_threads) {
            uint byte_row = i / FAST_TILE_N;
            uint byte_col = i % FAST_TILE_N;
            
            uint global_byte_row = (k_base / 2) + byte_row;
            uint global_col = tile_n + byte_col;
            
            uchar packed = 0;
            
            if (global_byte_row < (K / 2) && global_col < N) {
                packed = W_packed[global_byte_row * N + global_col];
            }
            
            // Use CACHED scales/zeros from shared memory (no random access!)
            half scale = scales_tile[byte_col];
            half zero = zeros_tile[byte_col];
            
            // Dequant both nibbles
            int q0 = int(packed & 0x0F);
            int q1 = int(packed >> 4);
            half w0 = (half(q0) - zero) * scale;
            half w1 = (half(q1) - zero) * scale;
            
            // Store to shared memory
            uint k0 = byte_row * 2;
            uint k1 = byte_row * 2 + 1;
            W_dequant[k0][byte_col] = w0;
            if (k1 < FAST_TILE_K) W_dequant[k1][byte_col] = w1;
        }
        
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // === Compute ===
        // Each thread computes 4 M x 2 N = 8 output elements
        uint k_end = min((uint)FAST_TILE_K, K - k_base);
        
        // Unroll K loop by 4
        for (uint k = 0; k < k_end; k += 4) {
            // Load X values for 4 M rows (registers)
            half4 x0 = half4(X_tile[local_m + 0][k], X_tile[local_m + 0][k+1], 
                            X_tile[local_m + 0][k+2], X_tile[local_m + 0][k+3]);
            half4 x1 = half4(X_tile[local_m + 1][k], X_tile[local_m + 1][k+1],
                            X_tile[local_m + 1][k+2], X_tile[local_m + 1][k+3]);
            half4 x2 = half4(X_tile[local_m + 2][k], X_tile[local_m + 2][k+1],
                            X_tile[local_m + 2][k+2], X_tile[local_m + 2][k+3]);
            half4 x3 = half4(X_tile[local_m + 3][k], X_tile[local_m + 3][k+1],
                            X_tile[local_m + 3][k+2], X_tile[local_m + 3][k+3]);
            
            // Load W values for 2 N columns (registers)
            half4 w0_col0 = half4(W_dequant[k][local_n], W_dequant[k+1][local_n],
                                  W_dequant[k+2][local_n], W_dequant[k+3][local_n]);
            half4 w0_col1 = half4(W_dequant[k][local_n+1], W_dequant[k+1][local_n+1],
                                  W_dequant[k+2][local_n+1], W_dequant[k+3][local_n+1]);
            
            // Accumulate - dot products
            acc0.x += dot(float4(x0), float4(w0_col0));
            acc0.y += dot(float4(x1), float4(w0_col0));
            acc0.z += dot(float4(x2), float4(w0_col0));
            acc0.w += dot(float4(x3), float4(w0_col0));
            
            acc1.x += dot(float4(x0), float4(w0_col1));
            acc1.y += dot(float4(x1), float4(w0_col1));
            acc1.z += dot(float4(x2), float4(w0_col1));
            acc1.w += dot(float4(x3), float4(w0_col1));
        }
    }
    
    // === Write output ===
    // Each thread writes 4 M x 2 N elements
    for (uint dm = 0; dm < 4; dm++) {
        uint gm = tile_m + local_m + dm;
        if (gm < M) {
            uint gn0 = tile_n + local_n;
            uint gn1 = tile_n + local_n + 1;
            
            if (gn0 < N) Y[gm * N + gn0] = half(dm == 0 ? acc0.x : (dm == 1 ? acc0.y : (dm == 2 ? acc0.z : acc0.w)));
            if (gn1 < N) Y[gm * N + gn1] = half(dm == 0 ? acc1.x : (dm == 1 ? acc1.y : (dm == 2 ? acc1.z : acc1.w)));
        }
    }
}

// =============================================================================
// FUSED INT4 MATMUL + SILU KERNEL
// =============================================================================
// Combines: Y = silu(X @ dequant(W))
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Eliminates memory round-trip between matmul and activation

kernel void matmul_int4_silu_fast(
    device const half* X [[buffer(0)]],            // [M, K]
    device const uchar* W_packed [[buffer(1)]],    // [K/2, N] packed INT4
    device const half* scales [[buffer(2)]],       // [num_groups, N]
    device const half* zeros [[buffer(3)]],        // [num_groups, N]
    device half* Y [[buffer(4)]],                  // [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint tindex [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Tile positions
    const uint tile_m = tgid.y * FAST_TILE_M;
    const uint tile_n = tgid.x * FAST_TILE_N;
    
    // Thread position: each thread handles 4 M rows, 2 N columns
    const uint local_m = tid.y * 4;
    const uint local_n = tid.x * ELEMS_PER_THREAD_N;
    
    // Shared memory
    threadgroup half X_tile[FAST_TILE_M][FAST_TILE_K + 1];
    threadgroup half W_dequant[FAST_TILE_K][FAST_TILE_N + 1];
    
    // Register accumulators
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    
    const uint num_threads = THREADS_M * THREADS_N;
    
    // Process K dimension (identical to matmul_int4_fast)
    for (uint k_base = 0; k_base < K; k_base += FAST_TILE_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Collaborative load of X tile
        const uint x_elements = FAST_TILE_M * FAST_TILE_K;
        for (uint i = tindex; i < x_elements; i += num_threads) {
            uint lm = i / FAST_TILE_K;
            uint lk = i % FAST_TILE_K;
            uint gm = tile_m + lm;
            uint gk = k_base + lk;
            X_tile[lm][lk] = (gm < M && gk < K) ? X[gm * K + gk] : half(0);
        }
        
        // Collaborative load and dequant of W tile
        uint group_idx = k_base / group_size;
        
        const uint w_bytes = (FAST_TILE_K / 2) * FAST_TILE_N;
        for (uint i = tindex; i < w_bytes; i += num_threads) {
            uint byte_row = i / FAST_TILE_N;
            uint byte_col = i % FAST_TILE_N;
            
            uint global_byte_row = (k_base / 2) + byte_row;
            uint global_col = tile_n + byte_col;
            
            uchar packed = 0;
            half scale = half(1);
            half zero = half(0);
            
            if (global_byte_row < (K / 2) && global_col < N) {
                packed = W_packed[global_byte_row * N + global_col];
                scale = scales[group_idx * N + global_col];
                zero = zeros[group_idx * N + global_col];
            }
            
            int q0 = int(packed & 0x0F);
            int q1 = int(packed >> 4);
            half w0 = (half(q0) - zero) * scale;
            half w1 = (half(q1) - zero) * scale;
            
            uint k0 = byte_row * 2;
            uint k1 = byte_row * 2 + 1;
            W_dequant[k0][byte_col] = w0;
            if (k1 < FAST_TILE_K) W_dequant[k1][byte_col] = w1;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute
        uint k_end = min((uint)FAST_TILE_K, K - k_base);
        
        for (uint k = 0; k < k_end; k += 4) {
            half4 x0 = half4(X_tile[local_m + 0][k], X_tile[local_m + 0][k+1], 
                            X_tile[local_m + 0][k+2], X_tile[local_m + 0][k+3]);
            half4 x1 = half4(X_tile[local_m + 1][k], X_tile[local_m + 1][k+1],
                            X_tile[local_m + 1][k+2], X_tile[local_m + 1][k+3]);
            half4 x2 = half4(X_tile[local_m + 2][k], X_tile[local_m + 2][k+1],
                            X_tile[local_m + 2][k+2], X_tile[local_m + 2][k+3]);
            half4 x3 = half4(X_tile[local_m + 3][k], X_tile[local_m + 3][k+1],
                            X_tile[local_m + 3][k+2], X_tile[local_m + 3][k+3]);
            
            half4 w0_col0 = half4(W_dequant[k][local_n], W_dequant[k+1][local_n],
                                  W_dequant[k+2][local_n], W_dequant[k+3][local_n]);
            half4 w0_col1 = half4(W_dequant[k][local_n+1], W_dequant[k+1][local_n+1],
                                  W_dequant[k+2][local_n+1], W_dequant[k+3][local_n+1]);
            
            acc0.x += dot(float4(x0), float4(w0_col0));
            acc0.y += dot(float4(x1), float4(w0_col0));
            acc0.z += dot(float4(x2), float4(w0_col0));
            acc0.w += dot(float4(x3), float4(w0_col0));
            
            acc1.x += dot(float4(x0), float4(w0_col1));
            acc1.y += dot(float4(x1), float4(w0_col1));
            acc1.z += dot(float4(x2), float4(w0_col1));
            acc1.w += dot(float4(x3), float4(w0_col1));
        }
    }
    
    // === Apply SiLU and write output ===
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float4 silu0 = acc0 / (1.0f + exp(-acc0));
    float4 silu1 = acc1 / (1.0f + exp(-acc1));
    
    for (uint dm = 0; dm < 4; dm++) {
        uint gm = tile_m + local_m + dm;
        if (gm < M) {
            uint gn0 = tile_n + local_n;
            uint gn1 = tile_n + local_n + 1;
            
            if (gn0 < N) Y[gm * N + gn0] = half(dm == 0 ? silu0.x : (dm == 1 ? silu0.y : (dm == 2 ? silu0.z : silu0.w)));
            if (gn1 < N) Y[gm * N + gn1] = half(dm == 0 ? silu1.x : (dm == 1 ? silu1.y : (dm == 2 ? silu1.z : silu1.w)));
        }
    }
}

// =============================================================================
// FUSED INT4 MATMUL + GELU KERNEL
// =============================================================================
// Combines: Y = gelu(X @ dequant(W))
// GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

#define GELU_SQRT_2_PI 0.7978845608f
#define GELU_COEFF 0.044715f

kernel void matmul_int4_gelu_fast(
    device const half* X [[buffer(0)]],
    device const uchar* W_packed [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const half* zeros [[buffer(3)]],
    device half* Y [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint tindex [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Tile positions
    const uint tile_m = tgid.y * FAST_TILE_M;
    const uint tile_n = tgid.x * FAST_TILE_N;
    
    const uint local_m = tid.y * 4;
    const uint local_n = tid.x * ELEMS_PER_THREAD_N;
    
    threadgroup half X_tile[FAST_TILE_M][FAST_TILE_K + 1];
    threadgroup half W_dequant[FAST_TILE_K][FAST_TILE_N + 1];
    
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    
    const uint num_threads = THREADS_M * THREADS_N;
    
    for (uint k_base = 0; k_base < K; k_base += FAST_TILE_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        const uint x_elements = FAST_TILE_M * FAST_TILE_K;
        for (uint i = tindex; i < x_elements; i += num_threads) {
            uint lm = i / FAST_TILE_K;
            uint lk = i % FAST_TILE_K;
            uint gm = tile_m + lm;
            uint gk = k_base + lk;
            X_tile[lm][lk] = (gm < M && gk < K) ? X[gm * K + gk] : half(0);
        }
        
        uint group_idx = k_base / group_size;
        
        const uint w_bytes = (FAST_TILE_K / 2) * FAST_TILE_N;
        for (uint i = tindex; i < w_bytes; i += num_threads) {
            uint byte_row = i / FAST_TILE_N;
            uint byte_col = i % FAST_TILE_N;
            
            uint global_byte_row = (k_base / 2) + byte_row;
            uint global_col = tile_n + byte_col;
            
            uchar packed = 0;
            half scale = half(1);
            half zero = half(0);
            
            if (global_byte_row < (K / 2) && global_col < N) {
                packed = W_packed[global_byte_row * N + global_col];
                scale = scales[group_idx * N + global_col];
                zero = zeros[group_idx * N + global_col];
            }
            
            int q0 = int(packed & 0x0F);
            int q1 = int(packed >> 4);
            half w0 = (half(q0) - zero) * scale;
            half w1 = (half(q1) - zero) * scale;
            
            uint k0 = byte_row * 2;
            uint k1 = byte_row * 2 + 1;
            W_dequant[k0][byte_col] = w0;
            if (k1 < FAST_TILE_K) W_dequant[k1][byte_col] = w1;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint k_end = min((uint)FAST_TILE_K, K - k_base);
        
        for (uint k = 0; k < k_end; k += 4) {
            half4 x0 = half4(X_tile[local_m + 0][k], X_tile[local_m + 0][k+1], 
                            X_tile[local_m + 0][k+2], X_tile[local_m + 0][k+3]);
            half4 x1 = half4(X_tile[local_m + 1][k], X_tile[local_m + 1][k+1],
                            X_tile[local_m + 1][k+2], X_tile[local_m + 1][k+3]);
            half4 x2 = half4(X_tile[local_m + 2][k], X_tile[local_m + 2][k+1],
                            X_tile[local_m + 2][k+2], X_tile[local_m + 2][k+3]);
            half4 x3 = half4(X_tile[local_m + 3][k], X_tile[local_m + 3][k+1],
                            X_tile[local_m + 3][k+2], X_tile[local_m + 3][k+3]);
            
            half4 w0_col0 = half4(W_dequant[k][local_n], W_dequant[k+1][local_n],
                                  W_dequant[k+2][local_n], W_dequant[k+3][local_n]);
            half4 w0_col1 = half4(W_dequant[k][local_n+1], W_dequant[k+1][local_n+1],
                                  W_dequant[k+2][local_n+1], W_dequant[k+3][local_n+1]);
            
            acc0.x += dot(float4(x0), float4(w0_col0));
            acc0.y += dot(float4(x1), float4(w0_col0));
            acc0.z += dot(float4(x2), float4(w0_col0));
            acc0.w += dot(float4(x3), float4(w0_col0));
            
            acc1.x += dot(float4(x0), float4(w0_col1));
            acc1.y += dot(float4(x1), float4(w0_col1));
            acc1.z += dot(float4(x2), float4(w0_col1));
            acc1.w += dot(float4(x3), float4(w0_col1));
        }
    }
    
    // === Apply GELU and write output ===
    // GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float4 x3_0 = acc0 * acc0 * acc0;
    float4 x3_1 = acc1 * acc1 * acc1;
    float4 arg0 = GELU_SQRT_2_PI * (acc0 + GELU_COEFF * x3_0);
    float4 arg1 = GELU_SQRT_2_PI * (acc1 + GELU_COEFF * x3_1);
    float4 gelu0 = acc0 * 0.5f * (1.0f + tanh(arg0));
    float4 gelu1 = acc1 * 0.5f * (1.0f + tanh(arg1));
    
    for (uint dm = 0; dm < 4; dm++) {
        uint gm = tile_m + local_m + dm;
        if (gm < M) {
            uint gn0 = tile_n + local_n;
            uint gn1 = tile_n + local_n + 1;
            
            if (gn0 < N) Y[gm * N + gn0] = half(dm == 0 ? gelu0.x : (dm == 1 ? gelu0.y : (dm == 2 ? gelu0.z : gelu0.w)));
            if (gn1 < N) Y[gm * N + gn1] = half(dm == 0 ? gelu1.x : (dm == 1 ? gelu1.y : (dm == 2 ? gelu1.z : gelu1.w)));
        }
    }
}

// =============================================================================
// TENSOR CORE INT4 GEMM - Maximum Performance using Apple MPP API
// =============================================================================
// Uses Apple's internal mpp::tensor_ops::matmul2d API (same as llama.cpp)
// for hardware-accelerated matrix multiplication.
//
// Approach:
// 1. Load INT4 weights from memory
// 2. Dequant to half in threadgroup memory
// 3. Use hardware tensor cores for the actual matmul
//
// Tile sizes optimized for Apple's tensor cores

#define TC_TILE_M 64    // M per threadgroup (NR0)
#define TC_TILE_N 32    // N per threadgroup (NR1)  
#define TC_TILE_K 64    // K per tile (NK)

#ifdef METALCORE_HAS_TENSOR

kernel void matmul_int4_tensor(
    device const half* X [[buffer(0)]],            // [M, K]
    device const uchar* W_packed [[buffer(1)]],    // [K/2, N] packed INT4
    device const half* scales [[buffer(2)]],       // [num_groups, N]
    device const half* zeros [[buffer(3)]],        // [num_groups, N]
    device half* Y [[buffer(4)]],                  // [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    threadgroup char* shmem [[threadgroup(0)]],    // shared memory
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup position
    const uint tile_m = tgid.y * TC_TILE_M;
    const uint tile_n = tgid.x * TC_TILE_N;
    
    if (tile_m >= M || tile_n >= N) return;
    
    // Threadgroup memory layout
    // sa: X tile [TC_TILE_M, TC_TILE_K] half = 64*64*2 = 8KB
    // sb: W tile [TC_TILE_K, TC_TILE_N] half = 64*32*2 = 4KB
    // sc: Output [TC_TILE_M, TC_TILE_N] float = 64*32*4 = 8KB
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + TC_TILE_M * TC_TILE_K * sizeof(half));
    threadgroup float* sc = (threadgroup float*)(shmem + TC_TILE_M * TC_TILE_K * sizeof(half) + TC_TILE_K * TC_TILE_N * sizeof(half));
    
    // Create tensor views for matmul2d
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(TC_TILE_K, TC_TILE_M));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(TC_TILE_N, TC_TILE_K));
    
    // Set up matmul operator (NR1 x NR0 x NK, transA=false, transB=true)
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(TC_TILE_N, TC_TILE_M, TC_TILE_K, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    
    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    
    const uint num_threads = 128;  // 4 simdgroups * 32 threads
    
    // Process K in tiles
    for (uint k_base = 0; k_base < K; k_base += TC_TILE_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // === Load X tile [TC_TILE_M, TC_TILE_K] ===
        // X is [M, K], stored row-major
        const uint x_elements = TC_TILE_M * TC_TILE_K;
        for (uint i = tiitg; i < x_elements; i += num_threads) {
            uint lm = i / TC_TILE_K;
            uint lk = i % TC_TILE_K;
            uint gm = tile_m + lm;
            uint gk = k_base + lk;
            // Store transposed for tensor API: sa[k, m]
            sa[lk * TC_TILE_M + lm] = (gm < M && gk < K) ? X[gm * K + gk] : half(0);
        }
        
        // === Load and dequant W tile [TC_TILE_K, TC_TILE_N] ===
        uint group_idx = k_base / group_size;
        const uint w_bytes = (TC_TILE_K / 2) * TC_TILE_N;
        for (uint i = tiitg; i < w_bytes; i += num_threads) {
            uint byte_row = i / TC_TILE_N;
            uint byte_col = i % TC_TILE_N;
            
            uint global_byte_row = (k_base / 2) + byte_row;
            uint global_col = tile_n + byte_col;
            
            uchar packed = 0;
            half scale = half(1);
            half zero = half(0);
            
            if (global_byte_row < (K / 2) && global_col < N) {
                packed = W_packed[global_byte_row * N + global_col];
                scale = scales[group_idx * N + global_col];
                zero = zeros[group_idx * N + global_col];
            }
            
            int q0 = int(packed & 0x0F);
            int q1 = int(packed >> 4);
            half w0 = (half(q0) - zero) * scale;
            half w1 = (half(q1) - zero) * scale;
            
            // Store transposed for tensor API: sb[n, k]
            uint k0 = byte_row * 2;
            uint k1 = byte_row * 2 + 1;
            sb[byte_col * TC_TILE_K + k0] = w0;
            if (k1 < TC_TILE_K) sb[byte_col * TC_TILE_K + k1] = w1;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // === Hardware tensor core matmul ===
        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);
        mm.run(sB, sA, cT);
    }
    
    // Store results
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Store cT to threadgroup memory
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(TC_TILE_M, TC_TILE_N));
    cT.store(tC);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write to global memory
    for (uint i = tiitg; i < TC_TILE_M * TC_TILE_N; i += num_threads) {
        uint lm = i / TC_TILE_N;
        uint ln = i % TC_TILE_N;
        uint gm = tile_m + lm;
        uint gn = tile_n + ln;
        if (gm < M && gn < N) {
            Y[gm * N + gn] = half(sc[lm * TC_TILE_N + ln]);
        }
    }
}

#endif // METALCORE_HAS_TENSOR

// =============================================================================
// SIMDGROUP MATRIX INT4 GEMM - Hardware 8x8 Matrix Multiply (Metal 3, M1/M2/M3)
// =============================================================================
// Uses simdgroup_matrix operations which ARE available on Metal 3 (M1/M2/M3).
// This is llama.cpp's fallback path when tensor API is not available.
//
// Key operations:
// - simdgroup_half8x8: 8x8 matrix type for halfs
// - simdgroup_float8x8: 8x8 accumulator for floats
// - simdgroup_load(matrix, ptr, stride): load from threadgroup memory
// - simdgroup_multiply_accumulate(C, A, B, C): C += A @ B (hardware accelerated)
//
// Tiling strategy (from llama.cpp):
// - 4 simdgroups per threadgroup
// - Each simdgroup handles 32x16 output tile
// - Uses 8x8 matrix multiply as building block
// - Total 64x32 output per threadgroup

// Tile sizes matching llama.cpp
#define SG_NR0 64    // M per threadgroup
#define SG_NR1 32    // N per threadgroup
#define SG_NK  64    // K per tile

kernel void matmul_int4_simdgroup(
    device const half* X [[buffer(0)]],            // [M, K]
    device const uchar* W_packed [[buffer(1)]],    // [K/2, N] packed INT4
    device const half* scales [[buffer(2)]],       // [num_groups, N]
    device const half* zeros [[buffer(3)]],        // [num_groups, N]
    device half* Y [[buffer(4)]],                  // [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    // Threadgroup output position
    const uint r0 = tgid.y * SG_NR0;  // Starting M
    const uint r1 = tgid.x * SG_NR1;  // Starting N
    
    if (r0 >= M || r1 >= N) return;
    
    // Threadgroup memory layout:
    // sa: X tile, stored as [NK, NR0] for transposed access = 64*64*2 = 8KB
    // sb: W tile, stored as [NR1, NK] = 32*64*2 = 4KB  
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + SG_NK * SG_NR0 * sizeof(half));
    
    // Each simdgroup (sgitg 0-3) handles a 32x16 output block
    // sgitg % 2 selects M half (0-31 or 32-63)
    // sgitg / 2 selects N half (0-15 or 16-31)
    
    // Simdgroup accumulator matrices (8 x 8x8 = 64 output values per simdgroup)
    // Layout: 4 M tiles x 2 N tiles = 32x16 per simdgroup
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }
    
    // Matrix A and B tiles for loading
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    
    const uint num_threads = 128;  // 4 simdgroups * 32 threads
    
    // Process K in tiles
    for (uint k_base = 0; k_base < K; k_base += SG_NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // === Load X tile [NR0, NK] into sa as [NK, NR0] (transposed) ===
        const uint x_elements = SG_NR0 * SG_NK;
        for (uint i = tiitg; i < x_elements; i += num_threads) {
            uint lm = i / SG_NK;  // local M
            uint lk = i % SG_NK;  // local K
            uint gm = r0 + lm;
            uint gk = k_base + lk;
            // Store transposed: sa[k, m] = X[m, k]
            sa[lk * SG_NR0 + lm] = (gm < M && gk < K) ? X[gm * K + gk] : half(0);
        }
        
        // === Load and dequant W tile [NK, NR1] into sb as [NR1, NK] (transposed) ===
        uint group_idx = k_base / group_size;
        const uint w_bytes = (SG_NK / 2) * SG_NR1;
        for (uint i = tiitg; i < w_bytes; i += num_threads) {
            uint byte_row = i / SG_NR1;  // K/2 index
            uint ln = i % SG_NR1;        // local N
            
            uint global_byte_row = (k_base / 2) + byte_row;
            uint gn = r1 + ln;
            
            uchar packed = 0;
            half scale = half(1);
            half zero = half(0);
            
            if (global_byte_row < (K / 2) && gn < N) {
                packed = W_packed[global_byte_row * N + gn];
                scale = scales[group_idx * N + gn];
                zero = zeros[group_idx * N + gn];
            }
            
            int q0 = int(packed & 0x0F);
            int q1 = int(packed >> 4);
            half w0 = (half(q0) - zero) * scale;
            half w1 = (half(q1) - zero) * scale;
            
            // Store transposed: sb[n, k]
            uint k0 = byte_row * 2;
            uint k1 = byte_row * 2 + 1;
            sb[ln * SG_NK + k0] = w0;
            if (k1 < SG_NK) sb[ln * SG_NK + k1] = w1;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // === Simdgroup matrix multiply ===
        // Each simdgroup handles 32x16 output (4 x 2 of 8x8 blocks)
        // sgitg % 2: which half of M (0=first 32, 1=second 32)
        // sgitg / 2: which half of N (0=first 16, 1=second 16)
        
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);  // Start of A blocks for this simdgroup
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);  // Start of B blocks for this simdgroup
        
        // Process K in 8-wide chunks (8 iterations for NK=64)
        for (short ik = 0; ik < SG_NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            
            // Load 4 x 8x8 tiles from A (covers 32 M rows, 8 K cols)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            
            simdgroup_barrier(mem_flags::mem_none);
            
            // Load 2 x 8x8 tiles from B (covers 16 N cols, 8 K rows)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            
            simdgroup_barrier(mem_flags::mem_none);
            
            // Multiply: 8 output 8x8 blocks (4 M x 2 N)
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            
            lsma += 8 * 64;  // Advance by 8 K
            lsmb += 4 * 64;  // Advance by 8 K (but 2 tiles so 4*8*2 = 64 per half)
        }
    }
    
    // === Store results ===
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Each simdgroup stores its 8 8x8 output blocks
    // Temp storage in threadgroup memory
    threadgroup float* temp_out = (threadgroup float*)(shmem);
    
    // Store the 8x8 blocks
    for (short i = 0; i < 8; i++) {
        // Block position within simdgroup's 32x16 output
        short block_m = (i % 4) * 8;  // 0, 8, 16, 24
        short block_n = (i / 4) * 8;  // 0 or 8
        
        // Global offset for this simdgroup
        short sg_m_offset = (sgitg % 2) * 32;
        short sg_n_offset = (sgitg / 2) * 16;
        
        short out_m = block_m + sg_m_offset;
        short out_n = block_n + sg_n_offset;
        
        simdgroup_store(mc[i], temp_out + out_m * SG_NR1 + out_n, SG_NR1, 0, false);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write to global memory
    for (uint i = tiitg; i < SG_NR0 * SG_NR1; i += num_threads) {
        uint lm = i / SG_NR1;
        uint ln = i % SG_NR1;
        uint gm = r0 + lm;
        uint gn = r1 + ln;
        if (gm < M && gn < N) {
            Y[gm * N + gn] = half(temp_out[lm * SG_NR1 + ln]);
        }
    }
}

// =============================================================================
// LLAMA.CPP-STYLE OPTIMIZED INT4 GEMM
// =============================================================================
// Portions derived from llama.cpp (https://github.com/ggerganov/llama.cpp)
// Copyright (c) 2023-2024 The ggml authors
// Licensed under MIT License
//
// Key optimizations from llama.cpp:
// 1. Vectorized dequantization using half4x4 registers
// 2. Cooperative loading - all 128 threads participate in loading tiles
// 3. Proper memory layout for efficient simdgroup_load
// 4. Unrolled inner loops with pragma
// 5. Careful indexing to maximize memory coalescing
//
// This is adapted for metalcore's per-group INT4 format where scales and zeros
// are stored in separate tensors (not embedded in weight blocks like ggml).

#define LLAMA_NR0 64    // M per threadgroup
#define LLAMA_NR1 32    // N per threadgroup
#define LLAMA_NK  32    // K per tile (must match simdgroup 8x8 pattern)
#define LLAMA_NL0 2     // NK/16 = threads processing A dimension
#define LLAMA_NL1 4     // NK/8 = threads processing B dimension

// Unroll helper
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Vectorized dequantization for 16 INT4 values -> half4x4 register
// Adapted from llama.cpp's dequantize_q4_0 for our per-group format
inline void dequant_int4_vec16(
    device const uchar* packed,     // 8 bytes = 16 INT4 values
    half scale,
    half zero,
    thread half4x4& reg
) {
    // Read as 4 ushorts for vectorized unpacking
    device const ushort* qs = (device const ushort*)packed;
    
    // Precompute constants
    const half md = -zero * scale;  // offset term
    const ushort mask_lo = 0x000F;
    const ushort mask_hi = 0x00F0;
    
    // Unpack and dequantize 16 values into 4x4 matrix
    FOR_UNROLL (short i = 0; i < 4; i++) {
        ushort q = qs[i];
        // Low nibbles (even indices)
        reg[i][0] = half(q & mask_lo) * scale + md;
        reg[i][1] = half((q >> 4) & mask_lo) * scale + md;
        // High byte nibbles (odd indices)  
        reg[i][2] = half((q >> 8) & mask_lo) * scale + md;
        reg[i][3] = half((q >> 12) & mask_lo) * scale + md;
    }
}

kernel void matmul_int4_llama(
    device const half* X [[buffer(0)]],            // [M, K]
    device const uchar* W_packed [[buffer(1)]],    // [K/2, N] packed INT4
    device const half* scales [[buffer(2)]],       // [num_groups, N]
    device const half* zeros [[buffer(3)]],        // [num_groups, N]
    device half* Y [[buffer(4)]],                  // [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup memory layout (from llama.cpp):
    // sa: [LLAMA_NK, LLAMA_NR0] halfs for X tile (transposed) = 32*64*2 = 4KB
    // sb: [LLAMA_NR1, LLAMA_NK] halfs for W tile (transposed) = 32*32*2 = 2KB
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
    
    // Also used for output accumulation
    threadgroup float* sc = (threadgroup float*)(shmem);
    
    const ushort r0 = tgid.y * LLAMA_NR0;  // Starting M
    const ushort r1 = tgid.x * LLAMA_NR1;  // Starting N
    
    // Bounds for partial tiles at edges
    const short nr0 = (M - r0 < LLAMA_NR0) ? (M - r0) : LLAMA_NR0;
    const short nr1 = (N - r1 < LLAMA_NR1) ? (N - r1) : LLAMA_NR1;
    
    // Thread assignment for loading (from llama.cpp)
    const short lr0 = min((short)(tiitg / LLAMA_NL0), (short)(nr0 - 1));
    const short lr1 = min((short)(tiitg / LLAMA_NL1), (short)(nr1 - 1));
    
    // Simdgroup accumulators (8 x 8x8 matrices = 32x16 output per simdgroup)
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    
    FOR_UNROLL (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }
    
    // Process K in tiles
    for (uint k_base = 0; k_base < K; k_base += LLAMA_NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // === Cooperative load X tile ===
        // Each thread loads multiple elements using vectorized access
        // Layout: sa[k, m] = X[m, k] (transposed for simdgroup_load)
        {
            const short il0 = tiitg % LLAMA_NL0;  // 0 or 1
            const ushort gm = r0 + lr0;
            
            // Load 16 consecutive K values per thread iteration
            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = lr0 / 8;
                const short lx = lr0 % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                const ushort gk = k_base + 16 * il0 + i;
                
                half val = (gm < M && gk < K) ? X[gm * K + gk] : half(0);
                *(sa + 64 * ib + 8 * ly + lx) = val;
            }
        }
        
        // === Cooperative load and dequant W tile ===
        // Each thread loads 8 elements (4 bytes) and dequantizes
        {
            const short il1 = tiitg % LLAMA_NL1;  // 0-3
            const ushort gn = r1 + lr1;
            
            // Get scale/zero for this K position
            uint group_idx = k_base / group_size;
            half scale = (gn < N) ? scales[group_idx * N + gn] : half(1);
            half zero = (gn < N) ? zeros[group_idx * N + gn] : half(0);
            
            const short iy = 8 * il1;  // K offset within tile
            
            if (gn < N && k_base + iy < K) {
                device const half* y_src = X;  // temporary, we'll load from W
                
                // Load 8 halfs worth of W data
                FOR_UNROLL (short i = 0; i < 8; i++) {
                    const short sx = il1;
                    const short sy = lr1 / 8;
                    const short lx = i;
                    const short ly = lr1 % 8;
                    const short ib = 4 * sx + sy;
                    
                    // Read and dequant from W_packed
                    const ushort gk = k_base + iy + i;
                    half val = half(0);
                    if (gk < K && gn < N) {
                        uchar packed = W_packed[(gk / 2) * N + gn];
                        int q = (gk % 2 == 0) ? int(packed & 0x0F) : int(packed >> 4);
                        val = (half(q) - zero) * scale;
                    }
                    
                    *(sb + 64 * ib + 8 * ly + lx) = val;
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // === Simdgroup matrix multiply (llama.cpp style) ===
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        
        FOR_UNROLL (short ik = 0; ik < LLAMA_NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            
            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            
            simdgroup_barrier(mem_flags::mem_none);
            
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            
            simdgroup_barrier(mem_flags::mem_none);
            
            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }
    
    // === Store results ===
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Store simdgroup results to threadgroup memory
    threadgroup float* temp_str = sc + 32 * (sgitg & 1) + 16 * (sgitg >> 1) * LLAMA_NR0;
    
    FOR_UNROLL (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * LLAMA_NR0 * (i / 4), LLAMA_NR0, 0, false);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write to global memory with bounds checking
    if (sgitg == 0) {
        for (short c = tiitg; c < nr1; c += 32) {
            for (short r = 0; r < nr0; r++) {
                Y[(r0 + r) * N + (r1 + c)] = half(*(sc + r + c * LLAMA_NR0));
            }
        }
    }
}

// =============================================================================
// GGML-COMPATIBLE block_q4_0 GEMM KERNEL
// =============================================================================
// Directly ported from llama.cpp (https://github.com/ggerganov/llama.cpp)
// Copyright (c) 2023-2024 The ggml authors
// Licensed under MIT License
//
// block_q4_0 format:
// - 32 values per block (QK4_0 = 32)
// - Each block: 2 bytes scale (half) + 16 bytes packed (32 nibbles)  
// - Dequant: W = d * (q - 8), where d is scale, q is [0,15]
//
// This kernel works with metalcore's GGML-compatible quantization format.

#define GGML_QK4_0 32
#define GGML_BLOCK_SIZE 18  // 2 bytes scale + 16 bytes packed

// Block q4_0 structure (matches llama.cpp)
struct block_q4_0 {
    half d;                        // delta (scale)
    uchar qs[GGML_QK4_0 / 2];     // nibbles / quants (16 bytes)
};

// Vectorized dequantization from llama.cpp
// Dequantizes 16 INT4 values into a half4x4 register
template <typename type4x4>
inline void dequantize_q4_0_ggml(device const block_q4_0* xb, short il, thread type4x4& reg) {
    device const ushort* qs = ((device const ushort*)(xb->qs));
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;
    
    float4x4 reg_f;
    
    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }
    
    reg = (type4x4) reg_f;
}

// Tile sizes for GGML GEMM (from llama.cpp)
#define GGML_NR0 64    // M per threadgroup
#define GGML_NR1 32    // N per threadgroup  
#define GGML_NK  32    // K per tile

kernel void matmul_ggml_q4_0(
    device const half* X [[buffer(0)]],           // [M, K]
    device const char* W_blocks [[buffer(1)]],    // [num_blocks_k, N, 18] packed blocks
    device half* Y [[buffer(2)]],                 // [M, N]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    threadgroup char* shmem [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    // Threadgroup memory
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
    threadgroup float* sc = (threadgroup float*)(shmem);
    
    const ushort r0 = tgid.y * GGML_NR0;  // Starting M
    const ushort r1 = tgid.x * GGML_NR1;  // Starting N
    
    const short nr0 = min((int)(M - r0), (int)GGML_NR0);
    const short nr1 = min((int)(N - r1), (int)GGML_NR1);
    
    if (nr0 <= 0 || nr1 <= 0) return;
    
    // Thread assignment
    const short lr0 = min((short)(tiitg / 2), (short)(nr0 - 1));
    const short lr1 = min((short)(tiitg / 4), (short)(nr1 - 1));
    const short il0 = tiitg % 2;
    
    // Number of blocks per column
    const uint num_blocks_k = K / GGML_QK4_0;
    
    // Accumulators
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }
    
    // Process K in tiles
    for (uint k_base = 0; k_base < K; k_base += GGML_NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // === Load X tile into sa (transposed) ===
        {
            const ushort gm = r0 + lr0;
            
            for (short i = 0; i < 16; i++) {
                const short sx = 2 * il0 + i / 8;
                const short sy = lr0 / 8;
                const short lx = lr0 % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                const ushort gk = k_base + 16 * il0 + i;
                
                half val = (gm < M && gk < K) ? X[gm * K + gk] : half(0);
                *(sa + 64 * ib + 8 * ly + lx) = val;
            }
        }
        
        // === Load and dequant W blocks into sb ===
        {
            const short il1 = tiitg % 4;
            const ushort gn = r1 + lr1;
            const short iy = 8 * il1;
            
            // Get block for this K position
            const uint block_idx = k_base / GGML_QK4_0;
            
            if (gn < N && k_base + iy < K) {
                // Access block: [block_idx, gn, 18]
                device const block_q4_0* block = (device const block_q4_0*)(
                    W_blocks + (block_idx * N + gn) * GGML_BLOCK_SIZE
                );
                
                const half d = block->d;
                const short offset = il1 * 4;  // 8 values = 4 bytes
                
                for (short i = 0; i < 8; i++) {
                    const short sx = il1;
                    const short sy = lr1 / 8;
                    const short lx = i;
                    const short ly = lr1 % 8;
                    const short ib = 4 * sx + sy;
                    
                    // Dequant
                    const short byte_idx = offset + i / 2;
                    uchar packed = block->qs[byte_idx];
                    int q = (i % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
                    half val = d * half(q - 8);
                    
                    *(sb + 64 * ib + 8 * ly + lx) = val;
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // === Simdgroup matrix multiply ===
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        
        for (short ik = 0; ik < GGML_NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            
            simdgroup_barrier(mem_flags::mem_none);
            
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            
            simdgroup_barrier(mem_flags::mem_none);
            
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }
    
    // === Store results ===
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    threadgroup float* temp_str = sc + 32 * (sgitg & 1) + 16 * (sgitg >> 1) * GGML_NR0;
    
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * GGML_NR0 * (i / 4), GGML_NR0, 0, false);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write to global
    if (sgitg == 0) {
        for (short c = tiitg; c < nr1; c += 32) {
            for (short r = 0; r < nr0; r++) {
                Y[(r0 + r) * N + (r1 + c)] = half(*(sc + r + c * GGML_NR0));
            }
        }
    }
}

// =============================================================================
// PACKED INT4 FORMAT - Optimized for Cache Locality
// =============================================================================
// Instead of storing scales/zeros separately:
//   W_packed: [K/2, N]
//   scales: [num_groups, N]  
//   zeros: [num_groups, N]
//
// We pack them together in blocks:
//   W_blocks: [num_groups, N, BLOCK_SIZE] where BLOCK_SIZE = 4 + group_size/2
//   Each block: [scale (half, 2B)] [zero (half, 2B)] [group_size/2 packed bytes]
//
// For group_size=128: BLOCK_SIZE = 4 + 64 = 68 bytes per block
// 
// Benefits:
// 1. Single contiguous memory read per dequant group
// 2. Scale/zero cached with weights in L1/L2
// 3. No random access to separate scale/zero arrays
//
// Memory layout for W_blocks:
//   Block at [g, n]: starts at W_blocks + (g * N + n) * BLOCK_SIZE
//   - bytes 0-1: scale (half)
//   - bytes 2-3: zero (half)  
//   - bytes 4-(4+group_size/2-1): packed INT4 weights for K in [g*group_size, (g+1)*group_size)

#define PACKED_INT4_BLOCK_SIZE 68  // 4 bytes header + 64 bytes data (group_size=128)
#define PACKED_GROUP_SIZE 128

kernel void matmul_int4_packed(
    device const half* X [[buffer(0)]],              // [M, K]
    device const uchar* W_blocks [[buffer(1)]],      // [num_groups, N, BLOCK_SIZE]
    device half* Y [[buffer(2)]],                    // [M, N]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint tindex [[thread_index_in_threadgroup]]
) {
    // Simple but efficient kernel - one output per thread
    // Optimized for the packed format to show cache benefit
    
    const uint m = tgid.y * 8 + tid.y;  // 8 M rows per threadgroup
    const uint n = tgid.x * 32 + tid.x; // 32 N cols per threadgroup
    
    if (m >= M || n >= N) return;
    
    const uint num_groups = K / PACKED_GROUP_SIZE;
    
    float acc = 0.0f;
    
    // Process each group
    for (uint g = 0; g < num_groups; g++) {
        // Read block header (scale, zero) - now in same cache line as weights!
        device const uchar* block = W_blocks + (g * N + n) * PACKED_INT4_BLOCK_SIZE;
        half scale = *((device const half*)block);
        half zero = *((device const half*)(block + 2));
        device const uchar* packed = block + 4;
        
        // Process 128 K values from this group (64 packed bytes)
        uint k_base = g * PACKED_GROUP_SIZE;
        
        // Unroll by 8: process 8 bytes (16 weights) at a time
        #pragma unroll
        for (uint b = 0; b < 64; b += 8) {
            // Load 8 bytes and 16 X values
            float4 x0 = float4(
                X[m * K + k_base + b*2 + 0],
                X[m * K + k_base + b*2 + 1],
                X[m * K + k_base + b*2 + 2],
                X[m * K + k_base + b*2 + 3]
            );
            float4 x1 = float4(
                X[m * K + k_base + b*2 + 4],
                X[m * K + k_base + b*2 + 5],
                X[m * K + k_base + b*2 + 6],
                X[m * K + k_base + b*2 + 7]
            );
            float4 x2 = float4(
                X[m * K + k_base + b*2 + 8],
                X[m * K + k_base + b*2 + 9],
                X[m * K + k_base + b*2 + 10],
                X[m * K + k_base + b*2 + 11]
            );
            float4 x3 = float4(
                X[m * K + k_base + b*2 + 12],
                X[m * K + k_base + b*2 + 13],
                X[m * K + k_base + b*2 + 14],
                X[m * K + k_base + b*2 + 15]
            );
            
            // Dequantize 8 bytes -> 16 weights
            uchar p0 = packed[b];
            uchar p1 = packed[b+1];
            uchar p2 = packed[b+2];
            uchar p3 = packed[b+3];
            uchar p4 = packed[b+4];
            uchar p5 = packed[b+5];
            uchar p6 = packed[b+6];
            uchar p7 = packed[b+7];
            
            float fscale = float(scale);
            float fzero = float(zero);
            
            float4 w0 = float4(
                (float(p0 & 0xF) - fzero) * fscale,
                (float(p0 >> 4) - fzero) * fscale,
                (float(p1 & 0xF) - fzero) * fscale,
                (float(p1 >> 4) - fzero) * fscale
            );
            float4 w1 = float4(
                (float(p2 & 0xF) - fzero) * fscale,
                (float(p2 >> 4) - fzero) * fscale,
                (float(p3 & 0xF) - fzero) * fscale,
                (float(p3 >> 4) - fzero) * fscale
            );
            float4 w2 = float4(
                (float(p4 & 0xF) - fzero) * fscale,
                (float(p4 >> 4) - fzero) * fscale,
                (float(p5 & 0xF) - fzero) * fscale,
                (float(p5 >> 4) - fzero) * fscale
            );
            float4 w3 = float4(
                (float(p6 & 0xF) - fzero) * fscale,
                (float(p6 >> 4) - fzero) * fscale,
                (float(p7 & 0xF) - fzero) * fscale,
                (float(p7 >> 4) - fzero) * fscale
            );
            
            acc += dot(x0, w0) + dot(x1, w1) + dot(x2, w2) + dot(x3, w3);
        }
    }
    
    Y[m * N + n] = half(acc);
}
