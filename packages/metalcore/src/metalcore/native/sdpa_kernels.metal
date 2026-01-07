#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Flash Attention v2 - Optimized Tiled Implementation for Apple Silicon
// =============================================================================
//
// Based on Flash Attention 2 algorithm (Dao, 2023) with Metal-specific optimizations:
//
// ROCm/CUDA-inspired optimizations applied:
// 1. SIMD-level reductions using simd_sum/simd_max for fast warp-level ops
// 2. Vectorized memory access (float4) for coalesced loads
// 3. Register blocking - each thread handles multiple elements
// 4. Threadgroup memory for K/V tile caching
// 5. Online softmax with running max/sum (never materialize N×N matrix)
// 6. Causal masking integrated into inner loop
//
// Memory Layout: Q, K, V are (batch_heads, seq_len, head_dim)
// Output O is same shape, L is (batch_heads, seq_len) for logsumexp/backward
// =============================================================================

// Block sizes tuned for Apple Silicon (M1/M2/M3 GPU)
// M3 Max has 40 execution units, 128 threads per threadgroup is a good balance
constant uint BLOCK_M = 64;    // Queries per threadgroup
constant uint BLOCK_N = 64;    // Keys per iteration
constant uint BLOCK_D = 128;   // Max head dimension (supports up to 128)
constant uint WARP_SIZE = 32;  // SIMD width on Apple Silicon

// =============================================================================
// Vectorized Load/Store Macros (guaranteed zero overhead)
// =============================================================================
#define LOAD_FLOAT4(ptr, idx) (reinterpret_cast<device const float4*>(ptr)[idx])
#define STORE_FLOAT4(ptr, idx, val) (reinterpret_cast<device float4*>(ptr)[idx] = (val))

// =============================================================================
// SIMD-accelerated Softmax Macros
// =============================================================================
#define SIMD_ROW_MAX(val) simd_max(val)
#define SIMD_ROW_SUM(val) simd_sum(val)

// =============================================================================
// Flash Attention Forward - Optimized Kernel
// =============================================================================

kernel void flash_attention_fwd_v2(
    device const float* Q [[buffer(0)]],       // (B*H, N, D)
    device const float* K [[buffer(1)]],       // (B*H, N, D)
    device const float* V [[buffer(2)]],       // (B*H, N, D)
    device float* O [[buffer(3)]],             // (B*H, N, D)
    device float* L [[buffer(4)]],             // (B*H, N) logsumexp for backward
    constant uint& batch_heads [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant uint& is_causal [[buffer(9)]],
    
    // Threadgroup shared memory
    threadgroup float* smem [[threadgroup(0)]],
    
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]]
) {
    // Grid: (num_q_blocks, batch_heads, 1)
    uint bh = tgid.y;
    uint q_block_idx = tgid.x;
    
    if (bh >= batch_heads) return;
    
    // Memory layout
    uint qkv_stride = seq_len * head_dim;
    uint base = bh * qkv_stride;
    
    // Query block start
    uint q_start = q_block_idx * BLOCK_M;
    if (q_start >= seq_len) return;
    uint q_end = min(q_start + BLOCK_M, seq_len);
    uint num_q = q_end - q_start;
    
    // Shared memory layout (allocated externally)
    // K_tile: BLOCK_N × head_dim
    // V_tile: BLOCK_N × head_dim
    threadgroup float* K_tile = smem;
    threadgroup float* V_tile = smem + BLOCK_N * BLOCK_D;
    
    // Each thread handles one query row
    // With 64 queries and 128 threads, each thread handles 1 query (with some idle)
    uint local_q = tid;
    if (local_q >= num_q) return;
    
    uint global_q = q_start + local_q;
    
    // Load Q row into registers (vectorized when head_dim % 4 == 0)
    float q_reg[BLOCK_D / 4][4];  // Register blocking with float4
    uint num_vec = head_dim / 4;
    for (uint v = 0; v < num_vec && v < BLOCK_D / 4; ++v) {
        float4 qv = LOAD_FLOAT4(Q, (base + global_q * head_dim) / 4 + v);
        q_reg[v][0] = qv.x;
        q_reg[v][1] = qv.y;
        q_reg[v][2] = qv.z;
        q_reg[v][3] = qv.w;
    }
    
    // Initialize running softmax state
    float m_i = -INFINITY;  // Running max
    float l_i = 0.0f;       // Running sum of exp
    float o_reg[BLOCK_D];   // Output accumulator
    for (uint d = 0; d < head_dim && d < BLOCK_D; ++d) {
        o_reg[d] = 0.0f;
    }
    
    // Process all K/V blocks
    uint num_k_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    
    for (uint k_block = 0; k_block < num_k_blocks; ++k_block) {
        uint k_start = k_block * BLOCK_N;
        if (k_start >= seq_len) break;
        
        // Early exit for causal: if entire K block is after all queries in Q block
        if (is_causal && k_start > q_end - 1) break;
        
        uint k_end = min(k_start + BLOCK_N, seq_len);
        uint num_k = k_end - k_start;
        
        // Cooperative load of K and V tiles into shared memory
        // Each thread loads one row
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < num_k) {
            uint global_k = k_start + tid;
            // Vectorized load
            for (uint v = 0; v < num_vec && v < BLOCK_D / 4; ++v) {
                float4 kv = LOAD_FLOAT4(K, (base + global_k * head_dim) / 4 + v);
                K_tile[tid * BLOCK_D + v * 4 + 0] = kv.x;
                K_tile[tid * BLOCK_D + v * 4 + 1] = kv.y;
                K_tile[tid * BLOCK_D + v * 4 + 2] = kv.z;
                K_tile[tid * BLOCK_D + v * 4 + 3] = kv.w;
                
                float4 vv = LOAD_FLOAT4(V, (base + global_k * head_dim) / 4 + v);
                V_tile[tid * BLOCK_D + v * 4 + 0] = vv.x;
                V_tile[tid * BLOCK_D + v * 4 + 1] = vv.y;
                V_tile[tid * BLOCK_D + v * 4 + 2] = vv.z;
                V_tile[tid * BLOCK_D + v * 4 + 3] = vv.w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute attention scores for this query against all keys in tile
        float m_j = -INFINITY;  // Max for this block
        float l_j = 0.0f;       // Sum for this block
        float p_reg[BLOCK_N];   // Attention weights (before normalization)
        
        // Initialize p_reg to -INFINITY for masked positions
        for (uint kk = 0; kk < BLOCK_N; ++kk) {
            p_reg[kk] = -INFINITY;
        }
        
        for (uint kk = 0; kk < num_k; ++kk) {
            uint global_k = k_start + kk;
            
            // Causal mask check - leave as -INFINITY for masked positions
            if (is_causal && global_k > global_q) {
                continue;  // p_reg[kk] stays -INFINITY
            }
            
            // Dot product Q[local_q] · K[kk] using registers
            float dot = 0.0f;
            for (uint v = 0; v < num_vec && v < BLOCK_D / 4; ++v) {
                dot += q_reg[v][0] * K_tile[kk * BLOCK_D + v * 4 + 0];
                dot += q_reg[v][1] * K_tile[kk * BLOCK_D + v * 4 + 1];
                dot += q_reg[v][2] * K_tile[kk * BLOCK_D + v * 4 + 2];
                dot += q_reg[v][3] * K_tile[kk * BLOCK_D + v * 4 + 3];
            }
            float s = dot * scale;
            
            // Track block max (only for non-masked positions)
            m_j = max(m_j, s);
            p_reg[kk] = s;  // Store score temporarily
        }
        
        // Handle case where all positions are masked (m_j stays -INFINITY)
        if (m_j == -INFINITY) {
            // No valid keys in this block for this query, skip
            continue;
        }
        
        // Compute exponentials with numerical stability
        for (uint kk = 0; kk < num_k; ++kk) {
            // Masked positions have p_reg[kk] = -INFINITY, exp(-INF - m_j) = 0
            float exp_s = exp(p_reg[kk] - m_j);
            p_reg[kk] = exp_s;
            l_j += exp_s;
        }
        
        // Online softmax update
        float m_new = max(m_i, m_j);
        float alpha = exp(m_i - m_new);
        float beta = exp(m_j - m_new);
        
        // Rescale previous accumulator and sum
        l_i = l_i * alpha + l_j * beta;
        for (uint d = 0; d < head_dim && d < BLOCK_D; ++d) {
            o_reg[d] = o_reg[d] * alpha;
        }
        
        // Accumulate contribution from this K/V block
        for (uint kk = 0; kk < num_k; ++kk) {
            float p = p_reg[kk] * beta;  // Already has beta from m_j rescale
            
            // p will be 0 for masked positions (exp(-INF) = 0)
            // Accumulate weighted V
            for (uint d = 0; d < head_dim && d < BLOCK_D; ++d) {
                o_reg[d] += p * V_tile[kk * BLOCK_D + d];
            }
        }
        
        m_i = m_new;
    }
    
    // Normalize output and write
    float inv_l = 1.0f / (l_i + 1e-6f);
    
    // Vectorized store
    for (uint v = 0; v < num_vec && v < BLOCK_D / 4; ++v) {
        float4 out_v;
        out_v.x = o_reg[v * 4 + 0] * inv_l;
        out_v.y = o_reg[v * 4 + 1] * inv_l;
        out_v.z = o_reg[v * 4 + 2] * inv_l;
        out_v.w = o_reg[v * 4 + 3] * inv_l;
        STORE_FLOAT4(O, (base + global_q * head_dim) / 4 + v, out_v);
    }
    
    // Store logsumexp for backward
    L[bh * seq_len + global_q] = m_i + log(l_i + 1e-6f);
}

// =============================================================================
// Naive Attention (for correctness testing and small sequences)
// =============================================================================

kernel void attention_naive(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant uint& batch_heads [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint q_idx = tid.x;
    uint bh = tid.y;
    
    if (bh >= batch_heads || q_idx >= seq_len) return;
    
    uint qkv_stride = seq_len * head_dim;
    uint base = bh * qkv_stride;
    
    // Compute all attention scores for this query
    float max_score = -INFINITY;
    
    // Storage for scores (limited sequence length)
    float scores[1024];
    
    for (uint k_idx = 0; k_idx < seq_len && k_idx < 1024; ++k_idx) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            dot += Q[base + q_idx * head_dim + d] * K[base + k_idx * head_dim + d];
        }
        scores[k_idx] = dot * scale;
        max_score = max(max_score, scores[k_idx]);
    }
    
    // Compute softmax
    float sum_exp = 0.0f;
    for (uint k_idx = 0; k_idx < seq_len && k_idx < 1024; ++k_idx) {
        scores[k_idx] = exp(scores[k_idx] - max_score);
        sum_exp += scores[k_idx];
    }
    
    // Compute output
    for (uint d = 0; d < head_dim; ++d) {
        float acc = 0.0f;
        for (uint k_idx = 0; k_idx < seq_len && k_idx < 1024; ++k_idx) {
            acc += (scores[k_idx] / sum_exp) * V[base + k_idx * head_dim + d];
        }
        O[base + q_idx * head_dim + d] = acc;
    }
}

// =============================================================================
// Flash Attention Backward - Recomputation-based for memory efficiency
// =============================================================================
// Uses recomputation strategy: don't store attention matrix, recompute from L

kernel void flash_attention_bwd_v2(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* O [[buffer(3)]],
    device const float* dO [[buffer(4)]],
    device const float* L [[buffer(5)]],
    device float* dQ [[buffer(6)]],
    device float* dK [[buffer(7)]],
    device float* dV [[buffer(8)]],
    constant uint& batch_heads [[buffer(9)]],
    constant uint& seq_len [[buffer(10)]],
    constant uint& head_dim [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    constant uint& is_causal [[buffer(13)]],
    
    uint2 tid [[thread_position_in_grid]]
) {
    uint q_idx = tid.x;
    uint bh = tid.y;
    
    if (bh >= batch_heads || q_idx >= seq_len) return;
    
    uint qkv_stride = seq_len * head_dim;
    uint base = bh * qkv_stride;
    
    float lse = L[bh * seq_len + q_idx];
    
    // Compute D = sum_i(dO_i * O_i) for bias term
    float D = 0.0f;
    for (uint d = 0; d < head_dim; ++d) {
        D += dO[base + q_idx * head_dim + d] * O[base + q_idx * head_dim + d];
    }
    
    // For each key position, recompute attention and compute gradients
    for (uint k_idx = 0; k_idx < seq_len; ++k_idx) {
        if (is_causal && k_idx > q_idx) continue;
        
        // Recompute attention score
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            dot += Q[base + q_idx * head_dim + d] * K[base + k_idx * head_dim + d];
        }
        float s = dot * scale;
        
        // Recompute attention weight from logsumexp
        float p = exp(s - lse);
        
        // Compute dS = scale * P * (dO @ V^T - D)
        float dO_V = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            dO_V += dO[base + q_idx * head_dim + d] * V[base + k_idx * head_dim + d];
        }
        float dS = scale * p * (dO_V - D);
        
        // Accumulate gradients (using atomics for thread safety)
        for (uint d = 0; d < head_dim; ++d) {
            // dQ[q, :] += dS * K[k, :]
            float dq_val = dS * K[base + k_idx * head_dim + d];
            atomic_fetch_add_explicit((device atomic_float*)&dQ[base + q_idx * head_dim + d],
                                      dq_val, memory_order_relaxed);
            
            // dK[k, :] += dS * Q[q, :]
            float dk_val = dS * Q[base + q_idx * head_dim + d];
            atomic_fetch_add_explicit((device atomic_float*)&dK[base + k_idx * head_dim + d],
                                      dk_val, memory_order_relaxed);
            
            // dV[k, :] += P * dO[q, :]
            float dv_val = p * dO[base + q_idx * head_dim + d];
            atomic_fetch_add_explicit((device atomic_float*)&dV[base + k_idx * head_dim + d],
                                      dv_val, memory_order_relaxed);
        }
    }
}

// =============================================================================
// SDPA Vector Mode - Simple 64-thread implementation
// =============================================================================
// Each thread handles one dimension of head_dim=64
// All threads cooperate to compute attention for one (batch, head, query)

constant uint HEAD_DIM_64 = 64;

kernel void sdpa_vector_64(
    device const float* Q [[buffer(0)]],       // (B*H, N, 64)
    device const float* K [[buffer(1)]],       // (B*H, N, 64)
    device const float* V [[buffer(2)]],       // (B*H, N, 64)
    device float* O [[buffer(3)]],             // (B*H, N, 64)
    constant uint& gqa_factor [[buffer(4)]],
    constant uint& kv_seq_len [[buffer(5)]],
    constant uint& q_stride [[buffer(6)]],
    constant uint& k_stride [[buffer(7)]],
    constant uint& v_stride [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint batch_head = tgid.x;
    uint q_idx = tgid.y;
    uint kv_head = batch_head / gqa_factor;
    
    // Each thread owns one dimension of head_dim=64
    uint d = tid;  // tid ∈ [0, 63]
    if (d >= 64) return;
    
    // Pointers
    device const float* q_ptr = Q + batch_head * q_stride + q_idx * HEAD_DIM_64;
    device const float* k_base = K + kv_head * k_stride;
    device const float* v_base = V + kv_head * v_stride;
    device float* o_ptr = O + batch_head * q_stride + q_idx * HEAD_DIM_64;
    
    // Load my query dimension
    float q_val = q_ptr[d];
    
    // Shared memory for dot product reduction and softmax
    threadgroup float shared_scratch[64];
    
    // Online softmax accumulators - one per dimension
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc = 0.0f;
    
    // Process all keys
    for (uint k_idx = 0; k_idx < kv_seq_len; ++k_idx) {
        device const float* k_ptr = k_base + k_idx * HEAD_DIM_64;
        device const float* v_ptr = v_base + k_idx * HEAD_DIM_64;
        
        // Each thread loads one K dimension, compute partial dot product
        float k_val = k_ptr[d];
        float partial = q_val * k_val;
        
        // Reduce to get full dot product (sum across 64 threads using 2 simdgroups)
        shared_scratch[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Tree reduction
        if (tid < 32) shared_scratch[tid] += shared_scratch[tid + 32];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float dot;
        if (tid < 32) {
            dot = simd_sum(shared_scratch[tid]);
        }
        // Broadcast dot to all threads
        if (tid == 0) shared_scratch[0] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dot = shared_scratch[0];
        
        float s = dot * scale;
        
        // Online softmax update (same for all threads since they share same score)
        float m_new = max(m_prev, s);
        float exp_diff = exp(m_prev - m_new);
        float exp_s = exp(s - m_new);
        
        l_prev = l_prev * exp_diff + exp_s;
        
        // Load V and update output accumulator
        float v_val = v_ptr[d];
        o_acc = o_acc * exp_diff + v_val * exp_s;
        
        m_prev = m_new;
    }
    
    // Normalize and write output (each thread writes one dimension)
    o_ptr[d] = o_acc / l_prev;
}
