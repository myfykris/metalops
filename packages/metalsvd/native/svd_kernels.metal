#include <metal_stdlib>
using namespace metal;

// constant float EPSILON = 1e-6; // Removed

inline float get_epsilon(float) { return 1e-6; }
inline half get_epsilon(half) { return (half)1e-3; } // Relaxed for half
#if __METAL_VERSION__ >= 310
inline bfloat get_epsilon(bfloat) { return (bfloat)1e-3; } // Relaxed for bfloat
#endif

// Helper: SIMD Reduction for float
inline float simd_reduction(float val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return val;
}

// Helper: SIMD Reduction for half
inline half simd_reduction(half val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return val;
}

#if __METAL_VERSION__ >= 310
// Helper: SIMD Shuffle for bfloat (cast to ushort)
inline bfloat simd_shuffle_down(bfloat val, ushort delta) {
    return as_type<bfloat>(simd_shuffle_down(as_type<ushort>(val), delta));
}

// Helper: SIMD Reduction for bfloat
inline bfloat simd_reduction(bfloat val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return val;
}

// Helper: Dot Product for bfloat4
inline bfloat dot(vec<bfloat, 4> a, vec<bfloat, 4> b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// Helper: Dot Product for bfloat4 returning float (Mixed Precision)
inline float dot_float(vec<bfloat, 4> a, vec<bfloat, 4> b) {
    return (float)a.x * (float)b.x + (float)a.y * (float)b.y + (float)a.z * (float)b.z + (float)a.w * (float)b.w;
}
#endif

// Helper: Dot Product overloads for Template Macro
inline float dot_float(vec<float, 4> a, vec<float, 4> b) {
    return dot(a, b); // float * float -> float is standard
}

inline float dot_float(vec<half, 4> a, vec<half, 4> b) {
    return (float)a.x * (float)b.x + (float)a.y * (float)b.y + (float)a.z * (float)b.z + (float)a.w * (float)b.w;
}

// Struct for ICB Uniforms (defined globally)
struct ICBUniforms {
    uint M;
    uint N;
    uint BatchStrideA;
    uint BatchStrideV;
    uint NumPairs;
};

// -----------------------------------------------------------------------------
// Macros for Templating
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// MACRO: Transpose
// -----------------------------------------------------------------------------
#define DEFINE_TRANSPOSE(T, SUFFIX) \
kernel void transpose_kernel_##SUFFIX( \
    device const T* A [[buffer(0)]], \
    device T* Out [[buffer(1)]], \
    constant uint& M [[buffer(2)]], \
    constant uint& N [[buffer(3)]], \
    uint2 gid [[thread_position_in_grid]]) \
{ \
    if (gid.x >= N || gid.y >= M) return; \
    uint idx_in = gid.y * N + gid.x; \
    uint idx_out = gid.x * M + gid.y; \
    Out[idx_out] = A[idx_in]; \
}

// -----------------------------------------------------------------------------
// MACRO: Jacobi Optimized
// -----------------------------------------------------------------------------
#define DEFINE_JACOBI(T, SUFFIX) \
kernel void jacobi_rotate_kernel_optimized_##SUFFIX( \
    device T* A_T [[buffer(0)]], \
    device T* V_T [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& N [[buffer(4)]], \
    constant uint& NumPairs [[buffer(5)]], \
    constant uint& NumSteps [[buffer(6)]], \
    constant uint& ThreadsPerPair [[buffer(7)]], \
    constant uint& BatchStrideA [[buffer(8)]], \
    constant uint& BatchStrideV [[buffer(9)]], \
    threadgroup T* shared_mem [[threadgroup(0)]], \
    uint3 group_pos [[threadgroup_position_in_grid]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 threads_per_group_vec [[threads_per_threadgroup]]) \
{ \
    uint tid = tid_vec.x; \
    uint batch_idx = group_pos.z; \
    uint threads_per_group = threads_per_group_vec.x; \
    uint simd_lane_id = tid % 32; \
    uint simd_group_id = tid / 32; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* A_ptr = A_T + batch_offset_A; \
    device T* V_ptr = V_T + batch_offset_V; \
    int pair_idx = group_pos.x; \
    uint pairs_offset = 0; \
    int p = AllPairs[pairs_offset + pair_idx * 2]; \
    int q = AllPairs[pairs_offset + pair_idx * 2 + 1]; \
    device T* col_i = A_ptr + p * M; \
    device T* col_j = A_ptr + q * M; \
    T part_ii = (T)0.0; \
    T part_jj = (T)0.0; \
    T part_ij = (T)0.0; \
    for (uint k = tid; k < M; k += threads_per_group) { \
        T val_i = col_i[k]; \
        T val_j = col_j[k]; \
        part_ii += val_i * val_i; \
        part_jj += val_j * val_j; \
        part_ij += val_i * val_j; \
    } \
    part_ii = simd_reduction((T)part_ii); \
    part_jj = simd_reduction((T)part_jj); \
    part_ij = simd_reduction((T)part_ij); \
    if (simd_lane_id == 0) { \
        shared_mem[simd_group_id * 3 + 0] = part_ii; \
        shared_mem[simd_group_id * 3 + 1] = part_jj; \
        shared_mem[simd_group_id * 3 + 2] = part_ij; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (tid == 0) { \
        float sum_ii = 0.0f; \
        float sum_jj = 0.0f; \
        float sum_ij = 0.0f; \
        uint num_simd_groups = (threads_per_group + 31) / 32; \
        for (uint s = 0; s < num_simd_groups; ++s) { \
            sum_ii += (float)shared_mem[s * 3 + 0]; \
            sum_jj += (float)shared_mem[s * 3 + 1]; \
            sum_ij += (float)shared_mem[s * 3 + 2]; \
        } \
        float c = 1.0f, s = 0.0f; \
        if (abs(sum_ij) > (float)get_epsilon((T)0)) { \
            float tau = (sum_jj - sum_ii) / (2.0f * sum_ij); \
            float t; \
            if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
            else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
            c = 1.0f / sqrt(1.0f + t * t); \
            s = t * c; \
        } \
        shared_mem[0] = (T)c; \
        shared_mem[1] = (T)s; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    T c = shared_mem[0]; \
    T s = shared_mem[1]; \
    for (uint k = tid; k < M; k += threads_per_group) { \
        T val_i = col_i[k]; \
        T val_j = col_j[k]; \
        col_i[k] = c * val_i - s * val_j; \
        col_j[k] = s * val_i + c * val_j; \
    } \
    device T* v_col_i = V_ptr + p * N; \
    device T* v_col_j = V_ptr + q * N; \
    for (uint k = tid; k < N; k += threads_per_group) { \
        T val_vi = v_col_i[k]; \
        T val_vj = v_col_j[k]; \
        v_col_i[k] = c * val_vi - s * val_vj; \
        v_col_j[k] = s * val_vi + c * val_vj; \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Jacobi ICB
// -----------------------------------------------------------------------------
#define DEFINE_JACOBI_ICB(T, SUFFIX) \
kernel void jacobi_rotate_kernel_icb_##SUFFIX( \
    device T* A_T [[buffer(0)]], \
    device T* V_T [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant ICBUniforms* uniforms [[buffer(3)]], \
    device const uint* StepPtr [[buffer(4)]], \
    threadgroup T* shared_mem [[threadgroup(0)]], \
    uint3 group_pos [[threadgroup_position_in_grid]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 threads_per_group_vec [[threads_per_threadgroup]]) \
{ \
    int pair_idx = group_pos.x; \
    int batch_idx = group_pos.z; \
    uint tid = tid_vec.x; \
    uint threads_per_group = threads_per_group_vec.x; \
    uint simd_lane_id = tid % 32; \
    uint simd_group_id = tid / 32; \
    uint M = uniforms->M; \
    uint N = uniforms->N; \
    uint BatchStrideA = uniforms->BatchStrideA; \
    uint BatchStrideV = uniforms->BatchStrideV; \
    uint NumPairs = uniforms->NumPairs; \
    uint step = *StepPtr; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* A_ptr = A_T + batch_offset_A; \
    device T* V_ptr = V_T + batch_offset_V; \
    uint pairs_offset = step * NumPairs * 2; \
    int p = AllPairs[pairs_offset + pair_idx * 2]; \
    int q = AllPairs[pairs_offset + pair_idx * 2 + 1]; \
    device T* col_i = A_ptr + p * M; \
    device T* col_j = A_ptr + q * M; \
    T part_ii = (T)0.0; \
    T part_jj = (T)0.0; \
    T part_ij = (T)0.0; \
    for (uint k = tid; k < M; k += threads_per_group) { \
        T val_i = col_i[k]; \
        T val_j = col_j[k]; \
        part_ii += val_i * val_i; \
        part_jj += val_j * val_j; \
        part_ij += val_i * val_j; \
    } \
    part_ii = simd_reduction((T)part_ii); \
    part_jj = simd_reduction((T)part_jj); \
    part_ij = simd_reduction((T)part_ij); \
    if (simd_lane_id == 0) { \
        shared_mem[simd_group_id * 3 + 0] = part_ii; \
        shared_mem[simd_group_id * 3 + 1] = part_jj; \
        shared_mem[simd_group_id * 3 + 2] = part_ij; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (tid == 0) { \
        float sum_ii = 0.0f; \
        float sum_jj = 0.0f; \
        float sum_ij = 0.0f; \
        uint num_simd_groups = (threads_per_group + 31) / 32; \
        for (uint s = 0; s < num_simd_groups; ++s) { \
            sum_ii += (float)shared_mem[s * 3 + 0]; \
            sum_jj += (float)shared_mem[s * 3 + 1]; \
            sum_ij += (float)shared_mem[s * 3 + 2]; \
        } \
        float c = 1.0f, s = 0.0f; \
        if (abs(sum_ij) > (float)get_epsilon((T)0)) { \
            float tau = (sum_jj - sum_ii) / (2.0f * sum_ij); \
            float t; \
            if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
            else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
            c = 1.0f / sqrt(1.0f + t * t); \
            s = t * c; \
        } \
        shared_mem[0] = (T)c; \
        shared_mem[1] = (T)s; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    T c = shared_mem[0]; \
    T s = shared_mem[1]; \
    for (uint k = tid; k < M; k += threads_per_group) { \
        T val_i = col_i[k]; \
        T val_j = col_j[k]; \
        col_i[k] = c * val_i - s * val_j; \
        col_j[k] = s * val_i + c * val_j; \
    } \
    device T* v_col_i = V_ptr + p * N; \
    device T* v_col_j = V_ptr + q * N; \
    for (uint k = tid; k < N; k += threads_per_group) { \
        T val_vi = v_col_i[k]; \
        T val_vj = v_col_j[k]; \
        v_col_i[k] = c * val_vi - s * val_vj; \
        v_col_j[k] = s * val_vi + c * val_vj; \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Jacobi ICB Vectorized (float4/half4)
// Requires M % 4 == 0 and N % 4 == 0
// -----------------------------------------------------------------------------
#define DEFINE_JACOBI_ICB_VEC4(T, SUFFIX) \
kernel void jacobi_rotate_kernel_icb_vec4_##SUFFIX( \
    device T* A_T [[buffer(0)]], \
    device T* V_T [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant ICBUniforms* uniforms [[buffer(3)]], \
    device const uint* StepPtr [[buffer(4)]], \
    threadgroup float* shared_mem [[threadgroup(0)]], \
    uint3 group_pos [[threadgroup_position_in_grid]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 threads_per_group_vec [[threads_per_threadgroup]]) \
{ \
    int pair_idx = group_pos.x; \
    int batch_idx = group_pos.z; \
    uint tid = tid_vec.x; \
    uint threads_per_group = threads_per_group_vec.x; \
    uint simd_lane_id = tid % 32; \
    uint simd_group_id = tid / 32; \
    uint M = uniforms->M; \
    uint N = uniforms->N; \
    uint BatchStrideA = uniforms->BatchStrideA; \
    uint BatchStrideV = uniforms->BatchStrideV; \
    uint NumPairs = uniforms->NumPairs; \
    uint step = *StepPtr; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* A_ptr = A_T + batch_offset_A; \
    device T* V_ptr = V_T + batch_offset_V; \
    uint pairs_offset = step * NumPairs * 2; \
    int p = AllPairs[pairs_offset + pair_idx * 2]; \
    int q = AllPairs[pairs_offset + pair_idx * 2 + 1]; \
    device T* col_i = A_ptr + p * M; \
    device T* col_j = A_ptr + q * M; \
    \
    typedef vec<T, 4> vec4; \
    device vec4* col_i_vec = (device vec4*)col_i; \
    device vec4* col_j_vec = (device vec4*)col_j; \
    uint M_vec = M / 4; \
    \
    float part_ii = 0.0f; \
    float part_jj = 0.0f; \
    float part_ij = 0.0f; \
    for (uint k = tid; k < M_vec; k += threads_per_group) { \
        vec4 vi = col_i_vec[k]; \
        vec4 vj = col_j_vec[k]; \
        part_ii += dot_float(vi, vi); \
        part_jj += dot_float(vj, vj); \
        part_ij += dot_float(vi, vj); \
    } \
    part_ii = simd_reduction((float)part_ii); \
    part_jj = simd_reduction((float)part_jj); \
    part_ij = simd_reduction((float)part_ij); \
    if (simd_lane_id == 0) { \
        shared_mem[simd_group_id * 3 + 0] = part_ii; \
        shared_mem[simd_group_id * 3 + 1] = part_jj; \
        shared_mem[simd_group_id * 3 + 2] = part_ij; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (tid == 0) { \
        float sum_ii = 0.0f; \
        float sum_jj = 0.0f; \
        float sum_ij = 0.0f; \
        uint num_simd_groups = (threads_per_group + 31) / 32; \
        for (uint s = 0; s < num_simd_groups; ++s) { \
            sum_ii += shared_mem[s * 3 + 0]; \
            sum_jj += shared_mem[s * 3 + 1]; \
            sum_ij += shared_mem[s * 3 + 2]; \
        } \
        float c = 1.0f, s = 0.0f; \
        if (abs(sum_ij) > (float)get_epsilon((T)0)) { \
            float tau = (sum_jj - sum_ii) / (2.0f * sum_ij); \
            float t; \
            if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
            else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
            c = 1.0f / sqrt(1.0f + t * t); \
            s = t * c; \
        } \
        shared_mem[0] = c; \
        shared_mem[1] = s; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    float c_f = shared_mem[0]; \
    float s_f = shared_mem[1]; \
    T c = (T)c_f; \
    T s = (T)s_f; \
    \
    for (uint k = tid; k < M_vec; k += threads_per_group) { \
        vec4 vi = col_i_vec[k]; \
        vec4 vj = col_j_vec[k]; \
        col_i_vec[k] = c * vi - s * vj; \
        col_j_vec[k] = s * vi + c * vj; \
    } \
    device T* v_col_i = V_ptr + p * N; \
    device T* v_col_j = V_ptr + q * N; \
    device vec4* v_col_i_vec = (device vec4*)v_col_i; \
    device vec4* v_col_j_vec = (device vec4*)v_col_j; \
    uint N_vec = N / 4; \
    for (uint k = tid; k < N_vec; k += threads_per_group) { \
        vec4 val_vi = v_col_i_vec[k]; \
        vec4 val_vj = v_col_j_vec[k]; \
        v_col_i_vec[k] = c * val_vi - s * val_vj; \
        v_col_j_vec[k] = s * val_vi + c * val_vj; \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Fused Block Kernel (Generic)
// -----------------------------------------------------------------------------
#define DEFINE_FUSED_GENERIC(T, SUFFIX) \
kernel void svd_fused_block_kernel_##SUFFIX( \
    device T* A [[buffer(0)]], \
    device T* V [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& N [[buffer(4)]], \
    constant uint& NumPairs [[buffer(5)]], \
    constant uint& NumSteps [[buffer(6)]], \
    constant uint& ThreadsPerPair [[buffer(7)]], \
    constant uint& BatchStrideA [[buffer(8)]], \
    constant uint& BatchStrideV [[buffer(9)]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 group_id [[threadgroup_position_in_grid]]) \
{ \
    uint tid = tid_vec.x; \
    uint batch_idx = group_id.z; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* A_ptr = A + batch_offset_A; \
    device T* V_ptr = V + batch_offset_V; \
    uint pair_idx = tid / ThreadsPerPair; \
    uint lane_id = tid % ThreadsPerPair; \
    (void)lane_id; \
    for (uint sw = 0; sw < 10; ++sw) { \
        for (uint step = 0; step < NumSteps; ++step) { \
            threadgroup_barrier(mem_flags::mem_device); \
            uint pair_offset = step * NumPairs * 2 + pair_idx * 2; \
            int p = AllPairs[pair_offset]; \
            int q = AllPairs[pair_offset+1]; \
            if (pair_idx < NumPairs) { \
                device T* col_p = A_ptr + p * M; \
                device T* col_q = A_ptr + q * M; \
                device T* v_col_p = V_ptr + p * N; \
                device T* v_col_q = V_ptr + q * N; \
                float app = 0.0f, aqq = 0.0f, apq = 0.0f; \
                for(uint k=0; k<M; ++k) { \
                    float vp = (float)col_p[k]; \
                    float vq = (float)col_q[k]; \
                    app += vp * vp; \
                    aqq += vq * vq; \
                    apq += vp * vq; \
                } \
                float c = 1.0f, s = 0.0f; \
                bool rotate = false; \
                if (abs(apq) > max(1e-6f, 1e-6f * sqrt(app * aqq))) { \
                    rotate = true; \
                    float tau = (aqq - app) / (2.0f * apq); \
                    float t; \
                    if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
                    else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
                    c = 1.0f / sqrt(1.0f + t*t); \
                    s = t * c; \
                } \
                if (rotate) { \
                    for(uint k=0; k<M; ++k) { \
                        float vp = (float)col_p[k]; \
                        float vq = (float)col_q[k]; \
                        col_p[k] = (T)(c * vp - s * vq); \
                        col_q[k] = (T)(s * vp + c * vq); \
                    } \
                    for(uint k=0; k<N; ++k) { \
                        float vp = (float)v_col_p[k]; \
                        float vq = (float)v_col_q[k]; \
                        v_col_p[k] = (T)(c * vp - s * vq); \
                        v_col_q[k] = (T)(s * vp + c * vq); \
                    } \
                } \
            } \
        } \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Fused Block Kernel (N=64 Specialized)
// -----------------------------------------------------------------------------
#define DEFINE_FUSED_64(T, SUFFIX) \
kernel void svd_fused_block_kernel_64_##SUFFIX( \
    device T* A [[buffer(0)]], \
    device T* V [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& BatchStrideA [[buffer(4)]], \
    constant uint& BatchStrideV [[buffer(5)]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 group_id [[threadgroup_position_in_grid]]) \
{ \
    const uint NumPairs = 32; \
    const uint NumSteps = 63; \
    const uint N_LOCAL = 64; \
    \
    uint tid = tid_vec.x; \
    uint batch_idx = group_id.z; \
    \
    /* Shared Memory for A and V */ \
    threadgroup float sA[64 * 64]; \
    threadgroup float sV[64 * 64]; \
    \
    /* Coalesced Load A and V -> shared memory */ \
    device T* A_src = A + batch_idx * BatchStrideA; \
    device T* V_src = V + batch_idx * BatchStrideV; \
    \
    /* 1024 threads load 4096 floats each for A and V */ \
    for (uint i = 0; i < 4; ++i) { \
        uint idx = tid * 4 + i; \
        if (idx < 4096) { \
            sA[idx] = (float)A_src[idx]; \
            sV[idx] = (float)V_src[idx]; \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    /* Each pair handled by 1 thread (like Generic) */ \
    /* We use only first 32 threads, rest are idle during compute */ \
    uint pair_idx = tid; /* Only tid 0..31 will work */ \
    \
    for (uint sw = 0; sw < 10; ++sw) { \
        for (uint step = 0; step < NumSteps; ++step) { \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            \
            if (pair_idx < NumPairs) { \
                uint pair_offset = step * NumPairs * 2 + pair_idx * 2; \
                int p = AllPairs[pair_offset]; \
                int q = AllPairs[pair_offset + 1]; \
                \
                threadgroup float* col_p = sA + p * N_LOCAL; \
                threadgroup float* col_q = sA + q * N_LOCAL; \
                threadgroup float* v_col_p = sV + p * N_LOCAL; \
                threadgroup float* v_col_q = sV + q * N_LOCAL; \
                \
                /* Sequential dot product - exactly like Generic */ \
                float app = 0.0f, aqq = 0.0f, apq = 0.0f; \
                for(uint k=0; k<N_LOCAL; ++k) { \
                    float vp = col_p[k]; \
                    float vq = col_q[k]; \
                    app += vp * vp; \
                    aqq += vq * vq; \
                    apq += vp * vq; \
                } \
                \
                float c = 1.0f, s = 0.0f; \
                bool rotate = false; \
                if (abs(apq) > max(1e-6f, 1e-6f * sqrt(app * aqq))) { \
                    rotate = true; \
                    float tau = (aqq - app) / (2.0f * apq); \
                    float t; \
                    if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
                    else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
                    c = 1.0f / sqrt(1.0f + t*t); \
                    s = t * c; \
                } \
                \
                if (rotate) { \
                    /* Sequential update - exactly like Generic */ \
                    for(uint k=0; k<N_LOCAL; ++k) { \
                        float vp = col_p[k]; \
                        float vq = col_q[k]; \
                        col_p[k] = c * vp - s * vq; \
                        col_q[k] = s * vp + c * vq; \
                    } \
                    for(uint k=0; k<N_LOCAL; ++k) { \
                        float vp = v_col_p[k]; \
                        float vq = v_col_q[k]; \
                        v_col_p[k] = c * vp - s * vq; \
                        v_col_q[k] = s * vp + c * vq; \
                    } \
                } \
            } \
        } \
    } \
    \
    /* Store sA and sV back to global memory */ \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint i = 0; i < 4; ++i) { \
        uint idx = tid * 4 + i; \
        if (idx < 4096) { \
            A_src[idx] = (T)sA[idx]; \
            V_src[idx] = (T)sV[idx]; \
        } \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Fused Block Kernel (N=128 Specialized)
// -----------------------------------------------------------------------------
#define DEFINE_FUSED_128(T, SUFFIX) \
kernel void svd_fused_block_kernel_128_##SUFFIX( \
    device T* A [[buffer(0)]], \
    device T* V [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& BatchStrideA [[buffer(4)]], \
    constant uint& BatchStrideV [[buffer(5)]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 group_id [[threadgroup_position_in_grid]]) \
{ \
    const uint NumPairs = 64; \
    const uint NumSteps = 127; \
    const uint ThreadsPerPair = 16; \
    const uint N = 128; \
    uint tid = tid_vec.x; \
    uint batch_idx = group_id.z; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* A_ptr = A + batch_offset_A; \
    device T* V_ptr = V + batch_offset_V; \
    uint pair_idx = tid / ThreadsPerPair; \
    uint lane_id = tid % ThreadsPerPair; \
    threadgroup atomic_uint sweep_rotations; \
    for (uint sw = 0; sw < 10; ++sw) { \
        if (tid == 0) atomic_store_explicit(&sweep_rotations, 0, memory_order_relaxed); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint step = 0; step < NumSteps; ++step) { \
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device); \
            uint pair_offset = step * NumPairs * 2 + pair_idx * 2; \
            int p = AllPairs[pair_offset]; \
            int q = AllPairs[pair_offset+1]; \
            if (pair_idx < NumPairs) { \
                device T* col_p = A_ptr + p * M; \
                device T* col_q = A_ptr + q * M; \
                device T* v_col_p = V_ptr + p * N; \
                device T* v_col_q = V_ptr + q * N; \
                float app = 0.0f, aqq = 0.0f, apq = 0.0f; \
                for(uint k=lane_id; k<M; k+=ThreadsPerPair) { \
                    float vp = (float)col_p[k]; \
                    float vq = (float)col_q[k]; \
                    app += vp * vp; \
                    aqq += vq * vq; \
                    apq += vp * vq; \
                } \
                app += simd_shuffle_down(app, 8); \
                app += simd_shuffle_down(app, 4); \
                app += simd_shuffle_down(app, 2); \
                app += simd_shuffle_down(app, 1); \
                aqq += simd_shuffle_down(aqq, 8); \
                aqq += simd_shuffle_down(aqq, 4); \
                aqq += simd_shuffle_down(aqq, 2); \
                aqq += simd_shuffle_down(aqq, 1); \
                apq += simd_shuffle_down(apq, 8); \
                apq += simd_shuffle_down(apq, 4); \
                apq += simd_shuffle_down(apq, 2); \
                apq += simd_shuffle_down(apq, 1); \
                float c = 1.0f, s = 0.0f; \
                bool rotate = false; \
                if (lane_id == 0) { \
                    if (abs(apq) > max(1e-6f, 1e-6f * sqrt(app * aqq))) { \
                        rotate = true; \
                        float tau = (aqq - app) / (2.0f * apq); \
                        float t; \
                        if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
                        else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
                        c = 1.0f / sqrt(1.0f + t*t); \
                        s = t * c; \
                        atomic_fetch_add_explicit(&sweep_rotations, 1, memory_order_relaxed); \
                    } \
                } \
                c = simd_shuffle(c, (tid % 32) & ~15); \
                s = simd_shuffle(s, (tid % 32) & ~15); \
                rotate = (bool)simd_shuffle((ushort)rotate, (tid % 32) & ~15); \
                if (rotate) { \
                    for(uint k=lane_id; k<M; k+=ThreadsPerPair) { \
                        float vp = (float)col_p[k]; \
                        float vq = (float)col_q[k]; \
                        col_p[k] = (T)(c * vp - s * vq); \
                        col_q[k] = (T)(s * vp + c * vq); \
                    } \
                    for(uint k=lane_id; k<N; k+=ThreadsPerPair) { \
                        float vp = (float)v_col_p[k]; \
                        float vq = (float)v_col_q[k]; \
                        v_col_p[k] = (T)(c * vp - s * vq); \
                        v_col_q[k] = (T)(s * vp + c * vq); \
                    } \
                } \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (atomic_load_explicit(&sweep_rotations, memory_order_relaxed) == 0) break; \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Fused Block Kernel (N=256 Specialized)
// -----------------------------------------------------------------------------
#define DEFINE_FUSED_256(T, SUFFIX) \
kernel void svd_fused_block_kernel_256_##SUFFIX( \
    device T* A [[buffer(0)]], \
    device T* V [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& BatchStrideA [[buffer(4)]], \
    constant uint& BatchStrideV [[buffer(5)]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 group_id [[threadgroup_position_in_grid]]) \
{ \
    const uint NumPairs = 128; \
    const uint NumSteps = 255; \
    const uint ThreadsPerPair = 8; \
    const uint N = 256; \
    uint tid = tid_vec.x; \
    uint batch_idx = group_id.z; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* A_ptr = A + batch_offset_A; \
    device T* V_ptr = V + batch_offset_V; \
    uint pair_idx = tid / ThreadsPerPair; \
    uint lane_id = tid % ThreadsPerPair; \
    threadgroup atomic_uint sweep_rotations; \
    for (uint sw = 0; sw < 10; ++sw) { \
        if (tid == 0) atomic_store_explicit(&sweep_rotations, 0, memory_order_relaxed); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint step = 0; step < NumSteps; ++step) { \
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device); \
            uint pair_offset = step * NumPairs * 2 + pair_idx * 2; \
            int p = AllPairs[pair_offset]; \
            int q = AllPairs[pair_offset+1]; \
            if (pair_idx < NumPairs) { \
                device T* col_p = A_ptr + p * M; \
                device T* col_q = A_ptr + q * M; \
                device T* v_col_p = V_ptr + p * N; \
                device T* v_col_q = V_ptr + q * N; \
                float app = 0.0f, aqq = 0.0f, apq = 0.0f; \
                for(uint k=lane_id; k<M; k+=ThreadsPerPair) { \
                    float vp = (float)col_p[k]; \
                    float vq = (float)col_q[k]; \
                    app += vp * vp; \
                    aqq += vq * vq; \
                    apq += vp * vq; \
                } \
                app += simd_shuffle_down(app, 4); \
                app += simd_shuffle_down(app, 2); \
                app += simd_shuffle_down(app, 1); \
                aqq += simd_shuffle_down(aqq, 4); \
                aqq += simd_shuffle_down(aqq, 2); \
                aqq += simd_shuffle_down(aqq, 1); \
                apq += simd_shuffle_down(apq, 4); \
                apq += simd_shuffle_down(apq, 2); \
                apq += simd_shuffle_down(apq, 1); \
                float c = 1.0f, s = 0.0f; \
                bool rotate = false; \
                if (lane_id == 0) { \
                    if (abs(apq) > max(1e-6f, 1e-6f * sqrt(app * aqq))) { \
                        rotate = true; \
                        float tau = (aqq - app) / (2.0f * apq); \
                        float t; \
                        if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
                        else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
                        c = 1.0f / sqrt(1.0f + t*t); \
                        s = t * c; \
                        atomic_fetch_add_explicit(&sweep_rotations, 1, memory_order_relaxed); \
                    } \
                } \
                c = simd_shuffle(c, (tid % 32) & ~7); \
                s = simd_shuffle(s, (tid % 32) & ~7); \
                rotate = (bool)simd_shuffle((ushort)rotate, (tid % 32) & ~7); \
                if (rotate) { \
                    for(uint k=lane_id; k<M; k+=ThreadsPerPair) { \
                        float vp = (float)col_p[k]; \
                        float vq = (float)col_q[k]; \
                        col_p[k] = (T)(c * vp - s * vq); \
                        col_q[k] = (T)(s * vp + c * vq); \
                    } \
                    for(uint k=lane_id; k<N; k+=ThreadsPerPair) { \
                        float vp = (float)v_col_p[k]; \
                        float vq = (float)v_col_q[k]; \
                        v_col_p[k] = (T)(c * vp - s * vq); \
                        v_col_q[k] = (T)(s * vp + c * vq); \
                    } \
                } \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (atomic_load_explicit(&sweep_rotations, memory_order_relaxed) == 0) break; \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Fused Block Kernel (N=512 Specialized)
// -----------------------------------------------------------------------------
#define DEFINE_FUSED_512(T, SUFFIX) \
kernel void svd_fused_block_kernel_512_##SUFFIX( \
    device T* A [[buffer(0)]], \
    device T* V [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& BatchStrideA [[buffer(4)]], \
    constant uint& BatchStrideV [[buffer(5)]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 group_id [[threadgroup_position_in_grid]]) \
{ \
    const uint NumPairs = 256; \
    const uint NumSteps = 511; \
    const uint ThreadsPerPair = 4; \
    const uint N = 512; \
    uint tid = tid_vec.x; \
    uint batch_idx = group_id.z; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* A_ptr = A + batch_offset_A; \
    device T* V_ptr = V + batch_offset_V; \
    uint pair_idx = tid / ThreadsPerPair; \
    uint lane_id = tid % ThreadsPerPair; \
    threadgroup atomic_uint sweep_rotations; \
    for (uint sw = 0; sw < 10; ++sw) { \
        if (tid == 0) atomic_store_explicit(&sweep_rotations, 0, memory_order_relaxed); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint step = 0; step < NumSteps; ++step) { \
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device); \
            uint pair_offset = step * NumPairs * 2 + pair_idx * 2; \
            int p = AllPairs[pair_offset]; \
            int q = AllPairs[pair_offset+1]; \
            if (pair_idx < NumPairs) { \
                device T* col_p = A_ptr + p * M; \
                device T* col_q = A_ptr + q * M; \
                device T* v_col_p = V_ptr + p * N; \
                device T* v_col_q = V_ptr + q * N; \
                /* Vectorized dot product using float4 */ \
                typedef vec<T, 4> vec4; \
                device vec4* col_p_vec = (device vec4*)col_p; \
                device vec4* col_q_vec = (device vec4*)col_q; \
                uint M4 = M / 4; \
                float app = 0.0f, aqq = 0.0f, apq = 0.0f; \
                for(uint k=lane_id; k<M4; k+=ThreadsPerPair) { \
                    vec4 vp4 = col_p_vec[k]; \
                    vec4 vq4 = col_q_vec[k]; \
                    float4 vp = float4(vp4); \
                    float4 vq = float4(vq4); \
                    app += dot(vp, vp); \
                    aqq += dot(vq, vq); \
                    apq += dot(vp, vq); \
                } \
                app += simd_shuffle_down(app, 2); \
                app += simd_shuffle_down(app, 1); \
                aqq += simd_shuffle_down(aqq, 2); \
                aqq += simd_shuffle_down(aqq, 1); \
                apq += simd_shuffle_down(apq, 2); \
                apq += simd_shuffle_down(apq, 1); \
                float c = 1.0f, s = 0.0f; \
                bool rotate = false; \
                if (lane_id == 0) { \
                    if (abs(apq) > max(1e-6f, 1e-6f * sqrt(app * aqq))) { \
                        rotate = true; \
                        float tau = (aqq - app) / (2.0f * apq); \
                        float t; \
                        if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
                        else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
                        c = 1.0f / sqrt(1.0f + t*t); \
                        s = t * c; \
                        atomic_fetch_add_explicit(&sweep_rotations, 1, memory_order_relaxed); \
                    } \
                } \
                c = simd_shuffle(c, (tid % 32) & ~3); \
                s = simd_shuffle(s, (tid % 32) & ~3); \
                rotate = (bool)simd_shuffle((ushort)rotate, (tid % 32) & ~3); \
                if (rotate) { \
                    /* Vectorized column update */ \
                    for(uint k=lane_id; k<M4; k+=ThreadsPerPair) { \
                        vec4 vp4 = col_p_vec[k]; \
                        vec4 vq4 = col_q_vec[k]; \
                        float4 vp = float4(vp4); \
                        float4 vq = float4(vq4); \
                        col_p_vec[k] = vec4(c * vp - s * vq); \
                        col_q_vec[k] = vec4(s * vp + c * vq); \
                    } \
                    /* Vectorized V update */ \
                    device vec4* v_col_p_vec = (device vec4*)v_col_p; \
                    device vec4* v_col_q_vec = (device vec4*)v_col_q; \
                    uint N4 = N / 4; \
                    for(uint k=lane_id; k<N4; k+=ThreadsPerPair) { \
                        vec4 vp4 = v_col_p_vec[k]; \
                        vec4 vq4 = v_col_q_vec[k]; \
                        float4 vp = float4(vp4); \
                        float4 vq = float4(vq4); \
                        v_col_p_vec[k] = vec4(c * vp - s * vq); \
                        v_col_q_vec[k] = vec4(s * vp + c * vq); \
                    } \
                } \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (atomic_load_explicit(&sweep_rotations, memory_order_relaxed) == 0) break; \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Fused Block Kernel (N=1024 Specialized)
// -----------------------------------------------------------------------------
#define DEFINE_FUSED_1024(T, SUFFIX) \
kernel void svd_fused_block_kernel_1024_##SUFFIX( \
    device T* A [[buffer(0)]], \
    device T* V [[buffer(1)]], \
    device const int* AllPairs [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& BatchStrideA [[buffer(4)]], \
    constant uint& BatchStrideV [[buffer(5)]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 group_id [[threadgroup_position_in_grid]]) \
{ \
    const uint NumPairs = 512; \
    const uint NumSteps = 1023; \
    const uint ThreadsPerPair = 2; \
    const uint N = 1024; \
    uint tid = tid_vec.x; \
    uint batch_idx = group_id.z; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* A_ptr = A + batch_offset_A; \
    device T* V_ptr = V + batch_offset_V; \
    uint pair_idx = tid / ThreadsPerPair; \
    uint lane_id = tid % ThreadsPerPair; \
    threadgroup atomic_uint sweep_rotations; \
    for (uint sw = 0; sw < 10; ++sw) { \
        if (tid == 0) atomic_store_explicit(&sweep_rotations, 0, memory_order_relaxed); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint step = 0; step < NumSteps; ++step) { \
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device); \
            uint pair_offset = step * NumPairs * 2 + pair_idx * 2; \
            int p = AllPairs[pair_offset]; \
            int q = AllPairs[pair_offset+1]; \
            if (pair_idx < NumPairs) { \
                device T* col_p = A_ptr + p * M; \
                device T* col_q = A_ptr + q * M; \
                device T* v_col_p = V_ptr + p * N; \
                device T* v_col_q = V_ptr + q * N; \
                /* Vectorized dot product using float4 */ \
                typedef vec<T, 4> vec4; \
                device vec4* col_p_vec = (device vec4*)col_p; \
                device vec4* col_q_vec = (device vec4*)col_q; \
                uint M4 = M / 4; \
                float app = 0.0f, aqq = 0.0f, apq = 0.0f; \
                for(uint k=lane_id; k<M4; k+=ThreadsPerPair) { \
                    vec4 vp4 = col_p_vec[k]; \
                    vec4 vq4 = col_q_vec[k]; \
                    float4 vp = float4(vp4); \
                    float4 vq = float4(vq4); \
                    app += dot(vp, vp); \
                    aqq += dot(vq, vq); \
                    apq += dot(vp, vq); \
                } \
                app += simd_shuffle_down(app, 1); \
                aqq += simd_shuffle_down(aqq, 1); \
                apq += simd_shuffle_down(apq, 1); \
                float c = 1.0f, s = 0.0f; \
                bool rotate = false; \
                if (lane_id == 0) { \
                    if (abs(apq) > max(1e-6f, 1e-6f * sqrt(app * aqq))) { \
                        rotate = true; \
                        float tau = (aqq - app) / (2.0f * apq); \
                        float t; \
                        if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
                        else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
                        c = 1.0f / sqrt(1.0f + t*t); \
                        s = t * c; \
                        atomic_fetch_add_explicit(&sweep_rotations, 1, memory_order_relaxed); \
                    } \
                } \
                c = simd_shuffle(c, (tid % 32) & ~1); \
                s = simd_shuffle(s, (tid % 32) & ~1); \
                rotate = (bool)simd_shuffle((ushort)rotate, (tid % 32) & ~1); \
                if (rotate) { \
                    /* Vectorized column update */ \
                    for(uint k=lane_id; k<M4; k+=ThreadsPerPair) { \
                        vec4 vp4 = col_p_vec[k]; \
                        vec4 vq4 = col_q_vec[k]; \
                        float4 vp = float4(vp4); \
                        float4 vq = float4(vq4); \
                        col_p_vec[k] = vec4(c * vp - s * vq); \
                        col_q_vec[k] = vec4(s * vp + c * vq); \
                    } \
                    /* Vectorized V update */ \
                    device vec4* v_col_p_vec = (device vec4*)v_col_p; \
                    device vec4* v_col_q_vec = (device vec4*)v_col_q; \
                    uint N4 = N / 4; \
                    for(uint k=lane_id; k<N4; k+=ThreadsPerPair) { \
                        vec4 vp4 = v_col_p_vec[k]; \
                        vec4 vq4 = v_col_q_vec[k]; \
                        float4 vp = float4(vp4); \
                        float4 vq = float4(vq4); \
                        v_col_p_vec[k] = vec4(c * vp - s * vq); \
                        v_col_q_vec[k] = vec4(s * vp + c * vq); \
                    } \
                } \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (atomic_load_explicit(&sweep_rotations, memory_order_relaxed) == 0) break; \
    } \
}

// -----------------------------------------------------------------------------
// MACRO: Norm and Normalize
// -----------------------------------------------------------------------------
#define DEFINE_NORMALIZATION(T, SUFFIX) \
kernel void column_norm_kernel_##SUFFIX( \
    device const T* A_T [[buffer(0)]], \
    device T* S [[buffer(1)]], \
    constant uint& M [[buffer(2)]], \
    constant uint& N [[buffer(3)]], \
    constant uint& BatchStrideA [[buffer(4)]], \
    constant uint& BatchStrideS [[buffer(5)]], \
    uint2 gid [[thread_position_in_grid]]) \
{ \
    uint i = gid.x; \
    uint batch_idx = gid.y; \
    if (i >= N) return; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_S = batch_idx * BatchStrideS; \
    device const T* col_i = A_T + batch_offset_A + i * M; \
    float sum_sq = 0.0f; \
    for (uint k = 0; k < M; ++k) { \
        float val = (float)col_i[k]; \
        sum_sq += val * val; \
    } \
    S[batch_offset_S + i] = (T)sqrt(sum_sq); \
} \
kernel void normalize_kernel_##SUFFIX( \
    device const T* A_T [[buffer(0)]], \
    device const T* S [[buffer(1)]], \
    device T* U_T [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& N [[buffer(4)]], \
    constant uint& BatchStrideA [[buffer(5)]], \
    constant uint& BatchStrideS [[buffer(6)]], \
    uint2 gid [[thread_position_in_grid]]) \
{ \
    uint i = gid.x; \
    if (i >= N) return; \
    uint batch_idx = gid.y; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_S = batch_idx * BatchStrideS; \
    device const T* col_i = A_T + batch_offset_A + i * M; \
    device T* u_col_i = U_T + batch_offset_A + i * M; \
    float sigma = (float)S[batch_offset_S + i]; \
    float inv_sigma = (sigma > 1.0e-8f) ? (1.0f / sigma) : 0.0f; \
    for (uint k = 0; k < M; ++k) { \
        u_col_i[k] = (T)((float)col_i[k] * inv_sigma); \
    } \
}

// -----------------------------------------------------------------------------
// Instantiations
// -----------------------------------------------------------------------------
// Transpose
DEFINE_TRANSPOSE(float, float)
DEFINE_TRANSPOSE(half, half)
#if __METAL_VERSION__ >= 310
DEFINE_TRANSPOSE(bfloat, bfloat)
#endif

// Jacobi
DEFINE_JACOBI(float, float)
DEFINE_JACOBI(half, half)
#if __METAL_VERSION__ >= 310
DEFINE_JACOBI(bfloat, bfloat)
#endif

// Jacobi ICB
DEFINE_JACOBI_ICB(float, float)
DEFINE_JACOBI_ICB_VEC4(float, float)

DEFINE_JACOBI_ICB(half, half)
DEFINE_JACOBI_ICB_VEC4(half, half)

#if __METAL_VERSION__ >= 310
DEFINE_JACOBI_ICB(bfloat, bfloat)
DEFINE_JACOBI_ICB_VEC4(bfloat, bfloat)
#endif

// Fused
DEFINE_FUSED_GENERIC(float, float)
DEFINE_FUSED_64(float, float)
DEFINE_FUSED_128(float, float)
DEFINE_FUSED_256(float, float)
DEFINE_FUSED_512(float, float)
DEFINE_FUSED_1024(float, float)

DEFINE_FUSED_GENERIC(half, half)
DEFINE_FUSED_64(half, half)
DEFINE_FUSED_128(half, half)
DEFINE_FUSED_256(half, half)
DEFINE_FUSED_512(half, half)
DEFINE_FUSED_1024(half, half)

#if __METAL_VERSION__ >= 310
DEFINE_FUSED_GENERIC(bfloat, bfloat)
DEFINE_FUSED_64(bfloat, bfloat)
DEFINE_FUSED_128(bfloat, bfloat)
DEFINE_FUSED_256(bfloat, bfloat)
DEFINE_FUSED_512(bfloat, bfloat)
DEFINE_FUSED_1024(bfloat, bfloat)
#endif

// Norm
DEFINE_NORMALIZATION(float, float)
DEFINE_NORMALIZATION(half, half)
#if __METAL_VERSION__ >= 310
DEFINE_NORMALIZATION(bfloat, bfloat)
#endif

// -----------------------------------------------------------------------------
// NEW: Clean Re-implementation of Vectorized Jacobi Kernel
// -----------------------------------------------------------------------------
// Simplified logic to avoid macro/template complexity and potential bugs.
// Assumes M % 4 == 0 and N % 4 == 0.
// Uses Float32 accumulation for stability.

template <typename T>
kernel void jacobi_rotate_kernel_vec4_clean(
    device T* A_T [[buffer(0)]],
    device T* V_T [[buffer(1)]],
    device const int* AllPairs [[buffer(2)]],
    constant ICBUniforms* uniforms [[buffer(3)]],
    device const uint* StepPtr [[buffer(4)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint3 group_pos [[threadgroup_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]],
    uint3 threads_per_group_vec [[threads_per_threadgroup]])
{
    // 1. Setup Indices
    uint tid = tid_vec.x;
    uint threads_per_group = threads_per_group_vec.x;
    
    // Grid/Pair Info
    int pair_idx = group_pos.x;
    int batch_idx = group_pos.z;
    
    // Uniforms
    uint M = uniforms->M;
    uint N = uniforms->N;
    uint BatchStrideA = uniforms->BatchStrideA;
    uint BatchStrideV = uniforms->BatchStrideV;
    uint NumPairs = uniforms->NumPairs;
    uint step = *StepPtr;
    
    // Pointers
    uint batch_offset_A = batch_idx * BatchStrideA;
    uint batch_offset_V = batch_idx * BatchStrideV;
    device T* A = A_T + batch_offset_A;
    device T* V = V_T + batch_offset_V;
    
    // Pair Indices (p < q)
    uint pairs_offset = step * NumPairs * 2;
    int p = AllPairs[pairs_offset + pair_idx * 2];
    int q = AllPairs[pairs_offset + pair_idx * 2 + 1];
    
    // Columns (M x 1)
    device T* col_p = A + p * M;
    device T* col_q = A + q * M;
    
    // Vectorized Pointers (M/4 x 1)
    typedef vec<T, 4> vec4;
    device vec4* col_p_vec = (device vec4*)col_p;
    device vec4* col_q_vec = (device vec4*)col_q;
    uint M_vec = M / 4;
    
    // 2. Accumulate Dot Products (G_{pp}, G_{qq}, G_{pq})
    float G_pp = 0.0f;
    float G_qq = 0.0f;
    float G_pq = 0.0f;
    
    for (uint k = tid; k < M_vec; k += threads_per_group) {
        vec4 val_p = col_p_vec[k];
        vec4 val_q = col_q_vec[k];
        
        G_pp += dot_float(val_p, val_p);
        G_qq += dot_float(val_q, val_q);
        G_pq += dot_float(val_p, val_q);
    }
    
    // 3. Reduction (Block-wide)
    // SIMD Reduction first
    G_pp = simd_reduction(G_pp);
    G_qq = simd_reduction(G_qq);
    G_pq = simd_reduction(G_pq);
    
    // Threadgroup Reduction (via Shared Mem)
    uint simd_lane_id = tid % 32;
    uint simd_group_id = tid / 32;
    
    if (simd_lane_id == 0) {
        shared_mem[simd_group_id * 3 + 0] = G_pp;
        shared_mem[simd_group_id * 3 + 1] = G_qq;
        shared_mem[simd_group_id * 3 + 2] = G_pq;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // First thread sums SIMD group results
    if (tid == 0) {
        float sum_pp = 0.0f;
        float sum_qq = 0.0f;
        float sum_pq = 0.0f;
        
        uint num_simd_groups = (threads_per_group + 31) / 32;
        for (uint s = 0; s < num_simd_groups; ++s) {
            sum_pp += shared_mem[s * 3 + 0];
            sum_qq += shared_mem[s * 3 + 1];
            sum_pq += shared_mem[s * 3 + 2];
        }
        
        // 4. Compute Rotation (c, s)
        float c = 1.0f;
        float s = 0.0f;
        
        // Check trace threshold (epsilon) to avoid noise
        // Using existing helper or hardcoded small epsilon
        if (abs(sum_pq) > 1e-9f) { // Strict epsilon
            float tau = (sum_qq - sum_pp) / (2.0f * sum_pq);
            float t;
            if (tau >= 0.0f) {
                t = 1.0f / (tau + sqrt(1.0f + tau*tau));
            } else {
                t = -1.0f / (-tau + sqrt(1.0f + tau*tau));
            }
            c = 1.0f / sqrt(1.0f + t*t);
            s = t * c;
        }
        
        shared_mem[0] = c;
        shared_mem[1] = s;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float c_f = shared_mem[0];
    float s_f = shared_mem[1];
    T c = (T)c_f;
    T s = (T)s_f;
    
    // 5. Apply Rotation to A (M x 2)
    for (uint k = tid; k < M_vec; k += threads_per_group) {
        vec4 val_p = col_p_vec[k];
        vec4 val_q = col_q_vec[k];
        
        col_p_vec[k] = c * val_p - s * val_q;
        col_q_vec[k] = s * val_p + c * val_q;
    }
    
    // 6. Apply Rotation to V (N x 2)
    device T* v_col_p = V + p * N;
    device T* v_col_q = V + q * N;
    
    device vec4* v_col_p_vec = (device vec4*)v_col_p;
    device vec4* v_col_q_vec = (device vec4*)v_col_q;
    uint N_vec = N / 4;
    
    for (uint k = tid; k < N_vec; k += threads_per_group) {
        vec4 val_p = v_col_p_vec[k];
        vec4 val_q = v_col_q_vec[k];
        
        v_col_p_vec[k] = c * val_p - s * val_q;
        v_col_q_vec[k] = s * val_p + c * val_q;
    }
}

// Explicit Instantiations for the Clean Kernel
template [[host_name("jacobi_rotate_kernel_vec4_clean_float")]] kernel void jacobi_rotate_kernel_vec4_clean<float>(
    device float*, device float*, device const int*, constant ICBUniforms*, device const uint*, threadgroup float*, uint3, uint3, uint3);

#if __METAL_VERSION__ >= 310
template [[host_name("jacobi_rotate_kernel_vec4_clean_bfloat")]] kernel void jacobi_rotate_kernel_vec4_clean<bfloat>(
    device bfloat*, device bfloat*, device const int*, constant ICBUniforms*, device const uint*, threadgroup float*, uint3, uint3, uint3);
#endif

template [[host_name("jacobi_rotate_kernel_vec4_clean_half")]] kernel void jacobi_rotate_kernel_vec4_clean<half>(
    device half*, device half*, device const int*, constant ICBUniforms*, device const uint*, threadgroup float*, uint3, uint3, uint3);
