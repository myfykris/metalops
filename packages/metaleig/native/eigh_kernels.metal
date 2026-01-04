#include <metal_stdlib>
using namespace metal;

// Helper: SIMD Reduction for float
inline float simd_reduction(float val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return val;
}

// Struct for ICB Uniforms
struct ICBUniforms {
    uint M;
    uint N;
    uint BatchStrideA;
    uint BatchStrideV;
    uint NumPairs;
    uint _pad0;
    uint _pad1;
    uint _pad2; // Align to 32 bytes (8 uints)
};

// ... (helpers)


inline float get_epsilon(float) { return 1e-6; }
inline half get_epsilon(half) { return (half)1e-3; } 
#if __METAL_VERSION__ >= 310
inline bfloat get_epsilon(bfloat) { return (bfloat)1e-3; } 
#endif

// Helper: SIMD Shuffle for bfloat (cast to ushort)
#if __METAL_VERSION__ >= 310
inline bfloat simd_shuffle_down(bfloat val, ushort delta) {
    return as_type<bfloat>(simd_shuffle_down(as_type<ushort>(val), delta));
}
#endif

#if __METAL_VERSION__ >= 310
inline bfloat simd_shuffle(bfloat val, ushort lane) {
    return as_type<bfloat>(simd_shuffle(as_type<ushort>(val), lane));
}
#endif

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
// Helper: SIMD Reduction for bfloat
inline bfloat simd_reduction(bfloat val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return val;
}
#endif

// -----------------------------------------------------------------------------
// MACRO: Jacobi Optimized (One-Sided)
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
    part_ii = simd_reduction(part_ii); \
    part_jj = simd_reduction(part_jj); \
    part_ij = simd_reduction(part_ij); \
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
    /* DEBUG: Verify kernel execution (Removed) */ \
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
    part_ii = simd_reduction(part_ii); \
    part_jj = simd_reduction(part_jj); \
    part_ij = simd_reduction(part_ij); \
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

DEFINE_JACOBI_ICB(float, float)
DEFINE_JACOBI_ICB(half, half)
#if __METAL_VERSION__ >= 310
DEFINE_JACOBI_ICB(bfloat, bfloat)
#endif
#define DEFINE_DOT_COLUMNS(T, SUFFIX) \
kernel void dot_columns_kernel_##SUFFIX( \
    device const T* A_Rotated [[buffer(0)]], \
    device const T* V_Rotated [[buffer(1)]], \
    device T* Eigenvalues [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& N [[buffer(4)]], \
    constant uint& BatchStrideA [[buffer(5)]], \
    constant uint& BatchStrideV [[buffer(6)]], \
    constant uint& BatchStrideE [[buffer(7)]], \
    threadgroup float* shared_mem [[threadgroup(0)]], \
    uint3 group_pos [[threadgroup_position_in_grid]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 threads_per_group_vec [[threads_per_threadgroup]]) \
{ \
    uint tid = tid_vec.x; \
    uint i = group_pos.x; \
    uint batch_idx = group_pos.z; \
    uint threads_per_group = threads_per_group_vec.x; \
    \
    if (i >= N) return; \
    \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    uint batch_offset_E = batch_idx * BatchStrideE; \
    \
    device const T* col_a = A_Rotated + batch_offset_A + i * M; \
    device const T* col_v = V_Rotated + batch_offset_V + i * N; \
    \
    float sum_dot = 0.0f; \
    for (uint k = tid; k < M; k += threads_per_group) { \
        sum_dot += (float)col_a[k] * (float)col_v[k]; \
    } \
    \
    sum_dot = simd_reduction(sum_dot); \
    \
    if ((tid % 32) == 0) { \
        shared_mem[tid / 32] = sum_dot; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    if (tid == 0) { \
        float total_dot = 0.0f; \
        uint num_simd_groups = (threads_per_group + 31) / 32; \
        for (uint s = 0; s < num_simd_groups; ++s) { \
            total_dot += shared_mem[s]; \
        } \
        Eigenvalues[batch_offset_E + i] = (T)total_dot; \
    } \
}

DEFINE_JACOBI(float, float)
DEFINE_JACOBI(half, half)
#if __METAL_VERSION__ >= 310
DEFINE_JACOBI(bfloat, bfloat)
#endif

DEFINE_DOT_COLUMNS(float, float)
DEFINE_DOT_COLUMNS(half, half)
#if __METAL_VERSION__ >= 310
DEFINE_DOT_COLUMNS(bfloat, bfloat)
#endif

// -----------------------------------------------------------------------------
// MACRO: Fused Block Kernel (Generic)
// -----------------------------------------------------------------------------
#define DEFINE_FUSED_GENERIC(T, SUFFIX) \
kernel void svd_fused_block_kernel_generic_##SUFFIX( \
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
    for (uint sw = 0; sw < 15; ++sw) { \
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
                float epsilon = (sizeof(T) == 4) ? 1e-9f : 1e-4f; \
                if (abs(apq) > epsilon) { \
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
    const uint ThreadsPerPair = 32; \
    const uint N = 64; \
    uint tid = tid_vec.x; \
    uint batch_idx = group_id.z; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* A_ptr = A + batch_offset_A; \
    device T* V_ptr = V + batch_offset_V; \
    uint pair_idx = tid / ThreadsPerPair; \
    uint lane_id = tid % ThreadsPerPair; \
    threadgroup atomic_uint sweep_rotations; \
    for (uint sw = 0; sw < 15; ++sw) { \
        /* if (tid == 0) atomic_store_explicit(&sweep_rotations, 0, memory_order_relaxed); */ \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint step = 0; step < NumSteps; ++step) { \
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup); \
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
                app += simd_shuffle_down(app, 16); \
                app += simd_shuffle_down(app, 8); \
                app += simd_shuffle_down(app, 4); \
                app += simd_shuffle_down(app, 2); \
                app += simd_shuffle_down(app, 1); \
                aqq += simd_shuffle_down(aqq, 16); \
                aqq += simd_shuffle_down(aqq, 8); \
                aqq += simd_shuffle_down(aqq, 4); \
                aqq += simd_shuffle_down(aqq, 2); \
                aqq += simd_shuffle_down(aqq, 1); \
                apq += simd_shuffle_down(apq, 16); \
                apq += simd_shuffle_down(apq, 8); \
                apq += simd_shuffle_down(apq, 4); \
                apq += simd_shuffle_down(apq, 2); \
                apq += simd_shuffle_down(apq, 1); \
                float c = 1.0f, s = 0.0f; \
                bool rotate = false; \
                if (lane_id == 0) { \
                    float epsilon = (sizeof(T) == 4) ? 1e-9f : 1e-4f; \
                    if (abs(apq) > epsilon) { \
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
                c = simd_shuffle(c, tid & ~31); \
                s = simd_shuffle(s, tid & ~31); \
                rotate = (bool)simd_shuffle((ushort)rotate, tid & ~31); \
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
        /* if (atomic_load_explicit(&sweep_rotations, memory_order_relaxed) == 0) break; */ \
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
    for (uint sw = 0; sw < 15; ++sw) { \
        /* if (tid == 0) atomic_store_explicit(&sweep_rotations, 0, memory_order_relaxed); */ \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint step = 0; step < NumSteps; ++step) { \
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup); \
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
                    float epsilon = (sizeof(T) == 4) ? 1e-9f : 1e-4f; \
                    if (abs(apq) > epsilon) { \
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
                c = simd_shuffle(c, tid & ~15); \
                s = simd_shuffle(s, tid & ~15); \
                rotate = (bool)simd_shuffle((ushort)rotate, tid & ~15); \
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
        /* if (atomic_load_explicit(&sweep_rotations, memory_order_relaxed) == 0) break; */ \
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
    \
    threadgroup atomic_uint sweep_rotations; \
    \
    for (uint sw = 0; sw < 15; ++sw) { \
        /* if (tid == 0) atomic_store_explicit(&sweep_rotations, 0, memory_order_relaxed); */ \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        \
        for (uint step = 0; step < NumSteps; ++step) { \
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup); \
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
                    float epsilon = (sizeof(T) == 4) ? 1e-9f : 1e-4f; \
                    if (abs(apq) > epsilon) { \
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
                c = simd_shuffle(c, tid & ~7); \
                s = simd_shuffle(s, tid & ~7); \
                rotate = (bool)simd_shuffle((ushort)rotate, tid & ~7); \
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
        /* if (atomic_load_explicit(&sweep_rotations, memory_order_relaxed) == 0) break; */ \
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
    for (uint sw = 0; sw < 15; ++sw) { \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint step = 0; step < NumSteps; ++step) { \
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup); \
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
                app += simd_shuffle_down(app, 2); \
                app += simd_shuffle_down(app, 1); \
                aqq += simd_shuffle_down(aqq, 2); \
                aqq += simd_shuffle_down(aqq, 1); \
                apq += simd_shuffle_down(apq, 2); \
                apq += simd_shuffle_down(apq, 1); \
                float c = 1.0f, s = 0.0f; \
                bool rotate = false; \
                if (lane_id == 0) { \
                    float epsilon = (sizeof(T) == 4) ? 1e-9f : 1e-4f; \
                    if (abs(apq) > epsilon) { \
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
                c = simd_shuffle(c, tid & ~3); \
                s = simd_shuffle(s, tid & ~3); \
                rotate = (bool)simd_shuffle((ushort)rotate, tid & ~3); \
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
    for (uint sw = 0; sw < 15; ++sw) { \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint step = 0; step < NumSteps; ++step) { \
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup); \
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
                app += simd_shuffle_down(app, 1); \
                aqq += simd_shuffle_down(aqq, 1); \
                apq += simd_shuffle_down(apq, 1); \
                float c = 1.0f, s = 0.0f; \
                bool rotate = false; \
                if (lane_id == 0) { \
                    float epsilon = (sizeof(T) == 4) ? 1e-9f : 1e-4f; \
                    if (abs(apq) > epsilon) { \
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
                c = simd_shuffle(c, tid & ~1); \
                s = simd_shuffle(s, tid & ~1); \
                rotate = (bool)simd_shuffle((ushort)rotate, tid & ~1); \
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
    } \
}


// Fused
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
