#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

using namespace at::mps;
using namespace at::native::mps;

// -----------------------------------------------------------------------------
// Metal Source Code (Embedded)
// -----------------------------------------------------------------------------
const char* SVD_METAL_SOURCE = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant float EPSILON = 1e-6;

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

// -----------------------------------------------------------------------------
// Macros for Templating
// -----------------------------------------------------------------------------
#define INSTANTIATE_KERNELS(T, SUFFIX) \
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
} \
\
kernel void jacobi_rotate_kernel_optimized_##SUFFIX( \
    device T* A_T [[buffer(0)]], \
    device T* V_T [[buffer(1)]], \
    device const int2* Pairs [[buffer(2)]], \
    constant uint& M [[buffer(3)]], \
    constant uint& N [[buffer(4)]], \
    constant uint& BatchStrideA [[buffer(5)]], \
    constant uint& BatchStrideV [[buffer(6)]], \
    threadgroup T* shared_mem [[threadgroup(0)]], \
    uint3 group_pos [[threadgroup_position_in_grid]], \
    uint3 tid_vec [[thread_position_in_threadgroup]], \
    uint3 threads_per_group_vec [[threads_per_threadgroup]] \
) { \
    int pair_idx = group_pos.x; \
    int batch_idx = group_pos.z; \
    uint tid = tid_vec.x; \
    uint threads_per_group = threads_per_group_vec.x; \
    uint simd_lane_id = tid % 32; \
    uint simd_group_id = tid / 32; \
    int2 pair = Pairs[pair_idx]; \
    int i = pair.x; \
    int j = pair.y; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_V = batch_idx * BatchStrideV; \
    device T* col_i = A_T + batch_offset_A + i * M; \
    device T* col_j = A_T + batch_offset_A + j * M; \
    T part_ii = 0.0; \
    T part_jj = 0.0; \
    T part_ij = 0.0; \
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
        if (abs(sum_ij) > EPSILON) { \
            float tau = (sum_jj - sum_ii) / (2.0f * sum_ij); \
            float t; \
            float sqrt_term = sqrt(1.0f + tau * tau); \
            if (tau >= 0.0f) { \
                t = 1.0f / (tau + sqrt_term); \
            } else { \
                t = -1.0f / (-tau + sqrt_term); \
            } \
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
    device T* v_col_i = V_T + batch_offset_V + i * N; \
    device T* v_col_j = V_T + batch_offset_V + j * N; \
    for (uint k = tid; k < N; k += threads_per_group) { \
        T val_vi = v_col_i[k]; \
        T val_vj = v_col_j[k]; \
        v_col_i[k] = c * val_vi - s * val_vj; \
        v_col_j[k] = s * val_vi + c * val_vj; \
    } \
} \
\
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
    \
    device T* A_ptr = A + batch_offset_A; \
    device T* V_ptr = V + batch_offset_V; \
    \
    uint pair_idx = tid / ThreadsPerPair; \
    uint lane_id = tid % ThreadsPerPair; \
    \
    /* Jacobi Logic (ThreadsPerPair = 1 assumed for simplicity) */ \
    for (uint sw = 0; sw < 10; ++sw) { \
        for (uint step = 0; step < NumSteps; ++step) { \
            /* Sync before step */ \
            threadgroup_barrier(mem_flags::mem_device); \
            \
            uint pair_offset = step * NumPairs * 2 + pair_idx * 2; \
            int p = AllPairs[pair_offset]; \
            int q = AllPairs[pair_offset+1]; \
            \
            if (pair_idx < NumPairs) { \
                device T* col_p = A_ptr + p * M; \
                device T* col_q = A_ptr + q * M; \
                device T* v_col_p = V_ptr + p * N; \
                device T* v_col_q = V_ptr + q * N; \
                \
                /* 1. Dot Product */ \
                float app = 0.0f, aqq = 0.0f, apq = 0.0f; \
                for(uint k=0; k<M; ++k) { \
                    float vp = (float)col_p[k]; \
                    float vq = (float)col_q[k]; \
                    app += vp * vp; \
                    aqq += vq * vq; \
                    apq += vp * vq; \
                } \
                \
                /* 2. Compute Rotation */ \
                float c = 1.0f, s = 0.0f; \
                bool rotate = false; \
                if (abs(apq) > MAX(1e-6f, 1e-6f * sqrt(app * aqq))) { \
                    rotate = true; \
                    float tau = (aqq - app) / (2.0f * apq); \
                    float t; \
                    if (tau >= 0.0f) t = 1.0f / (tau + sqrt(1.0f + tau*tau)); \
                    else t = -1.0f / (-tau + sqrt(1.0f + tau*tau)); \
                    c = 1.0f / sqrt(1.0f + t*t); \
                    s = t * c; \
                } \
                \
                /* 3. Rotate A */ \
                if (rotate) { \
                    for(uint k=0; k<M; ++k) { \
                        float vp = (float)col_p[k]; \
                        float vq = (float)col_q[k]; \
                        col_p[k] = (T)(c * vp - s * vq); \
                        col_q[k] = (T)(s * vp + c * vq); \
                    } \
                    /* 4. Rotate V */ \
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
    return; \
} \
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
    if (i >= N) return; \
    uint batch_idx = gid.y; \
    uint batch_offset_A = batch_idx * BatchStrideA; \
    uint batch_offset_S = batch_idx * BatchStrideS; \
    device const T* col_i = A_T + batch_offset_A + i * M; \
    float sum_sq = 0.0f; \
    for (uint k = 0; k < M; ++k) { \
        sum_sq += (float)(col_i[k] * col_i[k]); \
    } \
    S[batch_offset_S + i] = (T)sqrt(sum_sq); \
} \
\
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

// Instantiate for float, half
INSTANTIATE_KERNELS(float, float)
INSTANTIATE_KERNELS(half, half)

#if __METAL_VERSION__ >= 310
INSTANTIATE_KERNELS(bfloat, bfloat)
#endif

)METAL";
// -----------------------------------------------------------------------------
// Global States
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Global States
// -----------------------------------------------------------------------------
struct SVDKernels {
    id<MTLFunction> jacobi = nil;
    id<MTLFunction> jacobi_fused = nil; // New
    id<MTLFunction> norm = nil;
    id<MTLFunction> normalize = nil;
};

static SVDKernels kFloat;
static SVDKernels kHalf;
static SVDKernels kBFloat;
static id<MTLLibrary> svdLib = nil;
static std::once_flag init_flag;

void load_kernels(id<MTLLibrary> lib, SVDKernels& k, NSString* suffix, bool required) {
    k.jacobi = [lib newFunctionWithName:[NSString stringWithFormat:@"jacobi_rotate_kernel_optimized_%@", suffix]];
    
    k.jacobi_fused = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_%@", suffix]];
    
    k.norm = [lib newFunctionWithName:[NSString stringWithFormat:@"column_norm_kernel_%@", suffix]];
    k.normalize = [lib newFunctionWithName:[NSString stringWithFormat:@"normalize_kernel_%@", suffix]];
    if (required) {
        TORCH_CHECK(k.jacobi && k.jacobi_fused && k.norm && k.normalize, "Failed to load required kernels for suffix: ", [suffix UTF8String]);
    }
}

void init_mps_svd() {
    std::call_once(init_flag, [](){
        id<MTLDevice> device = MPSDevice::getInstance()->device();
        if (!device) TORCH_CHECK(false, "MPS Device not found");
        
        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:SVD_METAL_SOURCE];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        
        // Attempt to target newer Metal version for bfloat support
        // options.languageVersion = MTLLanguageVersion3_0; 
        
        svdLib = [device newLibraryWithSource:src options:options error:&error];
        if (!svdLib) {
             TORCH_CHECK(false, "Failed to compile Metal SVD library: ", [[error localizedDescription] UTF8String]);
        }
        
        load_kernels(svdLib, kFloat, @"float", true);
        load_kernels(svdLib, kHalf, @"half", true);
        load_kernels(svdLib, kBFloat, @"bfloat", false); // Optional
    });
}

// -----------------------------------------------------------------------------
// Helper: Pairing Strategy
// -----------------------------------------------------------------------------
std::pair<std::vector<int>, int> generate_ordering(int N) {
    std::vector<int> all_pairs;
    int num_steps = N - 1; 
    std::vector<int> players(N);
    for(int i=0; i<N; ++i) players[i] = i;

    // Round Robin
    for(int s=0; s<num_steps; ++s) {
        for(int k=0; k<N/2; ++k) {
            all_pairs.push_back(players[k]);
            all_pairs.push_back(players[N - 1 - k]);
        }
        int last = players.back();
        for(int i=N-1; i>1; --i) players[i] = players[i-1];
        players[1] = last;
    }
    return {all_pairs, num_steps};
}

// -----------------------------------------------------------------------------
// SVD Forward
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> svd_forward(torch::Tensor A) { 
    TORCH_CHECK(A.device().is_mps(), "Input tensor must be on MPS");
    
    // Dispatch Kernels
    init_mps_svd();
    SVDKernels* kernels = nullptr;
    if (A.scalar_type() == torch::kFloat32) {
        kernels = &kFloat;
    } else if (A.scalar_type() == torch::kHalf) {
        kernels = &kHalf;
    } else if (A.scalar_type() == torch::kBFloat16) {
        if (!kBFloat.jacobi) {
             TORCH_CHECK(false, "BFloat16 not supported on this device/OS (requires Metal 3.1+)");
        }
        kernels = &kBFloat;
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Only Float32, Float16, and BFloat16 supported.");
    }
    
    if (A.dim() == 2) A = A.unsqueeze(0);
    
    int64_t Batch = A.size(0);
    int64_t M = A.size(1);
    int64_t N = A.size(2);
    
    if (N % 2 != 0) {
        TORCH_CHECK(N % 2 == 0, "Internal Error: N must be even (padding failed?)");
    }

    // 1. Setup V
    torch::Tensor V = torch::eye(N, A.options()).expand({Batch, N, N}).contiguous();
    
    // 2. Prepare Transposed Data
    torch::Tensor A_T = A.transpose(1, 2).contiguous(); 
    torch::Tensor V_T = V.transpose(1, 2).contiguous(); 
    
    // 3. Logic & Pre-calculation
    auto [pairs_cpu, num_steps] = generate_ordering(N);
    int num_pairs_per_step = N / 2;
    int num_pairs = N / 2;
    int max_threads = 1024; 
    int threads_per_pair = max_threads / num_pairs;
    
    // Power of 2 logic
    if (threads_per_pair >= 32) threads_per_pair = 32;
    else if (threads_per_pair >= 16) threads_per_pair = 16;
    else if (threads_per_pair >= 8) threads_per_pair = 8;
    else if (threads_per_pair >= 4) threads_per_pair = 4;
    else if (threads_per_pair >= 2) threads_per_pair = 2;
    else threads_per_pair = 1;
    
    // Fused Kernel Decision
    bool use_fused = (threads_per_pair >= 1);
    if (use_fused) threads_per_pair = 1; // Force 1 thread per pair for stability
    
    torch::Tensor PairsTens;
    torch::Tensor FullPairs;
    std::vector<int> full_pair_sequence;
    
    if (use_fused) {
        // Generate Full Pair Sequence
        std::vector<int> players(N);
        for(int i=0; i<N; ++i) players[i] = i;
        
        for(int s=0; s<num_steps; ++s) {
            for(int k=0; k<N/2; ++k) {
                full_pair_sequence.push_back(players[k]);
                full_pair_sequence.push_back(players[N - 1 - k]);
            }
            int last = players.back();
            for(int i=N-1; i>1; --i) players[i] = players[i-1];
            players[1] = last;
        }
        
        // Upload FullPairs
        FullPairs = torch::tensor(full_pair_sequence, torch::dtype(torch::kInt32).device(torch::kCPU)).contiguous();
        FullPairs = FullPairs.to(A.device());
        
    } else {
        // Fallback: Upload PairsTens
        PairsTens = torch::tensor(pairs_cpu, torch::dtype(torch::kInt32).device(torch::kCPU));
        PairsTens = PairsTens.to(A.device()); 
    }
    
    // -------------------------------------------------------------------------
    // CRITICAL: Fetch Encoder AFTER all tensor copies (.to) are done!
    // -------------------------------------------------------------------------
    MPSStream* stream = getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    
    NSError* error = nil;
    id<MTLComputePipelineState> rotatePSO = [svdLib.device newComputePipelineStateWithFunction:kernels->jacobi error:&error];
    id<MTLComputePipelineState> normPSO = [svdLib.device newComputePipelineStateWithFunction:kernels->norm error:&error];
    id<MTLComputePipelineState> normalizePSO = [svdLib.device newComputePipelineStateWithFunction:kernels->normalize error:&error];
    TORCH_CHECK(rotatePSO, "Failed to create PSO");
    
    uint32_t BatchStrideA = (uint32_t)(N * M);
    uint32_t BatchStrideV = (uint32_t)(N * N);
    uint32_t M_u = (uint32_t)M;
    uint32_t N_u = (uint32_t)N;

    if (use_fused) {
         // Fused Dispatch
         id<MTLComputePipelineState> fusedPSO = [svdLib.device newComputePipelineStateWithFunction:kernels->jacobi_fused error:&error];
         TORCH_CHECK(fusedPSO, "Failed to create Fused PSO");
         
         [encoder setComputePipelineState:fusedPSO];
         mtl_setBuffer(encoder, A_T, 0);
         mtl_setBuffer(encoder, V_T, 1);
         [encoder setBuffer:getMTLBufferStorage(FullPairs) offset:(FullPairs.storage_offset() * FullPairs.element_size()) atIndex:2];
         
         [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:3];
         [encoder setBytes:&N_u length:sizeof(uint32_t) atIndex:4];
         uint32_t NumPairs_u = (uint32_t)num_pairs;
         [encoder setBytes:&NumPairs_u length:sizeof(uint32_t) atIndex:5];
         uint32_t NumSteps_u = (uint32_t)num_steps;
         [encoder setBytes:&NumSteps_u length:sizeof(uint32_t) atIndex:6];
         uint32_t TPP_u = (uint32_t)threads_per_pair;
         [encoder setBytes:&TPP_u length:sizeof(uint32_t) atIndex:7];
         [encoder setBytes:&BatchStrideA length:sizeof(uint32_t) atIndex:8];
         [encoder setBytes:&BatchStrideV length:sizeof(uint32_t) atIndex:9];
         
         int total_threads = num_pairs * threads_per_pair;
         MTLSize groupSize = MTLSizeMake(total_threads, 1, 1);
         
         // Persistent Threadgroup (1 per batch item)
         [encoder dispatchThreadgroups:MTLSizeMake(1, 1, Batch) threadsPerThreadgroup:groupSize];
         
    } else {
        // Fallback to original iterating dispatch
        int sweeps = 10;
        int threads_per_group_val = std::min((int)rotatePSO.maxTotalThreadsPerThreadgroup, 256);
        
        MTLSize threadgroupsGrid = MTLSizeMake(num_pairs_per_step, 1, Batch); 
        MTLSize threadsPerGroup = MTLSizeMake(threads_per_group_val, 1, 1);
        
        int elem_size = A.element_size();
        NSUInteger sharedMemSize = ((threads_per_group_val + 31) / 32) * 3 * elem_size; 

        [encoder setComputePipelineState:rotatePSO];
        
        for (int sw = 0; sw < sweeps; ++sw) {
            for (int step = 0; step < num_steps; ++step) {
                mtl_setBuffer(encoder, A_T, 0);
                mtl_setBuffer(encoder, V_T, 1);
                
                size_t pairs_offset = step * num_pairs_per_step * sizeof(int) * 2;
                [encoder setBuffer:getMTLBufferStorage(PairsTens) 
                            offset:(PairsTens.storage_offset() * PairsTens.element_size() + pairs_offset) 
                           atIndex:2];
                
                [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:3];
                [encoder setBytes:&N_u length:sizeof(uint32_t) atIndex:4];
                [encoder setBytes:&BatchStrideA length:sizeof(uint32_t) atIndex:5];
                [encoder setBytes:&BatchStrideV length:sizeof(uint32_t) atIndex:6];
                
                [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];
                
                [encoder dispatchThreadgroups:threadgroupsGrid threadsPerThreadgroup:threadsPerGroup];
            }
        }
    }
    
    // 5. Compute Norms
    torch::Tensor S = torch::empty({Batch, N}, A.options()); 
    
    [encoder setComputePipelineState:normPSO];
    mtl_setBuffer(encoder, A_T, 0);
    mtl_setBuffer(encoder, S, 1);
    [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&N_u length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&BatchStrideA length:sizeof(uint32_t) atIndex:4];
    uint32_t BatchStrideS = (uint32_t)N;
    [encoder setBytes:&BatchStrideS length:sizeof(uint32_t) atIndex:5];
    
    MTLSize normGridSize = MTLSizeMake(N, Batch, 1);
    MTLSize normGroupSize = MTLSizeMake(std::min((int)N, (int)normPSO.maxTotalThreadsPerThreadgroup), 1, 1);
    [encoder dispatchThreads:normGridSize threadsPerThreadgroup:normGroupSize];

    // 6. Normalize U
    torch::Tensor U_T = torch::empty_like(A_T);
    
    [encoder setComputePipelineState:normalizePSO];
    mtl_setBuffer(encoder, A_T, 0);
    mtl_setBuffer(encoder, S, 1);
    mtl_setBuffer(encoder, U_T, 2);
    [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&N_u length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&BatchStrideA length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&BatchStrideS length:sizeof(uint32_t) atIndex:6];
    
    [encoder dispatchThreads:normGridSize threadsPerThreadgroup:normGroupSize];
    
    return {U_T.transpose(1, 2).contiguous(), S, V_T.transpose(1, 2).contiguous()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("svd_forward", &svd_forward, "SVD Forward (Metal)");
}
