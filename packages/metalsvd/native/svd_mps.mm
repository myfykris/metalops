#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

using namespace at::mps;
using namespace at::native::mps;

#include <fstream>
#include <sstream>

// -----------------------------------------------------------------------------
// Metal Source Code (Embedded - Fallback/Legacy)
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Metal Source Code - LOADED FROM FILE ONLY
// -----------------------------------------------------------------------------
// const char* SVD_METAL_SOURCE = ... (Removed);
// -----------------------------------------------------------------------------
// Global States
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Global States
// -----------------------------------------------------------------------------
struct SVDKernels {
    id<MTLFunction> jacobi = nil;
    id<MTLFunction> jacobi_icb = nil; // New ICB-compatible kernel
    id<MTLFunction> jacobi_icb_vec4 = nil; // Vectorized ICB kernel
    id<MTLFunction> jacobi_fused = nil; 
    id<MTLFunction> jacobi_fused_256 = nil; // Specialized N=256 kernel 
    id<MTLFunction> norm = nil;
    id<MTLFunction> normalize = nil;
    
    // Cached PSOs
    id<MTLComputePipelineState> jacobiPSO = nil;
    id<MTLComputePipelineState> jacobiICBPSO = nil;
    id<MTLComputePipelineState> jacobiICBVec4PSO = nil;
    id<MTLComputePipelineState> jacobiFusedPSO = nil;
    id<MTLComputePipelineState> jacobiFused256PSO = nil;
    id<MTLComputePipelineState> jacobiFused128PSO = nil;
    id<MTLComputePipelineState> jacobiFused64PSO = nil;
    id<MTLComputePipelineState> jacobiFused512PSO = nil;
    id<MTLComputePipelineState> jacobiFused1024PSO = nil;
    id<MTLComputePipelineState> normPSO = nil;
    id<MTLComputePipelineState> normalizePSO = nil;
    
    // ICB Cache: Map N -> ICB
    NSMutableDictionary<NSNumber*, id<MTLIndirectCommandBuffer>>* icbCache = nil;
    // Step Buffer Cache: Map N -> MTLBuffer
    NSMutableDictionary<NSNumber*, id<MTLBuffer>>* stepBufferCache = nil;
};

static SVDKernels kFloat;
static SVDKernels kHalf;
static SVDKernels kBFloat;
static id<MTLLibrary> svdLib = nil;
static std::once_flag init_flag;

void load_kernels(id<MTLLibrary> lib, SVDKernels& k, NSString* suffix, bool required) {
    // Init Caches
    k.icbCache = [NSMutableDictionary new];
    k.stepBufferCache = [NSMutableDictionary new];

    k.jacobi = [lib newFunctionWithName:[NSString stringWithFormat:@"jacobi_rotate_kernel_optimized_%@", suffix]];
    k.jacobi_icb = [lib newFunctionWithName:[NSString stringWithFormat:@"jacobi_rotate_kernel_icb_%@", suffix]]; // Load ICB kernel
    k.jacobi_icb_vec4 = [lib newFunctionWithName:[NSString stringWithFormat:@"jacobi_rotate_kernel_vec4_clean_%@", suffix]]; // Load Clean Vec4 kernel
    
    k.jacobi_fused = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_%@", suffix]];
    k.jacobi_fused_256 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_256_%@", suffix]];
    
    k.norm = [lib newFunctionWithName:[NSString stringWithFormat:@"column_norm_kernel_%@", suffix]];
    k.normalize = [lib newFunctionWithName:[NSString stringWithFormat:@"normalize_kernel_%@", suffix]];
    
    if (required) {
        TORCH_CHECK(k.jacobi && k.jacobi_fused && k.norm && k.normalize, "Failed to load required kernels for suffix: ", [suffix UTF8String]);
    }
    
    // Cache PSOs (Only if function loaded)
    NSError* error = nil;
    id<MTLDevice> device = MPSDevice::getInstance()->device();
    
    if (k.jacobi) {
        k.jacobiPSO = [device newComputePipelineStateWithFunction:k.jacobi error:&error];
        if(!k.jacobiPSO) printf("Failed to create jacobiPSO: %s\n", [[error localizedDescription] UTF8String]);
    }
    
    if (k.jacobi_icb) {
        // Must use Descriptor to enable ICB Support explicitly
        MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
        desc.computeFunction = k.jacobi_icb;
        desc.supportIndirectCommandBuffers = YES; // Correct property name
        
        MTLComputePipelineReflection* reflection = nil;
        k.jacobiICBPSO = [device newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:&reflection error:&error];
        
        if(!k.jacobiICBPSO) printf("Failed to create jacobiICBPSO: %s\n", [[error localizedDescription] UTF8String]);
    }

    if (k.jacobi_icb_vec4) {
        MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
        desc.computeFunction = k.jacobi_icb_vec4;
        desc.supportIndirectCommandBuffers = YES; 
        
        MTLComputePipelineReflection* reflection = nil;
        k.jacobiICBVec4PSO = [device newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:&reflection error:&error];
        
        if(!k.jacobiICBVec4PSO) printf("Failed to create jacobiICBVec4PSO: %s\n", [[error localizedDescription] UTF8String]);
    }
    
    if (k.jacobi_fused) {
        k.jacobiFusedPSO = [device newComputePipelineStateWithFunction:k.jacobi_fused error:&error];
    }
    if (k.jacobi_fused_256) {
        k.jacobiFused256PSO = [device newComputePipelineStateWithFunction:k.jacobi_fused_256 error:&error];
    }

    id<MTLFunction> f128 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_128_%@", suffix]];
    if (f128) k.jacobiFused128PSO = [device newComputePipelineStateWithFunction:f128 error:&error];
    
    id<MTLFunction> f64 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_64_%@", suffix]];
    if (f64) k.jacobiFused64PSO = [device newComputePipelineStateWithFunction:f64 error:&error];
    
    id<MTLFunction> f512 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_512_%@", suffix]];
    if (f512) k.jacobiFused512PSO = [device newComputePipelineStateWithFunction:f512 error:&error];
    
    id<MTLFunction> f1024 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_1024_%@", suffix]];
    if (f1024) k.jacobiFused1024PSO = [device newComputePipelineStateWithFunction:f1024 error:&error];
    
    if (k.norm) {
        k.normPSO = [device newComputePipelineStateWithFunction:k.norm error:&error];
    }
    if (k.normalize) {
        k.normalizePSO = [device newComputePipelineStateWithFunction:k.normalize error:&error];
    }
}

void init_mps_svd() {
    std::call_once(init_flag, [](){
        id<MTLDevice> device = MPSDevice::getInstance()->device();
        if (!device) TORCH_CHECK(false, "MPS Device not found");
        
        NSError* error = nil;
        NSString* src = nil;
        
        // Try to load from file first
        const char* file_path = "/Users/kris/localprojects/metalops/packages/metalsvd/native/svd_kernels.metal";
        std::ifstream t(file_path);
        if (t.is_open()) {
            std::stringstream buffer;
            buffer << t.rdbuf();
            std::string content = buffer.str();
            src = [NSString stringWithUTF8String:content.c_str()];
            printf("Loaded Metal kernels from file: %s\n", file_path);
        } else {
             printf("CRITICAL ERROR: Could not load Metal kernels from %s\n", file_path);
             TORCH_CHECK(false, "Metal kernel file missing");
        }

        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        
        // Attempt to target newer Metal version for bfloat support
        options.languageVersion = MTLLanguageVersion3_1; 
        
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
    // printf("DEBUG: svd_forward N=%d Batch=%d\n", (int)N, (int)Batch); // Trace
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
    // Empirical: Fused (TPP=1) is slow/stuck for N >= 256.
    // Use Fused Kernel for small/medium matrices
    // Testing N=256 with Serial Mode (TPP=1) was too slow (0.05x).
    // Reverting to N=128 (Stable, 1.2x) as the limit.
    bool use_fused = (N <= 128);
    // if (use_fused) threads_per_pair = 1; // Standard Serial Fused for small/medium N -> DISABLED for Specialized Check
    
    // Ensure power of 2 for reduction
    if (use_fused) {
         // Round down to power of 2?
         // 1024/num_pairs is always power of 2 if N is power of 2.
         // But for general N?
         // Safety:
         if (threads_per_pair >= 32) threads_per_pair = 32;
         else if (threads_per_pair >= 16) threads_per_pair = 16;
         else if (threads_per_pair >= 8) threads_per_pair = 8;
         else if (threads_per_pair >= 4) threads_per_pair = 4;
         else if (threads_per_pair >= 2) threads_per_pair = 2;
         else threads_per_pair = 1;
    }
    // Specialized Kernel Flags - All specialized kernels now working!
    int specialized_mode = 0; // 0=None/Generic, 1=256, 2=128, 3=64, 4=512, 5=1024
    if (N == 1024) { specialized_mode = 5; threads_per_pair = 2; }
    else if (N == 512) { specialized_mode = 4; threads_per_pair = 4; }
    else if (N == 256) { specialized_mode = 1; threads_per_pair = 8; }
    else if (N == 128) { specialized_mode = 2; threads_per_pair = 16; }
    else if (N == 64) { specialized_mode = 3; threads_per_pair = 32; }
    
    bool use_fused_any = (specialized_mode > 0) || (N <= 256);

    // Unified Pair Tensor Creation (Full Sequence)
    torch::Tensor PairsTens = torch::tensor(pairs_cpu, torch::dtype(torch::kInt32).device(torch::kCPU)).contiguous();
    PairsTens = PairsTens.to(A.device());
    
    // -------------------------------------------------------------------------
    // CRITICAL: Fetch Encoder AFTER all tensor copies (.to) are done!
    // -------------------------------------------------------------------------
    MPSStream* stream = getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    
    // Use Cached PSOs
    id<MTLComputePipelineState> rotatePSO = kernels->jacobiPSO;
    id<MTLComputePipelineState> normPSO = kernels->normPSO;
    id<MTLComputePipelineState> normalizePSO = kernels->normalizePSO;
    
    TORCH_CHECK(rotatePSO, "Failed to get rotatePSO");
    
    uint32_t BatchStrideA = (uint32_t)(N * M);
    uint32_t BatchStrideV = (uint32_t)(N * N);
    uint32_t M_u = (uint32_t)M;
    uint32_t N_u = (uint32_t)N;


    if (use_fused_any) {
         // Fused Dispatch
         id<MTLComputePipelineState> fusedPSO = kernels->jacobiFusedPSO;
         if (specialized_mode == 5) fusedPSO = kernels->jacobiFused1024PSO;
         else if (specialized_mode == 4) fusedPSO = kernels->jacobiFused512PSO;
         else if (specialized_mode == 1) fusedPSO = kernels->jacobiFused256PSO;
         else if (specialized_mode == 2) fusedPSO = kernels->jacobiFused128PSO;
         else if (specialized_mode == 3) fusedPSO = kernels->jacobiFused64PSO;
         // If mode 0, use Generic (jacobiFusedPSO)
         
         TORCH_CHECK(fusedPSO, "Failed to get Fused PSO");
         
         [encoder setComputePipelineState:fusedPSO];
         mtl_setBuffer(encoder, A_T, 0);
         mtl_setBuffer(encoder, V_T, 1);
         [encoder setBuffer:getMTLBufferStorage(PairsTens) offset:(PairsTens.storage_offset() * PairsTens.element_size()) atIndex:2];
         
         if (specialized_mode > 0) {
             // All Specialized Kernels (N=64, 128, 256) use simplified buffer layout:
             // Signature: buffer(3)=M, buffer(4)=BatchStrideA, buffer(5)=BatchStrideV
             [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:3];
             [encoder setBytes:&BatchStrideA length:sizeof(uint32_t) atIndex:4];
             [encoder setBytes:&BatchStrideV length:sizeof(uint32_t) atIndex:5];
         } else {
             // Generic Kernel - Full buffer layout
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
         }
         
         int total_threads = num_pairs * threads_per_pair;
         MTLSize groupSize = MTLSizeMake(total_threads, 1, 1);
         
         
         // Persistent Threadgroup (1 per batch item)
         [encoder dispatchThreadgroups:MTLSizeMake(1, 1, Batch) threadsPerThreadgroup:groupSize];
         
    } else {
        bool disable_icb_debug = false; // ENABLE ICB for Large Matrices
        
        id<MTLComputePipelineState> rotatePSO = kernels->jacobiPSO;
        
        if (!disable_icb_debug) {
             // Fallback to ICB Dispatch for performance
             rotatePSO = kernels->jacobiICBPSO;
             
             // ICB Strategy is critical for N >= 256
             int sweeps = 6;
             
             // Check ICB Kernel Threads per Group
             int threads_per_group_val = 256;
             if (M >= 4096) threads_per_group_val = 1024; // Try Max Occupancy (Scalar)

             // Check Vectorization
             // PERFORMANCE REGRESSION: Vectorized Kernel is slower (46s vs 36s Scalar). Disabling for now.
             if (M % 4 == 0 && N % 4 == 0 && kernels->jacobiICBVec4PSO) {
                  rotatePSO = kernels->jacobiICBVec4PSO; 
                  threads_per_group_val = 1024; // Try Max Occupancy
                  printf("DEBUG: Using Vectorized PSO (TPG=%d)\n", threads_per_group_val);
             }
             
             if (!rotatePSO) {
                  printf("CRITICAL ERROR: rotatePSO is nil! Fallback will crash.\n");
                  TORCH_CHECK(false, "rotatePSO missing");
             }
             
             // ICB Strategy is critical for N >= 256
             // sweeps declared above or here? Let's check context.
             // If I see: int sweeps = 6; ... int sweeps = 6;
             // I will remove the second one.
             
             // threads_per_group_val is set above (256 default, 1024 huge-scalar, 256 huge-vec)
             
             // CLAMP to PSO Max
             if (rotatePSO.maxTotalThreadsPerThreadgroup < (NSUInteger)threads_per_group_val) {
                 printf("WARNING: Clamping ICB threads from %d to %lu\n", threads_per_group_val, (unsigned long)rotatePSO.maxTotalThreadsPerThreadgroup);
                 threads_per_group_val = (int)rotatePSO.maxTotalThreadsPerThreadgroup;
             }
             
             MTLSize threadsPerGroup = MTLSizeMake(threads_per_group_val, 1, 1);
             // printf("Debug: ICB threadsPerGroup=%d\n", threads_per_group_val);
             
             int elem_size = A.element_size();
             if (rotatePSO == kernels->jacobiICBVec4PSO) {
                 // Vectorized kernel uses FLOAT shared memory for accumulation, regardless of input type
                 elem_size = 4; 
             }
             
             NSUInteger sharedMemSize = ((threads_per_group_val + 31) / 32) * 3 * elem_size; 
             
             // Check if ICB cached
             id<MTLIndirectCommandBuffer> icb = [kernels->icbCache objectForKey:@(N)];
             long num_steps_val = (N % 2 == 0) ? (N - 1) : N;
             int total_commands = sweeps * (int)num_steps_val;
             
             // 256-byte alignment for Step Buffer
             NSUInteger stepStride = 256;
             NSUInteger stepBufferSize = stepStride * total_commands;
             
             // Check Cache
             id<MTLBuffer> stepBuffer = nil;
             if (!disable_icb_debug) {
                 stepBuffer = [kernels->stepBufferCache objectForKey:@(N)];
             }
     
             if (!icb) {
                  printf("Debug: Creating New ICB for N=%d TotalCmds=%d\n", (int)N, total_commands); fflush(stdout);
                  MTLIndirectCommandBufferDescriptor* icbDesc = [MTLIndirectCommandBufferDescriptor new];
                  icbDesc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
                  icbDesc.inheritBuffers = YES; 
                  icbDesc.inheritPipelineState = NO; // Explicitly setting it
                  icbDesc.maxVertexBufferBindCount = 0;
                  icbDesc.maxFragmentBufferBindCount = 0;
                  icbDesc.maxKernelBufferBindCount = 31; // Safe limit
                  
                  if (!stepBuffer) {
                      stepBuffer = [svdLib.device newBufferWithLength:stepBufferSize options:MTLResourceStorageModeShared];
                      if (Batch == 1) [kernels->stepBufferCache setObject:stepBuffer forKey:@(N)];
                  }
                  
                  icb = [svdLib.device newIndirectCommandBufferWithDescriptor:icbDesc maxCommandCount:total_commands options:0];
                  
                  // Encode ICB
                  int cmd_idx = 0;
                  uint32_t* ptr = (uint32_t*)stepBuffer.contents;
                  
                  for (int sw=0; sw<sweeps; sw++) {
                      @autoreleasepool {
                          for (int step=0; step<num_steps_val; step++) { // Loop logic matches Scheduler
                               // Write Step Index to Buffer (Aligned)
                               *(uint32_t*)((char*)ptr + cmd_idx * stepStride) = step;
                               
                               id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:cmd_idx];
                               [cmd setComputePipelineState:rotatePSO];
                               
                               // Bind Step Buffer (Index 4) with Aligned Offset
                               [cmd setKernelBuffer:stepBuffer offset:(cmd_idx * stepStride) atIndex:4];
                               
                               [cmd setThreadgroupMemoryLength:sharedMemSize atIndex:0];
                               [cmd concurrentDispatchThreadgroups:MTLSizeMake(num_pairs, 1, Batch) threadsPerThreadgroup:threadsPerGroup];
                               
                               cmd_idx++;
                          }
                      }
                  }
                  // Cache it
                  if (Batch == 1) { 
                      [kernels->icbCache setObject:icb forKey:@(N)];
                  }
             }
             
             // Executing ICB...
             
             // Create Scalar Buffer for M, N, Strides, NumPairs (Indices 3-7)
             // setBytes is NOT inherited by ICB!
             uint32_t scalars[5] = {M_u, N_u, BatchStrideA, BatchStrideV, (uint32_t)num_pairs};
             id<MTLBuffer> scalarBuffer = [svdLib.device newBufferWithBytes:scalars length:sizeof(scalars) options:MTLResourceStorageModeShared];
             
             // Execution
             [encoder setComputePipelineState:rotatePSO];
             mtl_setBuffer(encoder, A_T, 0);
             mtl_setBuffer(encoder, V_T, 1);
             [encoder setBuffer:getMTLBufferStorage(PairsTens) offset:(PairsTens.storage_offset() * PairsTens.element_size()) atIndex:2]; // FullPairs
             
             // Bind Uniforms Buffer (Index 3)
             [encoder setBuffer:scalarBuffer offset:0 atIndex:3]; 
             
             // Step (4) is set in ICB.
             
             [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];
             
             // Ensure Buffers are resident
             if (stepBuffer) [encoder useResource:stepBuffer usage:MTLResourceUsageRead];
             // Ensure Buffers are resident
             if (stepBuffer) [encoder useResource:stepBuffer usage:MTLResourceUsageRead];
             [encoder useResource:scalarBuffer usage:MTLResourceUsageRead];
             [encoder useResource:getMTLBufferStorage(PairsTens) usage:MTLResourceUsageRead]; // Critical for ICB!

             // Execute ICB
             [encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0, total_commands)];
        } else {
             // SLOW PATH (Iterative)
             int sweeps = 6;
             int threads_per_group_val = std::min((int)rotatePSO.maxTotalThreadsPerThreadgroup, 256);
             MTLSize threadsPerGroup = MTLSizeMake(threads_per_group_val, 1, 1);
             int elem_size = A.element_size();
             NSUInteger sharedMemSize = ((threads_per_group_val + 31) / 32) * 3 * elem_size; 
             
             [encoder setComputePipelineState:rotatePSO];
             mtl_setBuffer(encoder, A_T, 0);
             mtl_setBuffer(encoder, V_T, 1);
             // Index 2 set in loop
             [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:3];
             [encoder setBytes:&N_u length:sizeof(uint32_t) atIndex:4];
             [encoder setBytes:&BatchStrideA length:sizeof(uint32_t) atIndex:5];
             [encoder setBytes:&BatchStrideV length:sizeof(uint32_t) atIndex:6];
             [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];
             
             MTLSize resultGrid = MTLSizeMake(num_pairs_per_step, N, Batch); 
             MTLSize threadgroups = MTLSizeMake(num_pairs_per_step, 1, Batch); 
             
             for (int sw = 0; sw < sweeps; ++sw) {
                 for (int step = 0; step < num_steps; ++step) {
                     // Offset Pairs
                     // PairsTens is effectively FullPairs (from previous step 1222 edit)
                     // But wait, PairsTens = FullPairs.
                     size_t pairs_offset = step * num_pairs * sizeof(int) * 2;
                     
                     [encoder setBuffer:getMTLBufferStorage(PairsTens) 
                                 offset:(PairsTens.storage_offset() * PairsTens.element_size() + pairs_offset) 
                                atIndex:2];
                                
                     [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
                 }
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
