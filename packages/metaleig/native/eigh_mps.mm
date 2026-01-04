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
#include <mutex>

struct ICBUniforms {
    uint M;
    uint N;
    uint BatchStrideA;
    uint BatchStrideV;
    uint NumPairs;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// -----------------------------------------------------------------------------
// Global States
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Global States
// -----------------------------------------------------------------------------
struct EighKernels {
    id<MTLFunction> jacobi = nil;
    id<MTLFunction> dot_columns = nil;
    
    // Fused Kernels
    id<MTLFunction> fused_generic = nil;
    id<MTLFunction> fused_64 = nil;
    id<MTLFunction> fused_128 = nil;
    id<MTLFunction> fused_256 = nil;
    id<MTLFunction> fused_512 = nil;
    id<MTLFunction> fused_1024 = nil;
    
    id<MTLComputePipelineState> jacobiPSO = nil;
    id<MTLComputePipelineState> dotColumnsPSO = nil;
    
    // Fused PSOs
    id<MTLComputePipelineState> fusedGenericPSO = nil;
    id<MTLComputePipelineState> fused64PSO = nil;
    id<MTLComputePipelineState> fused128PSO = nil;
    id<MTLComputePipelineState> fused256PSO = nil;
    id<MTLComputePipelineState> fused512PSO = nil;
    id<MTLComputePipelineState> fused1024PSO = nil;
    
    // ICB Support
    id<MTLFunction> jacobiICB = nil;
    id<MTLComputePipelineState> jacobiICBPSO = nil;
    NSMutableDictionary<NSNumber*, id<MTLIndirectCommandBuffer>>* icbCache = nil;
    NSMutableDictionary<NSNumber*, id<MTLBuffer>>* stepBufferCache = nil;
    NSMutableDictionary<NSNumber*, id<MTLBuffer>>* uniformBufferCache = nil;
};

static EighKernels kFloat;
static EighKernels kHalf;
static EighKernels kBFloat;
static id<MTLLibrary> eighLib = nil;
static std::once_flag init_flag;

void load_kernels(id<MTLLibrary> lib, EighKernels& k, NSString* suffix, bool required) {
    NSError* error = nil;
    id<MTLDevice> device = MPSDevice::getInstance()->device();

    k.jacobi = [lib newFunctionWithName:[NSString stringWithFormat:@"jacobi_rotate_kernel_optimized_%@", suffix]];
    k.dot_columns = [lib newFunctionWithName:[NSString stringWithFormat:@"dot_columns_kernel_%@", suffix]];
    
    // Fused Kernels
    k.fused_generic = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_generic_%@", suffix]];
    k.fused_64 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_64_%@", suffix]];
    k.fused_128 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_128_%@", suffix]];
    k.fused_256 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_256_%@", suffix]];
    k.fused_512 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_512_%@", suffix]];
    k.fused_1024 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_1024_%@", suffix]];

    if (required) {
        TORCH_CHECK(k.jacobi && k.dot_columns, "Failed to load required kernels");
        TORCH_CHECK(k.fused_generic, "Failed to load fused_generic kernel");
    }
    
    // Load ICB Kernel
    k.jacobiICB = [lib newFunctionWithName:[NSString stringWithFormat:@"jacobi_rotate_kernel_icb_%@", suffix]];
    if (k.jacobiICB) {
        MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
        desc.computeFunction = k.jacobiICB;
        desc.supportIndirectCommandBuffers = YES;
        k.jacobiICBPSO = [device newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:&error];
        if(!k.jacobiICBPSO) printf("Failed to create jacobiICBPSO: %s\n", [[error localizedDescription] UTF8String]);
        else printf("SUCCESS: Created jacobiICBPSO for %s\n", [suffix UTF8String]);
    }
    
    // Init Caches
    k.icbCache = [NSMutableDictionary new];
    k.stepBufferCache = [NSMutableDictionary new];
    k.uniformBufferCache = [NSMutableDictionary new];
    
    if (k.jacobi) k.jacobiPSO = [device newComputePipelineStateWithFunction:k.jacobi error:&error];
    if (k.dot_columns) k.dotColumnsPSO = [device newComputePipelineStateWithFunction:k.dot_columns error:&error];
    
    if (k.fused_generic) k.fusedGenericPSO = [device newComputePipelineStateWithFunction:k.fused_generic error:&error];
    if (k.fused_64) k.fused64PSO = [device newComputePipelineStateWithFunction:k.fused_64 error:&error];
    if (k.fused_128) k.fused128PSO = [device newComputePipelineStateWithFunction:k.fused_128 error:&error];
    if (k.fused_256) k.fused256PSO = [device newComputePipelineStateWithFunction:k.fused_256 error:&error];
    if (k.fused_512) k.fused512PSO = [device newComputePipelineStateWithFunction:k.fused_512 error:&error];
    if (k.fused_1024) k.fused1024PSO = [device newComputePipelineStateWithFunction:k.fused_1024 error:&error];
    
    if (required) {
        TORCH_CHECK(k.jacobiPSO, "Failed to create jacobiPSO");
        TORCH_CHECK(k.dotColumnsPSO, "Failed to create dotColumnsPSO");
        TORCH_CHECK(k.fusedGenericPSO, "Failed to create fusedGenericPSO");
    }
}

void init_mps_eigh() {
    std::call_once(init_flag, [](){
        id<MTLDevice> device = MPSDevice::getInstance()->device();
        if (!device) TORCH_CHECK(false, "MPS Device not found");
        
        NSError* error = nil;
        NSString* src = nil;
        
        const char* file_path = "native/eigh_kernels.metal"; // Relative to where we run? 
        // We need a robust way to find the kernel file. For now, assume relative or absolute path.
        // Actually, Python package data should handle this, but let's try a fixed path for development
        // or look relative to this source file location if possible?
        // Let's rely on Python passing the path or a hardcoded path for now, simplified.
        // A better approach: Read from absolute path generated by python __file__ equivalent
        
        // FIXME: Valid path handling
        file_path = "/Users/kris/localprojects/metalops/packages/metaleig/native/eigh_kernels.metal";
        
        std::ifstream t(file_path);
        if (t.is_open()) {
            std::stringstream buffer;
            buffer << t.rdbuf();
            std::string content = buffer.str();
            src = [NSString stringWithUTF8String:content.c_str()];
        } else {
             printf("CRITICAL ERROR: Could not load Metal kernels from %s\n", file_path);
             TORCH_CHECK(false, "Metal kernel file missing");
        }

        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion3_1; 
        
        eighLib = [device newLibraryWithSource:src options:options error:&error];
        if (!eighLib) {
             TORCH_CHECK(false, "Failed to compile Metal EIGH library: ", [[error localizedDescription] UTF8String]);
        }
        
        load_kernels(eighLib, kFloat, @"float", true);
        load_kernels(eighLib, kHalf, @"half", true);
        load_kernels(eighLib, kBFloat, @"bfloat", false); 
    });
}

// -----------------------------------------------------------------------------
// Helper: Pairing Strategy (Round Robin)
// -----------------------------------------------------------------------------
std::pair<std::vector<int>, int> generate_ordering(int N) {
    std::vector<int> all_pairs;
    int num_steps = N - 1; 
    std::vector<int> players(N);
    for(int i=0; i<N; ++i) players[i] = i;

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
// EIGH Forward
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> eigh_forward(torch::Tensor A) { 
    TORCH_CHECK(A.device().is_mps(), "Input tensor must be on MPS");
    init_mps_eigh();
    
    EighKernels* kernels = nullptr;
    if (A.scalar_type() == torch::kFloat32) kernels = &kFloat;
    else if (A.scalar_type() == torch::kHalf) kernels = &kHalf;
    else if (A.scalar_type() == torch::kBFloat16) kernels = &kBFloat;
    else TORCH_CHECK(false, "Unsupported dtype.");
    
    if (A.dim() == 2) A = A.unsqueeze(0);
    
    int64_t Batch = A.size(0);
    int64_t M = A.size(1);
    int64_t N = A.size(2);
    
    if (N % 2 != 0) {
        TORCH_CHECK(N % 2 == 0, "N must be even for now (please pad)");
    }

    // 1. Setup V as Identity
    torch::Tensor V = torch::eye(N, A.options()).expand({Batch, N, N}).contiguous();
    
    // 2. Prepare Transposed Data
    torch::Tensor A_T = A.transpose(1, 2).contiguous(); 
    torch::Tensor V_T = V.transpose(1, 2).contiguous(); 
    
    // 3. Scheduler
    auto [pairs_cpu, num_steps] = generate_ordering(N);
    int num_pairs = N / 2;
    int max_threads = 1024;
    int threads_per_pair = std::max(1, max_threads / num_pairs);
    threads_per_pair = std::min(32, threads_per_pair); 
    
    torch::Tensor PairsTens = torch::tensor(pairs_cpu, torch::dtype(torch::kInt32).device(torch::kCPU)).contiguous();
    PairsTens = PairsTens.to(A.device());
    
    MPSStream* stream = getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    
    // Define Uniforms
    uint32_t BatchStrideA = (uint32_t)(N * M);
    uint32_t BatchStrideV = (uint32_t)(N * N);
    uint32_t M_u = (uint32_t)M;
    uint32_t N_u = (uint32_t)N;
    uint32_t NumPairs_u = (uint32_t)num_pairs;
    uint32_t NumSteps_u = (uint32_t)num_steps;
    uint32_t TPP_u = (uint32_t)threads_per_pair;
    
    // Fused Kernel Implementation (Fast Path for N <= 2048)
    // 2048 pairs = 1024. Fits in 1024 threads (max per threadgroup).
    id<MTLComputePipelineState> fusedPSO = nil;
    
    // Exact match for specialized kernels because they hardcode N
    // Exact match for specialized kernels because they hardcode N
    // FIXME: Fused Kernels failed verification (numerical instability & 0.01x speedup).
    // Reverting to stable Iterative dispatch.
    
    if (N == 64 && kernels->fused64PSO) fusedPSO = kernels->fused64PSO;
    else if (N == 128 && kernels->fused128PSO) fusedPSO = kernels->fused128PSO;
    // else if (N == 256 && kernels->fused256PSO) fusedPSO = kernels->fused256PSO;
    // else if (N == 512 && kernels->fused512PSO) fusedPSO = kernels->fused512PSO;
    // else if (N == 1024 && kernels->fused1024PSO) fusedPSO = kernels->fused1024PSO;
    // else if (N <= 2048 && kernels->fusedGenericPSO) fusedPSO = kernels->fusedGenericPSO;
    
    
    // Override: Use iterative if Fused PSO not available or N > 2048
    if (fusedPSO) {
         [encoder setComputePipelineState:fusedPSO];
         mtl_setBuffer(encoder, A_T, 0);
         mtl_setBuffer(encoder, V_T, 1);
         [encoder setBuffer:getMTLBufferStorage(PairsTens) offset:(PairsTens.storage_offset() * PairsTens.element_size()) atIndex:2];
         
         [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:3];
         
         // Helper to decide if specialized
         bool is_specialized = (N == 64 || N == 128 || N == 256 || N == 512 || N == 1024);
         
         if (is_specialized) {
              // Specialized arguments: M (3), BatchStrideA (4), BatchStrideV (5)
              [encoder setBytes:&BatchStrideA length:sizeof(uint32_t) atIndex:4];
              [encoder setBytes:&BatchStrideV length:sizeof(uint32_t) atIndex:5];
              
              // Launch max threads (1024) for specialized as they expect implicit padding/logic?
              // Specialized kernels have `if (pair_idx < NumPairs)` check but read AllPairs?
              // Wait, specialized kernels 64/128/256 use HARDCODED NumPairs/TPP.
              // fused64: NumPairs=32, TPP=32 -> 1024 threads.
              // fused128: NumPairs=64, TPP=16 -> 1024 threads.
              // fused256: NumPairs=128, TPP=8 -> 1024 threads.
              // So we MUST launch 1024 threads.
              [encoder dispatchThreadgroups:MTLSizeMake(1, 1, Batch) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
         } else {
              // Generic arguments:
              uint32_t TPP_Generic = 1; // FORCE 1 to avoid race condition in Generic Kernel (no reduction logic)
              
              [encoder setBytes:&N_u length:sizeof(uint32_t) atIndex:4];
              [encoder setBytes:&NumPairs_u length:sizeof(uint32_t) atIndex:5];
              [encoder setBytes:&NumSteps_u length:sizeof(uint32_t) atIndex:6];
              [encoder setBytes:&TPP_Generic length:sizeof(uint32_t) atIndex:7];
              [encoder setBytes:&BatchStrideA length:sizeof(uint32_t) atIndex:8];
              [encoder setBytes:&BatchStrideV length:sizeof(uint32_t) atIndex:9];
              
              // Generic Safety: Launch EXACT threads to avoid OOB AllPairs read
              // With TPP=1, required_threads = num_pairs.
              int required_threads = num_pairs * TPP_Generic;
              [encoder dispatchThreadgroups:MTLSizeMake(1, 1, Batch) threadsPerThreadgroup:MTLSizeMake(required_threads, 1, 1)];
         }
         
    } else {
        // ICB Optimization
        id<MTLComputePipelineState> icbPSO = kernels->jacobiICBPSO;
        
        NSUInteger total_sweeps = 15;
        NSUInteger steps = (NSUInteger)num_steps;
        NSUInteger total_commands = total_sweeps * steps;
        
        id<MTLIndirectCommandBuffer> icb = [kernels->icbCache objectForKey:@(N)];
        id<MTLBuffer> stepBuffer = [kernels->stepBufferCache objectForKey:@(N)];
        id<MTLBuffer> uniformBuffer = [kernels->uniformBufferCache objectForKey:@(N)];
        
         bool rebuild = (icb == nil);
         // if (rebuild) printf("Rebuilding ICB for N=%d\n", N);
         // else printf("Using Cached ICB for N=%d\n", N);
         
         if (rebuild) {
              NSUInteger stepStride = 256; // 256 bytes per step index (aligned)
              // ... (Use verified logic for steps)
              // Note: Use steps logic from below or ensure consistency
              NSUInteger total_sweeps = 15;
              NSUInteger steps = (NSUInteger)num_steps;
              NSUInteger total_commands = total_sweeps * steps;
              
              MTLIndirectCommandBufferDescriptor* icbDesc = [MTLIndirectCommandBufferDescriptor new];
              icbDesc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
              icbDesc.inheritBuffers = NO;
              icbDesc.inheritPipelineState = NO;
              icbDesc.maxVertexBufferBindCount = 0;
              icbDesc.maxFragmentBufferBindCount = 0;
              icbDesc.maxKernelBufferBindCount = 5; // A, V, Pairs, Uniforms, Step
              
              icb = [encoder.device newIndirectCommandBufferWithDescriptor:icbDesc maxCommandCount:total_commands options:MTLResourceStorageModeShared];
              stepBuffer = [encoder.device newBufferWithLength:total_commands * stepStride options:MTLResourceStorageModeShared];
              
              // Create and Cache Uniform Buffer (Single buffer for all steps)
              ICBUniforms uniforms = {
                 .M = N_u, 
                 .N = N_u,
                 .BatchStrideA = BatchStrideA,
                 .BatchStrideV = BatchStrideV,
                 .NumPairs = (uint)num_pairs,
                 ._pad0 = 0, ._pad1 = 0, ._pad2 = 0
             };
             uniformBuffer = [encoder.device newBufferWithBytes:&uniforms length:sizeof(uniforms) options:MTLResourceStorageModeShared];
              
              [kernels->icbCache setObject:icb forKey:@(N)];
              [kernels->stepBufferCache setObject:stepBuffer forKey:@(N)];
              [kernels->uniformBufferCache setObject:uniformBuffer forKey:@(N)];
              
              NSUInteger threadGroupSize = 1024;
              if (kernels->jacobiICBPSO.maxTotalThreadsPerThreadgroup < 1024) threadGroupSize = kernels->jacobiICBPSO.maxTotalThreadsPerThreadgroup;
              NSUInteger sharedMemSize = ((threadGroupSize + 31) / 32) * 3 * A.element_size(); // Approx logic from kernel

              for (NSUInteger sw = 0; sw < total_sweeps; ++sw) {
                  for (NSUInteger s = 0; s < steps; ++s) {
                      NSUInteger cmd_idx = sw * steps + s;
                      id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:cmd_idx];
                      
                      // Update Step Buffer
                      uint32_t step_val = (uint32_t)s;
                      uint32_t* step_ptr = (uint32_t*)((char*)[stepBuffer contents] + cmd_idx * stepStride);
                      *step_ptr = step_val;
                      
                      [cmd setComputePipelineState:kernels->jacobiICBPSO];
                      // Initial bindings - will be updated if reused
                      [cmd setKernelBuffer:at::native::mps::getMTLBufferStorage(A_T) offset:(A_T.storage_offset() * A_T.element_size()) atIndex:0];
                      [cmd setKernelBuffer:at::native::mps::getMTLBufferStorage(V_T) offset:(V_T.storage_offset() * V_T.element_size()) atIndex:1];
                      [cmd setKernelBuffer:at::native::mps::getMTLBufferStorage(PairsTens) offset:(PairsTens.storage_offset() * PairsTens.element_size()) atIndex:2];
                      [cmd setKernelBuffer:uniformBuffer offset:0 atIndex:3];
                      [cmd setKernelBuffer:stepBuffer offset:(cmd_idx * stepStride) atIndex:4];
                      
                      [cmd setThreadgroupMemoryLength:sharedMemSize atIndex:0];
                      [cmd concurrentDispatchThreadgroups:MTLSizeMake(num_pairs, 1, Batch) threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
                  }
              }
         } else {
              // Update Bindings for Reuse
              NSUInteger total_sweeps = 15;
              NSUInteger steps = (NSUInteger)num_steps;
              NSUInteger total_commands = total_sweeps * steps;
              
              // We MUST iterate and update dynamic buffers (A, V, Pairs)
              for (NSUInteger i = 0; i < total_commands; ++i) {
                   id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:i];
                   [cmd setKernelBuffer:at::native::mps::getMTLBufferStorage(A_T) offset:(A_T.storage_offset() * A_T.element_size()) atIndex:0];
                   [cmd setKernelBuffer:at::native::mps::getMTLBufferStorage(V_T) offset:(V_T.storage_offset() * V_T.element_size()) atIndex:1];
                   [cmd setKernelBuffer:at::native::mps::getMTLBufferStorage(PairsTens) offset:(PairsTens.storage_offset() * PairsTens.element_size()) atIndex:2];
              }
         }
         
         // Bind Encoder State (Inherited) - Not needed if ICB has inheritPipelineState=NO?
         // Actually ICB sets PSO. But do we need to set anything else? No.
         // Barriers etc.
         
         // Execute
         [encoder useResource:icb usage:MTLResourceUsageRead];
         [encoder useResource:stepBuffer usage:MTLResourceUsageRead];
         [encoder useResource:uniformBuffer usage:MTLResourceUsageRead];
        
        // Execute serially to ensure barriers between steps
        for (int i = 0; i < total_commands; ++i) {
             [encoder executeCommandsInBuffer:icb withRange:NSMakeRange(i, 1)];
             // Implicit barrier between compute encoder commands?
             // Usually yes for data dependencies.
             // Adding explicit barrier just in case
             [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers]; 
        }
    }
    
    // 5. Compute Exact Eigenvalues using Dot Product
    // Lambda_k = u_k . v_k (where u_k are cols of A_final, v_k are cols of V_final)
    // Note: A_final = A_initial * V_final
    // Thus A_final's columns are actually U * Sigma.
    // If A was positive definite, A_final = V * Sigma.
    // Generally A v = lambda v.
    // Our Jacobi rotates A to approximate Diagonal.
    // A_T holds the diagonalized matrix (mostly).
    // The diagonal elements of A_T are the eigenvalues if V was close to Identity at start?
    // Wait, One-Sided Jacobi:
    // A -> A V.
    // We update A by A = A J.
    // Converges to A' = U Sigma (orthogonal columns).
    // And V accumulates J.
    // So A_original * V = U * Sigma.
    // For Symmetric A: U = V (up to sign).
    // So A_original * V = V * Sigma * Signs.
    // A' = V * Sigma * Signs.
    // So columns of A' are parallel to columns of V.
    // Lambda_i = (Column i of A') dot (Column i of V).
    
    torch::Tensor Eigenvalues = torch::empty({Batch, N}, A.options());
    
    id<MTLComputePipelineState> dotPSO = kernels->dotColumnsPSO;
    [encoder setComputePipelineState:dotPSO];
    mtl_setBuffer(encoder, A_T, 0); // A_Rotated
    mtl_setBuffer(encoder, V_T, 1); // V_Rotated
    mtl_setBuffer(encoder, Eigenvalues, 2);
    [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&N_u length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&BatchStrideA length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&BatchStrideV length:sizeof(uint32_t) atIndex:6];
    uint32_t BatchStrideE = (uint32_t)N;
    [encoder setBytes:&BatchStrideE length:sizeof(uint32_t) atIndex:7];
    
    int dot_tpg = std::min((int)dotPSO.maxTotalThreadsPerThreadgroup, 256);
    NSUInteger dotSharedMem = ((dot_tpg + 31) / 32) * sizeof(float);
    [encoder setThreadgroupMemoryLength:dotSharedMem atIndex:0];
    
    MTLSize dotGrid = MTLSizeMake(N, 1, Batch); 
    MTLSize dotGroup = MTLSizeMake(dot_tpg, 1, 1);
    
    // We dispatch 1 threadgroup per COLUMN i.
    [encoder dispatchThreadgroups:dotGrid threadsPerThreadgroup:dotGroup];
    
    // 6. Return
    // Return Eigenvalues (Eigenvalues) and Eigenvectors (V_T transposed back)
    return {Eigenvalues, V_T.transpose(1, 2).contiguous()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("eigh_forward", &eigh_forward, "Eigh Forward (Metal)");
}
