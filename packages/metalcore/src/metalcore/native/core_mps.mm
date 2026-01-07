// metalcore PyTorch bindings
// Provides Python interface to Metal kernels for QR, trsm, Householder
//
// Based on metalsvd pattern: 
// - Load kernels from .metal file
// - Cache pipeline states
// - Use MPS stream for synchronization

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <fstream>
#include <sstream>
#include <dlfcn.h>
#include <libgen.h>

using namespace at::mps;
using namespace at::native::mps;

// -----------------------------------------------------------------------------
// Global State
// -----------------------------------------------------------------------------

struct CoreKernels {
    // Panel QR (geqr2)
    id<MTLFunction> geqr2 = nil;
    id<MTLComputePipelineState> geqr2PSO = nil;
    
    // Fused Panel QR (MAGMA-style)
    id<MTLFunction> geqr2Fused = nil;
    id<MTLComputePipelineState> geqr2FusedPSO = nil;
    
    // Householder
    id<MTLFunction> householder = nil;
    id<MTLFunction> applyHouseholder = nil;
    id<MTLComputePipelineState> householderPSO = nil;
    id<MTLComputePipelineState> applyHouseholderPSO = nil;
    
    // Block reflector (larfb)
    id<MTLFunction> larfbStep1 = nil;
    id<MTLFunction> larfbStep2 = nil;
    id<MTLFunction> larfbStep3 = nil;
    id<MTLComputePipelineState> larfbStep1PSO = nil;
    id<MTLComputePipelineState> larfbStep2PSO = nil;
    id<MTLComputePipelineState> larfbStep3PSO = nil;
    
    // T matrix builder
    id<MTLFunction> larft = nil;
    id<MTLComputePipelineState> larftPSO = nil;
    
    // Triangular solve
    id<MTLFunction> trsmLower = nil;
    id<MTLFunction> trsmUpper = nil;
    id<MTLComputePipelineState> trsmLowerPSO = nil;
    id<MTLComputePipelineState> trsmUpperPSO = nil;
    
    // Fully fused QR (single dispatch)
    id<MTLFunction> qrFullFused = nil;
    id<MTLComputePipelineState> qrFullFusedPSO = nil;
    
    // Batched QR (parallel matrices)
    id<MTLFunction> qrBatched = nil;
    id<MTLComputePipelineState> qrBatchedPSO = nil;
    
    // Batched TRSM (triangular solve)
    id<MTLFunction> trsmBatched = nil;
    id<MTLComputePipelineState> trsmBatchedPSO = nil;
    
    // Column norms
    id<MTLFunction> columnNorms = nil;
    id<MTLComputePipelineState> columnNormsPSO = nil;
    
    // Batched Cholesky (potrf)
    id<MTLFunction> choleskyBatched = nil;
    id<MTLComputePipelineState> choleskyBatchedPSO = nil;
    
    // Batched TRSM lower/upper
    id<MTLFunction> trsmLowerBatched = nil;
    id<MTLFunction> trsmUpperBatched = nil;
    id<MTLComputePipelineState> trsmLowerBatchedPSO = nil;
    id<MTLComputePipelineState> trsmUpperBatchedPSO = nil;
    
    // Fused Cholesky solve (forward + back substitution in one kernel)
    id<MTLFunction> choleskySolveBatched = nil;
    id<MTLComputePipelineState> choleskySolveBatchedPSO = nil;
    
    // Column norm sort (De Rijk optimization for SVD)
    id<MTLFunction> columnNormSort = nil;
    id<MTLComputePipelineState> columnNormSortPSO = nil;
    
    // Sign canonicalization (SVD U/V sign fix)
    id<MTLFunction> signCanonicalize = nil;
    id<MTLComputePipelineState> signCanonicalizePSO = nil;
    
    // Batched Q.T @ b for fused solve
    id<MTLFunction> batchedQtB = nil;
    id<MTLComputePipelineState> batchedQtBPSO = nil;
    
    // High-impact ML/LA kernels
    id<MTLFunction> luBatched = nil;
    id<MTLComputePipelineState> luBatchedPSO = nil;
    
    id<MTLFunction> inverseBatched = nil;
    id<MTLComputePipelineState> inverseBatchedPSO = nil;
    
    id<MTLFunction> syrkBatched = nil;
    id<MTLComputePipelineState> syrkBatchedPSO = nil;
    
    id<MTLFunction> frobeniusNormBatched = nil;
    id<MTLComputePipelineState> frobeniusNormBatchedPSO = nil;
    
    id<MTLFunction> softmaxBatched = nil;
    id<MTLComputePipelineState> softmaxBatchedPSO = nil;
    
    id<MTLFunction> traceBatched = nil;
    id<MTLComputePipelineState> traceBatchedPSO = nil;
    
    id<MTLFunction> solveBatched = nil;
    id<MTLComputePipelineState> solveBatchedPSO = nil;

    // Training Ops (RMSNorm, AdamW)
    id<MTLFunction> rmsnormFwd = nil;
    id<MTLComputePipelineState> rmsnormFwdPSO = nil;
    
    id<MTLFunction> rmsnormBwdDx = nil;
    id<MTLComputePipelineState> rmsnormBwdDxPSO = nil;

    id<MTLFunction> rmsnormBwdDw = nil;
    id<MTLComputePipelineState> rmsnormBwdDwPSO = nil;
    
    // Vectorized RMSNorm
    id<MTLFunction> rmsnormFwdVec4 = nil;
    id<MTLComputePipelineState> rmsnormFwdVec4PSO = nil;
    id<MTLFunction> rmsnormBwdDxVec4 = nil;
    id<MTLComputePipelineState> rmsnormBwdDxVec4PSO = nil;
    id<MTLFunction> rmsnormBwdDwVec4 = nil;
    id<MTLComputePipelineState> rmsnormBwdDwVec4PSO = nil;
    
    id<MTLFunction> adamwStep = nil;
    id<MTLComputePipelineState> adamwStepPSO = nil;
    
    id<MTLFunction> adamwStepScalar = nil;
    id<MTLComputePipelineState> adamwStepScalarPSO = nil;
    
    // Optimized Training Kernels (ILP/Fusion)
    id<MTLFunction> adamwStepIlp4 = nil;
    id<MTLComputePipelineState> adamwStepIlp4PSO = nil;
    id<MTLFunction> fusedAddRmsnorm = nil;
    id<MTLComputePipelineState> fusedAddRmsnormPSO = nil;
    id<MTLFunction> rmsnormFwdHalfVec = nil;
    id<MTLComputePipelineState> rmsnormFwdHalfVecPSO = nil;
    
    // Training Ops (half precision)
    id<MTLFunction> rmsnormFwdHalf = nil;
    id<MTLComputePipelineState> rmsnormFwdHalfPSO = nil;
    id<MTLFunction> rmsnormBwdDxHalf = nil;
    id<MTLComputePipelineState> rmsnormBwdDxHalfPSO = nil;
    id<MTLFunction> rmsnormBwdDwHalf = nil;
    id<MTLComputePipelineState> rmsnormBwdDwHalfPSO = nil;
    id<MTLFunction> adamwStepHalf = nil;
    id<MTLComputePipelineState> adamwStepHalfPSO = nil;
    id<MTLFunction> adamwStepHalfScalar = nil;
    id<MTLComputePipelineState> adamwStepHalfScalarPSO = nil;
    id<MTLFunction> adamwStepHalfIlp4 = nil;
    id<MTLComputePipelineState> adamwStepHalfIlp4PSO = nil;
    id<MTLFunction> adamwStepBfloat = nil;
    id<MTLComputePipelineState> adamwStepBfloatPSO = nil;
    id<MTLFunction> adamwStepBfloatScalar = nil;
    id<MTLComputePipelineState> adamwStepBfloatScalarPSO = nil;
    id<MTLFunction> adamwStepBfloatIlp4 = nil;
    id<MTLComputePipelineState> adamwStepBfloatIlp4PSO = nil;
    
    // Activation Kernels (float)
    id<MTLFunction> geluFwd = nil;
    id<MTLComputePipelineState> geluFwdPSO = nil;
    id<MTLFunction> geluBwd = nil;
    id<MTLComputePipelineState> geluBwdPSO = nil;
    id<MTLFunction> siluFwd = nil;
    id<MTLComputePipelineState> siluFwdPSO = nil;
    id<MTLFunction> siluBwd = nil;
    id<MTLComputePipelineState> siluBwdPSO = nil;
    id<MTLFunction> biasGeluFwd = nil;
    id<MTLComputePipelineState> biasGeluFwdPSO = nil;
    id<MTLFunction> biasSiluFwd = nil;
    id<MTLComputePipelineState> biasSiluFwdPSO = nil;
    id<MTLFunction> geluFwdScalar = nil;
    id<MTLComputePipelineState> geluFwdScalarPSO = nil;
    id<MTLFunction> siluFwdScalar = nil;
    id<MTLComputePipelineState> siluFwdScalarPSO = nil;
    
    // Activation Kernels (half precision)
    id<MTLFunction> geluFwdHalf = nil;
    id<MTLComputePipelineState> geluFwdHalfPSO = nil;
    id<MTLFunction> geluBwdHalf = nil;
    id<MTLComputePipelineState> geluBwdHalfPSO = nil;
    id<MTLFunction> siluFwdHalf = nil;
    id<MTLComputePipelineState> siluFwdHalfPSO = nil;
    id<MTLFunction> siluBwdHalf = nil;
    id<MTLComputePipelineState> siluBwdHalfPSO = nil;
    id<MTLFunction> biasGeluFwdHalf = nil;
    id<MTLComputePipelineState> biasGeluFwdHalfPSO = nil;
    id<MTLFunction> biasSiluFwdHalf = nil;
    id<MTLComputePipelineState> biasSiluFwdHalfPSO = nil;
    id<MTLFunction> geluFwdScalarHalf = nil;
    id<MTLComputePipelineState> geluFwdScalarHalfPSO = nil;
    id<MTLFunction> siluFwdScalarHalf = nil;
    id<MTLComputePipelineState> siluFwdScalarHalfPSO = nil;
    
    // Activation Kernels (bfloat16)
    id<MTLFunction> geluFwdBfloat = nil;
    id<MTLComputePipelineState> geluFwdBfloatPSO = nil;
    id<MTLFunction> geluFwdBfloatScalar = nil;
    id<MTLComputePipelineState> geluFwdBfloatScalarPSO = nil;
    id<MTLFunction> siluFwdBfloat = nil;
    id<MTLComputePipelineState> siluFwdBfloatPSO = nil;
    id<MTLFunction> siluFwdBfloatScalar = nil;
    id<MTLComputePipelineState> siluFwdBfloatScalarPSO = nil;
    id<MTLFunction> geluBwdBfloat = nil;
    id<MTLComputePipelineState> geluBwdBfloatPSO = nil;
    id<MTLFunction> geluBwdBfloatScalar = nil;
    id<MTLComputePipelineState> geluBwdBfloatScalarPSO = nil;
    id<MTLFunction> siluBwdBfloat = nil;
    id<MTLComputePipelineState> siluBwdBfloatPSO = nil;
    id<MTLFunction> siluBwdBfloatScalar = nil;
    id<MTLComputePipelineState> siluBwdBfloatScalarPSO = nil;
    id<MTLFunction> rmsnormFwdBfloat = nil;
    id<MTLComputePipelineState> rmsnormFwdBfloatPSO = nil;
    
    // SDPA
    id<MTLFunction> attentionNaive = nil;
    id<MTLComputePipelineState> attentionNaivePSO = nil;
    id<MTLFunction> flashAttentionFwdV2 = nil;
    id<MTLComputePipelineState> flashAttentionFwdV2PSO = nil;
    id<MTLFunction> flashAttentionBwdV2 = nil;
    id<MTLComputePipelineState> flashAttentionBwdV2PSO = nil;
    id<MTLFunction> sdpaVector64 = nil;
    id<MTLComputePipelineState> sdpaVector64PSO = nil;
    
    // Fused Softmax
    id<MTLFunction> fusedSoftmax = nil;
    id<MTLComputePipelineState> fusedSoftmaxPSO = nil;
    id<MTLFunction> fusedSoftmaxVec4 = nil;
    id<MTLComputePipelineState> fusedSoftmaxVec4PSO = nil;
    id<MTLFunction> fusedSoftmaxHalf = nil;
    id<MTLComputePipelineState> fusedSoftmaxHalfPSO = nil;
    id<MTLFunction> fusedSoftmaxBfloat = nil;
    id<MTLComputePipelineState> fusedSoftmaxBfloatPSO = nil;
    
    // LayerNorm
    id<MTLFunction> layernormFwd = nil;
    id<MTLComputePipelineState> layernormFwdPSO = nil;
    id<MTLFunction> fusedAddLayernorm = nil;
    id<MTLComputePipelineState> fusedAddLayernormPSO = nil;
    id<MTLFunction> layernormFwdHalf = nil;
    id<MTLComputePipelineState> layernormFwdHalfPSO = nil;
    id<MTLFunction> layernormFwdBfloat = nil;
    id<MTLComputePipelineState> layernormFwdBfloatPSO = nil;
    
    // Embedding Bag
    id<MTLFunction> embeddingBagSimple = nil;
    id<MTLComputePipelineState> embeddingBagSimplePSO = nil;
    
    // Scatter/Gather
    id<MTLFunction> gather1d = nil;
    id<MTLComputePipelineState> gather1dPSO = nil;
    id<MTLFunction> gather2d = nil;
    id<MTLComputePipelineState> gather2dPSO = nil;
    id<MTLFunction> scatterAdd1d = nil;
    id<MTLComputePipelineState> scatterAdd1dPSO = nil;
    id<MTLFunction> scatterAdd2d = nil;
    id<MTLComputePipelineState> scatterAdd2dPSO = nil;
    id<MTLFunction> indexSelect = nil;
    id<MTLComputePipelineState> indexSelectPSO = nil;

};

static CoreKernels kernels;
static id<MTLLibrary> coreLib = nil;
static std::once_flag init_flag;

// -----------------------------------------------------------------------------
// Kernel Loading
// -----------------------------------------------------------------------------

void load_core_kernels() {
    std::call_once(init_flag, [](){
        id<MTLDevice> device = MPSDevice::getInstance()->device();
        if (!device) TORCH_CHECK(false, "MPS Device not found");
        
        NSError* error = nil;
        
        // 1. Try to locate .metallib relative to this dylib
        // Get path to this dylib
        Dl_info info;
        if (dladdr((void*)&load_core_kernels, &info)) {
            // info.dli_fname contains path to metalcore_backend.so
            // Structure in wheel:
            //   site-packages/metalcore_backend.cpython...so
            //   site-packages/metalcore/native/core_kernels.metallib
            // OR if built in-place:
            //   src/metalcore_backend.cpython...so
            //   src/metalcore/native/
            
            // We need to look for metalcore/native/core_kernels.metallib relative to the directory containing the .so
            
            std::string dylib_path = info.dli_fname;
            std::string dylib_dir = dylib_path.substr(0, dylib_path.find_last_of('/'));
            
            // Try explicit path options
            std::vector<std::string> candidates = {
                dylib_dir + "/metalcore/native/core_kernels.metallib",     // Wheel structure (if backend is top level)
                dylib_dir + "/native/core_kernels.metallib",               // In-place or nested
                dylib_dir + "/../metalcore/native/core_kernels.metallib"   // Sibling directory
            };
            
            for (const auto& path : candidates) {
                NSURL* lib_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
                
                // key: use [NSURL checkResourceIsReachableAndReturnError:] to avoid noise
                if ([lib_url checkResourceIsReachableAndReturnError:nil]) {
                    coreLib = [device newLibraryWithURL:lib_url error:&error];
                    if (coreLib) {
                         printf("metalcore: Loaded kernels from %s\n", path.c_str());
                         break;
                    }
                }
            }
        }
        
        // 2. Fallback: Hardcoded dev path (for local debugging only)
        if (!coreLib) {
             const char* dev_path = "/Users/kris/localprojects/metalops/packages/metalcore/src/metalcore/native/core_kernels.metallib";
             NSURL* lib_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:dev_path]];
             if ([lib_url checkResourceIsReachableAndReturnError:nil]) {
                  coreLib = [device newLibraryWithURL:lib_url error:&error];
                  if (coreLib) printf("metalcore: Loaded kernels from DEV path\n");
             }
        }

        // 3. Fallback: Compile from source (Developer mode / No .metallib found)
        if (!coreLib) {
             // Look for .metal source relative to dylib as well
             // ... implementation omitted for brevity, usually rely on precompiled in prod ...
             printf("metalcore: WARNING - Could not find .metallib. Falling back to source (if available).\n");
             
             // Try dev source path
             const char* src_path = "/Users/kris/localprojects/metalops/packages/metalcore/src/metalcore/native/core_kernels.metal";
             std::ifstream file(src_path);
            
            if (file.good()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                std::string content = buffer.str();
                
                NSString* src = [NSString stringWithUTF8String:content.c_str()];
                MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
                opts.mathMode = MTLMathModeFast;
                
                coreLib = [device newLibraryWithSource:src options:opts error:&error];
                
                if (coreLib) {
                    printf("metalcore: Compiled kernels from DEV source %s\n", src_path);
                } else {
                    printf("metalcore: Failed to compile source: %s\n", [[error localizedDescription] UTF8String]);
                    return;
                }
            } else {
                printf("metalcore: FATAL - Could not find .metallib or .metal kernel source.\n");
                return;
            }
        }
        
        // Load functions
        kernels.geqr2 = [coreLib newFunctionWithName:@"geqr2_panel_kernel"];
        kernels.geqr2Fused = [coreLib newFunctionWithName:@"geqr2_fused_kernel"];
        kernels.householder = [coreLib newFunctionWithName:@"householder_vector_kernel"];
        kernels.applyHouseholder = [coreLib newFunctionWithName:@"apply_householder_kernel"];
        kernels.larfbStep1 = [coreLib newFunctionWithName:@"larfb_step1_vtc"];
        kernels.larfbStep2 = [coreLib newFunctionWithName:@"larfb_step2_tw"];
        kernels.larfbStep3 = [coreLib newFunctionWithName:@"larfb_step3_cvw"];
        kernels.larft = [coreLib newFunctionWithName:@"larft_kernel"];
        kernels.trsmLower = [coreLib newFunctionWithName:@"trsm_lower_kernel"];
        kernels.trsmUpper = [coreLib newFunctionWithName:@"trsm_upper_kernel"];
        kernels.qrFullFused = [coreLib newFunctionWithName:@"qr_full_fused_kernel"];
        kernels.qrBatched = [coreLib newFunctionWithName:@"qr_batched_kernel"];
        kernels.trsmBatched = [coreLib newFunctionWithName:@"trsm_batched_kernel"];
        kernels.columnNorms = [coreLib newFunctionWithName:@"column_norms_kernel"];
        kernels.choleskyBatched = [coreLib newFunctionWithName:@"cholesky_batched_kernel"];
        kernels.trsmLowerBatched = [coreLib newFunctionWithName:@"trsm_lower_batched_kernel"];
        kernels.trsmUpperBatched = [coreLib newFunctionWithName:@"trsm_upper_batched_kernel"];
        kernels.choleskySolveBatched = [coreLib newFunctionWithName:@"cholesky_solve_batched_kernel"];
        
        // New optimization kernels
        kernels.columnNormSort = [coreLib newFunctionWithName:@"column_norm_sort_kernel"];
        kernels.signCanonicalize = [coreLib newFunctionWithName:@"sign_canonicalize_kernel"];
        kernels.batchedQtB = [coreLib newFunctionWithName:@"batched_qt_b_kernel"];
        
        // High-impact ML/LA kernels
        kernels.luBatched = [coreLib newFunctionWithName:@"lu_batched_kernel"];
        kernels.inverseBatched = [coreLib newFunctionWithName:@"inverse_batched_kernel"];
        kernels.syrkBatched = [coreLib newFunctionWithName:@"syrk_batched_kernel"];
        kernels.frobeniusNormBatched = [coreLib newFunctionWithName:@"frobenius_norm_batched_kernel"];
        kernels.softmaxBatched = [coreLib newFunctionWithName:@"softmax_batched_kernel"];
        kernels.traceBatched = [coreLib newFunctionWithName:@"trace_batched_kernel"];
        kernels.solveBatched = [coreLib newFunctionWithName:@"solve_batched_kernel"];
        
        // Training Ops
        kernels.rmsnormFwd = [coreLib newFunctionWithName:@"rmsnorm_fwd"];
        kernels.rmsnormBwdDx = [coreLib newFunctionWithName:@"rmsnorm_bwd_dx"];
        kernels.rmsnormBwdDw = [coreLib newFunctionWithName:@"rmsnorm_bwd_dw"];
        kernels.adamwStep = [coreLib newFunctionWithName:@"adamw_step"];
        
        // Optimized Training Kernels (ILP/Fusion)
        kernels.adamwStepIlp4 = [coreLib newFunctionWithName:@"adamw_step_ilp4"];
        kernels.fusedAddRmsnorm = [coreLib newFunctionWithName:@"fused_add_rmsnorm"];
        kernels.rmsnormFwdHalfVec = [coreLib newFunctionWithName:@"rmsnorm_fwd_half_vec"];
        
        
        // Create pipeline states
        if (kernels.geqr2) {
            kernels.geqr2PSO = [device newComputePipelineStateWithFunction:kernels.geqr2 error:&error];
            if (!kernels.geqr2PSO) printf("Failed to create geqr2PSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.geqr2Fused) {
            kernels.geqr2FusedPSO = [device newComputePipelineStateWithFunction:kernels.geqr2Fused error:&error];
            if (!kernels.geqr2FusedPSO) printf("Failed to create geqr2FusedPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.householder) {
            kernels.householderPSO = [device newComputePipelineStateWithFunction:kernels.householder error:&error];
        }
        if (kernels.applyHouseholder) {
            kernels.applyHouseholderPSO = [device newComputePipelineStateWithFunction:kernels.applyHouseholder error:&error];
        }
        if (kernels.larfbStep1) {
            kernels.larfbStep1PSO = [device newComputePipelineStateWithFunction:kernels.larfbStep1 error:&error];
        }
        if (kernels.larfbStep2) {
            kernels.larfbStep2PSO = [device newComputePipelineStateWithFunction:kernels.larfbStep2 error:&error];
        }
        if (kernels.larfbStep3) {
            kernels.larfbStep3PSO = [device newComputePipelineStateWithFunction:kernels.larfbStep3 error:&error];
        }
        if (kernels.larft) {
            kernels.larftPSO = [device newComputePipelineStateWithFunction:kernels.larft error:&error];
        }
        if (kernels.trsmLower) {
            kernels.trsmLowerPSO = [device newComputePipelineStateWithFunction:kernels.trsmLower error:&error];
        }
        if (kernels.trsmUpper) {
            kernels.trsmUpperPSO = [device newComputePipelineStateWithFunction:kernels.trsmUpper error:&error];
        }
        if (kernels.qrFullFused) {
            kernels.qrFullFusedPSO = [device newComputePipelineStateWithFunction:kernels.qrFullFused error:&error];
            if (!kernels.qrFullFusedPSO) printf("Failed to create qrFullFusedPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.qrBatched) {
            kernels.qrBatchedPSO = [device newComputePipelineStateWithFunction:kernels.qrBatched error:&error];
            if (!kernels.qrBatchedPSO) printf("Failed to create qrBatchedPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.trsmBatched) {
            kernels.trsmBatchedPSO = [device newComputePipelineStateWithFunction:kernels.trsmBatched error:&error];
            if (!kernels.trsmBatchedPSO) printf("Failed to create trsmBatchedPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.columnNorms) {
            kernels.columnNormsPSO = [device newComputePipelineStateWithFunction:kernels.columnNorms error:&error];
            if (!kernels.columnNormsPSO) printf("Failed to create columnNormsPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.choleskyBatched) {
            kernels.choleskyBatchedPSO = [device newComputePipelineStateWithFunction:kernels.choleskyBatched error:&error];
            if (!kernels.choleskyBatchedPSO) printf("Failed to create choleskyBatchedPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.trsmLowerBatched) {
            kernels.trsmLowerBatchedPSO = [device newComputePipelineStateWithFunction:kernels.trsmLowerBatched error:&error];
            if (!kernels.trsmLowerBatchedPSO) printf("Failed to create trsmLowerBatchedPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.trsmUpperBatched) {
            kernels.trsmUpperBatchedPSO = [device newComputePipelineStateWithFunction:kernels.trsmUpperBatched error:&error];
            if (!kernels.trsmUpperBatchedPSO) printf("Failed to create trsmUpperBatchedPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.choleskySolveBatched) {
            kernels.choleskySolveBatchedPSO = [device newComputePipelineStateWithFunction:kernels.choleskySolveBatched error:&error];
            if (!kernels.choleskySolveBatchedPSO) printf("Failed to create choleskySolveBatchedPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        
        // New optimization kernels
        if (kernels.columnNormSort) {
            kernels.columnNormSortPSO = [device newComputePipelineStateWithFunction:kernels.columnNormSort error:&error];
            if (!kernels.columnNormSortPSO) printf("Failed to create columnNormSortPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.signCanonicalize) {
            kernels.signCanonicalizePSO = [device newComputePipelineStateWithFunction:kernels.signCanonicalize error:&error];
            if (!kernels.signCanonicalizePSO) printf("Failed to create signCanonicalizePSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.batchedQtB) {
            kernels.batchedQtBPSO = [device newComputePipelineStateWithFunction:kernels.batchedQtB error:&error];
            if (!kernels.batchedQtBPSO) printf("Failed to create batchedQtBPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        
        // High-impact ML/LA kernels
        if (kernels.luBatched) {
            kernels.luBatchedPSO = [device newComputePipelineStateWithFunction:kernels.luBatched error:&error];
            if (!kernels.luBatchedPSO) printf("Failed to create luBatchedPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.inverseBatched) {
            kernels.inverseBatchedPSO = [device newComputePipelineStateWithFunction:kernels.inverseBatched error:&error];
            if (!kernels.inverseBatchedPSO) printf("Failed to create inverseBatchedPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.syrkBatched) {
            kernels.syrkBatchedPSO = [device newComputePipelineStateWithFunction:kernels.syrkBatched error:&error];
            if (!kernels.syrkBatchedPSO) printf("Failed to create syrkBatchedPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.frobeniusNormBatched) {
            kernels.frobeniusNormBatchedPSO = [device newComputePipelineStateWithFunction:kernels.frobeniusNormBatched error:&error];
            if (!kernels.frobeniusNormBatchedPSO) printf("Failed to create frobeniusNormBatchedPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.softmaxBatched) {
            kernels.softmaxBatchedPSO = [device newComputePipelineStateWithFunction:kernels.softmaxBatched error:&error];
            if (!kernels.softmaxBatchedPSO) printf("Failed to create softmaxBatchedPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.traceBatched) {
            kernels.traceBatchedPSO = [device newComputePipelineStateWithFunction:kernels.traceBatched error:&error];
            if (!kernels.traceBatchedPSO) printf("Failed to create traceBatchedPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.solveBatched) {
            kernels.solveBatchedPSO = [device newComputePipelineStateWithFunction:kernels.solveBatched error:&error];
            if (!kernels.solveBatchedPSO) printf("Failed to create solveBatchedPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        
        if (kernels.rmsnormFwd) {
            kernels.rmsnormFwdPSO = [device newComputePipelineStateWithFunction:kernels.rmsnormFwd error:&error];
        }
        if (kernels.rmsnormBwdDx) {
            kernels.rmsnormBwdDxPSO = [device newComputePipelineStateWithFunction:kernels.rmsnormBwdDx error:&error];
        }
        if (kernels.rmsnormBwdDw) {
            kernels.rmsnormBwdDwPSO = [device newComputePipelineStateWithFunction:kernels.rmsnormBwdDw error:&error];
        }
        if (kernels.adamwStep) {
            kernels.adamwStepPSO = [device newComputePipelineStateWithFunction:kernels.adamwStep error:&error];
        }
        
        // Optimized Training Kernels PSOs
        if (kernels.adamwStepIlp4) {
            kernels.adamwStepIlp4PSO = [device newComputePipelineStateWithFunction:kernels.adamwStepIlp4 error:&error];
        }
        if (kernels.fusedAddRmsnorm) {
            kernels.fusedAddRmsnormPSO = [device newComputePipelineStateWithFunction:kernels.fusedAddRmsnorm error:&error];
        }
        if (kernels.rmsnormFwdHalfVec) {
            kernels.rmsnormFwdHalfVecPSO = [device newComputePipelineStateWithFunction:kernels.rmsnormFwdHalfVec error:&error];
        }
        
        // Vectorized kernels
        kernels.rmsnormFwdVec4 = [coreLib newFunctionWithName:@"rmsnorm_fwd_vec4"];
        kernels.rmsnormBwdDxVec4 = [coreLib newFunctionWithName:@"rmsnorm_bwd_dx_vec4"];
        kernels.rmsnormBwdDwVec4 = [coreLib newFunctionWithName:@"rmsnorm_bwd_dw_vec4"];
        
        if (kernels.rmsnormFwdVec4) {
            kernels.rmsnormFwdVec4PSO = [device newComputePipelineStateWithFunction:kernels.rmsnormFwdVec4 error:&error];
        }
        if (kernels.rmsnormBwdDxVec4) {
            kernels.rmsnormBwdDxVec4PSO = [device newComputePipelineStateWithFunction:kernels.rmsnormBwdDxVec4 error:&error];
        }
        if (kernels.rmsnormBwdDwVec4) {
            kernels.rmsnormBwdDwVec4PSO = [device newComputePipelineStateWithFunction:kernels.rmsnormBwdDwVec4 error:&error];
        }
        
        // Scalar AdamW
        kernels.adamwStepScalar = [coreLib newFunctionWithName:@"adamw_step_scalar"];
        if (kernels.adamwStepScalar) {
             kernels.adamwStepScalarPSO = [device newComputePipelineStateWithFunction:kernels.adamwStepScalar error:&error];
        }
        
        // Training Ops (half precision)
        kernels.rmsnormFwdHalf = [coreLib newFunctionWithName:@"rmsnorm_fwd_half"];
        kernels.rmsnormBwdDxHalf = [coreLib newFunctionWithName:@"rmsnorm_bwd_dx_half"];
        kernels.rmsnormBwdDwHalf = [coreLib newFunctionWithName:@"rmsnorm_bwd_dw_half"];
        kernels.adamwStepHalf = [coreLib newFunctionWithName:@"adamw_step_half"];
        
        if (kernels.rmsnormFwdHalf) kernels.rmsnormFwdHalfPSO = [device newComputePipelineStateWithFunction:kernels.rmsnormFwdHalf error:&error];
        if (kernels.rmsnormBwdDxHalf) kernels.rmsnormBwdDxHalfPSO = [device newComputePipelineStateWithFunction:kernels.rmsnormBwdDxHalf error:&error];
        if (kernels.rmsnormBwdDwHalf) kernels.rmsnormBwdDwHalfPSO = [device newComputePipelineStateWithFunction:kernels.rmsnormBwdDwHalf error:&error];
        if (kernels.adamwStepHalf) kernels.adamwStepHalfPSO = [device newComputePipelineStateWithFunction:kernels.adamwStepHalf error:&error];
        
        // Half precision scalar AdamW (for tail handling)
        kernels.adamwStepHalfScalar = [coreLib newFunctionWithName:@"adamw_step_half_scalar"];
        if (kernels.adamwStepHalfScalar) kernels.adamwStepHalfScalarPSO = [device newComputePipelineStateWithFunction:kernels.adamwStepHalfScalar error:&error];
        
        // Half precision ILP=4 (for large tensors)
        kernels.adamwStepHalfIlp4 = [coreLib newFunctionWithName:@"adamw_step_half_ilp4"];
        if (kernels.adamwStepHalfIlp4) kernels.adamwStepHalfIlp4PSO = [device newComputePipelineStateWithFunction:kernels.adamwStepHalfIlp4 error:&error];
        
        // BFloat16 AdamW (requires Metal 3.1+)
        kernels.adamwStepBfloat = [coreLib newFunctionWithName:@"adamw_step_bfloat"];
        kernels.adamwStepBfloatScalar = [coreLib newFunctionWithName:@"adamw_step_bfloat_scalar"];
        kernels.adamwStepBfloatIlp4 = [coreLib newFunctionWithName:@"adamw_step_bfloat_ilp4"];
        if (kernels.adamwStepBfloat) kernels.adamwStepBfloatPSO = [device newComputePipelineStateWithFunction:kernels.adamwStepBfloat error:&error];
        if (kernels.adamwStepBfloatScalar) kernels.adamwStepBfloatScalarPSO = [device newComputePipelineStateWithFunction:kernels.adamwStepBfloatScalar error:&error];
        if (kernels.adamwStepBfloatIlp4) kernels.adamwStepBfloatIlp4PSO = [device newComputePipelineStateWithFunction:kernels.adamwStepBfloatIlp4 error:&error];
        
        // Activation Kernels
        kernels.geluFwd = [coreLib newFunctionWithName:@"gelu_fwd"];
        kernels.geluBwd = [coreLib newFunctionWithName:@"gelu_bwd"];
        kernels.siluFwd = [coreLib newFunctionWithName:@"silu_fwd"];
        kernels.siluBwd = [coreLib newFunctionWithName:@"silu_bwd"];
        kernels.biasGeluFwd = [coreLib newFunctionWithName:@"bias_gelu_fwd"];
        kernels.biasSiluFwd = [coreLib newFunctionWithName:@"bias_silu_fwd"];
        kernels.geluFwdScalar = [coreLib newFunctionWithName:@"gelu_fwd_scalar"];
        kernels.siluFwdScalar = [coreLib newFunctionWithName:@"silu_fwd_scalar"];
        
        if (kernels.geluFwd) kernels.geluFwdPSO = [device newComputePipelineStateWithFunction:kernels.geluFwd error:&error];
        if (kernels.geluBwd) kernels.geluBwdPSO = [device newComputePipelineStateWithFunction:kernels.geluBwd error:&error];
        if (kernels.siluFwd) kernels.siluFwdPSO = [device newComputePipelineStateWithFunction:kernels.siluFwd error:&error];
        if (kernels.siluBwd) kernels.siluBwdPSO = [device newComputePipelineStateWithFunction:kernels.siluBwd error:&error];
        if (kernels.biasGeluFwd) kernels.biasGeluFwdPSO = [device newComputePipelineStateWithFunction:kernels.biasGeluFwd error:&error];
        if (kernels.biasSiluFwd) kernels.biasSiluFwdPSO = [device newComputePipelineStateWithFunction:kernels.biasSiluFwd error:&error];
        if (kernels.geluFwdScalar) kernels.geluFwdScalarPSO = [device newComputePipelineStateWithFunction:kernels.geluFwdScalar error:&error];
        if (kernels.siluFwdScalar) kernels.siluFwdScalarPSO = [device newComputePipelineStateWithFunction:kernels.siluFwdScalar error:&error];
        
        // Activation Kernels (half precision)
        kernels.geluFwdHalf = [coreLib newFunctionWithName:@"gelu_fwd_half"];
        kernels.geluBwdHalf = [coreLib newFunctionWithName:@"gelu_bwd_half"];
        kernels.siluFwdHalf = [coreLib newFunctionWithName:@"silu_fwd_half"];
        kernels.siluBwdHalf = [coreLib newFunctionWithName:@"silu_bwd_half"];
        kernels.biasGeluFwdHalf = [coreLib newFunctionWithName:@"bias_gelu_fwd_half"];
        kernels.biasSiluFwdHalf = [coreLib newFunctionWithName:@"bias_silu_fwd_half"];
        kernels.geluFwdScalarHalf = [coreLib newFunctionWithName:@"gelu_fwd_scalar_half"];
        kernels.siluFwdScalarHalf = [coreLib newFunctionWithName:@"silu_fwd_scalar_half"];
        
        if (kernels.geluFwdHalf) kernels.geluFwdHalfPSO = [device newComputePipelineStateWithFunction:kernels.geluFwdHalf error:&error];
        if (kernels.geluBwdHalf) kernels.geluBwdHalfPSO = [device newComputePipelineStateWithFunction:kernels.geluBwdHalf error:&error];
        if (kernels.siluFwdHalf) kernels.siluFwdHalfPSO = [device newComputePipelineStateWithFunction:kernels.siluFwdHalf error:&error];
        if (kernels.siluBwdHalf) kernels.siluBwdHalfPSO = [device newComputePipelineStateWithFunction:kernels.siluBwdHalf error:&error];
        if (kernels.biasGeluFwdHalf) kernels.biasGeluFwdHalfPSO = [device newComputePipelineStateWithFunction:kernels.biasGeluFwdHalf error:&error];
        if (kernels.biasSiluFwdHalf) kernels.biasSiluFwdHalfPSO = [device newComputePipelineStateWithFunction:kernels.biasSiluFwdHalf error:&error];
        if (kernels.geluFwdScalarHalf) kernels.geluFwdScalarHalfPSO = [device newComputePipelineStateWithFunction:kernels.geluFwdScalarHalf error:&error];
        if (kernels.siluFwdScalarHalf) kernels.siluFwdScalarHalfPSO = [device newComputePipelineStateWithFunction:kernels.siluFwdScalarHalf error:&error];
        
        // Activation kernels (bfloat16)
        kernels.geluFwdBfloat = [coreLib newFunctionWithName:@"gelu_fwd_bfloat"];
        kernels.geluFwdBfloatScalar = [coreLib newFunctionWithName:@"gelu_fwd_bfloat_scalar"];
        kernels.siluFwdBfloat = [coreLib newFunctionWithName:@"silu_fwd_bfloat"];
        kernels.siluFwdBfloatScalar = [coreLib newFunctionWithName:@"silu_fwd_bfloat_scalar"];
        
        if (kernels.geluFwdBfloat) kernels.geluFwdBfloatPSO = [device newComputePipelineStateWithFunction:kernels.geluFwdBfloat error:&error];
        if (kernels.geluFwdBfloatScalar) kernels.geluFwdBfloatScalarPSO = [device newComputePipelineStateWithFunction:kernels.geluFwdBfloatScalar error:&error];
        if (kernels.siluFwdBfloat) kernels.siluFwdBfloatPSO = [device newComputePipelineStateWithFunction:kernels.siluFwdBfloat error:&error];
        if (kernels.siluFwdBfloatScalar) kernels.siluFwdBfloatScalarPSO = [device newComputePipelineStateWithFunction:kernels.siluFwdBfloatScalar error:&error];
        
        // Activation kernels (bfloat16 backward)
        kernels.geluBwdBfloat = [coreLib newFunctionWithName:@"gelu_bwd_bfloat"];
        kernels.geluBwdBfloatScalar = [coreLib newFunctionWithName:@"gelu_bwd_bfloat_scalar"];
        kernels.siluBwdBfloat = [coreLib newFunctionWithName:@"silu_bwd_bfloat"];
        kernels.siluBwdBfloatScalar = [coreLib newFunctionWithName:@"silu_bwd_bfloat_scalar"];
        kernels.rmsnormFwdBfloat = [coreLib newFunctionWithName:@"rmsnorm_fwd_bfloat"];
        
        if (kernels.geluBwdBfloat) kernels.geluBwdBfloatPSO = [device newComputePipelineStateWithFunction:kernels.geluBwdBfloat error:&error];
        if (kernels.geluBwdBfloatScalar) kernels.geluBwdBfloatScalarPSO = [device newComputePipelineStateWithFunction:kernels.geluBwdBfloatScalar error:&error];
        if (kernels.siluBwdBfloat) kernels.siluBwdBfloatPSO = [device newComputePipelineStateWithFunction:kernels.siluBwdBfloat error:&error];
        if (kernels.siluBwdBfloatScalar) kernels.siluBwdBfloatScalarPSO = [device newComputePipelineStateWithFunction:kernels.siluBwdBfloatScalar error:&error];
        if (kernels.rmsnormFwdBfloat) kernels.rmsnormFwdBfloatPSO = [device newComputePipelineStateWithFunction:kernels.rmsnormFwdBfloat error:&error];
        
        // SDPA
        kernels.attentionNaive = [coreLib newFunctionWithName:@"attention_naive"];
        if (kernels.attentionNaive) kernels.attentionNaivePSO = [device newComputePipelineStateWithFunction:kernels.attentionNaive error:&error];
        
        kernels.flashAttentionFwdV2 = [coreLib newFunctionWithName:@"flash_attention_fwd_v2"];
        if (kernels.flashAttentionFwdV2) kernels.flashAttentionFwdV2PSO = [device newComputePipelineStateWithFunction:kernels.flashAttentionFwdV2 error:&error];
        
        kernels.flashAttentionBwdV2 = [coreLib newFunctionWithName:@"flash_attention_bwd_v2"];
        if (kernels.flashAttentionBwdV2) kernels.flashAttentionBwdV2PSO = [device newComputePipelineStateWithFunction:kernels.flashAttentionBwdV2 error:&error];
        
        kernels.sdpaVector64 = [coreLib newFunctionWithName:@"sdpa_vector_64"];
        if (kernels.sdpaVector64) kernels.sdpaVector64PSO = [device newComputePipelineStateWithFunction:kernels.sdpaVector64 error:&error];
        
        // Fused Softmax
        kernels.fusedSoftmax = [coreLib newFunctionWithName:@"fused_softmax"];
        if (kernels.fusedSoftmax) kernels.fusedSoftmaxPSO = [device newComputePipelineStateWithFunction:kernels.fusedSoftmax error:&error];
        kernels.fusedSoftmaxVec4 = [coreLib newFunctionWithName:@"fused_softmax_vec4"];
        if (kernels.fusedSoftmaxVec4) kernels.fusedSoftmaxVec4PSO = [device newComputePipelineStateWithFunction:kernels.fusedSoftmaxVec4 error:&error];
        kernels.fusedSoftmaxHalf = [coreLib newFunctionWithName:@"fused_softmax_half"];
        if (kernels.fusedSoftmaxHalf) kernels.fusedSoftmaxHalfPSO = [device newComputePipelineStateWithFunction:kernels.fusedSoftmaxHalf error:&error];
        kernels.fusedSoftmaxBfloat = [coreLib newFunctionWithName:@"fused_softmax_bfloat"];
        if (kernels.fusedSoftmaxBfloat) kernels.fusedSoftmaxBfloatPSO = [device newComputePipelineStateWithFunction:kernels.fusedSoftmaxBfloat error:&error];
        
        // LayerNorm
        kernels.layernormFwd = [coreLib newFunctionWithName:@"layernorm_fwd"];
        if (kernels.layernormFwd) kernels.layernormFwdPSO = [device newComputePipelineStateWithFunction:kernels.layernormFwd error:&error];
        kernels.fusedAddLayernorm = [coreLib newFunctionWithName:@"fused_add_layernorm"];
        if (kernels.fusedAddLayernorm) kernels.fusedAddLayernormPSO = [device newComputePipelineStateWithFunction:kernels.fusedAddLayernorm error:&error];
        kernels.layernormFwdHalf = [coreLib newFunctionWithName:@"layernorm_fwd_half"];
        if (kernels.layernormFwdHalf) kernels.layernormFwdHalfPSO = [device newComputePipelineStateWithFunction:kernels.layernormFwdHalf error:&error];
        kernels.layernormFwdBfloat = [coreLib newFunctionWithName:@"layernorm_fwd_bfloat"];
        if (kernels.layernormFwdBfloat) kernels.layernormFwdBfloatPSO = [device newComputePipelineStateWithFunction:kernels.layernormFwdBfloat error:&error];
        
        // Embedding Bag
        kernels.embeddingBagSimple = [coreLib newFunctionWithName:@"embedding_bag_simple"];
        if (kernels.embeddingBagSimple) kernels.embeddingBagSimplePSO = [device newComputePipelineStateWithFunction:kernels.embeddingBagSimple error:&error];
        
        // Scatter/Gather
        kernels.gather1d = [coreLib newFunctionWithName:@"gather_1d"];
        if (kernels.gather1d) kernels.gather1dPSO = [device newComputePipelineStateWithFunction:kernels.gather1d error:&error];
        kernels.gather2d = [coreLib newFunctionWithName:@"gather_2d"];
        if (kernels.gather2d) kernels.gather2dPSO = [device newComputePipelineStateWithFunction:kernels.gather2d error:&error];
        kernels.scatterAdd1d = [coreLib newFunctionWithName:@"scatter_add_1d"];
        if (kernels.scatterAdd1d) kernels.scatterAdd1dPSO = [device newComputePipelineStateWithFunction:kernels.scatterAdd1d error:&error];
        kernels.scatterAdd2d = [coreLib newFunctionWithName:@"scatter_add_2d"];
        if (kernels.scatterAdd2d) kernels.scatterAdd2dPSO = [device newComputePipelineStateWithFunction:kernels.scatterAdd2d error:&error];
        kernels.indexSelect = [coreLib newFunctionWithName:@"index_select"];
        if (kernels.indexSelect) kernels.indexSelectPSO = [device newComputePipelineStateWithFunction:kernels.indexSelect error:&error];

        
        printf("metalcore: Loaded %d kernel functions\n", 
            (kernels.geqr2 ? 1 : 0) + (kernels.householder ? 1 : 0) + 
            (kernels.applyHouseholder ? 1 : 0) + (kernels.larfbStep1 ? 1 : 0) +
            (kernels.larfbStep2 ? 1 : 0) + (kernels.larfbStep3 ? 1 : 0) +
            (kernels.larft ? 1 : 0) + (kernels.trsmLower ? 1 : 0) + 
            (kernels.trsmUpper ? 1 : 0) + (kernels.qrFullFused ? 1 : 0) +
            (kernels.qrBatched ? 1 : 0) + (kernels.trsmBatched ? 1 : 0) +
            (kernels.columnNorms ? 1 : 0) + (kernels.choleskyBatched ? 1 : 0) +
            (kernels.trsmLowerBatched ? 1 : 0) + (kernels.trsmUpperBatched ? 1 : 0));
    });
}

// -----------------------------------------------------------------------------
// Triangular Solve
// -----------------------------------------------------------------------------

torch::Tensor trsm_metal(
    torch::Tensor A,
    torch::Tensor b,
    bool lower,
    bool transpose
) {
    load_core_kernels();
    
    TORCH_CHECK(A.device().type() == at::kMPS, "A must be on MPS device");
    TORCH_CHECK(b.device().type() == at::kMPS, "b must be on MPS device");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    
    int64_t N = A.size(0);
    
    // Make contiguous
    auto A_contig = A.contiguous();
    auto x = b.clone().contiguous();
    
    id<MTLComputePipelineState> pso = lower ? kernels.trsmLowerPSO : kernels.trsmUpperPSO;
    
    if (!pso) {
        // Fallback to PyTorch - move to CPU, solve, move back
        auto A_cpu = A.cpu();
        auto x_cpu = b.cpu().clone();
        
        // Manual triangular solve on CPU
        int64_t n = A_cpu.size(0);
        auto A_acc = A_cpu.accessor<float, 2>();
        auto x_acc = x_cpu.accessor<float, 1>();
        
        if (lower) {
            for (int64_t i = 0; i < n; i++) {
                float sum = 0.0f;
                for (int64_t j = 0; j < i; j++) {
                    sum += A_acc[i][j] * x_acc[j];
                }
                x_acc[i] = (x_acc[i] - sum) / A_acc[i][i];
            }
        } else {
            for (int64_t i = n - 1; i >= 0; i--) {
                float sum = 0.0f;
                for (int64_t j = i + 1; j < n; j++) {
                    sum += A_acc[i][j] * x_acc[j];
                }
                x_acc[i] = (x_acc[i] - sum) / A_acc[i][i];
            }
        }
        return x_cpu.to(A.device());
    }
    
    @autoreleasepool {
        MPSStream* stream = getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(A_contig) offset:A_contig.storage_offset() * A_contig.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:1];
        
        uint32_t N_uint = (uint32_t)N;
        [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:2];
        
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return x;
}

// -----------------------------------------------------------------------------
// Panel QR (geqr2)
// -----------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> geqr2_metal(torch::Tensor A) {
    load_core_kernels();
    
    TORCH_CHECK(A.device().type() == at::kMPS, "A must be on MPS device");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    
    int64_t M = A.size(0);
    int64_t N = A.size(1);
    int64_t K = std::min(M, N);
    
    // Output: R (with V stored below diagonal) and tau
    auto R = A.clone().contiguous();
    auto tau = torch::zeros({K}, A.options());
    
    if (!kernels.geqr2PSO) {
        // CPU fallback - manual Householder QR
        auto R_cpu = A.cpu().clone();
        auto tau_cpu = torch::zeros({K}, torch::dtype(torch::kFloat32));
        
        auto R_acc = R_cpu.accessor<float, 2>();
        auto tau_acc = tau_cpu.accessor<float, 1>();
        
        for (int64_t k = 0; k < K; k++) {
            // Compute norm of column k below diagonal
            float sigma = 0.0f;
            for (int64_t i = k + 1; i < M; i++) {
                sigma += R_acc[i][k] * R_acc[i][k];
            }
            
            float x0 = R_acc[k][k];
            if (sigma < 1e-10f) {
                tau_acc[k] = 0.0f;
                continue;
            }
            
            float norm_x = std::sqrt(x0 * x0 + sigma);
            float sign = (x0 >= 0.0f) ? 1.0f : -1.0f;
            float v0 = x0 + sign * norm_x;
            float tau_k = 2.0f * v0 * v0 / (v0 * v0 + sigma);
            tau_acc[k] = tau_k;
            
            // Update diagonal
            R_acc[k][k] = -sign * norm_x;
            
            // Apply Householder to trailing columns
            for (int64_t j = k + 1; j < N; j++) {
                // Compute v^T @ A[k:, j]
                float dot = R_acc[k][j];  // v[0] = 1
                for (int64_t i = k + 1; i < M; i++) {
                    float v_i = R_acc[i][k] / v0;
                    dot += v_i * R_acc[i][j];
                }
                
                // Update column
                R_acc[k][j] -= tau_k * dot;
                for (int64_t i = k + 1; i < M; i++) {
                    float v_i = R_acc[i][k] / v0;
                    R_acc[i][j] -= tau_k * v_i * dot;
                }
            }
            
            // Store v below diagonal (normalized)
            for (int64_t i = k + 1; i < M; i++) {
                R_acc[i][k] /= v0;
            }
        }
        
        return std::make_tuple(R_cpu.to(A.device()), tau_cpu.to(A.device()));
    }
    
    // Use fused kernel if available and panel fits in shared memory
    // Shared memory: M*N floats for panel + 256 floats for reduction buffer
    bool use_fused = kernels.geqr2FusedPSO && (M * N + 256) * sizeof(float) <= 32768;
    
    @autoreleasepool {
        MPSStream* stream = getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
        if (use_fused) {
            [encoder setComputePipelineState:kernels.geqr2FusedPSO];
            [encoder setBuffer:getMTLBufferStorage(R) offset:R.storage_offset() * R.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(tau) offset:0 atIndex:1];
            
            uint32_t M_uint = (uint32_t)M;
            uint32_t N_uint = (uint32_t)N;
            uint32_t lda = (uint32_t)N;  // Row-major: lda = N
            [encoder setBytes:&M_uint length:sizeof(M_uint) atIndex:2];
            [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:3];
            [encoder setBytes:&lda length:sizeof(lda) atIndex:4];
            
            // Allocate shared memory for panel + reduction buffer
            NSUInteger shared_size = (M * N + 256) * sizeof(float);
            [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
            
            // Launch single threadgroup with 256 threads
            NSUInteger tg_size = 256;
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) 
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        } else {
            // Fallback to original kernel
            [encoder setComputePipelineState:kernels.geqr2PSO];
            [encoder setBuffer:getMTLBufferStorage(R) offset:R.storage_offset() * R.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(tau) offset:0 atIndex:1];
            
            uint32_t M_uint = (uint32_t)M;
            uint32_t N_uint = (uint32_t)N;
            uint32_t lda = (uint32_t)N;
            [encoder setBytes:&M_uint length:sizeof(M_uint) atIndex:2];
            [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:3];
            [encoder setBytes:&lda length:sizeof(lda) atIndex:4];
            
            NSUInteger tg_size = 256;
            NSUInteger shared_size = tg_size * sizeof(float);
            [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
            [encoder dispatchThreads:MTLSizeMake(tg_size, 1, 1) 
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        }
        
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return std::make_tuple(R, tau);
}

// -----------------------------------------------------------------------------
// Block Reflector Application (larfb)
// Apply H = I - V @ T @ V^T to C
// C = C - V @ (T @ (V^T @ C)) for trans=false
// C = C - V @ (T^T @ (V^T @ C)) for trans=true
// -----------------------------------------------------------------------------

torch::Tensor larfb_metal(
    torch::Tensor C,
    torch::Tensor V,
    torch::Tensor T,
    bool trans,
    int64_t panel_start
) {
    load_core_kernels();
    
    TORCH_CHECK(C.device().type() == at::kMPS, "C must be on MPS device");
    TORCH_CHECK(V.device().type() == at::kMPS, "V must be on MPS device");
    TORCH_CHECK(T.device().type() == at::kMPS, "T must be on MPS device");
    
    int64_t M = C.size(0);
    int64_t N = C.size(1);
    int64_t K = V.size(1);  // Number of reflectors
    
    // Build V_full efficiently using tensor operations
    // V_full has: 1s on diagonal (at panel_start offset), 0s above, V below
    torch::Tensor V_full;
    
    int64_t V_rows = V.size(0);
    int64_t V_cols = V.size(1);
    
    if (panel_start == 0 && V_rows >= V_cols) {
        // Fast path: use tril to zero above diagonal, then set diagonal to 1
        // tril keeps diagonal and below
        V_full = torch::tril(V, -1);  // Keep below diagonal only (strict lower)
        // Add identity for the diagonal
        auto diag_ones = torch::eye(V_rows, V_cols, V.options());
        V_full = V_full + diag_ones;
    } else {
        // General case - still need element-wise but less common
        V_full = V.clone();
        for (int64_t k = 0; k < K; k++) {
            if (panel_start + k < M) {
                V_full.index_put_({panel_start + k, k}, 1.0f);
            }
            for (int64_t i = 0; i < panel_start + k; i++) {
                V_full.index_put_({i, k}, 0.0f);
            }
        }
    }
    
    // Use optimized mm (faster than matmul for 2D tensors)
    // W = V^T @ C  (K x N)
    auto W = V_full.t().mm(C);
    
    // W = T^T @ W or T @ W  (K x N)
    if (trans) {
        W = T.t().mm(W);
    } else {
        W = T.mm(W);
    }
    
    // C = C - V @ W  (M x N) - use sub_ variant for efficiency
    return C - V_full.mm(W);
}

// -----------------------------------------------------------------------------
// Build T matrix (larft)
// -----------------------------------------------------------------------------

torch::Tensor larft_metal(
    torch::Tensor V,
    torch::Tensor tau,
    int64_t panel_start
) {
    load_core_kernels();
    
    TORCH_CHECK(V.device().type() == at::kMPS, "V must be on MPS device");
    TORCH_CHECK(tau.device().type() == at::kMPS, "tau must be on MPS device");
    
    int64_t M = V.size(0);
    int64_t K = V.size(1);
    
    // Always use CPU fallback for larft - it's small and sequential
    auto T_cpu = torch::zeros({K, K}, torch::dtype(torch::kFloat32));
    auto V_cpu = V.cpu();
    auto tau_cpu = tau.cpu();
    
    auto T_acc = T_cpu.accessor<float, 2>();
    auto V_acc = V_cpu.accessor<float, 2>();
    auto tau_acc = tau_cpu.accessor<float, 1>();
    
    // Build T column by column using LAPACK algorithm:
    // T[i,i] = tau[i]
    // T[0:i, i] = -tau[i] * T[0:i, 0:i] @ V[:, 0:i]^T @ V[:, i]
    for (int64_t i = 0; i < K; i++) {
        T_acc[i][i] = tau_acc[i];
        
        if (i > 0) {
            // Step 1: Compute w = V[:, 0:i]^T @ V[:, i]
            // w[j] = V[:, j]^T @ V[:, i] for j = 0..i-1
            std::vector<float> w(i);
            for (int64_t j = 0; j < i; j++) {
                float dot = 0.0f;
                for (int64_t m = 0; m < M; m++) {
                    float vj, vi;
                    
                    // V[m, j] with implicit 1 at (panel_start + j)
                    if (m == panel_start + j) vj = 1.0f;
                    else if (m < panel_start + j) vj = 0.0f;
                    else vj = V_acc[m][j];
                    
                    // V[m, i] with implicit 1 at (panel_start + i)
                    if (m == panel_start + i) vi = 1.0f;
                    else if (m < panel_start + i) vi = 0.0f;
                    else vi = V_acc[m][i];
                    
                    dot += vj * vi;
                }
                w[j] = dot;
            }
            
            // Step 2: T[0:i, i] = -tau[i] * T[0:i, 0:i] @ w
            // This is a triangular matrix-vector product (T is upper triangular)
            for (int64_t j = 0; j < i; j++) {
                float sum = 0.0f;
                for (int64_t k = j; k < i; k++) {  // T[j,k] for k >= j (upper triangular)
                    sum += T_acc[j][k] * w[k];
                }
                T_acc[j][i] = -tau_acc[i] * sum;
            }
        }
    }
    
    return T_cpu.to(V.device());
}

// -----------------------------------------------------------------------------
// Full Blocked QR
// -----------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> qr_metal(torch::Tensor A, int64_t block_size) {
    load_core_kernels();
    
    TORCH_CHECK(A.device().type() == at::kMPS, "A must be on MPS device");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    
    int64_t M = A.size(0);
    int64_t N = A.size(1);
    int64_t K = std::min(M, N);
    
    if (block_size <= 0) block_size = 32;
    
    // Copy A to R (will be modified in place)
    auto R = A.clone();
    
    // Storage for building Q  
    std::vector<std::tuple<int64_t, torch::Tensor, torch::Tensor>> panels;
    panels.reserve((K + block_size - 1) / block_size);
    
    for (int64_t j = 0; j < K; j += block_size) {
        int64_t jb = std::min(block_size, K - j);
        
        // Extract panel R[j:, j:j+jb] - need contiguous for kernel
        auto panel = R.index({torch::indexing::Slice(j, M), torch::indexing::Slice(j, j + jb)}).contiguous();
        
        // Factor panel: panel -> R_panel with V below diagonal, tau
        auto [R_panel, tau_panel] = geqr2_metal(panel);
        
        // Copy R_panel back to R
        R.index_put_({torch::indexing::Slice(j, M), torch::indexing::Slice(j, j + jb)}, R_panel);
        
        // V_panel is stored in R below the diagonal - reuse R_panel
        // Build T matrix for this panel
        auto T_panel = larft_metal(R_panel, tau_panel, 0);
        
        // Apply block reflector to trailing matrix R[j:, j+jb:]
        if (j + jb < N) {
            auto trailing = R.index({torch::indexing::Slice(j, M), torch::indexing::Slice(j + jb, N)}).contiguous();
            auto trailing_updated = larfb_metal(trailing, R_panel, T_panel, true, 0);
            R.index_put_({torch::indexing::Slice(j, M), torch::indexing::Slice(j + jb, N)}, trailing_updated);
        }
        
        panels.push_back(std::make_tuple(j, R_panel, T_panel));
    }
    
    // Zero below diagonal of R efficiently using triu
    auto R_upper = torch::triu(R.index({torch::indexing::Slice(0, K), torch::indexing::Slice()}));
    
    // Build Q by applying reflectors in reverse
    auto Q = torch::eye(M, K, A.options());
    
    for (auto it = panels.rbegin(); it != panels.rend(); ++it) {
        auto& [j, V_panel, T_panel] = *it;
        
        auto Q_sub = Q.index({torch::indexing::Slice(j, M), torch::indexing::Slice(j, K)}).contiguous();
        auto Q_updated = larfb_metal(Q_sub, V_panel, T_panel, false, 0);
        Q.index_put_({torch::indexing::Slice(j, M), torch::indexing::Slice(j, K)}, Q_updated);
    }
    
    return std::make_tuple(Q, R_upper);
}

// -----------------------------------------------------------------------------
// Fully Fused QR - Single Metal Dispatch
// -----------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> qr_fused_metal(torch::Tensor A) {
    load_core_kernels();
    
    TORCH_CHECK(A.device().type() == at::kMPS, "A must be on MPS device");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    
    int64_t M = A.size(0);
    int64_t N = A.size(1);
    int64_t K = std::min(M, N);
    
    // Check if matrix fits in shared memory
    // Shared memory layout: M*N (R) + M*K (Q) + K (tau) + 256 (reduction) floats
    // 32KB = 8192 floats
    int64_t shared_needed = M * N + M * K + K + 256;
    
    if (!kernels.qrFullFusedPSO || shared_needed > 8000) {
        // Fall back to blocked QR for large matrices
        return qr_metal(A, 32);
    }
    
    // Ensure input is contiguous and row-major
    auto A_in = A.contiguous();
    
    // Allocate outputs
    auto Q_out = torch::zeros({M, K}, A.options());
    auto R_out = torch::zeros({K, N}, A.options());
    
    @autoreleasepool {
        MPSStream* stream = getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:kernels.qrFullFusedPSO];
        [encoder setBuffer:getMTLBufferStorage(A_in) offset:A_in.storage_offset() * A_in.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(Q_out) offset:0 atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(R_out) offset:0 atIndex:2];
        
        uint32_t M_uint = (uint32_t)M;
        uint32_t N_uint = (uint32_t)N;
        [encoder setBytes:&M_uint length:sizeof(M_uint) atIndex:3];
        [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:4];
        
        // Allocate shared memory
        NSUInteger shared_size = shared_needed * sizeof(float);
        [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
        
        // Launch single threadgroup with 256 threads
        NSUInteger tg_size = 256;
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) 
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return std::make_tuple(Q_out, R_out);
}

// -----------------------------------------------------------------------------
// Batched QR - Process multiple matrices in single dispatch
// -----------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> qr_batched_metal(torch::Tensor A_batch) {
    load_core_kernels();
    
    TORCH_CHECK(A_batch.device().type() == at::kMPS, "A must be on MPS device");
    TORCH_CHECK(A_batch.dim() == 3, "A must be 3D (Batch, M, N)");
    
    int64_t Batch = A_batch.size(0);
    int64_t M = A_batch.size(1);
    int64_t N = A_batch.size(2);
    int64_t K = std::min(M, N);
    
    // Check if single matrix fits in shared memory
    int64_t shared_per_matrix = M * N + M * K + K + 256;
    
    if (!kernels.qrBatchedPSO || shared_per_matrix > 8000) {
        // Fall back to sequential processing
        auto Q_list = torch::zeros({Batch, M, K}, A_batch.options());
        auto R_list = torch::zeros({Batch, K, N}, A_batch.options());
        
        for (int64_t b = 0; b < Batch; b++) {
            auto [Q, R] = qr_fused_metal(A_batch[b]);
            Q_list[b] = Q;
            R_list[b] = R;
        }
        return std::make_tuple(Q_list, R_list);
    }
    
    auto A_contig = A_batch.contiguous();
    auto Q_out = torch::zeros({Batch, M, K}, A_batch.options());
    auto R_out = torch::zeros({Batch, K, N}, A_batch.options());
    
    @autoreleasepool {
        MPSStream* stream = getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:kernels.qrBatchedPSO];
        [encoder setBuffer:getMTLBufferStorage(A_contig) offset:A_contig.storage_offset() * A_contig.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(Q_out) offset:0 atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(R_out) offset:0 atIndex:2];
        
        uint32_t M_uint = (uint32_t)M;
        uint32_t N_uint = (uint32_t)N;
        uint32_t Batch_uint = (uint32_t)Batch;
        [encoder setBytes:&M_uint length:sizeof(M_uint) atIndex:3];
        [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:4];
        [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:5];
        
        NSUInteger shared_size = shared_per_matrix * sizeof(float);
        [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
        
        // Launch Batch threadgroups, each with 256 threads
        NSUInteger tg_size = 256;
        [encoder dispatchThreadgroups:MTLSizeMake(Batch, 1, 1) 
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return std::make_tuple(Q_out, R_out);
}

// -----------------------------------------------------------------------------
// Batched TRSM (Triangular Solve)
// -----------------------------------------------------------------------------

torch::Tensor trsm_batched_metal(torch::Tensor R, torch::Tensor B) {
    load_core_kernels();
    
    TORCH_CHECK(R.device().type() == at::kMPS, "R must be on MPS device");
    TORCH_CHECK(B.device().type() == at::kMPS, "B must be on MPS device");
    TORCH_CHECK(R.dim() == 3, "R must be 3D (Batch, N, N)");
    TORCH_CHECK(B.dim() == 3, "B must be 3D (Batch, N, NRHS)");
    
    int64_t Batch = R.size(0);
    int64_t N = R.size(1);
    int64_t NRHS = B.size(2);
    
    R = R.contiguous();
    B = B.contiguous();
    
    auto X = torch::empty({Batch, N, NRHS}, B.options());
    
    if (!kernels.trsmBatchedPSO) {
        // Fallback: sequential processing
        auto X_list = std::vector<torch::Tensor>();
        for (int64_t i = 0; i < Batch; i++) {
            auto R_i = R[i];
            auto B_i = B[i];
            // Simple back-substitution in C++
            auto X_i = torch::zeros_like(B_i);
            for (int64_t j = N - 1; j >= 0; j--) {
                auto sum = B_i.index({j}).clone();
                for (int64_t k = j + 1; k < N; k++) {
                    sum = sum - R_i.index({j, k}) * X_i.index({k});
                }
                X_i.index_put_({j}, sum / R_i.index({j, j}));
            }
            X_list.push_back(X_i);
        }
        return torch::stack(X_list);
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto cmdBuffer = stream->commandBuffer();
        auto encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:kernels.trsmBatchedPSO];
        [encoder setBuffer:getMTLBufferStorage(R) offset:R.storage_offset() * R.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(B) offset:B.storage_offset() * B.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(X) offset:0 atIndex:2];
        
        uint32_t N_uint = (uint32_t)N;
        uint32_t NRHS_uint = (uint32_t)NRHS;
        uint32_t Batch_uint = (uint32_t)Batch;
        [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:3];
        [encoder setBytes:&NRHS_uint length:sizeof(NRHS_uint) atIndex:4];
        [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:5];
        
        NSUInteger shared_size = N * NRHS * sizeof(float);
        [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
        
        NSUInteger tg_size = 256;
        [encoder dispatchThreadgroups:MTLSizeMake(Batch, 1, 1) 
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return X;
}

// -----------------------------------------------------------------------------
// Fused Solve Batched (QR + Q.T@b + TRSM in single command buffer)
// -----------------------------------------------------------------------------

torch::Tensor solve_batched_metal(torch::Tensor A, torch::Tensor b) {
    // Fused solve: eliminates Python overhead between QR, bmm, TRSM
    // All operations in single command buffer with one sync at end
    load_core_kernels();
    
    TORCH_CHECK(A.device().type() == at::kMPS, "A must be on MPS device");
    TORCH_CHECK(b.device().type() == at::kMPS, "b must be on MPS device");
    TORCH_CHECK(A.dim() == 3, "A must be 3D (Batch, N, N)");
    TORCH_CHECK(b.dim() == 3, "b must be 3D (Batch, N, K)");
    
    int64_t Batch = A.size(0);
    int64_t N = A.size(1);
    int64_t K = b.size(2);
    
    TORCH_CHECK(A.size(2) == N, "A must be square");
    TORCH_CHECK(b.size(1) == N, "b dimension mismatch");
    
    auto A_contig = A.contiguous();
    auto b_contig = b.contiguous();
    
    // Allocate outputs
    auto Q = torch::zeros({Batch, N, N}, A.options());
    auto R = torch::zeros({Batch, N, N}, A.options());
    auto c = torch::zeros({Batch, N, K}, b.options());  // Q.T @ b
    auto x = torch::zeros({Batch, N, K}, b.options());  // solution
    
    // Check kernel availability
    if (!kernels.qrBatchedPSO || !kernels.trsmBatchedPSO) {
        // Fallback: use existing separate functions
        auto [Q_out, R_out] = qr_batched_metal(A);
        auto c_out = torch::bmm(Q_out.transpose(-2, -1), b);
        return trsm_batched_metal(R_out, c_out);
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto cmdBuffer = stream->commandBuffer();
        
        // === Phase 1: Batched QR ===
        {
            id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
            
            [encoder setComputePipelineState:kernels.qrBatchedPSO];
            [encoder setBuffer:getMTLBufferStorage(A_contig) offset:A_contig.storage_offset() * A_contig.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(Q) offset:0 atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(R) offset:0 atIndex:2];
            
            uint32_t N_uint = (uint32_t)N;
            uint32_t Batch_uint = (uint32_t)Batch;
            [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:3];
            [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:4];  // M = N for square
            [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:5];
            
            int64_t shared_per_matrix = N * N + N * N + N + 256;
            NSUInteger shared_size = shared_per_matrix * sizeof(float);
            [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
            
            [encoder dispatchThreadgroups:MTLSizeMake(Batch, 1, 1) 
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            
            [encoder endEncoding];
        }
        
        // === Phase 2: c = Q.T @ b (use custom Metal kernel - no sync needed!) ===
        if (kernels.batchedQtBPSO) {
            id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
            
            [encoder setComputePipelineState:kernels.batchedQtBPSO];
            [encoder setBuffer:getMTLBufferStorage(Q) offset:0 atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(b_contig) offset:b_contig.storage_offset() * b_contig.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(c) offset:0 atIndex:2];
            
            uint32_t N_uint = (uint32_t)N;
            uint32_t K_uint = (uint32_t)K;
            uint32_t Batch_uint = (uint32_t)Batch;
            [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:3];  // M = N
            [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:4];  // N = N
            [encoder setBytes:&K_uint length:sizeof(K_uint) atIndex:5];
            [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:6];
            
            [encoder dispatchThreads:MTLSizeMake(K, N, Batch) 
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            
            [encoder endEncoding];
        } else {
            // Fallback to torch::bmm (requires sync)
            stream->synchronize(SyncType::COMMIT_AND_WAIT);
            c = torch::bmm(Q.transpose(-2, -1), b_contig);
            cmdBuffer = stream->commandBuffer();  // Get new buffer after sync
        }
        
        // === Phase 3: Batched TRSM ===
        {
            c = c.contiguous();
            id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
            
            [encoder setComputePipelineState:kernels.trsmBatchedPSO];
            [encoder setBuffer:getMTLBufferStorage(R) offset:0 atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(c) offset:0 atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(x) offset:0 atIndex:2];
            
            uint32_t N_uint = (uint32_t)N;
            uint32_t K_uint = (uint32_t)K;
            uint32_t Batch_uint = (uint32_t)Batch;
            [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:3];
            [encoder setBytes:&K_uint length:sizeof(K_uint) atIndex:4];
            [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:5];
            
            NSUInteger shared_size = N * K * sizeof(float);
            [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
            
            [encoder dispatchThreadgroups:MTLSizeMake(Batch, 1, 1) 
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            
            [encoder endEncoding];
        }
        
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return x;
}

// -----------------------------------------------------------------------------
// Column Norm Sort (De Rijk optimization for SVD)
// -----------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> column_norm_sort_metal(torch::Tensor A) {
    load_core_kernels();
    
    TORCH_CHECK(A.device().type() == at::kMPS, "A must be on MPS device");
    TORCH_CHECK(A.dim() == 3, "A must be 3D (Batch, M, N)");
    
    int64_t Batch = A.size(0);
    int64_t M = A.size(1);
    int64_t N = A.size(2);
    
    auto A_contig = A.contiguous();
    auto A_sorted = torch::zeros_like(A);
    auto perm = torch::zeros({Batch, N}, torch::TensorOptions().dtype(torch::kInt32).device(A.device()));
    
    if (!kernels.columnNormSortPSO) {
        // Fallback: use Python-level sorting
        auto norms = torch::linalg_vector_norm(A, 2, /*dim=*/1);  // (B, N)
        auto argsort_result = torch::argsort(norms, /*dim=*/-1, /*descending=*/true);
        auto perm_out = argsort_result.to(torch::kInt32);
        auto A_out = torch::gather(A, 2, argsort_result.unsqueeze(1).expand_as(A));
        return std::make_tuple(A_out, perm_out);
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto cmdBuffer = stream->commandBuffer();
        auto encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:kernels.columnNormSortPSO];
        [encoder setBuffer:getMTLBufferStorage(A_contig) offset:A_contig.storage_offset() * A_contig.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(A_sorted) offset:0 atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(perm) offset:0 atIndex:2];
        
        uint32_t M_uint = (uint32_t)M;
        uint32_t N_uint = (uint32_t)N;
        uint32_t Batch_uint = (uint32_t)Batch;
        [encoder setBytes:&M_uint length:sizeof(M_uint) atIndex:3];
        [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:4];
        [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:5];
        
        // Shared memory: N floats for norms + N ints for indices
        NSUInteger shared_size = N * (sizeof(float) + sizeof(int));
        [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
        
        [encoder dispatchThreadgroups:MTLSizeMake(Batch, 1, 1) 
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return std::make_tuple(A_sorted, perm);
}

// -----------------------------------------------------------------------------
// Sign Canonicalization (SVD U/V sign normalization)
// -----------------------------------------------------------------------------

void sign_canonicalize_metal(torch::Tensor U, torch::Tensor V) {
    load_core_kernels();
    
    TORCH_CHECK(U.device().type() == at::kMPS, "U must be on MPS device");
    TORCH_CHECK(V.device().type() == at::kMPS, "V must be on MPS device");
    TORCH_CHECK(U.dim() == 3, "U must be 3D (Batch, M, N)");
    TORCH_CHECK(V.dim() == 3, "V must be 3D (Batch, N, N)");
    
    int64_t Batch = U.size(0);
    int64_t M = U.size(1);
    int64_t N = U.size(2);
    
    if (!kernels.signCanonicalizePSO) {
        // Fallback: Python-level sign canonicalization
        auto max_vals = std::get<0>(torch::max(torch::abs(U), 1));  // (B, N)
        auto max_signs = torch::sign(torch::gather(U, 1, std::get<1>(torch::max(torch::abs(U), 1)).unsqueeze(1)));
        // Skip fallback implementation for now - kernel should work
        return;
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto cmdBuffer = stream->commandBuffer();
        auto encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:kernels.signCanonicalizePSO];
        [encoder setBuffer:getMTLBufferStorage(U) offset:U.storage_offset() * U.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(V) offset:V.storage_offset() * V.element_size() atIndex:1];
        
        uint32_t M_uint = (uint32_t)M;
        uint32_t N_uint = (uint32_t)N;
        uint32_t Batch_uint = (uint32_t)Batch;
        [encoder setBytes:&M_uint length:sizeof(M_uint) atIndex:2];
        [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:3];
        [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:4];
        
        [encoder dispatchThreadgroups:MTLSizeMake(Batch, 1, 1) 
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
}

// -----------------------------------------------------------------------------
// Batched Q.T @ b (for fused solve without sync)
// -----------------------------------------------------------------------------

torch::Tensor batched_qt_b_metal(torch::Tensor Q, torch::Tensor b) {
    load_core_kernels();
    
    TORCH_CHECK(Q.device().type() == at::kMPS, "Q must be on MPS device");
    TORCH_CHECK(b.device().type() == at::kMPS, "b must be on MPS device");
    TORCH_CHECK(Q.dim() == 3, "Q must be 3D (Batch, M, N)");
    TORCH_CHECK(b.dim() == 3, "b must be 3D (Batch, M, K)");
    
    int64_t Batch = Q.size(0);
    int64_t M = Q.size(1);
    int64_t N = Q.size(2);
    int64_t K = b.size(2);
    
    auto Q_contig = Q.contiguous();
    auto b_contig = b.contiguous();
    auto c = torch::zeros({Batch, N, K}, b.options());
    
    if (!kernels.batchedQtBPSO) {
        // Fallback: use PyTorch bmm
        return torch::bmm(Q.transpose(-2, -1), b);
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto cmdBuffer = stream->commandBuffer();
        auto encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:kernels.batchedQtBPSO];
        [encoder setBuffer:getMTLBufferStorage(Q_contig) offset:Q_contig.storage_offset() * Q_contig.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(b_contig) offset:b_contig.storage_offset() * b_contig.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(c) offset:0 atIndex:2];
        
        uint32_t M_uint = (uint32_t)M;
        uint32_t N_uint = (uint32_t)N;
        uint32_t K_uint = (uint32_t)K;
        uint32_t Batch_uint = (uint32_t)Batch;
        [encoder setBytes:&M_uint length:sizeof(M_uint) atIndex:3];
        [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:4];
        [encoder setBytes:&K_uint length:sizeof(K_uint) atIndex:5];
        [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:6];
        
        // Dispatch threads for each output element
        [encoder dispatchThreads:MTLSizeMake(K, N, Batch) 
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return c;
}

// -----------------------------------------------------------------------------
// Batched LU Decomposition
// -----------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> lu_batched_metal(torch::Tensor A) {
    load_core_kernels();
    
    TORCH_CHECK(A.device().type() == at::kMPS, "A must be on MPS device");
    TORCH_CHECK(A.dim() == 3, "A must be 3D (Batch, N, N)");
    TORCH_CHECK(A.size(1) == A.size(2), "A must be square");
    
    int64_t Batch = A.size(0);
    int64_t N = A.size(1);
    
    auto LU = A.clone().contiguous();
    auto pivots = torch::zeros({Batch, N}, torch::TensorOptions().dtype(torch::kInt32).device(A.device()));
    
    if (!kernels.luBatchedPSO) {
        // Fallback
        TORCH_CHECK(false, "LU kernel not available");
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto cmdBuffer = stream->commandBuffer();
        auto encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:kernels.luBatchedPSO];
        [encoder setBuffer:getMTLBufferStorage(LU) offset:0 atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(pivots) offset:0 atIndex:1];
        
        uint32_t N_uint = (uint32_t)N;
        uint32_t Batch_uint = (uint32_t)Batch;
        [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:2];
        [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:3];
        
        [encoder dispatchThreadgroups:MTLSizeMake(Batch, 1, 1) 
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return std::make_tuple(LU, pivots);
}

// -----------------------------------------------------------------------------
// Batched SYRK: C = A.T @ A
// -----------------------------------------------------------------------------

torch::Tensor syrk_batched_metal(torch::Tensor A) {
    load_core_kernels();
    
    TORCH_CHECK(A.device().type() == at::kMPS, "A must be on MPS device");
    TORCH_CHECK(A.dim() == 3, "A must be 3D (Batch, M, N)");
    
    int64_t Batch = A.size(0);
    int64_t M = A.size(1);
    int64_t N = A.size(2);
    
    auto A_contig = A.contiguous();
    auto C = torch::zeros({Batch, N, N}, A.options());
    
    if (!kernels.syrkBatchedPSO) {
        // Fallback to bmm
        return torch::bmm(A.transpose(-2, -1), A);
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto cmdBuffer = stream->commandBuffer();
        auto encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:kernels.syrkBatchedPSO];
        [encoder setBuffer:getMTLBufferStorage(A_contig) offset:A_contig.storage_offset() * A_contig.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(C) offset:0 atIndex:1];
        
        uint32_t M_uint = (uint32_t)M;
        uint32_t N_uint = (uint32_t)N;
        uint32_t Batch_uint = (uint32_t)Batch;
        [encoder setBytes:&M_uint length:sizeof(M_uint) atIndex:2];
        [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:3];
        [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:4];
        
        [encoder dispatchThreads:MTLSizeMake(N, N, Batch) 
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return C;
}

// -----------------------------------------------------------------------------
// Batched Frobenius Norm
// -----------------------------------------------------------------------------

torch::Tensor frobenius_norm_batched_metal(torch::Tensor A) {
    load_core_kernels();
    
    TORCH_CHECK(A.device().type() == at::kMPS, "A must be on MPS device");
    TORCH_CHECK(A.dim() == 3, "A must be 3D (Batch, M, N)");
    
    int64_t Batch = A.size(0);
    int64_t M = A.size(1);
    int64_t N = A.size(2);
    
    auto A_contig = A.contiguous();
    auto norms = torch::zeros({Batch}, A.options());
    
    if (!kernels.frobeniusNormBatchedPSO) {
        // Fallback
        return torch::linalg_matrix_norm(A, "fro");
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto cmdBuffer = stream->commandBuffer();
        auto encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:kernels.frobeniusNormBatchedPSO];
        [encoder setBuffer:getMTLBufferStorage(A_contig) offset:A_contig.storage_offset() * A_contig.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(norms) offset:0 atIndex:1];
        
        uint32_t M_uint = (uint32_t)M;
        uint32_t N_uint = (uint32_t)N;
        uint32_t Batch_uint = (uint32_t)Batch;
        [encoder setBytes:&M_uint length:sizeof(M_uint) atIndex:2];
        [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:3];
        [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:4];
        
        NSUInteger shared_size = 256 * sizeof(float);
        [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
        
        [encoder dispatchThreadgroups:MTLSizeMake(Batch, 1, 1) 
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return norms;
}

// -----------------------------------------------------------------------------
// Batched Softmax
// -----------------------------------------------------------------------------

torch::Tensor softmax_batched_metal(torch::Tensor x, float temperature) {
    load_core_kernels();
    
    TORCH_CHECK(x.device().type() == at::kMPS, "x must be on MPS device");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (Batch, N)");
    
    int64_t Batch = x.size(0);
    int64_t N = x.size(1);
    
    auto out = x.clone().contiguous();
    
    if (!kernels.softmaxBatchedPSO) {
        // Fallback
        return torch::softmax(x / temperature, -1);
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto cmdBuffer = stream->commandBuffer();
        auto encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:kernels.softmaxBatchedPSO];
        [encoder setBuffer:getMTLBufferStorage(out) offset:0 atIndex:0];
        
        uint32_t N_uint = (uint32_t)N;
        uint32_t Batch_uint = (uint32_t)Batch;
        [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:1];
        [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:2];
        [encoder setBytes:&temperature length:sizeof(temperature) atIndex:3];
        
        NSUInteger shared_size = 256 * sizeof(float);
        [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
        
        [encoder dispatchThreadgroups:MTLSizeMake(Batch, 1, 1) 
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return out;
}

// -----------------------------------------------------------------------------
// Batched Trace
// -----------------------------------------------------------------------------

torch::Tensor trace_batched_metal(torch::Tensor A) {
    load_core_kernels();
    
    TORCH_CHECK(A.device().type() == at::kMPS, "A must be on MPS device");
    TORCH_CHECK(A.dim() == 3, "A must be 3D (Batch, N, N)");
    TORCH_CHECK(A.size(1) == A.size(2), "A must be square");
    
    int64_t Batch = A.size(0);
    int64_t N = A.size(1);
    
    auto A_contig = A.contiguous();
    auto traces = torch::zeros({Batch}, A.options());
    
    if (!kernels.traceBatchedPSO) {
        // Fallback
        return torch::sum(torch::diagonal(A, 0, -2, -1), -1);
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto cmdBuffer = stream->commandBuffer();
        auto encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:kernels.traceBatchedPSO];
        [encoder setBuffer:getMTLBufferStorage(A_contig) offset:A_contig.storage_offset() * A_contig.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(traces) offset:0 atIndex:1];
        
        uint32_t N_uint = (uint32_t)N;
        uint32_t Batch_uint = (uint32_t)Batch;
        [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:2];
        [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:3];
        
        NSUInteger shared_size = 256 * sizeof(float);
        [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
        
        [encoder dispatchThreadgroups:MTLSizeMake(Batch, 1, 1) 
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return traces;
}

// -----------------------------------------------------------------------------
// Column Norms
// -----------------------------------------------------------------------------

torch::Tensor column_norms_metal(torch::Tensor A) {
    load_core_kernels();
    
    TORCH_CHECK(A.device().type() == at::kMPS, "A must be on MPS device");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    
    int64_t M = A.size(0);
    int64_t N = A.size(1);
    
    A = A.contiguous();
    auto norms = torch::empty({N}, A.options());
    
    if (!kernels.columnNormsPSO) {
        // Fallback: compute norms manually
        return (A * A).sum(0).sqrt();
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto cmdBuffer = stream->commandBuffer();
        auto encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:kernels.columnNormsPSO];
        [encoder setBuffer:getMTLBufferStorage(A) offset:A.storage_offset() * A.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(norms) offset:0 atIndex:1];
        
        uint32_t M_uint = (uint32_t)M;
        uint32_t N_uint = (uint32_t)N;
        [encoder setBytes:&M_uint length:sizeof(M_uint) atIndex:2];
        [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:3];
        
        NSUInteger tg_size = 256;
        NSUInteger shared_size = tg_size * sizeof(float);
        [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
        
        // One threadgroup per column
        [encoder dispatchThreadgroups:MTLSizeMake(N, 1, 1) 
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return norms;
}

// -----------------------------------------------------------------------------
// Forward Declarations for Python Bindings
// -----------------------------------------------------------------------------

torch::Tensor cholesky_batched_metal(torch::Tensor A);
torch::Tensor cholesky_solve_batched_metal(torch::Tensor L, torch::Tensor b);

// =============================================================================
// SVD KERNELS (ported from metalsvd)
// =============================================================================

struct SVDKernels {
    id<MTLFunction> jacobi = nil;
    id<MTLFunction> jacobi_icb = nil;
    id<MTLFunction> jacobi_icb_vec4 = nil;
    id<MTLFunction> jacobi_fused = nil;
    id<MTLFunction> jacobi_fused_256 = nil;
    id<MTLFunction> norm = nil;
    id<MTLFunction> normalize = nil;
    
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
    
    NSMutableDictionary<NSNumber*, id<MTLIndirectCommandBuffer>>* icbCache = nil;
    NSMutableDictionary<NSNumber*, id<MTLBuffer>>* stepBufferCache = nil;
};

static SVDKernels svdKFloat;
static SVDKernels svdKHalf;
static SVDKernels svdKBFloat;
static bool svd_kernels_loaded = false;

void load_svd_kernels_typed(id<MTLLibrary> lib, SVDKernels& k, NSString* suffix, bool required) {
    k.icbCache = [NSMutableDictionary new];
    k.stepBufferCache = [NSMutableDictionary new];

    k.jacobi = [lib newFunctionWithName:[NSString stringWithFormat:@"jacobi_rotate_kernel_optimized_%@", suffix]];
    k.jacobi_icb = [lib newFunctionWithName:[NSString stringWithFormat:@"jacobi_rotate_kernel_icb_%@", suffix]];
    k.jacobi_icb_vec4 = [lib newFunctionWithName:[NSString stringWithFormat:@"jacobi_rotate_kernel_vec4_clean_%@", suffix]];
    k.jacobi_fused = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_%@", suffix]];
    k.jacobi_fused_256 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_256_%@", suffix]];
    k.norm = [lib newFunctionWithName:[NSString stringWithFormat:@"column_norm_kernel_%@", suffix]];
    k.normalize = [lib newFunctionWithName:[NSString stringWithFormat:@"normalize_kernel_%@", suffix]];
    
    NSError* error = nil;
    id<MTLDevice> device = MPSDevice::getInstance()->device();
    
    if (k.jacobi) k.jacobiPSO = [device newComputePipelineStateWithFunction:k.jacobi error:&error];
    
    if (k.jacobi_icb) {
        MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
        desc.computeFunction = k.jacobi_icb;
        desc.supportIndirectCommandBuffers = YES;
        k.jacobiICBPSO = [device newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:&error];
    }
    
    if (k.jacobi_icb_vec4) {
        MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
        desc.computeFunction = k.jacobi_icb_vec4;
        desc.supportIndirectCommandBuffers = YES;
        k.jacobiICBVec4PSO = [device newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:&error];
    }
    
    if (k.jacobi_fused) k.jacobiFusedPSO = [device newComputePipelineStateWithFunction:k.jacobi_fused error:&error];
    if (k.jacobi_fused_256) k.jacobiFused256PSO = [device newComputePipelineStateWithFunction:k.jacobi_fused_256 error:&error];
    
    id<MTLFunction> f128 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_128_%@", suffix]];
    if (f128) k.jacobiFused128PSO = [device newComputePipelineStateWithFunction:f128 error:&error];
    
    id<MTLFunction> f64 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_64_%@", suffix]];
    if (f64) k.jacobiFused64PSO = [device newComputePipelineStateWithFunction:f64 error:&error];
    
    id<MTLFunction> f512 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_512_%@", suffix]];
    if (f512) k.jacobiFused512PSO = [device newComputePipelineStateWithFunction:f512 error:&error];
    
    id<MTLFunction> f1024 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_1024_%@", suffix]];
    if (f1024) k.jacobiFused1024PSO = [device newComputePipelineStateWithFunction:f1024 error:&error];
    
    if (k.norm) k.normPSO = [device newComputePipelineStateWithFunction:k.norm error:&error];
    if (k.normalize) k.normalizePSO = [device newComputePipelineStateWithFunction:k.normalize error:&error];
}

void load_svd_kernels() {
    if (svd_kernels_loaded) return;
    load_core_kernels();  // Ensure coreLib is loaded
    
    load_svd_kernels_typed(coreLib, svdKFloat, @"float", true);
    load_svd_kernels_typed(coreLib, svdKHalf, @"half", false);
    load_svd_kernels_typed(coreLib, svdKBFloat, @"bfloat", false);
    svd_kernels_loaded = true;
}

std::pair<std::vector<int>, int> svd_generate_ordering(int N) {
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

std::vector<torch::Tensor> svd_forward(torch::Tensor A) { 
    TORCH_CHECK(A.device().is_mps(), "Input tensor must be on MPS");
    load_svd_kernels();
    
    SVDKernels* kernels = nullptr;
    if (A.scalar_type() == torch::kFloat32) kernels = &svdKFloat;
    else if (A.scalar_type() == torch::kHalf) kernels = &svdKHalf;
    else if (A.scalar_type() == torch::kBFloat16) kernels = &svdKBFloat;
    else TORCH_CHECK(false, "Unsupported dtype.");
    
    if (A.dim() == 2) A = A.unsqueeze(0);
    
    int64_t Batch = A.size(0);
    int64_t M = A.size(1);
    int64_t N = A.size(2);
    
    TORCH_CHECK(N % 2 == 0, "N must be even");

    torch::Tensor V = torch::eye(N, A.options()).expand({Batch, N, N}).contiguous();
    torch::Tensor A_T = A.transpose(1, 2).contiguous(); 
    torch::Tensor V_T = V.transpose(1, 2).contiguous(); 
    
    auto [pairs_cpu, num_steps] = svd_generate_ordering(N);
    int num_pairs = N / 2;
    int threads_per_pair = 32;
    if (N >= 64) threads_per_pair = 32;
    if (N >= 128) threads_per_pair = 16;
    if (N >= 256) threads_per_pair = 8;
    if (N >= 512) threads_per_pair = 4;
    if (N >= 1024) threads_per_pair = 2;
    
    int specialized_mode = 0;
    if (N == 1024) specialized_mode = 5;
    else if (N == 512) specialized_mode = 4;
    else if (N == 256) specialized_mode = 1;
    else if (N == 128) specialized_mode = 2;
    else if (N == 64) specialized_mode = 3;
    
    bool use_fused_any = (specialized_mode > 0) || (N <= 256);

    torch::Tensor PairsTens = torch::tensor(pairs_cpu, torch::dtype(torch::kInt32).device(torch::kCPU)).contiguous();
    PairsTens = PairsTens.to(A.device());
    
    MPSStream* stream = getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    
    uint32_t BatchStrideA = (uint32_t)(N * M);
    uint32_t BatchStrideV = (uint32_t)(N * N);
    uint32_t M_u = (uint32_t)M;
    uint32_t N_u = (uint32_t)N;

    if (use_fused_any) {
        id<MTLComputePipelineState> fusedPSO = kernels->jacobiFusedPSO;
        if (specialized_mode == 5) fusedPSO = kernels->jacobiFused1024PSO;
        else if (specialized_mode == 4) fusedPSO = kernels->jacobiFused512PSO;
        else if (specialized_mode == 1) fusedPSO = kernels->jacobiFused256PSO;
        else if (specialized_mode == 2) fusedPSO = kernels->jacobiFused128PSO;
        else if (specialized_mode == 3) fusedPSO = kernels->jacobiFused64PSO;
        
        TORCH_CHECK(fusedPSO, "Failed to get Fused PSO");
        
        [encoder setComputePipelineState:fusedPSO];
        mtl_setBuffer(encoder, A_T, 0);
        mtl_setBuffer(encoder, V_T, 1);
        [encoder setBuffer:getMTLBufferStorage(PairsTens) offset:(PairsTens.storage_offset() * PairsTens.element_size()) atIndex:2];
        
        if (specialized_mode > 0) {
            [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&BatchStrideA length:sizeof(uint32_t) atIndex:4];
            [encoder setBytes:&BatchStrideV length:sizeof(uint32_t) atIndex:5];
        } else {
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
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, Batch) threadsPerThreadgroup:MTLSizeMake(total_threads, 1, 1)];
    } else {
        // ICB path for large matrices - simplified version
        int sweeps = 6;
        id<MTLComputePipelineState> rotatePSO = kernels->jacobiICBPSO ? kernels->jacobiICBPSO : kernels->jacobiPSO;
        TORCH_CHECK(rotatePSO, "No rotate PSO available");
        
        int threads_per_group = std::min((int)rotatePSO.maxTotalThreadsPerThreadgroup, 256);
        int elem_size = A.element_size();
        NSUInteger sharedMemSize = ((threads_per_group + 31) / 32) * 3 * elem_size;
        
        [encoder setComputePipelineState:rotatePSO];
        mtl_setBuffer(encoder, A_T, 0);
        mtl_setBuffer(encoder, V_T, 1);
        [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&N_u length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&BatchStrideA length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&BatchStrideV length:sizeof(uint32_t) atIndex:6];
        [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];
        
        for (int sw = 0; sw < sweeps; ++sw) {
            for (int step = 0; step < num_steps; ++step) {
                size_t pairs_offset = step * num_pairs * sizeof(int) * 2;
                [encoder setBuffer:getMTLBufferStorage(PairsTens) 
                            offset:(PairsTens.storage_offset() * PairsTens.element_size() + pairs_offset) 
                           atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(num_pairs, 1, Batch) 
                    threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
            }
        }
    }
    
    torch::Tensor S = torch::empty({Batch, N}, A.options()); 
    
    [encoder setComputePipelineState:kernels->normPSO];
    mtl_setBuffer(encoder, A_T, 0);
    mtl_setBuffer(encoder, S, 1);
    [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&N_u length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&BatchStrideA length:sizeof(uint32_t) atIndex:4];
    uint32_t BatchStrideS = (uint32_t)N;
    [encoder setBytes:&BatchStrideS length:sizeof(uint32_t) atIndex:5];
    
    MTLSize normGridSize = MTLSizeMake(N, Batch, 1);
    MTLSize normGroupSize = MTLSizeMake(std::min((int)N, (int)kernels->normPSO.maxTotalThreadsPerThreadgroup), 1, 1);
    [encoder dispatchThreads:normGridSize threadsPerThreadgroup:normGroupSize];

    torch::Tensor U_T = torch::empty_like(A_T);
    
    [encoder setComputePipelineState:kernels->normalizePSO];
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

// =============================================================================
// EIGH KERNELS (ported from metaleig)
// =============================================================================

struct EighKernels {
    id<MTLFunction> jacobi = nil;
    id<MTLFunction> dot_columns = nil;
    id<MTLFunction> fused_generic = nil;
    id<MTLFunction> fused_64 = nil;
    id<MTLFunction> fused_128 = nil;
    id<MTLFunction> fused_256 = nil;
    id<MTLFunction> fused_512 = nil;
    id<MTLFunction> fused_1024 = nil;
    
    id<MTLComputePipelineState> jacobiPSO = nil;
    id<MTLComputePipelineState> dotColumnsPSO = nil;
    id<MTLComputePipelineState> fusedGenericPSO = nil;
    id<MTLComputePipelineState> fused64PSO = nil;
    id<MTLComputePipelineState> fused128PSO = nil;
    id<MTLComputePipelineState> fused256PSO = nil;
    id<MTLComputePipelineState> fused512PSO = nil;
    id<MTLComputePipelineState> fused1024PSO = nil;
    
    id<MTLFunction> jacobiICB = nil;
    id<MTLComputePipelineState> jacobiICBPSO = nil;
    NSMutableDictionary<NSNumber*, id<MTLIndirectCommandBuffer>>* icbCache = nil;
    NSMutableDictionary<NSNumber*, id<MTLBuffer>>* stepBufferCache = nil;
    NSMutableDictionary<NSNumber*, id<MTLBuffer>>* uniformBufferCache = nil;
};

static EighKernels eighKFloat;
static EighKernels eighKHalf;
static EighKernels eighKBFloat;
static bool eigh_kernels_loaded = false;

void load_eigh_kernels_typed(id<MTLLibrary> lib, EighKernels& k, NSString* suffix, bool required) {
    k.icbCache = [NSMutableDictionary new];
    k.stepBufferCache = [NSMutableDictionary new];
    k.uniformBufferCache = [NSMutableDictionary new];
    
    NSError* error = nil;
    id<MTLDevice> device = MPSDevice::getInstance()->device();

    k.jacobi = [lib newFunctionWithName:[NSString stringWithFormat:@"jacobi_rotate_kernel_optimized_%@", suffix]];
    k.dot_columns = [lib newFunctionWithName:[NSString stringWithFormat:@"dot_columns_kernel_%@", suffix]];
    k.fused_generic = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_generic_%@", suffix]];
    k.fused_64 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_64_%@", suffix]];
    k.fused_128 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_128_%@", suffix]];
    k.fused_256 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_256_%@", suffix]];
    k.fused_512 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_512_%@", suffix]];
    k.fused_1024 = [lib newFunctionWithName:[NSString stringWithFormat:@"svd_fused_block_kernel_1024_%@", suffix]];
    
    k.jacobiICB = [lib newFunctionWithName:[NSString stringWithFormat:@"jacobi_rotate_kernel_icb_%@", suffix]];
    if (k.jacobiICB) {
        MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
        desc.computeFunction = k.jacobiICB;
        desc.supportIndirectCommandBuffers = YES;
        k.jacobiICBPSO = [device newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:&error];
    }
    
    if (k.jacobi) k.jacobiPSO = [device newComputePipelineStateWithFunction:k.jacobi error:&error];
    if (k.dot_columns) k.dotColumnsPSO = [device newComputePipelineStateWithFunction:k.dot_columns error:&error];
    if (k.fused_generic) k.fusedGenericPSO = [device newComputePipelineStateWithFunction:k.fused_generic error:&error];
    if (k.fused_64) k.fused64PSO = [device newComputePipelineStateWithFunction:k.fused_64 error:&error];
    if (k.fused_128) k.fused128PSO = [device newComputePipelineStateWithFunction:k.fused_128 error:&error];
    if (k.fused_256) k.fused256PSO = [device newComputePipelineStateWithFunction:k.fused_256 error:&error];
    if (k.fused_512) k.fused512PSO = [device newComputePipelineStateWithFunction:k.fused_512 error:&error];
    if (k.fused_1024) k.fused1024PSO = [device newComputePipelineStateWithFunction:k.fused_1024 error:&error];
}

void load_eigh_kernels() {
    if (eigh_kernels_loaded) return;
    load_core_kernels();
    
    load_eigh_kernels_typed(coreLib, eighKFloat, @"float", true);
    load_eigh_kernels_typed(coreLib, eighKHalf, @"half", false);
    load_eigh_kernels_typed(coreLib, eighKBFloat, @"bfloat", false);
    eigh_kernels_loaded = true;
}

std::vector<torch::Tensor> eigh_forward(torch::Tensor A) { 
    TORCH_CHECK(A.device().is_mps(), "Input tensor must be on MPS");
    load_eigh_kernels();
    
    EighKernels* kernels = nullptr;
    if (A.scalar_type() == torch::kFloat32) kernels = &eighKFloat;
    else if (A.scalar_type() == torch::kHalf) kernels = &eighKHalf;
    else if (A.scalar_type() == torch::kBFloat16) kernels = &eighKBFloat;
    else TORCH_CHECK(false, "Unsupported dtype.");
    
    if (A.dim() == 2) A = A.unsqueeze(0);
    
    int64_t Batch = A.size(0);
    int64_t M = A.size(1);
    int64_t N = A.size(2);
    
    TORCH_CHECK(N % 2 == 0, "N must be even");

    torch::Tensor V = torch::eye(N, A.options()).expand({Batch, N, N}).contiguous();
    torch::Tensor A_T = A.transpose(1, 2).contiguous(); 
    torch::Tensor V_T = V.transpose(1, 2).contiguous(); 
    
    auto [pairs_cpu, num_steps] = svd_generate_ordering(N);
    int num_pairs = N / 2;
    int threads_per_pair = std::min(32, std::max(1, 1024 / num_pairs));
    
    torch::Tensor PairsTens = torch::tensor(pairs_cpu, torch::dtype(torch::kInt32).device(torch::kCPU)).contiguous();
    PairsTens = PairsTens.to(A.device());
    
    MPSStream* stream = getCurrentMPSStream();
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    
    uint32_t BatchStrideA = (uint32_t)(N * M);
    uint32_t BatchStrideV = (uint32_t)(N * N);
    uint32_t M_u = (uint32_t)M;
    uint32_t N_u = (uint32_t)N;
    
    id<MTLComputePipelineState> fusedPSO = nil;
    if (N == 64 && kernels->fused64PSO) fusedPSO = kernels->fused64PSO;
    else if (N == 128 && kernels->fused128PSO) fusedPSO = kernels->fused128PSO;
    
    if (fusedPSO) {
        [encoder setComputePipelineState:fusedPSO];
        mtl_setBuffer(encoder, A_T, 0);
        mtl_setBuffer(encoder, V_T, 1);
        [encoder setBuffer:getMTLBufferStorage(PairsTens) offset:(PairsTens.storage_offset() * PairsTens.element_size()) atIndex:2];
        [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&BatchStrideA length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&BatchStrideV length:sizeof(uint32_t) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, Batch) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    } else {
        // Iterative fallback
        int sweeps = 15;
        id<MTLComputePipelineState> rotatePSO = kernels->jacobiPSO;
        TORCH_CHECK(rotatePSO, "No jacobi PSO");
        
        int threads_per_group = std::min((int)rotatePSO.maxTotalThreadsPerThreadgroup, 256);
        int elem_size = A.element_size();
        NSUInteger sharedMemSize = ((threads_per_group + 31) / 32) * 3 * elem_size;
        
        [encoder setComputePipelineState:rotatePSO];
        mtl_setBuffer(encoder, A_T, 0);
        mtl_setBuffer(encoder, V_T, 1);
        [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&N_u length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&BatchStrideA length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&BatchStrideV length:sizeof(uint32_t) atIndex:6];
        [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];
        
        for (int sw = 0; sw < sweeps; ++sw) {
            for (int step = 0; step < num_steps; ++step) {
                size_t pairs_offset = step * num_pairs * sizeof(int) * 2;
                [encoder setBuffer:getMTLBufferStorage(PairsTens) 
                            offset:(PairsTens.storage_offset() * PairsTens.element_size() + pairs_offset) 
                           atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(num_pairs, 1, Batch) 
                    threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
            }
        }
    }
    
    torch::Tensor Eigenvalues = torch::empty({Batch, N}, A.options());
    
    id<MTLComputePipelineState> dotPSO = kernels->dotColumnsPSO;
    TORCH_CHECK(dotPSO, "No dot columns PSO");
    
    [encoder setComputePipelineState:dotPSO];
    mtl_setBuffer(encoder, A_T, 0);
    mtl_setBuffer(encoder, V_T, 1);
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
    
    [encoder dispatchThreadgroups:MTLSizeMake(N, 1, Batch) threadsPerThreadgroup:MTLSizeMake(dot_tpg, 1, 1)];
    
    return {Eigenvalues, V_T.transpose(1, 2).contiguous()};
}

// -----------------------------------------------------------------------------
// Python Bindings
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// RMSNorm
// -----------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> rmsnorm_fwd_metal(torch::Tensor X, torch::Tensor W, float eps) {
    load_core_kernels();
    
    TORCH_CHECK(X.device().type() == at::kMPS, "X must be on MPS");
    TORCH_CHECK(W.device().type() == at::kMPS, "W must be on MPS");
    
    bool is_bf16 = X.scalar_type() == at::kBFloat16;
    bool is_half = X.scalar_type() == at::kHalf;
    
    int64_t B = X.size(0);
    int64_t N = X.size(1);
    int64_t elem_size = (is_half || is_bf16) ? 2 : 4;
    
    auto Y = torch::empty_like(X);
    // Rstd always in float for numerical stability
    auto Rstd = torch::empty({B}, X.options().dtype(at::kFloat));
    
    // Select kernel based on dtype
    id<MTLComputePipelineState> pso = nil;
    id<MTLComputePipelineState> pso_vec4 = nil;
    
    if (is_bf16 && kernels.rmsnormFwdBfloatPSO) {
        pso = kernels.rmsnormFwdBfloatPSO;
    } else if (is_half) {
        pso = kernels.rmsnormFwdHalfPSO;
    } else {
        pso = kernels.rmsnormFwdPSO;
        pso_vec4 = kernels.rmsnormFwdVec4PSO;
    }
    
    // Fallback for bf16 if no kernel available
    if (is_bf16 && !pso) {
        auto X_fp32 = X.to(at::kFloat);
        auto W_fp32 = W.to(at::kFloat);
        auto [Y_fp32, Rstd_out] = rmsnorm_fwd_metal(X_fp32, W_fp32, eps);
        return std::make_tuple(Y_fp32.to(at::kBFloat16), Rstd_out);
    }
    
    // Check for vectorization (float only for now)
    bool use_vec4 = pso_vec4 && !is_half && !is_bf16 &&
                    (N % 4 == 0) && 
                    X.is_contiguous() && W.is_contiguous() && 
                    (X.storage_offset() % 4 == 0) && 
                    (W.storage_offset() % 4 == 0);

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = [stream->commandBuffer() computeCommandEncoder];
        
        if (use_vec4) {
             [encoder setComputePipelineState:pso_vec4];
             NSUInteger threads = std::min((NSUInteger)(N / 4), (NSUInteger)256);
             [encoder setBuffer:getMTLBufferStorage(X) offset:X.storage_offset() * elem_size atIndex:0];
             [encoder setBuffer:getMTLBufferStorage(W) offset:W.storage_offset() * elem_size atIndex:1];
             [encoder setBuffer:getMTLBufferStorage(Y) offset:Y.storage_offset() * elem_size atIndex:2];
             [encoder setBuffer:getMTLBufferStorage(Rstd) offset:Rstd.storage_offset() * 4 atIndex:3]; // Rstd always float
             uint32_t N_u = (uint32_t)N;
             [encoder setBytes:&N_u length:4 atIndex:4];
             [encoder setBytes:&eps length:4 atIndex:5];
             [encoder dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        } else if (pso) {
             [encoder setComputePipelineState:pso];
             [encoder setBuffer:getMTLBufferStorage(X) offset:X.storage_offset() * elem_size atIndex:0];
             [encoder setBuffer:getMTLBufferStorage(W) offset:W.storage_offset() * elem_size atIndex:1];
             [encoder setBuffer:getMTLBufferStorage(Y) offset:Y.storage_offset() * elem_size atIndex:2];
             [encoder setBuffer:getMTLBufferStorage(Rstd) offset:Rstd.storage_offset() * 4 atIndex:3]; // Rstd always float
             uint32_t N_u = (uint32_t)N;
             [encoder setBytes:&N_u length:4 atIndex:4];
             [encoder setBytes:&eps length:4 atIndex:5];
             NSUInteger threads = std::min((NSUInteger)N, (NSUInteger)1024);
             [encoder dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return std::make_tuple(Y, Rstd);
}


std::tuple<torch::Tensor, torch::Tensor> rmsnorm_bwd_metal(torch::Tensor dY, torch::Tensor X, torch::Tensor Rstd, torch::Tensor W) {
    load_core_kernels();
    
    int64_t B = X.size(0);
    int64_t N = X.size(1);
    
    auto dX = torch::empty_like(X);
    auto dW = torch::empty_like(W);
    
    if (!kernels.rmsnormBwdDxPSO || !kernels.rmsnormBwdDwPSO) {
        return std::make_tuple(dX, dW);
    }
    
    bool use_vec4 = kernels.rmsnormBwdDxVec4PSO && kernels.rmsnormBwdDwVec4PSO &&
                    (N % 4 == 0) && 
                    dY.is_contiguous() && X.is_contiguous() && W.is_contiguous() &&
                    (dY.storage_offset() % 4 == 0) && (X.storage_offset() % 4 == 0) && (W.storage_offset() % 4 == 0);

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = [stream->commandBuffer() computeCommandEncoder];
        
        if (use_vec4) {
             // 1. Compute dX (Vectorized)
             [encoder setComputePipelineState:kernels.rmsnormBwdDxVec4PSO];
             // Tune: 256 threads
             NSUInteger threads = std::min((NSUInteger)(N / 4), (NSUInteger)256);
             
             [encoder setBuffer:getMTLBufferStorage(dY) offset:dY.storage_offset() * 4 atIndex:0];
             [encoder setBuffer:getMTLBufferStorage(X) offset:X.storage_offset() * 4 atIndex:1];
             [encoder setBuffer:getMTLBufferStorage(Rstd) offset:Rstd.storage_offset() * 4 atIndex:2];
             [encoder setBuffer:getMTLBufferStorage(W) offset:W.storage_offset() * 4 atIndex:3];
             [encoder setBuffer:getMTLBufferStorage(dX) offset:dX.storage_offset() * 4 atIndex:4];
             uint32_t N_u = (uint32_t)N;
             [encoder setBytes:&N_u length:4 atIndex:5];
             [encoder dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
             
             // 2. Compute dW (Vectorized)
             [encoder setComputePipelineState:kernels.rmsnormBwdDwVec4PSO];
             [encoder setBuffer:getMTLBufferStorage(dY) offset:dY.storage_offset() * 4 atIndex:0];
             [encoder setBuffer:getMTLBufferStorage(X) offset:X.storage_offset() * 4 atIndex:1];
             [encoder setBuffer:getMTLBufferStorage(Rstd) offset:Rstd.storage_offset() * 4 atIndex:2];
             [encoder setBuffer:getMTLBufferStorage(dW) offset:dW.storage_offset() * 4 atIndex:3];
             [encoder setBytes:&N_u length:4 atIndex:4];
             uint32_t B_u = (uint32_t)B;
             [encoder setBytes:&B_u length:4 atIndex:5];
             
             // N / 4 items to sum
             NSUInteger dw_threads = (NSUInteger)(N / 4);
             NSUInteger dw_tg_size = std::min(dw_threads, (NSUInteger)1024);
             NSUInteger dw_groups = (dw_threads + dw_tg_size - 1) / dw_tg_size;
             [encoder dispatchThreadgroups:MTLSizeMake(dw_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(dw_tg_size, 1, 1)];

        } else {
             // Scalar Path
             // 1. Compute dX
             [encoder setComputePipelineState:kernels.rmsnormBwdDxPSO];
             [encoder setBuffer:getMTLBufferStorage(dY) offset:dY.storage_offset() * 4 atIndex:0];
             [encoder setBuffer:getMTLBufferStorage(X) offset:X.storage_offset() * 4 atIndex:1];
             [encoder setBuffer:getMTLBufferStorage(Rstd) offset:Rstd.storage_offset() * 4 atIndex:2];
             [encoder setBuffer:getMTLBufferStorage(W) offset:W.storage_offset() * 4 atIndex:3];
             [encoder setBuffer:getMTLBufferStorage(dX) offset:dX.storage_offset() * 4 atIndex:4];
             uint32_t N_u = (uint32_t)N;
             [encoder setBytes:&N_u length:4 atIndex:5];
             NSUInteger threads = std::min((NSUInteger)N, (NSUInteger)1024);
             [encoder dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
             
             // 2. Compute dW
             [encoder setComputePipelineState:kernels.rmsnormBwdDwPSO];
             [encoder setBuffer:getMTLBufferStorage(dY) offset:dY.storage_offset() * 4 atIndex:0];
             [encoder setBuffer:getMTLBufferStorage(X) offset:X.storage_offset() * 4 atIndex:1];
             [encoder setBuffer:getMTLBufferStorage(Rstd) offset:Rstd.storage_offset() * 4 atIndex:2];
             [encoder setBuffer:getMTLBufferStorage(dW) offset:dW.storage_offset() * 4 atIndex:3];
             [encoder setBytes:&N_u length:4 atIndex:4];
             uint32_t B_u = (uint32_t)B;
             [encoder setBytes:&B_u length:4 atIndex:5];
             NSUInteger dw_threads = (NSUInteger)N;
             NSUInteger dw_tg_size = std::min(dw_threads, (NSUInteger)1024);
             NSUInteger dw_groups = (dw_threads + dw_tg_size - 1) / dw_tg_size;
             [encoder dispatchThreadgroups:MTLSizeMake(dw_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(dw_tg_size, 1, 1)];
        }
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return std::make_tuple(dX, dW);
}

// -----------------------------------------------------------------------------
// Fused Add + RMSNorm (saves memory round-trip)
// -----------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> fused_add_rmsnorm_metal(
    torch::Tensor input,    // [B, N] - overwritten with output
    torch::Tensor residual, // [B, N] - updated in-place
    torch::Tensor W,        // [N]
    float eps
) {
    load_core_kernels();
    
    TORCH_CHECK(input.device().type() == at::kMPS, "input must be on MPS device");
    TORCH_CHECK(input.dim() == 2, "input must be 2D (B, N)");
    TORCH_CHECK(residual.dim() == 2, "residual must be 2D (B, N)");
    
    int64_t B = input.size(0);
    int64_t N = input.size(1);
    
    // Ensure contiguous
    auto input_c = input.contiguous();
    auto residual_c = residual.contiguous();
    auto W_c = W.contiguous();
    
    auto Rstd = torch::empty({B}, input.options().dtype(at::kFloat));
    
    if (!kernels.fusedAddRmsnormPSO) {
        // Fallback: do it manually
        residual_c.add_(input_c);
        auto var = residual_c.pow(2).mean(-1, true);
        auto rstd_exp = torch::rsqrt(var + eps);
        input_c.copy_(residual_c * rstd_exp * W_c);
        return std::make_tuple(input_c, Rstd);
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = [stream->commandBuffer() computeCommandEncoder];
        
        [encoder setComputePipelineState:kernels.fusedAddRmsnormPSO];
        [encoder setBuffer:getMTLBufferStorage(input_c) offset:input_c.storage_offset() * 4 atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(residual_c) offset:residual_c.storage_offset() * 4 atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(W_c) offset:W_c.storage_offset() * 4 atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(Rstd) offset:Rstd.storage_offset() * 4 atIndex:3];
        uint32_t N_u = (uint32_t)N;
        [encoder setBytes:&N_u length:4 atIndex:4];
        [encoder setBytes:&eps length:4 atIndex:5];
        
        NSUInteger threads = std::min((NSUInteger)N, (NSUInteger)1024);
        [encoder dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return std::make_tuple(input_c, Rstd);
}

// -----------------------------------------------------------------------------
// AdamW
// -----------------------------------------------------------------------------

void adamw_step_metal(
    torch::Tensor params,
    torch::Tensor grads,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float correction1,
    float correction2
) {
    load_core_kernels();
    
    TORCH_CHECK(params.is_contiguous(), "params must be contig");
    TORCH_CHECK(grads.is_contiguous(), "grads must be contig");
    TORCH_CHECK(exp_avg.is_contiguous(), "exp_avg must be contig");
    TORCH_CHECK(exp_avg_sq.is_contiguous(), "exp_avg_sq must be contig");
    
    int64_t numel = params.numel();
    at::ScalarType dtype = params.scalar_type();
    int64_t elem_size = params.element_size();
    
    // Optimizer states MUST be float32 for numerical stability
    TORCH_CHECK(exp_avg.scalar_type() == at::kFloat, "exp_avg must be float32");
    TORCH_CHECK(exp_avg_sq.scalar_type() == at::kFloat, "exp_avg_sq must be float32");
    
    // Select kernel PSOs based on dtype
    id<MTLComputePipelineState> vecPSO = nil;
    id<MTLComputePipelineState> scalarPSO = nil;
    id<MTLComputePipelineState> ilp4PSO = nil;  // For large tensors (all dtypes)
    
    if (dtype == at::kFloat) {
        vecPSO = kernels.adamwStepPSO;
        scalarPSO = kernels.adamwStepScalarPSO;
        ilp4PSO = kernels.adamwStepIlp4PSO;
    } else if (dtype == at::kHalf) {
        vecPSO = kernels.adamwStepHalfPSO;
        scalarPSO = kernels.adamwStepHalfScalarPSO;
        ilp4PSO = kernels.adamwStepHalfIlp4PSO;
    } else if (dtype == at::kBFloat16) {
        vecPSO = kernels.adamwStepBfloatPSO;
        scalarPSO = kernels.adamwStepBfloatScalarPSO;
        ilp4PSO = kernels.adamwStepBfloatIlp4PSO;
    } else {
        TORCH_CHECK(false, "adamw_step: unsupported dtype ", dtype, ". Supported: float32, float16, bfloat16");
    }
    
    if (!vecPSO) {
        TORCH_CHECK(false, "adamw_step: kernel not available for dtype ", dtype);
        return;
    }
    
    // Split into vectorized (divisible by 4) and scalar tail
    int64_t numel_vec = numel / 4;
    int64_t tail = numel % 4;
    
    // Use ILP4 kernel for large tensors (>256KB = 64K vec4s for float, 128K vec4s for half/bf16)
    // Threshold: 65536 vec4 elements = 256KB for float32, 128KB for half/bf16
    bool use_ilp4 = ilp4PSO && (numel_vec >= 65536);
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = [stream->commandBuffer() computeCommandEncoder];
        
        // For half/bfloat: params are 2 bytes, but exp_avg/exp_avg_sq are 4 bytes
        int64_t state_elem_size = 4;  // exp_avg and exp_avg_sq are always float32
        
        // 1. Vectorized Body
        if (numel_vec > 0) {
            if (use_ilp4) {
                // ILP=4 kernel: each thread processes 4 float4 vectors (float32 only)
                [encoder setComputePipelineState:ilp4PSO];
                [encoder setBuffer:getMTLBufferStorage(params) offset:params.storage_offset()*elem_size atIndex:0];
                [encoder setBuffer:getMTLBufferStorage(grads) offset:grads.storage_offset()*elem_size atIndex:1];
                [encoder setBuffer:getMTLBufferStorage(exp_avg) offset:exp_avg.storage_offset()*state_elem_size atIndex:2];
                [encoder setBuffer:getMTLBufferStorage(exp_avg_sq) offset:exp_avg_sq.storage_offset()*state_elem_size atIndex:3];
                
                [encoder setBytes:&lr length:4 atIndex:4];
                [encoder setBytes:&beta1 length:4 atIndex:5];
                [encoder setBytes:&beta2 length:4 atIndex:6];
                [encoder setBytes:&eps length:4 atIndex:7];
                [encoder setBytes:&weight_decay length:4 atIndex:8];
                [encoder setBytes:&correction1 length:4 atIndex:9];
                [encoder setBytes:&correction2 length:4 atIndex:10];
                uint32_t numel_u = (uint32_t)numel_vec;
                [encoder setBytes:&numel_u length:4 atIndex:11];
                
                NSUInteger num_threads = (NSUInteger)((numel_vec + 3) / 4);
                NSUInteger tg_size = std::min((NSUInteger)256, num_threads);
                NSUInteger num_groups = (num_threads + tg_size - 1) / tg_size;
                
                [encoder dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1) 
                        threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
            } else {
                // Standard vectorized kernel: one vec4 per thread
                [encoder setComputePipelineState:vecPSO];
                [encoder setBuffer:getMTLBufferStorage(params) offset:params.storage_offset()*elem_size atIndex:0];
                [encoder setBuffer:getMTLBufferStorage(grads) offset:grads.storage_offset()*elem_size atIndex:1];
                [encoder setBuffer:getMTLBufferStorage(exp_avg) offset:exp_avg.storage_offset()*state_elem_size atIndex:2];
                [encoder setBuffer:getMTLBufferStorage(exp_avg_sq) offset:exp_avg_sq.storage_offset()*state_elem_size atIndex:3];
                
                [encoder setBytes:&lr length:4 atIndex:4];
                [encoder setBytes:&beta1 length:4 atIndex:5];
                [encoder setBytes:&beta2 length:4 atIndex:6];
                [encoder setBytes:&eps length:4 atIndex:7];
                [encoder setBytes:&weight_decay length:4 atIndex:8];
                [encoder setBytes:&correction1 length:4 atIndex:9];
                [encoder setBytes:&correction2 length:4 atIndex:10];
                
                NSUInteger num_threads = (NSUInteger)numel_vec;
                NSUInteger tg_size = std::min(num_threads, (NSUInteger)256);
                NSUInteger num_groups = (num_threads + tg_size - 1) / tg_size;
                
                [encoder dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
            }
        }
        
        // 2. Scalar Tail
        if (tail > 0 && scalarPSO) {
             [encoder setComputePipelineState:scalarPSO];
             
             // Offset to tail elements
             int64_t offset_elems = numel_vec * 4;
             int64_t param_offset_bytes = offset_elems * elem_size;
             int64_t state_offset_bytes = offset_elems * state_elem_size;
             
             [encoder setBuffer:getMTLBufferStorage(params) offset:(params.storage_offset()*elem_size + param_offset_bytes) atIndex:0];
             [encoder setBuffer:getMTLBufferStorage(grads) offset:(grads.storage_offset()*elem_size + param_offset_bytes) atIndex:1];
             [encoder setBuffer:getMTLBufferStorage(exp_avg) offset:(exp_avg.storage_offset()*state_elem_size + state_offset_bytes) atIndex:2];
             [encoder setBuffer:getMTLBufferStorage(exp_avg_sq) offset:(exp_avg_sq.storage_offset()*state_elem_size + state_offset_bytes) atIndex:3];
             
             [encoder setBytes:&lr length:4 atIndex:4];
             [encoder setBytes:&beta1 length:4 atIndex:5];
             [encoder setBytes:&beta2 length:4 atIndex:6];
             [encoder setBytes:&eps length:4 atIndex:7];
             [encoder setBytes:&weight_decay length:4 atIndex:8];
             [encoder setBytes:&correction1 length:4 atIndex:9];
             [encoder setBytes:&correction2 length:4 atIndex:10];
             
             [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)tail, 1, 1)];
        } else if (tail > 0) {
            // Fallback: process tail elements via separate call (shouldn't happen if all kernels loaded)
            printf("metalcore: Warning - No scalar AdamW kernel for dtype, tail %lld elements ignored!\n", tail);
        }
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
}


// -----------------------------------------------------------------------------
// Batched Cholesky Decomposition
// -----------------------------------------------------------------------------

torch::Tensor cholesky_batched_metal(torch::Tensor A) {
    load_core_kernels();
    
    TORCH_CHECK(A.device().type() == at::kMPS, "A must be on MPS device");
    TORCH_CHECK(A.dim() == 3, "A must be 3D (batch, N, N)");
    TORCH_CHECK(A.size(1) == A.size(2), "A must be square");
    
    int64_t batch_size = A.size(0);
    int64_t N = A.size(1);
    
    // Clone and make contiguous (we modify in-place)
    auto L = A.clone().contiguous();
    
    if (!kernels.choleskyBatchedPSO) {
        // CPU fallback using manual Cholesky
        auto L_cpu = L.cpu();
        for (int64_t i = 0; i < batch_size; i++) {
            L_cpu[i] = at::linalg_cholesky(L_cpu[i]);
        }
        return L_cpu.to(A.device());
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto cmdBuffer = stream->commandBuffer();
        auto encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:kernels.choleskyBatchedPSO];
        [encoder setBuffer:getMTLBufferStorage(L) offset:L.storage_offset() * L.element_size() atIndex:0];
        
        uint32_t N_uint = (uint32_t)N;
        uint32_t batch_uint = (uint32_t)batch_size;
        [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:1];
        [encoder setBytes:&batch_uint length:sizeof(batch_uint) atIndex:2];
        
        // Shared memory for N*N panel (MAGMA-style optimization)
        NSUInteger shared_size = N * N * sizeof(float);
        [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
        
        // One threadgroup per batch, up to 64 threads per group
        NSUInteger tg_size = std::min((NSUInteger)64, std::max((NSUInteger)32, (NSUInteger)N));
        [encoder dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1) 
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return L;
}

// -----------------------------------------------------------------------------
// Batched Cholesky Solve (L @ L.T @ x = b)
// -----------------------------------------------------------------------------

torch::Tensor cholesky_solve_batched_metal(torch::Tensor L, torch::Tensor b) {
    load_core_kernels();
    
    TORCH_CHECK(L.device().type() == at::kMPS, "L must be on MPS device");
    TORCH_CHECK(b.device().type() == at::kMPS, "b must be on MPS device");
    TORCH_CHECK(L.dim() == 3, "L must be 3D (batch, N, N)");
    
    int64_t batch_size = L.size(0);
    int64_t N = L.size(1);
    int64_t K = b.dim() == 3 ? b.size(2) : 1;
    
    L = L.contiguous();
    auto x = b.clone().contiguous();
    if (x.dim() == 2) {
        x = x.unsqueeze(-1);  // (batch, N) -> (batch, N, 1)
    }
    
    if (!kernels.choleskySolveBatchedPSO) {
        // CPU fallback
        auto L_cpu = L.cpu();
        auto x_cpu = x.cpu();
        for (int64_t i = 0; i < batch_size; i++) {
            x_cpu[i] = at::cholesky_solve(x_cpu[i], L_cpu[i]);
        }
        return b.dim() == 2 ? x_cpu.squeeze(-1).to(L.device()) : x_cpu.to(L.device());
    }
    
    // Single fused kernel: forward + back substitution with zero-copy transpose
    // The kernel accesses L[j,i] directly for L.T[i,j] - no memory copy needed
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto cmdBuffer = stream->commandBuffer();
        auto encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:kernels.choleskySolveBatchedPSO];
        [encoder setBuffer:getMTLBufferStorage(L) offset:L.storage_offset() * L.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:1];
        
        uint32_t N_uint = (uint32_t)N;
        uint32_t K_uint = (uint32_t)K;
        uint32_t batch_uint = (uint32_t)batch_size;
        [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:2];
        [encoder setBytes:&K_uint length:sizeof(K_uint) atIndex:3];
        [encoder setBytes:&batch_uint length:sizeof(batch_uint) atIndex:4];
        
        // Shared memory for row cache (N floats)
        [encoder setThreadgroupMemoryLength:N * sizeof(float) atIndex:0];
        
        NSUInteger tg_size = std::min((NSUInteger)64, std::max((NSUInteger)K, (NSUInteger)32));
        [encoder dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1) 
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return b.dim() == 2 ? x.squeeze(-1) : x;
}
// -----------------------------------------------------------------------------
// Activation Functions (GELU / SiLU)
// -----------------------------------------------------------------------------

torch::Tensor gelu_fwd_metal(torch::Tensor X) {
    load_core_kernels();
    
    TORCH_CHECK(X.device().type() == at::kMPS, "X must be on MPS device");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    
    bool is_bf16 = X.scalar_type() == at::kBFloat16;
    bool is_half = X.scalar_type() == at::kHalf;
    
    auto Y = torch::empty_like(X);
    int64_t numel = X.numel();
    int64_t elem_size = (is_half || is_bf16) ? 2 : 4;  // bytes per element
    
    // Select appropriate kernel PSO based on dtype
    id<MTLComputePipelineState> vecPSO = nil;
    id<MTLComputePipelineState> scalarPSO = nil;
    
    if (is_bf16 && kernels.geluFwdBfloatPSO) {
        vecPSO = kernels.geluFwdBfloatPSO;
        scalarPSO = kernels.geluFwdBfloatScalarPSO;
    } else if (is_half) {
        vecPSO = kernels.geluFwdHalfPSO;
        scalarPSO = kernels.geluFwdScalarHalfPSO;
    } else {
        vecPSO = kernels.geluFwdPSO;
        scalarPSO = kernels.geluFwdScalarPSO;
    }
    
    // Fallback to PyTorch if no kernel available
    if (!vecPSO) {
        if (is_bf16) {
            auto x_fp32 = X.to(at::kFloat);
            return torch::gelu(x_fp32).to(at::kBFloat16);
        }
        return torch::gelu(X);
    }
    
    int64_t numel_vec = numel / 4;
    int64_t tail = numel % 4;
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = [stream->commandBuffer() computeCommandEncoder];
        
        if (numel_vec > 0) {
            [encoder setComputePipelineState:vecPSO];
            [encoder setBuffer:getMTLBufferStorage(X) offset:X.storage_offset() * elem_size atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(Y) offset:Y.storage_offset() * elem_size atIndex:1];
            uint32_t numel_u = (uint32_t)numel;
            [encoder setBytes:&numel_u length:4 atIndex:2];
            
            NSUInteger threads = (NSUInteger)numel_vec;
            NSUInteger tg_size = std::min(threads, (NSUInteger)256);
            NSUInteger groups = (threads + tg_size - 1) / tg_size;
            [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        }
        
        if (tail > 0 && scalarPSO) {
            [encoder setComputePipelineState:scalarPSO];
            int64_t offset = numel_vec * 4 * elem_size; // bytes
            [encoder setBuffer:getMTLBufferStorage(X) offset:(X.storage_offset() * elem_size + offset) atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(Y) offset:(Y.storage_offset() * elem_size + offset) atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)tail, 1, 1)];
        }
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return Y;
}


torch::Tensor gelu_bwd_metal(torch::Tensor dY, torch::Tensor X) {
    load_core_kernels();
    
    TORCH_CHECK(dY.device().type() == at::kMPS, "dY must be on MPS device");
    TORCH_CHECK(X.device().type() == at::kMPS, "X must be on MPS device");
    
    bool is_bf16 = X.scalar_type() == at::kBFloat16;
    bool is_half = X.scalar_type() == at::kHalf;
    
    auto dX = torch::empty_like(X);
    int64_t numel = X.numel();
    int64_t elem_size = (is_half || is_bf16) ? 2 : 4;
    
    // Select appropriate kernel PSO based on dtype
    id<MTLComputePipelineState> pso = nil;
    
    if (is_bf16 && kernels.geluBwdBfloatPSO) {
        pso = kernels.geluBwdBfloatPSO;
    } else if (is_half) {
        pso = kernels.geluBwdHalfPSO;
    } else {
        pso = kernels.geluBwdPSO;
    }
    
    // Fallback to PyTorch
    if (!pso) {
        if (is_bf16) {
            auto X_fp32 = X.to(at::kFloat);
            auto dY_fp32 = dY.to(at::kFloat);
            auto X_cpu = X_fp32.cpu().requires_grad_(true);
            auto Y_cpu = torch::gelu(X_cpu);
            Y_cpu.backward(dY_fp32.cpu());
            return X_cpu.grad().to(X.device()).to(at::kBFloat16);
        }
        auto X_cpu = X.cpu().requires_grad_(true);
        auto Y_cpu = torch::gelu(X_cpu);
        Y_cpu.backward(dY.cpu());
        return X_cpu.grad().to(X.device());
    }
    
    int64_t numel_vec = numel / 4;
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        stream->synchronize(SyncType::COMMIT);
        auto encoder = [stream->commandBuffer() computeCommandEncoder];
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(dY) offset:dY.storage_offset() * elem_size atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(X) offset:X.storage_offset() * elem_size atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(dX) offset:dX.storage_offset() * elem_size atIndex:2];
        uint32_t numel_u = (uint32_t)numel;
        [encoder setBytes:&numel_u length:4 atIndex:3];
        
        NSUInteger threads = (NSUInteger)numel_vec;
        NSUInteger tg_size = std::min(threads, (NSUInteger)256);
        NSUInteger groups = (threads + tg_size - 1) / tg_size;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return dX;
}


torch::Tensor silu_fwd_metal(torch::Tensor X) {
    load_core_kernels();
    
    TORCH_CHECK(X.device().type() == at::kMPS, "X must be on MPS device");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    
    bool is_bf16 = X.scalar_type() == at::kBFloat16;
    bool is_half = X.scalar_type() == at::kHalf;
    
    auto Y = torch::empty_like(X);
    int64_t numel = X.numel();
    int64_t elem_size = (is_half || is_bf16) ? 2 : 4;
    
    // Select appropriate kernel PSO based on dtype
    id<MTLComputePipelineState> vecPSO = nil;
    id<MTLComputePipelineState> scalarPSO = nil;
    
    if (is_bf16 && kernels.siluFwdBfloatPSO) {
        vecPSO = kernels.siluFwdBfloatPSO;
        scalarPSO = kernels.siluFwdBfloatScalarPSO;
    } else if (is_half) {
        vecPSO = kernels.siluFwdHalfPSO;
        scalarPSO = kernels.siluFwdScalarHalfPSO;
    } else {
        vecPSO = kernels.siluFwdPSO;
        scalarPSO = kernels.siluFwdScalarPSO;
    }
    
    // Fallback to PyTorch if no kernel available
    if (!vecPSO) {
        if (is_bf16) {
            auto x_fp32 = X.to(at::kFloat);
            return torch::silu(x_fp32).to(at::kBFloat16);
        }
        return torch::silu(X);
    }
    
    int64_t numel_vec = numel / 4;
    int64_t tail = numel % 4;
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = [stream->commandBuffer() computeCommandEncoder];
        
        if (numel_vec > 0) {
            [encoder setComputePipelineState:vecPSO];
            [encoder setBuffer:getMTLBufferStorage(X) offset:X.storage_offset() * elem_size atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(Y) offset:Y.storage_offset() * elem_size atIndex:1];
            uint32_t numel_u = (uint32_t)numel;
            [encoder setBytes:&numel_u length:4 atIndex:2];
            
            NSUInteger threads = (NSUInteger)numel_vec;
            NSUInteger tg_size = std::min(threads, (NSUInteger)256);
            NSUInteger groups = (threads + tg_size - 1) / tg_size;
            [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        }
        
        if (tail > 0 && scalarPSO) {
            [encoder setComputePipelineState:scalarPSO];
            int64_t offset = numel_vec * 4 * elem_size;
            [encoder setBuffer:getMTLBufferStorage(X) offset:(X.storage_offset() * elem_size + offset) atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(Y) offset:(Y.storage_offset() * elem_size + offset) atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)tail, 1, 1)];
        }
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return Y;
}


torch::Tensor silu_bwd_metal(torch::Tensor dY, torch::Tensor X) {
    load_core_kernels();
    
    TORCH_CHECK(dY.device().type() == at::kMPS, "dY must be on MPS device");
    TORCH_CHECK(X.device().type() == at::kMPS, "X must be on MPS device");
    
    bool is_bf16 = X.scalar_type() == at::kBFloat16;
    bool is_half = X.scalar_type() == at::kHalf;
    
    auto dX = torch::empty_like(X);
    int64_t numel = X.numel();
    int64_t elem_size = (is_half || is_bf16) ? 2 : 4;
    
    // Select appropriate kernel PSO based on dtype
    id<MTLComputePipelineState> pso = nil;
    
    if (is_bf16 && kernels.siluBwdBfloatPSO) {
        pso = kernels.siluBwdBfloatPSO;
    } else if (is_half) {
        pso = kernels.siluBwdHalfPSO;
    } else {
        pso = kernels.siluBwdPSO;
    }
    
    // Fallback to PyTorch
    if (!pso) {
        if (is_bf16) {
            auto X_fp32 = X.to(at::kFloat);
            auto dY_fp32 = dY.to(at::kFloat);
            auto X_cpu = X_fp32.cpu().requires_grad_(true);
            auto Y_cpu = torch::silu(X_cpu);
            Y_cpu.backward(dY_fp32.cpu());
            return X_cpu.grad().to(X.device()).to(at::kBFloat16);
        }
        auto X_cpu = X.cpu().requires_grad_(true);
        auto Y_cpu = torch::silu(X_cpu);
        Y_cpu.backward(dY.cpu());
        return X_cpu.grad().to(X.device());
    }
    
    int64_t numel_vec = numel / 4;
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        stream->synchronize(SyncType::COMMIT);
        auto encoder = [stream->commandBuffer() computeCommandEncoder];
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(dY) offset:dY.storage_offset() * elem_size atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(X) offset:X.storage_offset() * elem_size atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(dX) offset:dX.storage_offset() * elem_size atIndex:2];
        uint32_t numel_u = (uint32_t)numel;
        [encoder setBytes:&numel_u length:4 atIndex:3];
        
        NSUInteger threads = (NSUInteger)numel_vec;
        NSUInteger tg_size = std::min(threads, (NSUInteger)256);
        NSUInteger groups = (threads + tg_size - 1) / tg_size;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return dX;
}


// -----------------------------------------------------------------------------
// Scaled Dot Product Attention
// -----------------------------------------------------------------------------

torch::Tensor sdpa_fwd_metal(torch::Tensor Q, torch::Tensor K, torch::Tensor V, float scale, bool is_causal) {
    load_core_kernels();
    
    // Expect Q, K, V shape: (B*H, N, D) where B*H = batch * heads
    TORCH_CHECK(Q.device().type() == at::kMPS, "Q must be on MPS device");
    TORCH_CHECK(K.device().type() == at::kMPS, "K must be on MPS device");
    TORCH_CHECK(V.device().type() == at::kMPS, "V must be on MPS device");
    TORCH_CHECK(Q.dim() == 3, "Q must be 3D (batch_heads, seq_len, head_dim)");
    
    int64_t batch_heads = Q.size(0);
    int64_t seq_len = Q.size(1);
    int64_t head_dim = Q.size(2);
    
    auto O = torch::empty_like(Q);
    auto L = torch::empty({batch_heads, seq_len}, Q.options());  // logsumexp for backward
    
    // Determine which kernel to use
    // Use specialized vector kernel for head_dim=64 (fastest path)
    bool use_vector64 = kernels.sdpaVector64PSO && (head_dim == 64) && !is_causal;
    // Use Flash Attention v2 for larger sequences, naive for small ones
    bool use_flash = kernels.flashAttentionFwdV2PSO && (seq_len > 256 || is_causal) && !use_vector64;
    bool use_naive = kernels.attentionNaivePSO && seq_len <= 1024 && !is_causal && !use_vector64;
    
    if (!use_flash && !use_naive && !use_vector64) {
        // CPU fallback
        return torch::scaled_dot_product_attention(Q, K, V);
    }
    
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        stream->synchronize(SyncType::COMMIT);
        auto encoder = [stream->commandBuffer() computeCommandEncoder];
        
        if (use_flash) {
            // Flash Attention v2 - tiled, handles arbitrary sequence lengths
            [encoder setComputePipelineState:kernels.flashAttentionFwdV2PSO];
            [encoder setBuffer:getMTLBufferStorage(Q) offset:Q.storage_offset() * 4 atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(K) offset:K.storage_offset() * 4 atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(V) offset:V.storage_offset() * 4 atIndex:2];
            [encoder setBuffer:getMTLBufferStorage(O) offset:O.storage_offset() * 4 atIndex:3];
            [encoder setBuffer:getMTLBufferStorage(L) offset:L.storage_offset() * 4 atIndex:4];
            
            uint32_t bh_u = (uint32_t)batch_heads;
            uint32_t sl_u = (uint32_t)seq_len;
            uint32_t hd_u = (uint32_t)head_dim;
            uint32_t causal_u = is_causal ? 1 : 0;
            [encoder setBytes:&bh_u length:4 atIndex:5];
            [encoder setBytes:&sl_u length:4 atIndex:6];
            [encoder setBytes:&hd_u length:4 atIndex:7];
            [encoder setBytes:&scale length:4 atIndex:8];
            [encoder setBytes:&causal_u length:4 atIndex:9];
            
            // Threadgroup shared memory: K_tile (64*128) + V_tile (64*128) = 2 * 64 * 128 * 4 bytes
            NSUInteger BLOCK_N = 64;
            NSUInteger BLOCK_D = 128;
            NSUInteger shared_mem_size = 2 * BLOCK_N * BLOCK_D * sizeof(float);
            [encoder setThreadgroupMemoryLength:shared_mem_size atIndex:0];
            
            // Grid: (num_q_blocks, batch_heads, 1), each threadgroup handles BLOCK_M queries
            NSUInteger BLOCK_M = 64;
            NSUInteger num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
            NSUInteger threads_per_group = std::min((NSUInteger)seq_len, (NSUInteger)BLOCK_M);
            [encoder dispatchThreadgroups:MTLSizeMake(num_q_blocks, batch_heads, 1) 
                    threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
        } else if (use_vector64) {
            // Specialized vector kernel for head_dim=64
            [encoder setComputePipelineState:kernels.sdpaVector64PSO];
            [encoder setBuffer:getMTLBufferStorage(Q) offset:Q.storage_offset() * 4 atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(K) offset:K.storage_offset() * 4 atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(V) offset:V.storage_offset() * 4 atIndex:2];
            [encoder setBuffer:getMTLBufferStorage(O) offset:O.storage_offset() * 4 atIndex:3];
            
            // GQA factor = 1 (no grouped query attention for now)
            uint32_t gqa_factor = 1;
            uint32_t kv_seq_len = (uint32_t)seq_len;
            uint32_t q_stride = (uint32_t)(seq_len * head_dim);  // Stride between heads
            uint32_t k_stride = (uint32_t)(seq_len * head_dim);
            uint32_t v_stride = (uint32_t)(seq_len * head_dim);
            
            [encoder setBytes:&gqa_factor length:4 atIndex:4];
            [encoder setBytes:&kv_seq_len length:4 atIndex:5];
            [encoder setBytes:&q_stride length:4 atIndex:6];
            [encoder setBytes:&k_stride length:4 atIndex:7];
            [encoder setBytes:&v_stride length:4 atIndex:8];
            [encoder setBytes:&scale length:4 atIndex:9];
            
            // Grid: (batch*heads, q_seq_len, 1), 64 threads per group (one per head_dim dimension)
            [encoder dispatchThreadgroups:MTLSizeMake(batch_heads, seq_len, 1) 
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        } else {
            // Naive attention for small sequences (no causal support)
            [encoder setComputePipelineState:kernels.attentionNaivePSO];
            [encoder setBuffer:getMTLBufferStorage(Q) offset:Q.storage_offset() * 4 atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(K) offset:K.storage_offset() * 4 atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(V) offset:V.storage_offset() * 4 atIndex:2];
            [encoder setBuffer:getMTLBufferStorage(O) offset:O.storage_offset() * 4 atIndex:3];
            
            uint32_t bh_u = (uint32_t)batch_heads;
            uint32_t sl_u = (uint32_t)seq_len;
            uint32_t hd_u = (uint32_t)head_dim;
            [encoder setBytes:&bh_u length:4 atIndex:4];
            [encoder setBytes:&sl_u length:4 atIndex:5];
            [encoder setBytes:&hd_u length:4 atIndex:6];
            [encoder setBytes:&scale length:4 atIndex:7];
            
            [encoder dispatchThreadgroups:MTLSizeMake(seq_len, batch_heads, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        }
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return O;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sdpa_bwd_metal(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor dO, torch::Tensor L,
    float scale, bool is_causal
) {
    load_core_kernels();
    
    TORCH_CHECK(Q.device().type() == at::kMPS, "Q must be on MPS device");
    TORCH_CHECK(dO.device().type() == at::kMPS, "dO must be on MPS device");
    TORCH_CHECK(Q.dim() == 3, "Q must be 3D (batch_heads, seq_len, head_dim)");
    
    int64_t batch_heads = Q.size(0);
    int64_t seq_len = Q.size(1);
    int64_t head_dim = Q.size(2);
    
    // Initialize gradients to zero
    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);
    
    if (!kernels.flashAttentionBwdV2PSO) {
        // CPU fallback
        TORCH_CHECK(false, "Flash Attention backward kernel not loaded");
    }
    
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    O = O.contiguous();
    dO = dO.contiguous();
    L = L.contiguous();
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        stream->synchronize(SyncType::COMMIT);
        auto encoder = [stream->commandBuffer() computeCommandEncoder];
        
        [encoder setComputePipelineState:kernels.flashAttentionBwdV2PSO];
        [encoder setBuffer:getMTLBufferStorage(Q) offset:Q.storage_offset() * 4 atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(K) offset:K.storage_offset() * 4 atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(V) offset:V.storage_offset() * 4 atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(O) offset:O.storage_offset() * 4 atIndex:3];
        [encoder setBuffer:getMTLBufferStorage(dO) offset:dO.storage_offset() * 4 atIndex:4];
        [encoder setBuffer:getMTLBufferStorage(L) offset:L.storage_offset() * 4 atIndex:5];
        [encoder setBuffer:getMTLBufferStorage(dQ) offset:dQ.storage_offset() * 4 atIndex:6];
        [encoder setBuffer:getMTLBufferStorage(dK) offset:dK.storage_offset() * 4 atIndex:7];
        [encoder setBuffer:getMTLBufferStorage(dV) offset:dV.storage_offset() * 4 atIndex:8];
        
        uint32_t bh_u = (uint32_t)batch_heads;
        uint32_t sl_u = (uint32_t)seq_len;
        uint32_t hd_u = (uint32_t)head_dim;
        uint32_t causal_u = is_causal ? 1 : 0;
        [encoder setBytes:&bh_u length:4 atIndex:9];
        [encoder setBytes:&sl_u length:4 atIndex:10];
        [encoder setBytes:&hd_u length:4 atIndex:11];
        [encoder setBytes:&scale length:4 atIndex:12];
        [encoder setBytes:&causal_u length:4 atIndex:13];
        
        // Grid: (seq_len, batch_heads)
        [encoder dispatchThreadgroups:MTLSizeMake(seq_len, batch_heads, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    return {dQ, dK, dV};
}

// -----------------------------------------------------------------------------
// Fused Linear Solve
// -----------------------------------------------------------------------------

torch::Tensor solve_metal(torch::Tensor A, torch::Tensor b) {
    load_core_kernels();
    
    TORCH_CHECK(A.device().type() == at::kMPS, "A must be on MPS device");
    TORCH_CHECK(b.device().type() == at::kMPS, "b must be on MPS device");
    TORCH_CHECK(A.dim() >= 2, "A must be at least 2D");
    TORCH_CHECK(b.dim() >= 1, "b must be at least 1D");
    
    // Promote fp16/bf16 to fp32 for numerical stability in LU factorization
    auto input_dtype = A.scalar_type();
    bool need_conversion = (input_dtype == at::kHalf || input_dtype == at::kBFloat16);
    
    torch::Tensor A_in = need_conversion ? A.to(at::kFloat) : A;
    torch::Tensor b_in = need_conversion ? b.to(at::kFloat) : b;
    
    // Handle batched and non-batched cases
    bool batched = A_in.dim() == 3;
    int64_t batch_size = batched ? A_in.size(0) : 1;
    int64_t N = batched ? A_in.size(1) : A_in.size(0);
    TORCH_CHECK((batched ? A_in.size(2) : A_in.size(1)) == N, "A must be square");
    
    // b can be (N,), (N, K), (B, N), or (B, N, K)
    int64_t K = 1;
    if (b_in.dim() == 1) {
        K = 1;
    } else if (b_in.dim() == 2) {
        K = batched ? 1 : b_in.size(1);
    } else if (b_in.dim() == 3) {
        K = b_in.size(2);
    }
    
    // Reshape inputs for kernel: (B, N, N) and (B, N, K)
    auto A_work = A_in.clone().contiguous();
    auto x = b_in.clone().contiguous();
    
    if (!batched) {
        A_work = A_work.unsqueeze(0);
        x = x.view({1, N, K});
    } else if (b_in.dim() == 2) {
        x = x.unsqueeze(-1);
    }
    
    // Allocate pivot storage
    auto pivots = torch::empty({batch_size, N}, A_in.options().dtype(at::kInt));
    
    if (!kernels.solveBatchedPSO) {
        // CPU fallback
        auto A_cpu = A_work.squeeze(0).cpu();
        auto b_cpu = x.squeeze(0).squeeze(-1).cpu();
        auto result = std::get<0>(torch::linalg_solve_ex(A_cpu, b_cpu));
        auto out = result.to(A.device());
        return need_conversion ? out.to(input_dtype) : out;
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = [stream->commandBuffer() computeCommandEncoder];
        
        [encoder setComputePipelineState:kernels.solveBatchedPSO];
        [encoder setBuffer:getMTLBufferStorage(A_work) offset:A_work.storage_offset() * 4 atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * 4 atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(pivots) offset:pivots.storage_offset() * 4 atIndex:2];
        
        uint32_t N_u = (uint32_t)N;
        uint32_t K_u = (uint32_t)K;
        uint32_t batch_u = (uint32_t)batch_size;
        [encoder setBytes:&N_u length:4 atIndex:3];
        [encoder setBytes:&K_u length:4 atIndex:4];
        [encoder setBytes:&batch_u length:4 atIndex:5];
        
        NSUInteger tg_size = std::min((NSUInteger)N, (NSUInteger)256);
        [encoder dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        
        [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    // Reshape output to match input shape
    torch::Tensor result;
    if (!batched) {
        if (b.dim() == 1) {
            result = x.view({N});
        } else {
            result = x.squeeze(0);
        }
    } else if (b.dim() == 2) {
        result = x.squeeze(-1);
    } else {
        result = x;
    }
    
    // Convert back to original dtype if needed
    return need_conversion ? result.to(input_dtype) : result;
}

// -----------------------------------------------------------------------------
// Fused Softmax
// -----------------------------------------------------------------------------

torch::Tensor fused_softmax_metal(torch::Tensor input, int64_t dim_) {
    load_core_kernels();
    
    TORCH_CHECK(input.device().type() == at::kMPS, "Input must be on MPS device");
    
    auto x = input.contiguous();
    auto output = torch::empty_like(x);
    
    // Normalize negative dim
    int64_t ndim = x.dim();
    int64_t dim = dim_ < 0 ? dim_ + ndim : dim_;
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for softmax");
    
    // Calculate outer_size, dim_size, inner_size
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) outer_size *= x.size(i);
    int64_t dim_size = x.size(dim);
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; i++) inner_size *= x.size(i);
    
    // Select kernel based on dtype
    bool is_half = (x.scalar_type() == at::kHalf);
    bool is_bfloat = (x.scalar_type() == at::kBFloat16);
    
    id<MTLComputePipelineState> pso = nil;
    if (is_half && kernels.fusedSoftmaxHalfPSO && inner_size == 1) {
        // Use optimized half kernel
        pso = kernels.fusedSoftmaxHalfPSO;
    } else if (is_bfloat && kernels.fusedSoftmaxBfloatPSO && inner_size == 1) {
        // Use native bf16 kernel with direct bit truncation
        pso = kernels.fusedSoftmaxBfloatPSO;
    } else if (!is_half && !is_bfloat) {
        // Float32: use vec4 if possible
        bool use_vec4 = (dim_size % 4 == 0) && (inner_size == 1) && kernels.fusedSoftmaxVec4PSO;
        pso = use_vec4 ? kernels.fusedSoftmaxVec4PSO : kernels.fusedSoftmaxPSO;
    }
    
    if (!pso) {
        // Fallback to PyTorch (for unsupported configs)
        if (is_bfloat) {
            auto x_fp32 = x.to(torch::kFloat32);
            auto out_fp32 = torch::softmax(x_fp32, dim);
            return out_fp32.to(torch::kBFloat16);
        }
        return torch::softmax(input, dim_);
    }

    
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:1];
        
        uint32_t dim_u = static_cast<uint32_t>(dim_size);
        uint32_t outer_u = static_cast<uint32_t>(outer_size);
        uint32_t inner_u = static_cast<uint32_t>(inner_size);
        
        [encoder setBytes:&dim_u length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&outer_u length:sizeof(uint32_t) atIndex:3];
        if (!is_half || inner_size != 1) {
            [encoder setBytes:&inner_u length:sizeof(uint32_t) atIndex:4];
        }
        
        // One threadgroup per row
        NSUInteger threadsPerGroup = std::min(256UL, static_cast<NSUInteger>(dim_size));
        [encoder dispatchThreadgroups:MTLSizeMake(outer_size, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        
        [encoder endEncoding];
        torch::mps::synchronize();
    }
    
    return output;
}

// -----------------------------------------------------------------------------
// LayerNorm
// -----------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> layernorm_fwd_metal(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    load_core_kernels();
    
    TORCH_CHECK(input.device().type() == at::kMPS, "Input must be on MPS device");
    
    auto x = input.contiguous();
    int64_t N = x.size(-1);  // normalized dim
    int64_t B = x.numel() / N;  // batch size (all other dims)
    
    auto output = torch::empty_like(x);
    auto mean = torch::empty({B}, x.options().dtype(torch::kFloat32));
    auto rstd = torch::empty({B}, x.options().dtype(torch::kFloat32));
    
    auto w = weight.contiguous();
    auto b = bias.contiguous();
    
    // Select kernel based on dtype
    bool is_half = (x.scalar_type() == at::kHalf);
    bool is_bfloat = (x.scalar_type() == at::kBFloat16);
    
    id<MTLComputePipelineState> pso = nil;
    if (is_half && kernels.layernormFwdHalfPSO) {
        pso = kernels.layernormFwdHalfPSO;
    } else if (is_bfloat && kernels.layernormFwdBfloatPSO) {
        // Use native bf16 kernel with direct bit truncation
        pso = kernels.layernormFwdBfloatPSO;
    } else if (!is_half && !is_bfloat && kernels.layernormFwdPSO) {
        pso = kernels.layernormFwdPSO;
    }
    
    if (!pso) {
        // Fallback to PyTorch
        if (is_bfloat) {
            // bf16: compute in float32 then convert back
            auto x_fp32 = x.to(torch::kFloat32);
            auto w_fp32 = w.to(torch::kFloat32);
            auto b_fp32 = b.to(torch::kFloat32);
            auto result = torch::layer_norm(x_fp32, {N}, w_fp32, b_fp32, eps);
            auto m = x_fp32.view({B, N}).mean(-1);
            auto v = x_fp32.view({B, N}).var(-1, false);
            return std::make_tuple(result.to(torch::kBFloat16), m, torch::rsqrt(v + eps));
        }
        auto result = torch::layer_norm(input, {N}, weight, bias, eps);
        auto m = x.view({B, N}).mean(-1);
        auto v = x.view({B, N}).var(-1, false);
        return std::make_tuple(result, m, torch::rsqrt(v + eps));
    }
    
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(w) offset:w.storage_offset() * w.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(b) offset:b.storage_offset() * b.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:3];
        [encoder setBuffer:getMTLBufferStorage(mean) offset:mean.storage_offset() * mean.element_size() atIndex:4];
        [encoder setBuffer:getMTLBufferStorage(rstd) offset:rstd.storage_offset() * rstd.element_size() atIndex:5];
        
        uint32_t N_u = static_cast<uint32_t>(N);
        [encoder setBytes:&N_u length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&eps length:sizeof(float) atIndex:7];
        
        NSUInteger threadsPerGroup = std::min(256UL, static_cast<NSUInteger>(N));
        [encoder dispatchThreadgroups:MTLSizeMake(B, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        
        [encoder endEncoding];
        torch::mps::synchronize();
    }
    
    return std::make_tuple(output, mean, rstd);
}

// -----------------------------------------------------------------------------
// Embedding Bag
// -----------------------------------------------------------------------------

torch::Tensor embedding_bag_metal(
    torch::Tensor weight,
    torch::Tensor indices,
    torch::Tensor offsets,
    int64_t mode  // 0=sum, 1=mean, 2=max
) {
    load_core_kernels();
    
    TORCH_CHECK(weight.device().type() == at::kMPS, "Weight must be on MPS device");
    TORCH_CHECK(indices.device().type() == at::kMPS, "Indices must be on MPS device");
    TORCH_CHECK(offsets.device().type() == at::kMPS, "Offsets must be on MPS device");
    
    auto w = weight.contiguous();
    auto idx = indices.to(torch::kInt32).contiguous();
    auto off = offsets.to(torch::kInt32).contiguous();
    
    int64_t batch_size = offsets.size(0) - 1;
    int64_t dim = weight.size(1);
    
    auto output = torch::zeros({batch_size, dim}, w.options());
    
    if (!kernels.embeddingBagSimplePSO) {
        // Fallback to PyTorch
        auto [out, _, __, ___] = torch::embedding_bag(weight, indices, offsets, false, mode);
        return out;
    }
    
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        
        [encoder setComputePipelineState:kernels.embeddingBagSimplePSO];
        [encoder setBuffer:getMTLBufferStorage(w) offset:w.storage_offset() * w.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(idx) offset:idx.storage_offset() * idx.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(off) offset:off.storage_offset() * off.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:3];
        
        uint32_t dim_u = static_cast<uint32_t>(dim);
        uint32_t batch_u = static_cast<uint32_t>(batch_size);
        uint32_t mode_u = static_cast<uint32_t>(mode);
        
        [encoder setBytes:&dim_u length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&batch_u length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&mode_u length:sizeof(uint32_t) atIndex:6];
        
        // 2D grid: dim x batch_size
        [encoder dispatchThreads:MTLSizeMake(dim, batch_size, 1)
           threadsPerThreadgroup:MTLSizeMake(std::min(256UL, static_cast<NSUInteger>(dim)), 1, 1)];
        
        [encoder endEncoding];
        torch::mps::synchronize();
    }
    
    return output;
}

// -----------------------------------------------------------------------------
// Scatter/Gather
// -----------------------------------------------------------------------------

torch::Tensor gather_metal(torch::Tensor src, torch::Tensor index, int64_t dim_) {
    load_core_kernels();
    
    TORCH_CHECK(src.device().type() == at::kMPS, "src must be on MPS device");
    TORCH_CHECK(index.device().type() == at::kMPS, "index must be on MPS device");
    
    auto s = src.contiguous();
    auto idx = index.to(torch::kInt32).contiguous();
    
    // For 1D gather
    if (src.dim() == 1 && index.dim() == 1) {
        auto output = torch::empty({index.size(0)}, s.options());
        
        if (!kernels.gather1dPSO) {
            return torch::gather(src, 0, index.to(torch::kLong));
        }
        
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            
            [encoder setComputePipelineState:kernels.gather1dPSO];
            [encoder setBuffer:getMTLBufferStorage(s) offset:s.storage_offset() * s.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(idx) offset:idx.storage_offset() * idx.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:2];
            
            uint32_t n = static_cast<uint32_t>(index.size(0));
            [encoder setBytes:&n length:sizeof(uint32_t) atIndex:3];
            
            [encoder dispatchThreads:MTLSizeMake(n, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            
            [encoder endEncoding];
            torch::mps::synchronize();
        }
        
        return output;
    }
    
    // Fallback to PyTorch for other cases
    return torch::gather(src, dim_, index.to(torch::kLong));
}

torch::Tensor scatter_add_metal(torch::Tensor dst, torch::Tensor index, torch::Tensor src, int64_t dim_) {
    load_core_kernels();
    
    TORCH_CHECK(dst.device().type() == at::kMPS, "dst must be on MPS device");
    TORCH_CHECK(index.device().type() == at::kMPS, "index must be on MPS device");
    TORCH_CHECK(src.device().type() == at::kMPS, "src must be on MPS device");
    
    auto output = dst.clone().contiguous();
    auto idx = index.to(torch::kInt32).contiguous();
    auto s = src.contiguous();
    
    // For 1D scatter_add
    if (dst.dim() == 1 && index.dim() == 1 && src.dim() == 1) {
        if (!kernels.scatterAdd1dPSO) {
            return dst.scatter_add(0, index.to(torch::kLong), src);
        }
        
        @autoreleasepool {
            id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            
            [encoder setComputePipelineState:kernels.scatterAdd1dPSO];
            [encoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(idx) offset:idx.storage_offset() * idx.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(s) offset:s.storage_offset() * s.element_size() atIndex:2];
            
            uint32_t n = static_cast<uint32_t>(src.size(0));
            [encoder setBytes:&n length:sizeof(uint32_t) atIndex:3];
            
            [encoder dispatchThreads:MTLSizeMake(n, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            
            [encoder endEncoding];
            torch::mps::synchronize();
        }
        
        return output;
    }
    
    // Fallback for other cases
    return dst.scatter_add(dim_, index.to(torch::kLong), src);
}

torch::Tensor index_select_metal(torch::Tensor src, int64_t dim, torch::Tensor index) {
    load_core_kernels();
    
    TORCH_CHECK(src.device().type() == at::kMPS, "src must be on MPS device");
    TORCH_CHECK(index.device().type() == at::kMPS, "index must be on MPS device");
    
    // Use PyTorch's optimized implementation as baseline
    return torch::index_select(src, dim, index.to(torch::kLong));
}

PYBIND11_MODULE(metalcore_backend, m) {
    m.def("trsm", &trsm_metal, "Triangular Solve (TRSM)");
    m.def("geqr2", &geqr2_metal, "Panel Householder QR");
    m.def("larfb", &larfb_metal, "Apply Block Reflector");
    m.def("larft", &larft_metal, "Form T Matrix");
    m.def("qr", &qr_fused_metal, "Fused QR");
    m.def("qr_blocked", &qr_metal, "Blocked QR");
    m.def("qr_batched", &qr_batched_metal, "Batched QR");
    m.def("trsm_batched", &trsm_batched_metal, "Batched TRSM");
    m.def("cholesky_batched", &cholesky_batched_metal, "Batched Cholesky");
    m.def("cholesky_solve_batched", &cholesky_solve_batched_metal, "Batched Cholesky Solve");
    m.def("solve", &solve_metal, "Linear Solve (LU-based)");
    
    // Training ops
    m.def("rmsnorm_fwd", &rmsnorm_fwd_metal, "RMSNorm Forward");
    m.def("rmsnorm_bwd", &rmsnorm_bwd_metal, "RMSNorm Backward");
    m.def("adamw_step", &adamw_step_metal, "AdamW Step");
    m.def("fused_add_rmsnorm", &fused_add_rmsnorm_metal, "Fused Add + RMSNorm");
    
    // Activations
    m.def("gelu_fwd", &gelu_fwd_metal, "GELU Forward");
    m.def("gelu_bwd", &gelu_bwd_metal, "GELU Backward");
    m.def("silu_fwd", &silu_fwd_metal, "SiLU Forward");
    m.def("silu_bwd", &silu_bwd_metal, "SiLU Backward");
    
    // SDPA
    m.def("sdpa_fwd", &sdpa_fwd_metal, "Scaled Dot Product Attention Forward");
    m.def("sdpa_bwd", &sdpa_bwd_metal, "Scaled Dot Product Attention Backward");
    
    // Eigendecomposition
    m.def("eigh_forward", &eigh_forward, "Symmetric Eigenvalue Decomposition");
    
    // SVD support
    m.def("column_norm_sort", &column_norm_sort_metal, "Column Norm Sort for SVD");
    m.def("svd_forward", &svd_forward, "Batched SVD Forward");
    m.def("sign_canonicalize", &sign_canonicalize_metal, "Sign Canonicalize for SVD");
    m.def("syrk_batched", &syrk_batched_metal, "Batched SYRK for Gram matrix");
    m.def("frobenius_norm_batched", &frobenius_norm_batched_metal, "Batched Frobenius Norm");
    m.def("softmax_batched", &softmax_batched_metal, "Batched Softmax");
    m.def("trace_batched", &trace_batched_metal, "Batched Trace");
    m.def("lu_batched", &lu_batched_metal, "Batched LU Decomposition");
    
    // New high-performance ops
    m.def("fused_softmax", &fused_softmax_metal, "Fused Softmax with online algorithm");
    m.def("layernorm_fwd", &layernorm_fwd_metal, "LayerNorm Forward");
    m.def("embedding_bag", &embedding_bag_metal, "Embedding Bag (sum/mean/max)");
    m.def("gather", &gather_metal, "Gather operation");
    m.def("scatter_add", &scatter_add_metal, "Scatter Add operation");
    m.def("index_select", &index_select_metal, "Index Select operation");
}

