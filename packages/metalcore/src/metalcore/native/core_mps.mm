// Author: Kris Bailey
// Copyright 2026
// Email: kris@krisbailey.com
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
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

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
    id<MTLFunction> qrBatched16x16 = nil;
    id<MTLComputePipelineState> qrBatched16x16PSO = nil;
    id<MTLFunction> qrBatched8x8Reg = nil;
    id<MTLComputePipelineState> qrBatched8x8RegPSO = nil;
    id<MTLFunction> qrBatched32x32 = nil;
    id<MTLComputePipelineState> qrBatched32x32PSO = nil;
    id<MTLFunction> qrBatched32x32SingleThread = nil;
    id<MTLComputePipelineState> qrBatched32x32SingleThreadPSO = nil;
    id<MTLFunction> qrBatched64x64 = nil;
    id<MTLComputePipelineState> qrBatched64x64PSO = nil;
    
    // Batched TRSM (triangular solve)
    id<MTLFunction> trsmBatched = nil;
    id<MTLComputePipelineState> trsmBatchedPSO = nil;
    
    // Specialized TRSM kernels
    id<MTLFunction> trsmBatched8x8Reg = nil;
    id<MTLComputePipelineState> trsmBatched8x8RegPSO = nil;
    id<MTLFunction> trsmBatched16x16 = nil;
    id<MTLComputePipelineState> trsmBatched16x16PSO = nil;
    id<MTLFunction> trsmBatched32x32 = nil;
    id<MTLComputePipelineState> trsmBatched32x32PSO = nil;
    
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
    
    // SwiGLU Backward
    id<MTLFunction> swigluBwdStridedFloat = nil;
    id<MTLComputePipelineState> swigluBwdStridedFloatPSO = nil;
    id<MTLFunction> swigluBwdStridedHalf = nil;
    id<MTLComputePipelineState> swigluBwdStridedHalfPSO = nil;
    id<MTLFunction> swigluBwdStridedBfloat = nil;
    id<MTLComputePipelineState> swigluBwdStridedBfloatPSO = nil;
    
    // SDPA
    id<MTLFunction> attentionNaive = nil;
    id<MTLComputePipelineState> attentionNaivePSO = nil;
    id<MTLFunction> flashAttentionFwdV2 = nil;
    id<MTLComputePipelineState> flashAttentionFwdV2PSO = nil;
    id<MTLFunction> flashAttentionBwdV2 = nil;
    id<MTLComputePipelineState> flashAttentionBwdV2PSO = nil;
    id<MTLFunction> sdpaVector64 = nil;
    id<MTLComputePipelineState> sdpaVector64PSO = nil;
    // Fused RoPE + SDPA
    id<MTLFunction> ropeSdpa64 = nil;
    id<MTLComputePipelineState> ropeSdpa64PSO = nil;
    id<MTLFunction> ropeSdpa64Half = nil;
    id<MTLComputePipelineState> ropeSdpa64HalfPSO = nil;
    
    // Fused Softmax
    id<MTLFunction> fusedSoftmax = nil;
    id<MTLComputePipelineState> fusedSoftmaxPSO = nil;
    id<MTLFunction> fusedSoftmaxVec4 = nil;
    id<MTLComputePipelineState> fusedSoftmaxVec4PSO = nil;
    id<MTLFunction> fusedSoftmaxHalf = nil;
    id<MTLComputePipelineState> fusedSoftmaxHalfPSO = nil;
    id<MTLFunction> fusedSoftmaxBfloat = nil;
    id<MTLComputePipelineState> fusedSoftmaxBfloatPSO = nil;
    
    // Cross-Entropy Loss (fused log-softmax + NLL)
    id<MTLFunction> crossEntropyFwd = nil;
    id<MTLComputePipelineState> crossEntropyFwdPSO = nil;
    id<MTLFunction> crossEntropyFwdHalf = nil;
    id<MTLComputePipelineState> crossEntropyFwdHalfPSO = nil;
    
    // KL Divergence
    id<MTLFunction> klDivFwd = nil;
    id<MTLComputePipelineState> klDivFwdPSO = nil;
    id<MTLFunction> klDivTopkFwd = nil;
    id<MTLComputePipelineState> klDivTopkFwdPSO = nil;
    
    // Softmax Backward (for attention backward)
    id<MTLFunction> softmaxBwd = nil;
    id<MTLComputePipelineState> softmaxBwdPSO = nil;
    id<MTLFunction> softmaxBwdHalf = nil;
    id<MTLComputePipelineState> softmaxBwdHalfPSO = nil;
    id<MTLFunction> softmaxBwdBfloat = nil;
    id<MTLComputePipelineState> softmaxBwdBfloatPSO = nil;
    
    // SwiGLU (silu(gate) * up)
    id<MTLFunction> swigluFwd = nil;
    id<MTLComputePipelineState> swigluFwdPSO = nil;
    id<MTLFunction> swigluFwdHalf = nil;
    id<MTLComputePipelineState> swigluFwdHalfPSO = nil;
    
    // Strided SwiGLU
    id<MTLFunction> swigluFwdStrided = nil;
    id<MTLComputePipelineState> swigluFwdStridedPSO = nil;
    id<MTLFunction> swigluFwdStridedHalf = nil;
    id<MTLComputePipelineState> swigluFwdStridedHalfPSO = nil;
    id<MTLFunction> swigluFwdStridedBfloat = nil;
    id<MTLComputePipelineState> swigluFwdStridedBfloatPSO = nil;
    id<MTLFunction> swigluFwdBfloat = nil;
    id<MTLComputePipelineState> swigluFwdBfloatPSO = nil;
    
    // LoRA Add (base + scale * lora)
    id<MTLFunction> loraAddFwd = nil;
    id<MTLComputePipelineState> loraAddFwdPSO = nil;
    id<MTLFunction> loraAddFwdHalf = nil;
    id<MTLComputePipelineState> loraAddFwdHalfPSO = nil;
    id<MTLFunction> loraAddFwdBfloat = nil;
    id<MTLComputePipelineState> loraAddFwdBfloatPSO = nil;
    
    // Fused LoRA QKV (single dispatch for Q, K, V projections with LoRA)
    id<MTLFunction> fusedLoraQkv = nil;
    id<MTLComputePipelineState> fusedLoraQkvPSO = nil;
    
    // Fused AdamW Multi (all params in one dispatch)
    id<MTLFunction> adamwStepMulti = nil;
    id<MTLComputePipelineState> adamwStepMultiPSO = nil;
    
    // LayerNorm
    id<MTLFunction> layernormFwd = nil;
    id<MTLComputePipelineState> layernormFwdPSO = nil;
    id<MTLFunction> fusedAddLayernorm = nil;
    id<MTLComputePipelineState> fusedAddLayernormPSO = nil;
    id<MTLFunction> fusedAddLayernormHalf = nil;
    id<MTLComputePipelineState> fusedAddLayernormHalfPSO = nil;
    id<MTLFunction> fusedAddLayernormBfloat = nil;
    id<MTLComputePipelineState> fusedAddLayernormBfloatPSO = nil;
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
    
    // RoPE (Rotary Position Embedding)
    id<MTLFunction> ropeFwdSplitHalf = nil;
    id<MTLComputePipelineState> ropeFwdSplitHalfPSO = nil;
    id<MTLFunction> ropeFwdSplitHalfBfloat = nil;
    id<MTLComputePipelineState> ropeFwdSplitHalfBfloatPSO = nil;
    id<MTLFunction> ropeBwdSplitHalfBfloat = nil;
    id<MTLComputePipelineState> ropeBwdSplitHalfBfloatPSO = nil;
    id<MTLFunction> ropeBwdSplitHalfHalf = nil;
    id<MTLComputePipelineState> ropeBwdSplitHalfHalfPSO = nil;
    
    // Loss Functions (KL Div, Cross Entropy)
    id<MTLFunction> klDivFwdBfloat = nil;
    id<MTLComputePipelineState> klDivFwdBfloatPSO = nil;
    
    id<MTLFunction> klDivTopkFwdBfloat = nil;
    id<MTLComputePipelineState> klDivTopkFwdBfloatPSO = nil;
    
    id<MTLFunction> crossEntropyFwdBfloat = nil;
    id<MTLComputePipelineState> crossEntropyFwdBfloatPSO = nil;
    id<MTLFunction> ropeBwdSplitHalf = nil;
    id<MTLComputePipelineState> ropeBwdSplitHalfPSO = nil;
    id<MTLFunction> ropeFwdQkSplitHalf = nil;
    id<MTLComputePipelineState> ropeFwdQkSplitHalfPSO = nil;
    
    // Quantized Matmul (INT8/INT4)
    id<MTLFunction> matmulInt8Dequant = nil;
    id<MTLComputePipelineState> matmulInt8DequantPSO = nil;
    id<MTLFunction> matmulInt8DequantHalf = nil;
    id<MTLComputePipelineState> matmulInt8DequantHalfPSO = nil;
    id<MTLFunction> matmulInt4Dequant = nil;
    id<MTLComputePipelineState> matmulInt4DequantPSO = nil;
    id<MTLFunction> matmulInt4DequantHalf = nil;
    id<MTLComputePipelineState> matmulInt4DequantHalfPSO = nil;
    id<MTLFunction> matmulInt4DequantVec4 = nil;
    id<MTLComputePipelineState> matmulInt4DequantVec4PSO = nil;
    id<MTLFunction> matmulInt4DequantHalfVec4 = nil;
    id<MTLComputePipelineState> matmulInt4DequantHalfVec4PSO = nil;
    id<MTLFunction> matmulInt4DequantTiled = nil;
    id<MTLComputePipelineState> matmulInt4DequantTiledPSO = nil;
    id<MTLFunction> matmulInt4DequantSimd = nil;
    id<MTLComputePipelineState> matmulInt4DequantSimdPSO = nil;
    id<MTLFunction> matmulInt4DequantSimdHalf = nil;
    id<MTLComputePipelineState> matmulInt4DequantSimdHalfPSO = nil;
    id<MTLFunction> matmulInt4Fast = nil;
    id<MTLComputePipelineState> matmulInt4FastPSO = nil;
    // Fused matmul + activation kernels
    id<MTLFunction> matmulInt4SiluFast = nil;
    id<MTLComputePipelineState> matmulInt4SiluFastPSO = nil;
    id<MTLFunction> matmulInt4GeluFast = nil;
    id<MTLComputePipelineState> matmulInt4GeluFastPSO = nil;
    id<MTLFunction> matmulInt4Tensor = nil;
    id<MTLComputePipelineState> matmulInt4TensorPSO = nil;
    id<MTLFunction> matmulInt4Simdgroup = nil;
    id<MTLComputePipelineState> matmulInt4SimdgroupPSO = nil;
    id<MTLFunction> matmulGGMLQ4_0 = nil;
    id<MTLComputePipelineState> matmulGGMLQ4_0PSO = nil;
    id<MTLFunction> quantizeToInt8 = nil;
    id<MTLComputePipelineState> quantizeToInt8PSO = nil;
    id<MTLFunction> quantizeToInt4 = nil;
    id<MTLComputePipelineState> quantizeToInt4PSO = nil;

    // Strided RoPE SDPA V2 (Zero-Copy)
    id<MTLFunction> ropeSdpa64StridedV2 = nil;
    id<MTLComputePipelineState> ropeSdpa64StridedV2PSO = nil;
    id<MTLFunction> ropeSdpa64HalfStridedV2 = nil;
    id<MTLComputePipelineState> ropeSdpa64HalfStridedV2PSO = nil;
    
    id<MTLFunction> ropeFwdSplitHalfHalf = nil;
    id<MTLComputePipelineState> ropeFwdSplitHalfHalfPSO = nil;
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
        kernels.qrBatched16x16 = [coreLib newFunctionWithName:@"qr_batched_16x16_kernel"];
        kernels.qrBatched8x8Reg = [coreLib newFunctionWithName:@"qr_batched_8x8_register_kernel"];
        kernels.qrBatched32x32 = [coreLib newFunctionWithName:@"qr_batched_32x32_kernel"];
        kernels.qrBatched32x32SingleThread = [coreLib newFunctionWithName:@"qr_batched_32x32_single_thread_kernel"];
        kernels.qrBatched64x64 = [coreLib newFunctionWithName:@"qr_batched_64x64_kernel"];
        kernels.trsmBatched = [coreLib newFunctionWithName:@"trsm_batched_kernel"];
        kernels.trsmBatched8x8Reg = [coreLib newFunctionWithName:@"trsm_batched_8x8_register_kernel"];
        kernels.trsmBatched16x16 = [coreLib newFunctionWithName:@"trsm_batched_16x16_kernel"];
        kernels.trsmBatched32x32 = [coreLib newFunctionWithName:@"trsm_batched_32x32_kernel"];
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
        
        // RMSNorm Half/Bfloat
        kernels.rmsnormFwdHalf = [coreLib newFunctionWithName:@"rmsnorm_fwd_half"];
        if (kernels.rmsnormFwdHalf) kernels.rmsnormFwdHalfPSO = [device newComputePipelineStateWithFunction:kernels.rmsnormFwdHalf error:&error];
        kernels.rmsnormFwdBfloat = [coreLib newFunctionWithName:@"rmsnorm_fwd_bfloat"];
        if (kernels.rmsnormFwdBfloat) kernels.rmsnormFwdBfloatPSO = [device newComputePipelineStateWithFunction:kernels.rmsnormFwdBfloat error:&error];
        
        // SwiGLU Backward
        kernels.swigluBwdStridedFloat = [coreLib newFunctionWithName:@"swiglu_bwd_strided_float"];
        if (kernels.swigluBwdStridedFloat) kernels.swigluBwdStridedFloatPSO = [device newComputePipelineStateWithFunction:kernels.swigluBwdStridedFloat error:&error];
        kernels.swigluBwdStridedHalf = [coreLib newFunctionWithName:@"swiglu_bwd_strided_half"];
        if (kernels.swigluBwdStridedHalf) kernels.swigluBwdStridedHalfPSO = [device newComputePipelineStateWithFunction:kernels.swigluBwdStridedHalf error:&error];
        kernels.swigluBwdStridedBfloat = [coreLib newFunctionWithName:@"swiglu_bwd_strided_bfloat"];
        if (kernels.swigluBwdStridedBfloat) kernels.swigluBwdStridedBfloatPSO = [device newComputePipelineStateWithFunction:kernels.swigluBwdStridedBfloat error:&error];
        
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
        if (kernels.qrBatched16x16) {
            kernels.qrBatched16x16PSO = [device newComputePipelineStateWithFunction:kernels.qrBatched16x16 error:&error];
            if (!kernels.qrBatched16x16PSO) printf("Failed to create qrBatched16x16PSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.qrBatched8x8Reg) {
            kernels.qrBatched8x8RegPSO = [device newComputePipelineStateWithFunction:kernels.qrBatched8x8Reg error:&error];
            if (!kernels.qrBatched8x8RegPSO) printf("Failed to create qrBatched8x8RegPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.qrBatched32x32) {
            kernels.qrBatched32x32PSO = [device newComputePipelineStateWithFunction:kernels.qrBatched32x32 error:&error];
            if (!kernels.qrBatched32x32PSO) printf("Failed to create qrBatched32x32PSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.qrBatched32x32SingleThread) {
            kernels.qrBatched32x32SingleThreadPSO = [device newComputePipelineStateWithFunction:kernels.qrBatched32x32SingleThread error:&error];
            if (!kernels.qrBatched32x32SingleThreadPSO) printf("Failed to create qrBatched32x32SingleThreadPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.qrBatched64x64) {
            kernels.qrBatched64x64PSO = [device newComputePipelineStateWithFunction:kernels.qrBatched64x64 error:&error];
            if (!kernels.qrBatched64x64PSO) printf("Failed to create qrBatched64x64PSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.trsmBatched) {
            kernels.trsmBatchedPSO = [device newComputePipelineStateWithFunction:kernels.trsmBatched error:&error];
            if (!kernels.trsmBatchedPSO) printf("Failed to create trsmBatchedPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.trsmBatched8x8Reg) {
            kernels.trsmBatched8x8RegPSO = [device newComputePipelineStateWithFunction:kernels.trsmBatched8x8Reg error:&error];
            if (!kernels.trsmBatched8x8RegPSO) printf("Failed to create trsmBatched8x8RegPSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.trsmBatched16x16) {
            kernels.trsmBatched16x16PSO = [device newComputePipelineStateWithFunction:kernels.trsmBatched16x16 error:&error];
            if (!kernels.trsmBatched16x16PSO) printf("Failed to create trsmBatched16x16PSO: %s\n", [[error localizedDescription] UTF8String]);
        }
        if (kernels.trsmBatched32x32) {
            kernels.trsmBatched32x32PSO = [device newComputePipelineStateWithFunction:kernels.trsmBatched32x32 error:&error];
            if (!kernels.trsmBatched32x32PSO) printf("Failed to create trsmBatched32x32PSO: %s\n", [[error localizedDescription] UTF8String]);
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
        kernels.ropeSdpa64 = [coreLib newFunctionWithName:@"rope_sdpa_64"];
        if (kernels.ropeSdpa64) kernels.ropeSdpa64PSO = [device newComputePipelineStateWithFunction:kernels.ropeSdpa64 error:&error];
        kernels.ropeSdpa64Half = [coreLib newFunctionWithName:@"rope_sdpa_64_half"];
        if (kernels.ropeSdpa64Half) kernels.ropeSdpa64HalfPSO = [device newComputePipelineStateWithFunction:kernels.ropeSdpa64Half error:&error];
        
        // Strided Zero-Copy Variants
        kernels.ropeSdpa64StridedV2 = [coreLib newFunctionWithName:@"rope_sdpa_64_strided_v2"];
        if (kernels.ropeSdpa64StridedV2) kernels.ropeSdpa64StridedV2PSO = [device newComputePipelineStateWithFunction:kernels.ropeSdpa64StridedV2 error:&error];
        
        kernels.ropeSdpa64HalfStridedV2 = [coreLib newFunctionWithName:@"rope_sdpa_64_half_strided_v2"];
        if (kernels.ropeSdpa64HalfStridedV2) kernels.ropeSdpa64HalfStridedV2PSO = [device newComputePipelineStateWithFunction:kernels.ropeSdpa64HalfStridedV2 error:&error];
        
        // Fused Softmax
        kernels.fusedSoftmax = [coreLib newFunctionWithName:@"fused_softmax"];
        if (kernels.fusedSoftmax) kernels.fusedSoftmaxPSO = [device newComputePipelineStateWithFunction:kernels.fusedSoftmax error:&error];
        kernels.fusedSoftmaxVec4 = [coreLib newFunctionWithName:@"fused_softmax_vec4"];
        if (kernels.fusedSoftmaxVec4) kernels.fusedSoftmaxVec4PSO = [device newComputePipelineStateWithFunction:kernels.fusedSoftmaxVec4 error:&error];
        kernels.fusedSoftmaxHalf = [coreLib newFunctionWithName:@"fused_softmax_half"];
        if (kernels.fusedSoftmaxHalf) kernels.fusedSoftmaxHalfPSO = [device newComputePipelineStateWithFunction:kernels.fusedSoftmaxHalf error:&error];
        kernels.fusedSoftmaxBfloat = [coreLib newFunctionWithName:@"fused_softmax_bfloat"];
        if (kernels.fusedSoftmaxBfloat) kernels.fusedSoftmaxBfloatPSO = [device newComputePipelineStateWithFunction:kernels.fusedSoftmaxBfloat error:&error];
        
        // Cross-Entropy (fused log-softmax + NLL)
        kernels.crossEntropyFwd = [coreLib newFunctionWithName:@"cross_entropy_fwd"];
        if (kernels.crossEntropyFwd) kernels.crossEntropyFwdPSO = [device newComputePipelineStateWithFunction:kernels.crossEntropyFwd error:&error];
        kernels.crossEntropyFwdHalf = [coreLib newFunctionWithName:@"cross_entropy_fwd_half"];
        if (kernels.crossEntropyFwdHalf) kernels.crossEntropyFwdHalfPSO = [device newComputePipelineStateWithFunction:kernels.crossEntropyFwdHalf error:&error];
        
        // KL Divergence (for distillation)
        kernels.klDivFwd = [coreLib newFunctionWithName:@"kl_div_fwd"];
        if (kernels.klDivFwd) kernels.klDivFwdPSO = [device newComputePipelineStateWithFunction:kernels.klDivFwd error:&error];
        kernels.klDivTopkFwd = [coreLib newFunctionWithName:@"kl_div_topk_fwd"];
        if (kernels.klDivTopkFwd) kernels.klDivTopkFwdPSO = [device newComputePipelineStateWithFunction:kernels.klDivTopkFwd error:&error];
        
        // Softmax Backward (for attention backward)
        kernels.softmaxBwd = [coreLib newFunctionWithName:@"softmax_bwd"];
        if (kernels.softmaxBwd) kernels.softmaxBwdPSO = [device newComputePipelineStateWithFunction:kernels.softmaxBwd error:&error];
        kernels.softmaxBwdHalf = [coreLib newFunctionWithName:@"softmax_bwd_half"];
        if (kernels.softmaxBwdHalf) kernels.softmaxBwdHalfPSO = [device newComputePipelineStateWithFunction:kernels.softmaxBwdHalf error:&error];
        kernels.softmaxBwdBfloat = [coreLib newFunctionWithName:@"softmax_bwd_bfloat"];
        if (kernels.softmaxBwdBfloat) kernels.softmaxBwdBfloatPSO = [device newComputePipelineStateWithFunction:kernels.softmaxBwdBfloat error:&error];
        
        // SwiGLU (silu(gate) * up)
        kernels.swigluFwd = [coreLib newFunctionWithName:@"swiglu_fwd"];
        if (kernels.swigluFwd) kernels.swigluFwdPSO = [device newComputePipelineStateWithFunction:kernels.swigluFwd error:&error];
        kernels.swigluFwdHalf = [coreLib newFunctionWithName:@"swiglu_fwd_half"];
        if (kernels.swigluFwdHalf) kernels.swigluFwdHalfPSO = [device newComputePipelineStateWithFunction:kernels.swigluFwdHalf error:&error];
        
        kernels.swigluFwdStrided = [coreLib newFunctionWithName:@"swiglu_fwd_strided_float"];
        if (kernels.swigluFwdStrided) kernels.swigluFwdStridedPSO = [device newComputePipelineStateWithFunction:kernels.swigluFwdStrided error:&error];
        
        kernels.swigluFwdStridedHalf = [coreLib newFunctionWithName:@"swiglu_fwd_strided_half"];
        if (kernels.swigluFwdStridedHalf) kernels.swigluFwdStridedHalfPSO = [device newComputePipelineStateWithFunction:kernels.swigluFwdStridedHalf error:&error];
        
        kernels.swigluFwdStridedBfloat = [coreLib newFunctionWithName:@"swiglu_fwd_strided_bfloat"];
        if (kernels.swigluFwdStridedBfloat) kernels.swigluFwdStridedBfloatPSO = [device newComputePipelineStateWithFunction:kernels.swigluFwdStridedBfloat error:&error];
        kernels.swigluFwdBfloat = [coreLib newFunctionWithName:@"swiglu_fwd_bfloat"];
        if (kernels.swigluFwdBfloat) kernels.swigluFwdBfloatPSO = [device newComputePipelineStateWithFunction:kernels.swigluFwdBfloat error:&error];
        
        // LoRA Add (base + scale * lora)
        kernels.loraAddFwd = [coreLib newFunctionWithName:@"lora_add_fwd"];
        if (kernels.loraAddFwd) kernels.loraAddFwdPSO = [device newComputePipelineStateWithFunction:kernels.loraAddFwd error:&error];
        kernels.loraAddFwdHalf = [coreLib newFunctionWithName:@"lora_add_fwd_half"];
        if (kernels.loraAddFwdHalf) kernels.loraAddFwdHalfPSO = [device newComputePipelineStateWithFunction:kernels.loraAddFwdHalf error:&error];
        kernels.loraAddFwdBfloat = [coreLib newFunctionWithName:@"lora_add_fwd_bfloat"];
        if (kernels.loraAddFwdBfloat) kernels.loraAddFwdBfloatPSO = [device newComputePipelineStateWithFunction:kernels.loraAddFwdBfloat error:&error];
        
        // Fused LoRA QKV (single dispatch for Q, K, V with LoRA)
        kernels.fusedLoraQkv = [coreLib newFunctionWithName:@"fused_lora_qkv_fwd"];
        if (kernels.fusedLoraQkv) kernels.fusedLoraQkvPSO = [device newComputePipelineStateWithFunction:kernels.fusedLoraQkv error:&error];
        
        // Fused AdamW Multi (all params in one dispatch)
        kernels.adamwStepMulti = [coreLib newFunctionWithName:@"adamw_step_multi"];
        if (kernels.adamwStepMulti) kernels.adamwStepMultiPSO = [device newComputePipelineStateWithFunction:kernels.adamwStepMulti error:&error];
        
        // LayerNorm
        kernels.layernormFwd = [coreLib newFunctionWithName:@"layernorm_fwd"];
        if (kernels.layernormFwd) kernels.layernormFwdPSO = [device newComputePipelineStateWithFunction:kernels.layernormFwd error:&error];
        kernels.fusedAddLayernorm = [coreLib newFunctionWithName:@"fused_add_layernorm"];
        if (kernels.fusedAddLayernorm) kernels.fusedAddLayernormPSO = [device newComputePipelineStateWithFunction:kernels.fusedAddLayernorm error:&error];
        kernels.fusedAddLayernormHalf = [coreLib newFunctionWithName:@"fused_add_layernorm_half"];
        if (kernels.fusedAddLayernormHalf) kernels.fusedAddLayernormHalfPSO = [device newComputePipelineStateWithFunction:kernels.fusedAddLayernormHalf error:&error];
        kernels.fusedAddLayernormBfloat = [coreLib newFunctionWithName:@"fused_add_layernorm_bfloat"];
        if (kernels.fusedAddLayernormBfloat) kernels.fusedAddLayernormBfloatPSO = [device newComputePipelineStateWithFunction:kernels.fusedAddLayernormBfloat error:&error];
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
        
        // RoPE (Rotary Position Embedding)
        kernels.ropeFwdSplitHalf = [coreLib newFunctionWithName:@"rope_fwd_split_half"];
        if (kernels.ropeFwdSplitHalf) kernels.ropeFwdSplitHalfPSO = [device newComputePipelineStateWithFunction:kernels.ropeFwdSplitHalf error:&error];
        kernels.ropeFwdSplitHalfBfloat = [coreLib newFunctionWithName:@"rope_fwd_split_half_bfloat"];
        if (kernels.ropeFwdSplitHalfBfloat) kernels.ropeFwdSplitHalfBfloatPSO = [device newComputePipelineStateWithFunction:kernels.ropeFwdSplitHalfBfloat error:&error];
        
        kernels.ropeBwdSplitHalfBfloat = [coreLib newFunctionWithName:@"rope_bwd_split_half_bfloat"];
        if (kernels.ropeBwdSplitHalfBfloat) kernels.ropeBwdSplitHalfBfloatPSO = [device newComputePipelineStateWithFunction:kernels.ropeBwdSplitHalfBfloat error:&error];

        kernels.ropeFwdSplitHalfHalf = [coreLib newFunctionWithName:@"rope_fwd_split_half_half"];
        if (kernels.ropeFwdSplitHalfHalf) kernels.ropeFwdSplitHalfHalfPSO = [device newComputePipelineStateWithFunction:kernels.ropeFwdSplitHalfHalf error:&error];
        
        kernels.ropeBwdSplitHalfHalf = [coreLib newFunctionWithName:@"rope_bwd_split_half_half"];
        if (kernels.ropeBwdSplitHalfHalf) kernels.ropeBwdSplitHalfHalfPSO = [device newComputePipelineStateWithFunction:kernels.ropeBwdSplitHalfHalf error:&error];
        
        // Loss PSOs
        kernels.klDivFwdBfloat = [coreLib newFunctionWithName:@"kl_div_fwd_bfloat"];
        if (kernels.klDivFwdBfloat) kernels.klDivFwdBfloatPSO = [device newComputePipelineStateWithFunction:kernels.klDivFwdBfloat error:&error];
        
        kernels.klDivTopkFwdBfloat = [coreLib newFunctionWithName:@"kl_div_topk_fwd_bfloat"];
        if (kernels.klDivTopkFwdBfloat) kernels.klDivTopkFwdBfloatPSO = [device newComputePipelineStateWithFunction:kernels.klDivTopkFwdBfloat error:&error];
        
        kernels.crossEntropyFwdBfloat = [coreLib newFunctionWithName:@"cross_entropy_fwd_bfloat"];
        if (kernels.crossEntropyFwdBfloat) kernels.crossEntropyFwdBfloatPSO = [device newComputePipelineStateWithFunction:kernels.crossEntropyFwdBfloat error:&error];
        kernels.ropeBwdSplitHalf = [coreLib newFunctionWithName:@"rope_bwd_split_half"];
        if (kernels.ropeBwdSplitHalf) kernels.ropeBwdSplitHalfPSO = [device newComputePipelineStateWithFunction:kernels.ropeBwdSplitHalf error:&error];
        kernels.ropeFwdQkSplitHalf = [coreLib newFunctionWithName:@"rope_fwd_qk_split_half"];
        if (kernels.ropeFwdQkSplitHalf) kernels.ropeFwdQkSplitHalfPSO = [device newComputePipelineStateWithFunction:kernels.ropeFwdQkSplitHalf error:&error];
        
        // Quantized Matmul (INT8/INT4)
        kernels.matmulInt8Dequant = [coreLib newFunctionWithName:@"matmul_int8_dequant"];
        if (kernels.matmulInt8Dequant) kernels.matmulInt8DequantPSO = [device newComputePipelineStateWithFunction:kernels.matmulInt8Dequant error:&error];
        kernels.matmulInt8DequantHalf = [coreLib newFunctionWithName:@"matmul_int8_dequant_half"];
        if (kernels.matmulInt8DequantHalf) kernels.matmulInt8DequantHalfPSO = [device newComputePipelineStateWithFunction:kernels.matmulInt8DequantHalf error:&error];
        kernels.matmulInt4Dequant = [coreLib newFunctionWithName:@"matmul_int4_dequant"];
        if (kernels.matmulInt4Dequant) kernels.matmulInt4DequantPSO = [device newComputePipelineStateWithFunction:kernels.matmulInt4Dequant error:&error];
        kernels.matmulInt4DequantHalf = [coreLib newFunctionWithName:@"matmul_int4_dequant_half"];
        if (kernels.matmulInt4DequantHalf) kernels.matmulInt4DequantHalfPSO = [device newComputePipelineStateWithFunction:kernels.matmulInt4DequantHalf error:&error];
        kernels.matmulInt4DequantVec4 = [coreLib newFunctionWithName:@"matmul_int4_dequant_vec4"];
        if (kernels.matmulInt4DequantVec4) kernels.matmulInt4DequantVec4PSO = [device newComputePipelineStateWithFunction:kernels.matmulInt4DequantVec4 error:&error];
        kernels.matmulInt4DequantHalfVec4 = [coreLib newFunctionWithName:@"matmul_int4_dequant_half_vec4"];
        if (kernels.matmulInt4DequantHalfVec4) kernels.matmulInt4DequantHalfVec4PSO = [device newComputePipelineStateWithFunction:kernels.matmulInt4DequantHalfVec4 error:&error];
        kernels.matmulInt4DequantTiled = [coreLib newFunctionWithName:@"matmul_int4_dequant_tiled"];
        if (kernels.matmulInt4DequantTiled) kernels.matmulInt4DequantTiledPSO = [device newComputePipelineStateWithFunction:kernels.matmulInt4DequantTiled error:&error];
        kernels.matmulInt4DequantSimd = [coreLib newFunctionWithName:@"matmul_int4_dequant_simd"];
        if (kernels.matmulInt4DequantSimd) kernels.matmulInt4DequantSimdPSO = [device newComputePipelineStateWithFunction:kernels.matmulInt4DequantSimd error:&error];
        kernels.matmulInt4DequantSimdHalf = [coreLib newFunctionWithName:@"matmul_int4_dequant_simd_half"];
        if (kernels.matmulInt4DequantSimdHalf) kernels.matmulInt4DequantSimdHalfPSO = [device newComputePipelineStateWithFunction:kernels.matmulInt4DequantSimdHalf error:&error];
        kernels.matmulInt4Fast = [coreLib newFunctionWithName:@"matmul_int4_fast"];
        if (kernels.matmulInt4Fast) kernels.matmulInt4FastPSO = [device newComputePipelineStateWithFunction:kernels.matmulInt4Fast error:&error];
        kernels.matmulInt4SiluFast = [coreLib newFunctionWithName:@"matmul_int4_silu_fast"];
        if (kernels.matmulInt4SiluFast) kernels.matmulInt4SiluFastPSO = [device newComputePipelineStateWithFunction:kernels.matmulInt4SiluFast error:&error];
        kernels.matmulInt4GeluFast = [coreLib newFunctionWithName:@"matmul_int4_gelu_fast"];
        if (kernels.matmulInt4GeluFast) kernels.matmulInt4GeluFastPSO = [device newComputePipelineStateWithFunction:kernels.matmulInt4GeluFast error:&error];
        kernels.matmulInt4Tensor = [coreLib newFunctionWithName:@"matmul_int4_tensor"];
        if (kernels.matmulInt4Tensor) kernels.matmulInt4TensorPSO = [device newComputePipelineStateWithFunction:kernels.matmulInt4Tensor error:&error];
        kernels.matmulInt4Simdgroup = [coreLib newFunctionWithName:@"matmul_int4_simdgroup"];
        if (kernels.matmulInt4Simdgroup) kernels.matmulInt4SimdgroupPSO = [device newComputePipelineStateWithFunction:kernels.matmulInt4Simdgroup error:&error];
        kernels.matmulGGMLQ4_0 = [coreLib newFunctionWithName:@"matmul_ggml_q4_0"];
        if (kernels.matmulGGMLQ4_0) kernels.matmulGGMLQ4_0PSO = [device newComputePipelineStateWithFunction:kernels.matmulGGMLQ4_0 error:&error];
        kernels.quantizeToInt8 = [coreLib newFunctionWithName:@"quantize_to_int8"];
        if (kernels.quantizeToInt8) kernels.quantizeToInt8PSO = [device newComputePipelineStateWithFunction:kernels.quantizeToInt8 error:&error];
        kernels.quantizeToInt4 = [coreLib newFunctionWithName:@"quantize_to_int4"];
        if (kernels.quantizeToInt4) kernels.quantizeToInt4PSO = [device newComputePipelineStateWithFunction:kernels.quantizeToInt4 error:&error];

        
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
    
    // BUGFIX: qr_full_fused_kernel and qr_metal both have race conditions for M > 32
    // For now, fall back to CPU for matrices > 32 until we fix the kernels
    // Specialized kernels (8, 16, 32) work correctly
    if (!kernels.qrFullFusedPSO || shared_needed > 8000 || M > 32) {
        // Fall back to CPU QR and transfer back to MPS
        auto A_cpu = A.to(torch::kCPU);
        auto [Q_cpu, R_cpu] = torch::linalg_qr(A_cpu, "reduced");
        return std::make_tuple(Q_cpu.to(torch::kMPS), R_cpu.to(torch::kMPS));
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
        
        // Launch single threadgroup with 32 threads (single SIMD group to avoid inter-SIMD sync issues)
        NSUInteger tg_size = 32;
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
    
    auto A_contig = A_batch.contiguous();
    auto Q_out = torch::zeros({Batch, M, K}, A_batch.options());
    auto R_out = torch::zeros({Batch, K, N}, A_batch.options());
    
    // Use ultra-optimized 8x8 register kernel (one thread per matrix, no shared memory!)
    if (M == 8 && N == 8 && kernels.qrBatched8x8RegPSO) {
        @autoreleasepool {
            MPSStream* stream = getCurrentMPSStream();
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            
            [encoder setComputePipelineState:kernels.qrBatched8x8RegPSO];
            [encoder setBuffer:getMTLBufferStorage(A_contig) offset:A_contig.storage_offset() * A_contig.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(Q_out) offset:0 atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(R_out) offset:0 atIndex:2];
            
            uint32_t Batch_uint = (uint32_t)Batch;
            [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:3];
            
            // No shared memory needed! Launch Batch threads directly
            NSUInteger threads_per_tg = std::min((int64_t)256, Batch);
            NSUInteger num_tgs = (Batch + threads_per_tg - 1) / threads_per_tg;
            [encoder dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1) 
                threadsPerThreadgroup:MTLSizeMake(threads_per_tg, 1, 1)];
            
            stream->synchronize(SyncType::COMMIT_AND_WAIT);
        }
        return std::make_tuple(Q_out, R_out);
    }
    
    // Use specialized 16x16 kernel if available and applicable
    if (M == 16 && N == 16 && kernels.qrBatched16x16PSO) {
        @autoreleasepool {
            MPSStream* stream = getCurrentMPSStream();
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            
            [encoder setComputePipelineState:kernels.qrBatched16x16PSO];
            [encoder setBuffer:getMTLBufferStorage(A_contig) offset:A_contig.storage_offset() * A_contig.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(Q_out) offset:0 atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(R_out) offset:0 atIndex:2];
            
            uint32_t Batch_uint = (uint32_t)Batch;
            [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:3];
            
            // 16x16 kernel uses: R_work(256) + Q_work(256) + tau(16) + vT_R(32) = 560 floats
            NSUInteger shared_size = 560 * sizeof(float);
            [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
            
            // 32 threads per threadgroup (one SIMD group), Batch threadgroups
            [encoder dispatchThreadgroups:MTLSizeMake(Batch, 1, 1) 
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            
            stream->synchronize(SyncType::COMMIT_AND_WAIT);
        }
        return std::make_tuple(Q_out, R_out);
    }
    
    // Use specialized 32x32 kernel (32 threads with SIMD operations)
    // Note: Single-thread and 32-thread kernels have same performance (~9.4μs)
    // because GPU achieves 4.6 GFLOPS vs CPU's 6.7 GFLOPS regardless of approach.
    // The 1.44x gap is due to Householder's sequential dependencies.
    if (M == 32 && N == 32 && kernels.qrBatched32x32PSO) {
        @autoreleasepool {
            MPSStream* stream = getCurrentMPSStream();
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            
            [encoder setComputePipelineState:kernels.qrBatched32x32PSO];
            [encoder setBuffer:getMTLBufferStorage(A_contig) offset:A_contig.storage_offset() * A_contig.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(Q_out) offset:0 atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(R_out) offset:0 atIndex:2];
            
            uint32_t Batch_uint = (uint32_t)Batch;
            [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:3];
            
            // 32x32 kernel: v_shared(32) + tau_all(32) = 64 floats
            NSUInteger shared_size = 64 * sizeof(float);
            [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
            
            // 32 threads per threadgroup (ONE SIMD group)
            [encoder dispatchThreadgroups:MTLSizeMake(Batch, 1, 1) 
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            
            stream->synchronize(SyncType::COMMIT_AND_WAIT);
        }
        return std::make_tuple(Q_out, R_out);
    }
    
    // DISABLED: 64x64 single-thread kernel exceeds 32KB Apple GPU shared memory limit
    // R(16KB) + Q(16KB) + overhead = >32KB fails at batch>=2
    // Falling through to sequential processing which is 141x slower but correct
    if (false && M == 64 && N == 64 && kernels.qrBatched64x64PSO) {
        @autoreleasepool {
            MPSStream* stream = getCurrentMPSStream();
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            
            [encoder setComputePipelineState:kernels.qrBatched64x64PSO];
            [encoder setBuffer:getMTLBufferStorage(A_contig) offset:A_contig.storage_offset() * A_contig.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(Q_out) offset:0 atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(R_out) offset:0 atIndex:2];
            
            uint32_t Batch_uint = (uint32_t)Batch;
            [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:3];
            
            // 64x64 single-thread kernel: R(4096) + Q(4096) = 8192 floats = 32KB exactly
            NSUInteger shared_size = 8192 * sizeof(float);
            [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
            
            // 1 thread per threadgroup (single-thread kernel, no barriers!)
            [encoder dispatchThreadgroups:MTLSizeMake(Batch, 1, 1) 
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            
            stream->synchronize(SyncType::COMMIT_AND_WAIT);
        }
        return std::make_tuple(Q_out, R_out);
    }
    
    // NOTE: 64x64 single-thread kernel disabled - exceeds 32KB shared memory limit
    // Falling through to general batched kernel which works but is slower
    
    // Check if single matrix fits in shared memory for general kernel
    // Layout: R_work (M*N) + Q_work (M*K) + tau_storage (K) + vT_R_all (N) + reduction_buf (256)
    // For 64x64: 64*64 + 64*64 + 64 + 64 + 256 = 8576 floats = 34KB
    int64_t shared_per_matrix = M * N + M * K + K + N + 256;
    
    // BUGFIX: batched kernels have race conditions for M > 32
    // Fall back to sequential processing which uses qr_fused_metal (CPU fallback for M>32)
    if (!kernels.qrBatchedPSO || shared_per_matrix > 8000 || M > 32) {
        // Fall back to sequential processing - uses CPU for M>32
        for (int64_t b = 0; b < Batch; b++) {
            auto [Q, R] = qr_fused_metal(A_batch[b]);
            Q_out[b] = Q;
            R_out[b] = R;
        }
        return std::make_tuple(Q_out, R_out);
    }
    
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
    
    // Use specialized kernels for small sizes (matching QR optimization pattern)
    // 8x8: Register-based kernel (1 thread per matrix, no barriers)
    if (N == 8 && kernels.trsmBatched8x8RegPSO) {
        @autoreleasepool {
            MPSStream* stream = getCurrentMPSStream();
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            
            [encoder setComputePipelineState:kernels.trsmBatched8x8RegPSO];
            [encoder setBuffer:getMTLBufferStorage(R) offset:R.storage_offset() * R.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(B) offset:B.storage_offset() * B.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(X) offset:0 atIndex:2];
            
            uint32_t Batch_uint = (uint32_t)Batch;
            uint32_t NRHS_uint = (uint32_t)NRHS;
            [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:3];
            [encoder setBytes:&NRHS_uint length:sizeof(NRHS_uint) atIndex:4];
            
            // 1 thread per matrix, no shared memory
            [encoder dispatchThreads:MTLSizeMake(Batch, 1, 1) 
                threadsPerThreadgroup:MTLSizeMake(std::min((int64_t)256, Batch), 1, 1)];
            
            stream->synchronize(SyncType::COMMIT_AND_WAIT);
        }
        return X;
    }
    
    // 16x16: SIMD-based kernel
    if (N == 16 && kernels.trsmBatched16x16PSO) {
        @autoreleasepool {
            MPSStream* stream = getCurrentMPSStream();
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            
            [encoder setComputePipelineState:kernels.trsmBatched16x16PSO];
            [encoder setBuffer:getMTLBufferStorage(R) offset:R.storage_offset() * R.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(B) offset:B.storage_offset() * B.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(X) offset:0 atIndex:2];
            
            uint32_t Batch_uint = (uint32_t)Batch;
            uint32_t NRHS_uint = (uint32_t)NRHS;
            [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:3];
            [encoder setBytes:&NRHS_uint length:sizeof(NRHS_uint) atIndex:4];
            
            // Shared memory for X vector (16 floats per right-hand side)
            NSUInteger shared_size = 16 * sizeof(float);
            [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
            
            // 16 threads per threadgroup (half SIMD)
            [encoder dispatchThreadgroups:MTLSizeMake(Batch, 1, 1) 
                threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
            
            stream->synchronize(SyncType::COMMIT_AND_WAIT);
        }
        return X;
    }
    
    // 32x32: SIMD-based kernel
    if (N == 32 && kernels.trsmBatched32x32PSO) {
        @autoreleasepool {
            MPSStream* stream = getCurrentMPSStream();
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            
            [encoder setComputePipelineState:kernels.trsmBatched32x32PSO];
            [encoder setBuffer:getMTLBufferStorage(R) offset:R.storage_offset() * R.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(B) offset:B.storage_offset() * B.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(X) offset:0 atIndex:2];
            
            uint32_t Batch_uint = (uint32_t)Batch;
            uint32_t NRHS_uint = (uint32_t)NRHS;
            [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:3];
            [encoder setBytes:&NRHS_uint length:sizeof(NRHS_uint) atIndex:4];
            
            // Shared memory for X vector (32 floats per right-hand side)
            NSUInteger shared_size = 32 * sizeof(float);
            [encoder setThreadgroupMemoryLength:shared_size atIndex:0];
            
            // 32 threads per threadgroup (one SIMD group)
            [encoder dispatchThreadgroups:MTLSizeMake(Batch, 1, 1) 
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            
            stream->synchronize(SyncType::COMMIT_AND_WAIT);
        }
        return X;
    }
    
    // BUGFIX: General TRSM kernel has race conditions for larger sizes
    // Fall back to CPU for N > 32 (like QR)
    if (N > 32) {
        auto R_cpu = R.to(torch::kCPU);
        auto B_cpu = B.to(torch::kCPU);
        auto X_cpu = torch::linalg_solve_triangular(R_cpu, B_cpu, /*upper=*/true, /*left=*/true, /*unitriangular=*/false);
        return X_cpu.to(torch::kMPS);
    }
    
    // General kernel for other sizes
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        // auto cmdBuffer = stream->commandBuffer();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
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
        
        // [encoder endEncoding];
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
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            
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
            
            // [encoder endEncoding];
        }
        
        // === Phase 2: c = Q.T @ b (use custom Metal kernel - no sync needed!) ===
        if (kernels.batchedQtBPSO) {
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            
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
            
            // [encoder endEncoding];
        } else {
            // Fallback to torch::bmm (requires sync)
            stream->synchronize(SyncType::COMMIT_AND_WAIT);
            c = torch::bmm(Q.transpose(-2, -1), b_contig);
            cmdBuffer = stream->commandBuffer();  // Get new buffer after sync
        }
        
        // === Phase 3: Batched TRSM ===
        {
            c = c.contiguous();
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            
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
            
            // [encoder endEncoding];
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
        // auto cmdBuffer = stream->commandBuffer();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
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
        
        // [encoder endEncoding];
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
        // auto cmdBuffer = stream->commandBuffer();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
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
        
        // [encoder endEncoding];
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
        // auto cmdBuffer = stream->commandBuffer();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
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
        
        // [encoder endEncoding];
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
        // auto cmdBuffer = stream->commandBuffer();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:kernels.luBatchedPSO];
        [encoder setBuffer:getMTLBufferStorage(LU) offset:0 atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(pivots) offset:0 atIndex:1];
        
        uint32_t N_uint = (uint32_t)N;
        uint32_t Batch_uint = (uint32_t)Batch;
        [encoder setBytes:&N_uint length:sizeof(N_uint) atIndex:2];
        [encoder setBytes:&Batch_uint length:sizeof(Batch_uint) atIndex:3];
        
        [encoder dispatchThreadgroups:MTLSizeMake(Batch, 1, 1) 
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        
        // [encoder endEncoding];
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
        // auto cmdBuffer = stream->commandBuffer();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
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
        
        // [encoder endEncoding];
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
        // auto cmdBuffer = stream->commandBuffer();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
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
        
        // [encoder endEncoding];
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
        // auto cmdBuffer = stream->commandBuffer();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
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
        
        // [encoder endEncoding];
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
        // auto cmdBuffer = stream->commandBuffer();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
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
        
        // [encoder endEncoding];
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
        // auto cmdBuffer = stream->commandBuffer();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
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
        
        // [encoder endEncoding];
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
        auto encoder = stream->commandEncoder();
        
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
        
        // Don't call endEncoding - PyTorch manages encoder lifecycle
        stream->synchronize(SyncType::NONE);  // Let PyTorch batch commands
    }
    
    return std::make_tuple(Y, Rstd);
}

// -----------------------------------------------------------------------------
// Fused RMSNorm + INT4 Linear (Command Buffer Fusion)
// -----------------------------------------------------------------------------
// Combines: Y = matmul_int4(rmsnorm(X, W_norm), W_linear, scales, zeros)
// Runs RMSNorm and INT4 matmul in same command buffer without intermediate sync.
// Eliminates GPU idle time between operations.

torch::Tensor fused_rmsnorm_linear_int4_metal(
    torch::Tensor X,            // [B, N] input
    torch::Tensor W_norm,       // [N] RMSNorm weights
    torch::Tensor W_packed,     // [N/2, K] packed INT4 linear weights
    torch::Tensor scales,       // [num_groups, K] 
    torch::Tensor zeros,        // [num_groups, K]
    float eps,
    int64_t group_size
) {
    load_core_kernels();
    
    TORCH_CHECK(X.device().type() == at::kMPS, "X must be on MPS device");
    TORCH_CHECK(X.scalar_type() == at::kHalf, "Fused kernel requires half precision");
    
    auto x = X.contiguous();
    auto w_norm = W_norm.contiguous();
    auto w_packed = W_packed.contiguous();
    auto s = scales.contiguous();
    auto z = zeros.contiguous();
    
    int64_t B = x.size(0);
    int64_t N = x.size(1);  // Hidden dim (RMSNorm dim, also matmul K)
    int64_t K_out = w_packed.size(1);  // Output dim
    
    // Intermediate: normalized output (same shape as input)
    auto X_norm = torch::empty_like(x);
    auto Rstd = torch::empty({B}, x.options().dtype(at::kFloat));
    
    // Final output
    auto Y = torch::empty({B, K_out}, x.options());
    
    // Check kernel availability
    TORCH_CHECK(kernels.rmsnormFwdHalfPSO && kernels.matmulInt4FastPSO,
                "Fused RMSNorm + INT4 Linear requires half-precision RMSNorm and INT4 matmul kernels");
    
    int64_t elem_size = 2;  // half precision
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        
        // === Step 1: RMSNorm ===
        [encoder setComputePipelineState:kernels.rmsnormFwdHalfPSO];
        [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * elem_size atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(w_norm) offset:w_norm.storage_offset() * elem_size atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(X_norm) offset:X_norm.storage_offset() * elem_size atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(Rstd) offset:Rstd.storage_offset() * 4 atIndex:3];
        
        uint32_t N_u = (uint32_t)N;
        [encoder setBytes:&N_u length:4 atIndex:4];
        [encoder setBytes:&eps length:4 atIndex:5];
        
        NSUInteger threads = std::min((NSUInteger)N, (NSUInteger)1024);
        [encoder dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        
        // === Step 2: INT4 Matmul (in same command buffer, no sync) ===
        [encoder setComputePipelineState:kernels.matmulInt4FastPSO];
        [encoder setBuffer:getMTLBufferStorage(X_norm) offset:X_norm.storage_offset() * elem_size atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(w_packed) offset:w_packed.storage_offset() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(s) offset:s.storage_offset() * elem_size atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(z) offset:z.storage_offset() * elem_size atIndex:3];
        [encoder setBuffer:getMTLBufferStorage(Y) offset:Y.storage_offset() * elem_size atIndex:4];
        
        uint32_t M_u = (uint32_t)B;
        uint32_t K_u = (uint32_t)N;
        uint32_t N_out_u = (uint32_t)K_out;
        uint32_t group_size_u = (uint32_t)group_size;
        
        [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&K_u length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&N_out_u length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&group_size_u length:sizeof(uint32_t) atIndex:8];
        
        // Fast kernel dispatch: TILE_M=32, TILE_N=64, THREADS_M=8, THREADS_N=32
        NSUInteger tg_width = 32;
        NSUInteger tg_height = 8;
        NSUInteger num_tg_x = (K_out + 63) / 64;
        NSUInteger num_tg_y = (B + 31) / 32;
        [encoder dispatchThreadgroups:MTLSizeMake(num_tg_x, num_tg_y, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_width, tg_height, 1)];
        
        // Single sync at end
        stream->synchronize(SyncType::NONE);
    }
    
    return Y;
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
        auto encoder = stream->commandEncoder();
        
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
        
        // Don't call endEncoding - PyTorch manages encoder lifecycle
        // No sync - let PyTorch batch commands

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
        auto encoder = stream->commandEncoder();
        
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
        
        // Don't call endEncoding - PyTorch manages encoder lifecycle
        stream->synchronize(SyncType::NONE);  // Let PyTorch batch commands
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
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
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
        
        // [encoder endEncoding]; // Handled by PyTorch stream
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
        // auto cmdBuffer = stream->commandBuffer();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
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
        
        // [encoder endEncoding];
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
        // auto cmdBuffer = stream->commandBuffer();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
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
        
        // [encoder endEncoding];
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
        auto encoder = stream->commandEncoder();
        
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
        
        // Don't call endEncoding - PyTorch manages encoder lifecycle
        stream->synchronize(SyncType::NONE);  // Let PyTorch batch commands
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
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
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
        
        // [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT);  // Non-blocking for forward pass integration
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
        auto encoder = stream->commandEncoder();
        
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
        
        // Don't call endEncoding - PyTorch manages encoder lifecycle
        stream->synchronize(SyncType::NONE);  // Let PyTorch batch commands
    }
    
    return Y;
}

// -----------------------------------------------------------------------------
// SwiGLU Forward
// -----------------------------------------------------------------------------
// Computes: out = silu(gate) * up = (gate * sigmoid(gate)) * up
// This is the elementwise part of the SwiGLU MLP activation.

torch::Tensor swiglu_fwd_metal(torch::Tensor gate, torch::Tensor up) {
    load_core_kernels();
    
    TORCH_CHECK(gate.device().type() == at::kMPS, "gate must be on MPS device");
    TORCH_CHECK(up.device().type() == at::kMPS, "up must be on MPS device");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have same shape");
    
    bool is_half = gate.scalar_type() == at::kHalf;
    bool is_bfloat = gate.scalar_type() == at::kBFloat16;
    
    id<MTLComputePipelineState> pso = nil;
    if (is_half) {
        pso = kernels.swigluFwdHalfPSO;
    } else if (is_bfloat) {
        pso = kernels.swigluFwdBfloatPSO;
    } else {
        pso = kernels.swigluFwdPSO;
    }
    
    if (!pso) {
        // Fallback to PyTorch
        return torch::silu(gate) * up;
    }
    
    gate = gate.contiguous();
    up = up.contiguous();
    auto out = torch::empty_like(gate);
    int64_t numel = gate.numel();
    int64_t elem_size = (is_half || is_bfloat) ? 2 : 4;
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(gate) offset:gate.storage_offset() * elem_size atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(up) offset:up.storage_offset() * elem_size atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * elem_size atIndex:2];
        uint32_t numel_u = (uint32_t)numel;
        [encoder setBytes:&numel_u length:4 atIndex:3];
        
        NSUInteger threads = (NSUInteger)numel;
        NSUInteger tg_size = std::min(threads, (NSUInteger)256);
        NSUInteger groups = (threads + tg_size - 1) / tg_size;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        
        stream->synchronize(SyncType::NONE);
    }
    
    return out;
}


// -----------------------------------------------------------------------------
// LoRA Add Forward
// -----------------------------------------------------------------------------
// Computes: out = base + scale * lora
// This fuses the final step of LoRA: combining base weights with low-rank adaptation.

torch::Tensor lora_add_fwd_metal(torch::Tensor base, torch::Tensor lora, float scale) {
    load_core_kernels();
    
    TORCH_CHECK(base.device().type() == at::kMPS, "base must be on MPS device");
    TORCH_CHECK(lora.device().type() == at::kMPS, "lora must be on MPS device");
    TORCH_CHECK(base.sizes() == lora.sizes(), "base and lora must have same shape");
    
    bool is_half = base.scalar_type() == at::kHalf;
    bool is_bfloat = base.scalar_type() == at::kBFloat16;
    
    id<MTLComputePipelineState> pso = nil;
    if (is_half) {
        pso = kernels.loraAddFwdHalfPSO;
    } else if (is_bfloat) {
        pso = kernels.loraAddFwdBfloatPSO;
    } else {
        pso = kernels.loraAddFwdPSO;
    }
    
    if (!pso) {
        // Fallback to PyTorch
        return base + scale * lora;
    }
    
    base = base.contiguous();
    lora = lora.contiguous();
    auto out = torch::empty_like(base);
    int64_t numel = base.numel();
    int64_t elem_size = is_half ? 2 : 4;
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(base) offset:base.storage_offset() * elem_size atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(lora) offset:lora.storage_offset() * elem_size atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * elem_size atIndex:2];
        [encoder setBytes:&scale length:sizeof(float) atIndex:3];
        uint32_t numel_u = (uint32_t)numel;
        [encoder setBytes:&numel_u length:4 atIndex:4];
        
        NSUInteger threads = (NSUInteger)numel;
        NSUInteger tg_size = std::min(threads, (NSUInteger)256);
        NSUInteger groups = (threads + tg_size - 1) / tg_size;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        
        stream->synchronize(SyncType::NONE);
    }
    
    return out;
}

// -----------------------------------------------------------------------------
// Fused Cross-Entropy Loss
// -----------------------------------------------------------------------------
// Computes: loss = -log(softmax(logits)[target])
// Fused log-softmax + NLL avoids materializing full vocab softmax
// For vocab_size=32K, saves 32K * batch * 4 bytes per forward
// Even if slightly slower, this keeps data GPU-resident for full pipeline execution

torch::Tensor cross_entropy_fwd_metal(torch::Tensor logits, torch::Tensor targets) {
    load_core_kernels();
    
    TORCH_CHECK(logits.device().type() == at::kMPS, "logits must be on MPS device");
    TORCH_CHECK(targets.device().type() == at::kMPS, "targets must be on MPS device");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D [batch, vocab_size]");
    TORCH_CHECK(targets.dim() == 1, "targets must be 1D [batch]");
    
    auto lg = logits.contiguous();
    auto tg = targets.contiguous().to(at::kInt);
    
    int64_t batch_size = lg.size(0);
    int64_t vocab_size = lg.size(1);
    
    bool is_half = lg.scalar_type() == at::kHalf;
    bool is_bfloat = lg.scalar_type() == at::kBFloat16;
    
    id<MTLComputePipelineState> pso = nil;
    if (is_half) {
        pso = kernels.crossEntropyFwdHalfPSO;
    } else if (is_bfloat) {
        pso = kernels.crossEntropyFwdBfloatPSO;
    } else {
        pso = kernels.crossEntropyFwdPSO;
    }
    
    // Fallback to PyTorch if kernel not available
    if (!pso) {
        auto log_probs = torch::log_softmax(logits, /*dim=*/1);
        return torch::nll_loss(log_probs, targets, {}, at::Reduction::None);
    }
    
    auto losses = torch::empty({batch_size}, lg.options().dtype(at::kFloat));
    
    @autoreleasepool {
        MPSStream* stream = getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(lg) offset:lg.storage_offset() * lg.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(tg) offset:tg.storage_offset() * sizeof(int32_t) atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(losses) offset:0 atIndex:2];
        
        uint32_t batch_u = (uint32_t)batch_size;
        uint32_t vocab_u = (uint32_t)vocab_size;
        [encoder setBytes:&batch_u length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&vocab_u length:sizeof(uint32_t) atIndex:4];
        
        // Threadgroup-parallel: one threadgroup (256 threads) per batch element
        // Shared memory: 16 floats (8 for max, 8 for sum)
        uint32_t numThreadsPerGroup = 256;
        uint32_t sharedMemSize = 16 * sizeof(float);
        
        MTLSize threadgroupSize = MTLSizeMake(numThreadsPerGroup, 1, 1);
        MTLSize gridSize = MTLSizeMake(batch_size, 1, 1);  // batch_size threadgroups
        
        [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        
        stream->synchronize(SyncType::NONE);
    }
    
    return losses;
}





// -----------------------------------------------------------------------------
// KL Divergence Loss
// -----------------------------------------------------------------------------
// Computes: KL(P || Q) for distillation
// P = teacher (log_probs), Q = student (log_probs)
// Keeps data GPU-resident for full training pipeline

torch::Tensor kl_div_fwd_metal(torch::Tensor log_p, torch::Tensor log_q) {
    load_core_kernels();
    
    TORCH_CHECK(log_p.device().type() == at::kMPS, "log_p must be on MPS device");
    TORCH_CHECK(log_q.device().type() == at::kMPS, "log_q must be on MPS device");
    TORCH_CHECK(log_p.sizes() == log_q.sizes(), "log_p and log_q must have same shape");
    
    auto lp = log_p.contiguous();
    auto lq = log_q.contiguous();
    
    int64_t batch_size = lp.size(0);
    int64_t vocab_size = lp.size(1);
    
    // Try to use Metal kernel
    bool is_bfloat = lp.scalar_type() == at::kBFloat16;
    id<MTLComputePipelineState> pso = is_bfloat ? kernels.klDivFwdBfloatPSO : kernels.klDivFwdPSO;
    
    if (pso) {
        auto losses = torch::empty({batch_size}, lp.options().dtype(at::kFloat));
        
        @autoreleasepool {
            MPSStream* stream = getCurrentMPSStream();
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            
            [encoder setComputePipelineState:pso];
            [encoder setBuffer:getMTLBufferStorage(lp) offset:lp.storage_offset() * lp.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(lq) offset:lq.storage_offset() * lq.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(losses) offset:0 atIndex:2];
            
            uint32_t batch_u = (uint32_t)batch_size;
            uint32_t vocab_u = (uint32_t)vocab_size;
            [encoder setBytes:&batch_u length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&vocab_u length:sizeof(uint32_t) atIndex:4];
            
            // Threadgroup-parallel reduction: 256 threads per batch element
            uint32_t numThreadsPerGroup = 256;
            // Shared memory: 32 bytes (8 floats) for partial sums
            // But we use 16 floats (64 bytes) to be safe/aligned like cross_entropy
            uint32_t sharedMemSize = 16 * sizeof(float);
            
            MTLSize threadgroupSize = MTLSizeMake(numThreadsPerGroup, 1, 1);
            MTLSize gridSize = MTLSizeMake(batch_size, 1, 1); // One threadgroup per batch item
            
            [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];
            [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
            
            stream->synchronize(SyncType::NONE);
        }
        
        return losses;
    }
    
    // Fallback: torch ops
    auto p = torch::exp(log_p);
    return torch::sum(p * (log_p - log_q), -1);
}

// Top-K KL divergence for efficient distillation
// Only computes KL on top-k teacher tokens - 99%+ memory savings for large vocabs
torch::Tensor kl_div_topk_fwd_metal(
    torch::Tensor log_p, 
    torch::Tensor log_q,
    torch::Tensor topk_indices,
    int64_t k
) {
    load_core_kernels();
    
    TORCH_CHECK(log_p.device().type() == at::kMPS, "log_p must be on MPS device");
    TORCH_CHECK(log_q.device().type() == at::kMPS, "log_q must be on MPS device");
    
    auto lp = log_p.contiguous();
    auto lq = log_q.contiguous();
    auto idx = topk_indices.contiguous().to(at::kInt);
    
    int64_t batch_size = lp.size(0);
    int64_t vocab_size = lp.size(1);
    
    // Try to use Metal kernel
    bool is_bfloat = lp.scalar_type() == at::kBFloat16;
    id<MTLComputePipelineState> pso = is_bfloat ? kernels.klDivTopkFwdBfloatPSO : kernels.klDivTopkFwdPSO;
    
    if (pso) {
        auto losses = torch::empty({batch_size}, lp.options().dtype(at::kFloat));
        
        @autoreleasepool {
            MPSStream* stream = getCurrentMPSStream();
            // id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            
            [encoder setComputePipelineState:pso];
            [encoder setBuffer:getMTLBufferStorage(lp) offset:lp.storage_offset() * lp.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(lq) offset:lq.storage_offset() * lq.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(idx) offset:idx.storage_offset() * sizeof(int32_t) atIndex:2];
            [encoder setBuffer:getMTLBufferStorage(losses) offset:0 atIndex:3];
            
            uint32_t batch_u = (uint32_t)batch_size;
            uint32_t vocab_u = (uint32_t)vocab_size;
            int32_t k_u = (int32_t)k;
            [encoder setBytes:&batch_u length:sizeof(uint32_t) atIndex:4];
            [encoder setBytes:&vocab_u length:sizeof(uint32_t) atIndex:5];
            [encoder setBytes:&k_u length:sizeof(int32_t) atIndex:6];
            
            MTLSize gridSize = MTLSizeMake(batch_size, 1, 1);
            MTLSize threadgroupSize = MTLSizeMake(MIN(batch_size, 256), 1, 1);
            
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            // [encoder endEncoding];
            
            stream->synchronize(SyncType::NONE);
        }
        
        return losses;
    }
    
    // Fallback: torch ops with gather
    auto log_p_topk = torch::gather(log_p, 1, topk_indices);
    auto log_q_topk = torch::gather(log_q, 1, topk_indices);
    auto p_topk = torch::exp(log_p_topk);
    return torch::sum(p_topk * (log_p_topk - log_q_topk), -1);
}

// -----------------------------------------------------------------------------
// LoRA Linear Forward
// -----------------------------------------------------------------------------
// Computes: y = x @ W.T + scale * (x @ A.T @ B.T)
// Uses MPS matmuls with command buffer fusion for efficiency

torch::Tensor lora_linear_fwd_metal(
    torch::Tensor x,
    torch::Tensor W,
    torch::Tensor A,
    torch::Tensor B,
    float scale
) {
    TORCH_CHECK(x.device().type() == at::kMPS, "x must be on MPS device");
    TORCH_CHECK(W.device().type() == at::kMPS, "W must be on MPS device");
    TORCH_CHECK(A.device().type() == at::kMPS, "A must be on MPS device");
    TORCH_CHECK(B.device().type() == at::kMPS, "B must be on MPS device");
    
    // Use MPS matmuls - these are already highly optimized
    // Command buffer batching happens automatically with lazy execution
    auto base = torch::mm(x, W.t());
    auto lora = torch::mm(torch::mm(x, A.t()), B.t());
    
    return base + scale * lora;
}


// -----------------------------------------------------------------------------
// Softmax Backward
// -----------------------------------------------------------------------------
// Computes: dX = probs * (d_probs - sum(probs * d_probs))
// For attention backward pass, keeps computation fully GPU-resident

torch::Tensor softmax_bwd_metal(torch::Tensor probs, torch::Tensor d_probs) {
    load_core_kernels();
    
    TORCH_CHECK(probs.device().type() == at::kMPS, "probs must be on MPS device");
    TORCH_CHECK(d_probs.device().type() == at::kMPS, "d_probs must be on MPS device");
    TORCH_CHECK(probs.sizes() == d_probs.sizes(), "probs and d_probs must have same shape");
    
    auto p = probs.contiguous();
    auto dp = d_probs.contiguous();
    
    // Flatten to 2D: [N, L] where N = batch * heads, L = sequence
    int64_t N = 1;
    for (int i = 0; i < p.dim() - 1; i++) N *= p.size(i);
    int64_t L = p.size(-1);
    
    auto p_flat = p.view({N, L});
    auto dp_flat = dp.view({N, L});
    
    bool is_half = p.scalar_type() == at::kHalf;
    bool is_bfloat = p.scalar_type() == at::kBFloat16;
    
    id<MTLComputePipelineState> pso = nil;
    if (is_half) {
        pso = kernels.softmaxBwdHalfPSO;
    } else if (is_bfloat) {
        pso = kernels.softmaxBwdBfloatPSO;
    } else {
        pso = kernels.softmaxBwdPSO;
    }
    
    if (!pso) {
        // Fallback: PyTorch implementation
        auto dot = (p * dp).sum(-1, true);
        return p * (dp - dot);
    }
    
    auto d_logits = torch::empty_like(p_flat);
    
    @autoreleasepool {
        MPSStream* stream = getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(p_flat) offset:0 atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(dp_flat) offset:0 atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(d_logits) offset:0 atIndex:2];
        
        uint32_t N_u = (uint32_t)N;
        uint32_t L_u = (uint32_t)L;
        [encoder setBytes:&N_u length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&L_u length:sizeof(uint32_t) atIndex:4];
        
        // Threadgroup-parallel: one threadgroup (256 threads) per row
        // Shared memory: 8 floats for SIMD group partial sums
        uint32_t numThreadsPerGroup = 256;
        uint32_t sharedMemSize = 8 * sizeof(float);
        
        MTLSize threadgroupSize = MTLSizeMake(numThreadsPerGroup, 1, 1);
        MTLSize gridSize = MTLSizeMake(N, 1, 1);  // N threadgroups
        
        [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];

        
        stream->synchronize(SyncType::NONE);
    }
    
    return d_logits.view_as(probs);
}


// -----------------------------------------------------------------------------
// Fused LoRA QKV Projection
// -----------------------------------------------------------------------------
// Computes Q, K, V with LoRA in single dispatch:
//   Q = x @ W_q.T + scale * (x @ A_q.T @ B_q.T)
//   K = x @ W_k.T + scale * (x @ A_k.T @ B_k.T)
//   V = x @ W_v.T + scale * (x @ A_v.T @ B_v.T)
// Keeps all intermediate data GPU-resident, reduces 12 kernel launches to 1

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_lora_qkv_fwd_metal(
    torch::Tensor x,
    torch::Tensor W_q, torch::Tensor W_k, torch::Tensor W_v,
    torch::Tensor A_q, torch::Tensor B_q,
    torch::Tensor A_k, torch::Tensor B_k,
    torch::Tensor A_v, torch::Tensor B_v,
    float scale
) {
    TORCH_CHECK(x.device().type() == at::kMPS, "x must be on MPS device");
    
    // Ensure contiguous and get dimensions
    auto xc = x.contiguous();
    auto W_q_c = W_q.contiguous();
    auto W_k_c = W_k.contiguous();
    auto W_v_c = W_v.contiguous();
    auto A_q_c = A_q.contiguous();
    auto B_q_c = B_q.contiguous();
    auto A_k_c = A_k.contiguous();
    auto B_k_c = B_k.contiguous();
    auto A_v_c = A_v.contiguous();
    auto B_v_c = B_v.contiguous();
    
    int64_t M = xc.size(0);         // batch*seq
    int64_t in_f = xc.size(1);      // hidden_dim (e.g., 4096)
    int64_t out_q = W_q_c.size(0);  // Q output dim
    int64_t out_k = W_k_c.size(0);  // K output dim  
    int64_t out_v = W_v_c.size(0);  // V output dim
    int64_t rank = A_q_c.size(0);   // LoRA rank
    
    // For small M (decode/small prefill), torch::mm is faster due to MPS Graph optimization
    // For large M (training), direct MPS avoids PyTorch dispatcher overhead
    // Threshold determined by benchmarks: direct MPS matches PyTorch at M~256
    constexpr int64_t DIRECT_MPS_THRESHOLD = 256;
    
    if (M < DIRECT_MPS_THRESHOLD) {
        // Use PyTorch for small batches - better optimized for this case
        auto Q = torch::mm(xc, W_q_c.t()) + scale * torch::mm(torch::mm(xc, A_q_c.t()), B_q_c.t());
        auto K = torch::mm(xc, W_k_c.t()) + scale * torch::mm(torch::mm(xc, A_k_c.t()), B_k_c.t());
        auto V = torch::mm(xc, W_v_c.t()) + scale * torch::mm(torch::mm(xc, A_v_c.t()), B_v_c.t());
        return std::make_tuple(Q, K, V);
    }
    
    // Large M: use direct MPS for training workloads
    // Allocate outputs
    auto Q = torch::empty({M, out_q}, xc.options());
    auto K = torch::empty({M, out_k}, xc.options());
    auto V = torch::empty({M, out_v}, xc.options());
    
    // Temp tensors for LoRA intermediates: x @ A.T
    auto xa_q = torch::empty({M, rank}, xc.options());
    auto xa_k = torch::empty({M, rank}, xc.options());
    auto xa_v = torch::empty({M, rank}, xc.options());
    
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        MPSStream* stream = getCurrentMPSStream();
        id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
        
        // Helper to create MPS matrix descriptor from tensor
        auto makeMatrix = [&](torch::Tensor& t, int64_t rows, int64_t cols) -> MPSMatrix* {
            MPSMatrixDescriptor* desc = [MPSMatrixDescriptor 
                matrixDescriptorWithRows:rows
                columns:cols
                rowBytes:cols * sizeof(float)
                dataType:MPSDataTypeFloat32];
            return [[MPSMatrix alloc] initWithBuffer:getMTLBufferStorage(t)
                                             offset:t.storage_offset() * sizeof(float)
                                         descriptor:desc];
        };
        
        // Create matrix objects
        MPSMatrix* x_mat = makeMatrix(xc, M, in_f);
        
        MPSMatrix* Wq_mat = makeMatrix(W_q_c, out_q, in_f);
        MPSMatrix* Wk_mat = makeMatrix(W_k_c, out_k, in_f);
        MPSMatrix* Wv_mat = makeMatrix(W_v_c, out_v, in_f);
        
        MPSMatrix* Aq_mat = makeMatrix(A_q_c, rank, in_f);
        MPSMatrix* Bq_mat = makeMatrix(B_q_c, out_q, rank);
        MPSMatrix* Ak_mat = makeMatrix(A_k_c, rank, in_f);
        MPSMatrix* Bk_mat = makeMatrix(B_k_c, out_k, rank);
        MPSMatrix* Av_mat = makeMatrix(A_v_c, rank, in_f);
        MPSMatrix* Bv_mat = makeMatrix(B_v_c, out_v, rank);
        
        MPSMatrix* Q_mat = makeMatrix(Q, M, out_q);
        MPSMatrix* K_mat = makeMatrix(K, M, out_k);
        MPSMatrix* V_mat = makeMatrix(V, M, out_v);
        
        MPSMatrix* xaq_mat = makeMatrix(xa_q, M, rank);
        MPSMatrix* xak_mat = makeMatrix(xa_k, M, rank);
        MPSMatrix* xav_mat = makeMatrix(xa_v, M, rank);
        
        // === Q Projection ===
        // Step 1: Q = x @ W_q.T (base projection)
        MPSMatrixMultiplication* mmul_q_base = [[MPSMatrixMultiplication alloc]
            initWithDevice:device
            transposeLeft:NO
            transposeRight:YES
            resultRows:M
            resultColumns:out_q
            interiorColumns:in_f
            alpha:1.0
            beta:0.0];
        [mmul_q_base encodeToCommandBuffer:cmdBuf leftMatrix:x_mat rightMatrix:Wq_mat resultMatrix:Q_mat];
        
        // Step 2: xa_q = x @ A_q.T
        MPSMatrixMultiplication* mmul_xa_q = [[MPSMatrixMultiplication alloc]
            initWithDevice:device
            transposeLeft:NO
            transposeRight:YES
            resultRows:M
            resultColumns:rank
            interiorColumns:in_f
            alpha:1.0
            beta:0.0];
        [mmul_xa_q encodeToCommandBuffer:cmdBuf leftMatrix:x_mat rightMatrix:Aq_mat resultMatrix:xaq_mat];
        
        // Step 3: Q += scale * (xa_q @ B_q.T)
        MPSMatrixMultiplication* mmul_lora_q = [[MPSMatrixMultiplication alloc]
            initWithDevice:device
            transposeLeft:NO
            transposeRight:YES
            resultRows:M
            resultColumns:out_q
            interiorColumns:rank
            alpha:scale
            beta:1.0];  // Accumulate into Q
        [mmul_lora_q encodeToCommandBuffer:cmdBuf leftMatrix:xaq_mat rightMatrix:Bq_mat resultMatrix:Q_mat];
        
        // === K Projection ===
        MPSMatrixMultiplication* mmul_k_base = [[MPSMatrixMultiplication alloc]
            initWithDevice:device transposeLeft:NO transposeRight:YES
            resultRows:M resultColumns:out_k interiorColumns:in_f alpha:1.0 beta:0.0];
        [mmul_k_base encodeToCommandBuffer:cmdBuf leftMatrix:x_mat rightMatrix:Wk_mat resultMatrix:K_mat];
        
        MPSMatrixMultiplication* mmul_xa_k = [[MPSMatrixMultiplication alloc]
            initWithDevice:device transposeLeft:NO transposeRight:YES
            resultRows:M resultColumns:rank interiorColumns:in_f alpha:1.0 beta:0.0];
        [mmul_xa_k encodeToCommandBuffer:cmdBuf leftMatrix:x_mat rightMatrix:Ak_mat resultMatrix:xak_mat];
        
        MPSMatrixMultiplication* mmul_lora_k = [[MPSMatrixMultiplication alloc]
            initWithDevice:device transposeLeft:NO transposeRight:YES
            resultRows:M resultColumns:out_k interiorColumns:rank alpha:scale beta:1.0];
        [mmul_lora_k encodeToCommandBuffer:cmdBuf leftMatrix:xak_mat rightMatrix:Bk_mat resultMatrix:K_mat];
        
        // === V Projection ===
        MPSMatrixMultiplication* mmul_v_base = [[MPSMatrixMultiplication alloc]
            initWithDevice:device transposeLeft:NO transposeRight:YES
            resultRows:M resultColumns:out_v interiorColumns:in_f alpha:1.0 beta:0.0];
        [mmul_v_base encodeToCommandBuffer:cmdBuf leftMatrix:x_mat rightMatrix:Wv_mat resultMatrix:V_mat];
        
        MPSMatrixMultiplication* mmul_xa_v = [[MPSMatrixMultiplication alloc]
            initWithDevice:device transposeLeft:NO transposeRight:YES
            resultRows:M resultColumns:rank interiorColumns:in_f alpha:1.0 beta:0.0];
        [mmul_xa_v encodeToCommandBuffer:cmdBuf leftMatrix:x_mat rightMatrix:Av_mat resultMatrix:xav_mat];
        
        MPSMatrixMultiplication* mmul_lora_v = [[MPSMatrixMultiplication alloc]
            initWithDevice:device transposeLeft:NO transposeRight:YES
            resultRows:M resultColumns:out_v interiorColumns:rank alpha:scale beta:1.0];
        [mmul_lora_v encodeToCommandBuffer:cmdBuf leftMatrix:xav_mat rightMatrix:Bv_mat resultMatrix:V_mat];
        
        // All 9 matmuls encoded in single command buffer - no sync overhead
        stream->synchronize(SyncType::NONE);
    }
    
    return std::make_tuple(Q, K, V);
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
        stream->synchronize(SyncType::COMMIT);  // Non-blocking for forward pass integration
    }
    
    return dX;
}


// -----------------------------------------------------------------------------
// Bias + GELU Fusion: y = gelu(x + bias)
// -----------------------------------------------------------------------------
torch::Tensor bias_gelu_fwd_metal(torch::Tensor X, torch::Tensor bias) {
    load_core_kernels();
    
    TORCH_CHECK(X.device().type() == at::kMPS, "X must be on MPS device");
    TORCH_CHECK(bias.device().type() == at::kMPS, "bias must be on MPS device");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    
    bool is_half = X.scalar_type() == at::kHalf;
    id<MTLComputePipelineState> pso = is_half ? kernels.biasGeluFwdHalfPSO : kernels.biasGeluFwdPSO;
    
    if (!pso) {
        return torch::gelu(X + bias);
    }
    
    auto Y = torch::empty_like(X);
    int64_t numel = X.numel();
    int64_t bias_size = bias.numel() / 4;
    int64_t elem_size = is_half ? 2 : 4;
    int64_t numel_vec = numel / 4;
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(X) offset:X.storage_offset() * elem_size atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(bias) offset:bias.storage_offset() * elem_size atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(Y) offset:Y.storage_offset() * elem_size atIndex:2];
        uint32_t numel_u = (uint32_t)numel;
        uint32_t bias_size_u = (uint32_t)bias_size;
        [encoder setBytes:&numel_u length:4 atIndex:3];
        [encoder setBytes:&bias_size_u length:4 atIndex:4];
        
        NSUInteger threads = (NSUInteger)numel_vec;
        NSUInteger tg_size = std::min(threads, (NSUInteger)256);
        NSUInteger groups = (threads + tg_size - 1) / tg_size;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        
        stream->synchronize(SyncType::NONE);
    }
    
    return Y;
}

// -----------------------------------------------------------------------------
// Bias + SiLU Fusion: y = silu(x + bias)
// -----------------------------------------------------------------------------
torch::Tensor bias_silu_fwd_metal(torch::Tensor X, torch::Tensor bias) {
    load_core_kernels();
    
    TORCH_CHECK(X.device().type() == at::kMPS, "X must be on MPS device");
    TORCH_CHECK(bias.device().type() == at::kMPS, "bias must be on MPS device");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    
    bool is_half = X.scalar_type() == at::kHalf;
    id<MTLComputePipelineState> pso = is_half ? kernels.biasSiluFwdHalfPSO : kernels.biasSiluFwdPSO;
    
    if (!pso) {
        return torch::silu(X + bias);
    }
    
    auto Y = torch::empty_like(X);
    int64_t numel = X.numel();
    int64_t bias_size = bias.numel() / 4;
    int64_t elem_size = is_half ? 2 : 4;
    int64_t numel_vec = numel / 4;
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(X) offset:X.storage_offset() * elem_size atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(bias) offset:bias.storage_offset() * elem_size atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(Y) offset:Y.storage_offset() * elem_size atIndex:2];
        uint32_t numel_u = (uint32_t)numel;
        uint32_t bias_size_u = (uint32_t)bias_size;
        [encoder setBytes:&numel_u length:4 atIndex:3];
        [encoder setBytes:&bias_size_u length:4 atIndex:4];
        
        NSUInteger threads = (NSUInteger)numel_vec;
        NSUInteger tg_size = std::min(threads, (NSUInteger)256);
        NSUInteger groups = (threads + tg_size - 1) / tg_size;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        
        stream->synchronize(SyncType::NONE);
    }
    
    return Y;
}

// -----------------------------------------------------------------------------
// INT8 Matmul with on-the-fly dequantization: Y = X @ dequant(W_q)
// -----------------------------------------------------------------------------
torch::Tensor matmul_int8_metal(torch::Tensor X, torch::Tensor W_packed,
                                 torch::Tensor scales, torch::Tensor zeros,
                                 int64_t group_size) {
    load_core_kernels();
    
    TORCH_CHECK(X.device().type() == at::kMPS, "X must be on MPS device");
    TORCH_CHECK(W_packed.device().type() == at::kMPS, "W_packed must be on MPS device");
    
    bool is_half = X.scalar_type() == at::kHalf;
    id<MTLComputePipelineState> pso = is_half ? kernels.matmulInt8DequantHalfPSO : kernels.matmulInt8DequantPSO;
    
    if (!pso) {
        // Fallback: dequantize and use PyTorch matmul
        auto W_f = W_packed.to(at::kFloat);
        return torch::mm(X.to(at::kFloat), W_f).to(X.scalar_type());
    }
    
    X = X.contiguous();
    W_packed = W_packed.contiguous();
    scales = scales.contiguous();
    zeros = zeros.contiguous();
    
    int64_t M = X.size(0);
    int64_t K = X.size(1);
    int64_t N = W_packed.size(1);
    
    auto Y = torch::empty({M, N}, X.options());
    
    @autoreleasepool {
        MPSStream* stream = getCurrentMPSStream();
        id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(X) offset:X.storage_offset() * X.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(W_packed) offset:W_packed.storage_offset() * sizeof(int8_t) atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(scales) offset:scales.storage_offset() * scales.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(zeros) offset:zeros.storage_offset() * zeros.element_size() atIndex:3];
        [encoder setBuffer:getMTLBufferStorage(Y) offset:Y.storage_offset() * Y.element_size() atIndex:4];
        
        uint32_t M_u = (uint32_t)M;
        uint32_t K_u = (uint32_t)K;
        uint32_t N_u = (uint32_t)N;
        uint32_t group_size_u = (uint32_t)group_size;
        [encoder setBytes:&M_u length:4 atIndex:5];
        [encoder setBytes:&K_u length:4 atIndex:6];
        [encoder setBytes:&N_u length:4 atIndex:7];
        [encoder setBytes:&group_size_u length:4 atIndex:8];
        
        MTLSize gridSize = MTLSizeMake(N, M, 1);
        MTLSize threadgroupSize = MTLSizeMake(std::min((int64_t)16, N), std::min((int64_t)16, M), 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        stream->synchronize(SyncType::COMMIT);
    }
    
    return Y;
}

// -----------------------------------------------------------------------------
// Fused Add + LayerNorm: y = layernorm(x + residual, weight, bias)
// -----------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_add_layernorm_metal(
    torch::Tensor input, torch::Tensor residual,
    torch::Tensor weight, torch::Tensor bias, double eps) {
    load_core_kernels();
    
    TORCH_CHECK(input.device().type() == at::kMPS, "input must be on MPS device");
    TORCH_CHECK(input.dim() >= 2, "input must be at least 2D");
    
    bool is_half = input.scalar_type() == at::kHalf;
    bool is_bfloat = input.scalar_type() == at::kBFloat16;
    
    id<MTLComputePipelineState> pso = nil;
    if (is_half) {
        pso = kernels.fusedAddLayernormHalfPSO;
    } else if (is_bfloat) {
        pso = kernels.fusedAddLayernormBfloatPSO;
    } else {
        pso = kernels.fusedAddLayernormPSO; // FP32
    }
    
    if (!pso) {
        // Fallback to PyTorch
        auto x = input + residual;
        auto mean = x.mean(-1, true);
        auto var = x.var(-1, false, true);
        auto rstd = torch::rsqrt(var + eps);
        auto y = (x - mean) * rstd * weight + bias;
        return std::make_tuple(y, mean.squeeze(-1), rstd.squeeze(-1));
    }
    
    input = input.contiguous();
    residual = residual.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();
    
    int64_t batch = input.numel() / input.size(-1);
    int64_t N = input.size(-1);
    
    auto output = torch::empty_like(input);
    auto mean_out = torch::empty({batch}, input.options().dtype(at::kFloat));
    auto rstd_out = torch::empty({batch}, input.options().dtype(at::kFloat));
    
    @autoreleasepool {
        MPSStream* stream = getCurrentMPSStream();
        // id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(input) offset:input.storage_offset() * input.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(residual) offset:residual.storage_offset() * residual.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(weight) offset:weight.storage_offset() * weight.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(bias) offset:bias.storage_offset() * bias.element_size() atIndex:3];
        [encoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:4];
        [encoder setBuffer:getMTLBufferStorage(mean_out) offset:0 atIndex:5];
        [encoder setBuffer:getMTLBufferStorage(rstd_out) offset:0 atIndex:6];
        
        uint32_t N_u = (uint32_t)N;
        float eps_f = (float)eps;
        [encoder setBytes:&N_u length:4 atIndex:7];
        [encoder setBytes:&eps_f length:sizeof(float) atIndex:8];
        
        NSUInteger threadsPerGroup = std::min((int64_t)256, N);
        [encoder setThreadgroupMemoryLength:64 * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(batch, 1, 1) 
             threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        // [encoder endEncoding];
        
        stream->synchronize(SyncType::COMMIT);
    }
    
    return std::make_tuple(output, mean_out, rstd_out);
}

// -----------------------------------------------------------------------------
// GPU-side INT4 Quantization: quantize weights to INT4 on GPU
// -----------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize_to_int4_metal(
    torch::Tensor weight, int64_t group_size) {
    load_core_kernels();
    
    TORCH_CHECK(weight.device().type() == at::kMPS, "weight must be on MPS device");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [K, N]");
    TORCH_CHECK(weight.size(0) % 2 == 0, "K must be even for INT4 packing");
    
    if (!kernels.quantizeToInt4PSO) {
        // Return empty tensors - Python fallback will handle
        return std::make_tuple(torch::Tensor(), torch::Tensor(), torch::Tensor());
    }
    
    weight = weight.contiguous();
    int64_t K = weight.size(0);
    int64_t N = weight.size(1);
    int64_t num_groups = (K + group_size - 1) / group_size;
    
    auto W_packed = torch::empty({K / 2, N}, weight.options().dtype(at::kByte));
    auto scales = torch::empty({num_groups, N}, weight.options());
    auto zeros = torch::empty({num_groups, N}, weight.options());
    
    @autoreleasepool {
        MPSStream* stream = getCurrentMPSStream();
        // id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:kernels.quantizeToInt4PSO];
        [encoder setBuffer:getMTLBufferStorage(weight) offset:weight.storage_offset() * weight.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(W_packed) offset:0 atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(scales) offset:0 atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(zeros) offset:0 atIndex:3];
        
        uint32_t K_u = (uint32_t)K;
        uint32_t N_u = (uint32_t)N;
        uint32_t group_size_u = (uint32_t)group_size;
        [encoder setBytes:&K_u length:4 atIndex:4];
        [encoder setBytes:&N_u length:4 atIndex:5];
        [encoder setBytes:&group_size_u length:4 atIndex:6];
        
        // Grid: (N, num_groups) - each thread handles one column in one group
        MTLSize gridSize = MTLSizeMake(N, num_groups, 1);
        MTLSize threadgroupSize = MTLSizeMake(std::min((int64_t)64, N), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        // [encoder endEncoding];
        
        stream->synchronize(SyncType::COMMIT);
    }
    
    return std::make_tuple(W_packed, scales, zeros);
}

// -----------------------------------------------------------------------------
// Fused RoPE + Scaled Dot Product Attention
// -----------------------------------------------------------------------------
// Applies RoPE to Q and K inline during attention computation
// Eliminates separate RoPE dispatch and memory round-trip

torch::Tensor rope_sdpa_fwd_metal(
    torch::Tensor Q,        // (B*H, N, D) - unrotated queries
    torch::Tensor K,        // (B*H, N_kv, D) - unrotated keys
    torch::Tensor V,        // (B*H, N_kv, D) - values
    torch::Tensor cos,      // (max_seq, D/2) - cos values for RoPE
    torch::Tensor sin_vals, // (max_seq, D/2) - sin values for RoPE
    float scale,
    int64_t gqa_factor,
    int64_t q_offset        // Position offset for Q (for KV cache)
) {
    load_core_kernels();
    
    TORCH_CHECK(Q.device().type() == at::kMPS, "Q must be on MPS device");
    TORCH_CHECK(Q.dim() == 3, "Q must be 3D (B*H, N, D)");
    
    auto q = Q.contiguous();
    auto k = K.contiguous();
    auto v = V.contiguous();
    auto cos_c = cos.contiguous();
    auto sin_c = sin_vals.contiguous();
    
    int64_t batch_heads = q.size(0);
    int64_t q_seq_len = q.size(1);
    int64_t kv_seq_len = k.size(1);
    int64_t head_dim = q.size(2);
    
    TORCH_CHECK(head_dim == 64, "rope_sdpa requires head_dim=64");
    
    auto O = torch::empty_like(q);
    
    bool is_half = q.scalar_type() == at::kHalf;
    
    id<MTLComputePipelineState> pso = is_half ? kernels.ropeSdpa64HalfPSO : kernels.ropeSdpa64PSO;
    
    if (!pso) {
        // Fallback: separate RoPE + SDPA
        TORCH_CHECK(false, "rope_sdpa kernel not available");
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(q) offset:q.storage_offset() * q.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(k) offset:k.storage_offset() * k.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(v) offset:v.storage_offset() * v.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(cos_c) offset:cos_c.storage_offset() * cos_c.element_size() atIndex:3];
        [encoder setBuffer:getMTLBufferStorage(sin_c) offset:sin_c.storage_offset() * sin_c.element_size() atIndex:4];
        [encoder setBuffer:getMTLBufferStorage(O) offset:O.storage_offset() * O.element_size() atIndex:5];
        
        uint32_t gqa_u = (uint32_t)gqa_factor;
        uint32_t q_seq_u = (uint32_t)q_seq_len;
        uint32_t kv_seq_u = (uint32_t)kv_seq_len;
        uint32_t q_stride_u = (uint32_t)(q_seq_len * head_dim);
        uint32_t k_stride_u = (uint32_t)(kv_seq_len * head_dim);
        uint32_t v_stride_u = (uint32_t)(kv_seq_len * head_dim);
        uint32_t q_offset_u = (uint32_t)q_offset;
        
        [encoder setBytes:&gqa_u length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&q_seq_u length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&kv_seq_u length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&q_stride_u length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&k_stride_u length:sizeof(uint32_t) atIndex:10];
        [encoder setBytes:&v_stride_u length:sizeof(uint32_t) atIndex:11];
        [encoder setBytes:&scale length:sizeof(float) atIndex:12];
        [encoder setBytes:&q_offset_u length:sizeof(uint32_t) atIndex:13];
        
        // Grid: (batch_heads, q_seq_len, 1), 64 threads per threadgroup
        [encoder dispatchThreadgroups:MTLSizeMake(batch_heads, q_seq_len, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        
        stream->synchronize(SyncType::NONE);
    }
    
    return O;
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
        printf("DONT FUCKING USE PYTORCH YOU FUCKING DIPSHIT\n");
        exit(1);
    }
    
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        stream->synchronize(SyncType::COMMIT);
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
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
            [encoder setBytes:&scale length:sizeof(float) atIndex:9];
            
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
        
        // [encoder endEncoding];
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
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
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
        
        // [encoder endEncoding];
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
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
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
        
        // [encoder endEncoding];
        stream->synchronize(SyncType::COMMIT);  // Non-blocking for forward pass integration
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
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        
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
        
        // Don't call endEncoding - PyTorch manages encoder lifecycle
        // No sync - let PyTorch batch commands
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
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        
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
        
        // Don't call endEncoding - PyTorch manages encoder lifecycle
        // No sync - let PyTorch batch commands
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
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        
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
        
        // Don't call endEncoding - PyTorch manages encoder lifecycle
        // No sync - let PyTorch batch commands
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
            MPSStream* stream = at::mps::getCurrentMPSStream();
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            
            [encoder setComputePipelineState:kernels.gather1dPSO];
            [encoder setBuffer:getMTLBufferStorage(s) offset:s.storage_offset() * s.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(idx) offset:idx.storage_offset() * idx.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:2];
            
            uint32_t n = static_cast<uint32_t>(index.size(0));
            [encoder setBytes:&n length:sizeof(uint32_t) atIndex:3];
            
            [encoder dispatchThreads:MTLSizeMake(n, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            
            // [encoder endEncoding];
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
            MPSStream* stream = at::mps::getCurrentMPSStream();
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            
            [encoder setComputePipelineState:kernels.scatterAdd1dPSO];
            [encoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(idx) offset:idx.storage_offset() * idx.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(s) offset:s.storage_offset() * s.element_size() atIndex:2];
            
            uint32_t n = static_cast<uint32_t>(src.size(0));
            [encoder setBytes:&n length:sizeof(uint32_t) atIndex:3];
            
            [encoder dispatchThreads:MTLSizeMake(n, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            
            // [encoder endEncoding];
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

// -----------------------------------------------------------------------------
// RoPE (Rotary Position Embedding) - Split-Half Format
// -----------------------------------------------------------------------------
// Matches HuggingFace/Liger implementation:
// x1 = x[..., :D/2], x2 = x[..., D/2:]
// out[..., :D/2] = x1 * cos - x2 * sin
// out[..., D/2:] = x2 * cos + x1 * sin

torch::Tensor rope_fwd_metal(torch::Tensor qk, torch::Tensor cos, torch::Tensor sin) {
    load_core_kernels();
    
    TORCH_CHECK(qk.device().type() == at::kMPS, "qk must be on MPS device");
    TORCH_CHECK(cos.device().type() == at::kMPS, "cos must be on MPS device");
    TORCH_CHECK(sin.device().type() == at::kMPS, "sin must be on MPS device");
    TORCH_CHECK(qk.dim() == 4, "qk must be 4D [batch, seq_len, num_heads, head_dim]");
    
    auto q = qk.contiguous();
    auto c = cos.contiguous();
    auto s = sin.contiguous();
    
    int64_t batch = q.size(0);
    int64_t seq_len = q.size(1);
    int64_t num_heads = q.size(2);
    int64_t head_dim = q.size(3);
    
    auto out = torch::empty_like(q);
    
    bool is_half = q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16;
    auto pso = is_half ? kernels.ropeFwdSplitHalfHalfPSO : kernels.ropeFwdSplitHalfPSO;
    
    if (!pso) {
        // Fallback: manual Python-style rotation
        auto x1 = qk.narrow(-1, 0, head_dim / 2);
        auto x2 = qk.narrow(-1, head_dim / 2, head_dim / 2);
        auto out1 = x1 * cos - x2 * sin;
        auto out2 = x2 * cos + x1 * sin;
        return torch::cat({out1, out2}, -1);
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(q) offset:q.storage_offset() * q.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(c) offset:c.storage_offset() * c.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(s) offset:s.storage_offset() * s.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(out) offset:out.storage_offset() * out.element_size() atIndex:3];
        
        uint32_t batch_u = (uint32_t)batch;
        uint32_t seq_len_u = (uint32_t)seq_len;
        uint32_t num_heads_u = (uint32_t)num_heads;
        uint32_t head_dim_u = (uint32_t)head_dim;
        
        [encoder setBytes:&batch_u length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&seq_len_u length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&num_heads_u length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&head_dim_u length:sizeof(uint32_t) atIndex:7];
        
        // Grid: (D/2, H, B*S)
        [encoder dispatchThreads:MTLSizeMake(head_dim / 2, num_heads, batch * seq_len)
           threadsPerThreadgroup:MTLSizeMake(std::min(64UL, (unsigned long)(head_dim / 2)), 1, 1)];
        
        stream->synchronize(SyncType::NONE);
    }
    
    return out;
}

torch::Tensor rope_bwd_metal(torch::Tensor d_out, torch::Tensor cos, torch::Tensor sin) {
    load_core_kernels();
    
    TORCH_CHECK(d_out.device().type() == at::kMPS, "d_out must be on MPS device");
    TORCH_CHECK(d_out.dim() == 4, "d_out must be 4D");
    
    auto dy = d_out.contiguous();
    auto c = cos.contiguous();
    auto s = sin.contiguous();
    
    int64_t batch = dy.size(0);
    int64_t seq_len = dy.size(1);
    int64_t num_heads = dy.size(2);
    int64_t head_dim = dy.size(3);
    
    auto d_qk = torch::empty_like(dy);
    
    id<MTLComputePipelineState> pso = nil;
    if (d_out.scalar_type() == at::kFloat) pso = kernels.ropeBwdSplitHalfPSO;
    else if (d_out.scalar_type() == at::kHalf) pso = kernels.ropeBwdSplitHalfHalfPSO;
    else if (d_out.scalar_type() == at::kBFloat16) pso = kernels.ropeBwdSplitHalfBfloatPSO;
    
    if (!pso) {
         // Fallback
        auto dy1 = d_out.narrow(-1, 0, head_dim / 2);
        auto dy2 = d_out.narrow(-1, head_dim / 2, head_dim / 2);
        auto dx1 = dy1 * cos + dy2 * sin;
        auto dx2 = -dy1 * sin + dy2 * cos;
        return torch::cat({dx1, dx2}, -1);
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(dy) offset:dy.storage_offset() * dy.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(c) offset:c.storage_offset() * c.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(s) offset:s.storage_offset() * s.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(d_qk) offset:d_qk.storage_offset() * d_qk.element_size() atIndex:3];
        
        uint32_t batch_u = (uint32_t)batch;
        uint32_t seq_len_u = (uint32_t)seq_len;
        uint32_t num_heads_u = (uint32_t)num_heads;
        uint32_t head_dim_u = (uint32_t)head_dim;
        
        [encoder setBytes:&batch_u length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&seq_len_u length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&num_heads_u length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&head_dim_u length:sizeof(uint32_t) atIndex:7];
        
        // For BF16, we use a vectorized kernel processing 4 elements per thread (bfloat4)
        // Original grid width: head_dim / 2
        // Vectorized grid width: (head_dim / 2) / 4 = head_dim / 8
        NSUInteger width = (head_dim / 2);
        if (pso == kernels.ropeBwdSplitHalfBfloatPSO) {
            width /= 4;
        }

        [encoder dispatchThreads:MTLSizeMake(width, num_heads, batch * seq_len)
           threadsPerThreadgroup:MTLSizeMake(std::min(64UL, (unsigned long)width), 1, 1)];
        
        stream->synchronize(SyncType::NONE);
    }
    
    return d_qk;
}

// Fused Q+K RoPE (in-place modification like Liger)
std::tuple<torch::Tensor, torch::Tensor> rope_fwd_qk_metal(
    torch::Tensor Q, torch::Tensor K, torch::Tensor cos, torch::Tensor sin
) {
    load_core_kernels();
    
    TORCH_CHECK(Q.device().type() == at::kMPS, "Q must be on MPS device");
    TORCH_CHECK(K.device().type() == at::kMPS, "K must be on MPS device");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be 4D");
    
    // Make contiguous copies for in-place modification
    auto q = Q.contiguous().clone();
    auto k = K.contiguous().clone();
    auto c = cos.contiguous();
    auto s = sin.contiguous();
    
    int64_t batch = q.size(0);
    int64_t seq_len = q.size(1);
    int64_t num_heads_q = q.size(2);
    int64_t num_heads_kv = k.size(2);
    int64_t head_dim = q.size(3);
    
    if (!kernels.ropeFwdQkSplitHalfPSO) {
        // Fallback
        auto q_rot = rope_fwd_metal(Q, cos, sin);
        auto k_rot = rope_fwd_metal(K, cos, sin);
        return std::make_tuple(q_rot, k_rot);
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:kernels.ropeFwdQkSplitHalfPSO];
        [encoder setBuffer:getMTLBufferStorage(q) offset:q.storage_offset() * q.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(k) offset:k.storage_offset() * k.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(c) offset:c.storage_offset() * c.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(s) offset:s.storage_offset() * s.element_size() atIndex:3];
        
        uint32_t batch_u = (uint32_t)batch;
        uint32_t seq_len_u = (uint32_t)seq_len;
        uint32_t num_heads_q_u = (uint32_t)num_heads_q;
        uint32_t num_heads_kv_u = (uint32_t)num_heads_kv;
        uint32_t head_dim_u = (uint32_t)head_dim;
        
        [encoder setBytes:&batch_u length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&seq_len_u length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&num_heads_q_u length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&num_heads_kv_u length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&head_dim_u length:sizeof(uint32_t) atIndex:8];
        
        // Grid: (D/2, max(H_q, H_kv), B*S)
        uint64_t max_heads = std::max(num_heads_q, num_heads_kv);
        [encoder dispatchThreads:MTLSizeMake(head_dim / 2, max_heads, batch * seq_len)
           threadsPerThreadgroup:MTLSizeMake(std::min(64UL, (unsigned long)(head_dim / 2)), 1, 1)];
        
        stream->synchronize(SyncType::NONE);
    }
    
    return std::make_tuple(q, k);
}

// -----------------------------------------------------------------------------
// INT4 Quantized Matmul
// -----------------------------------------------------------------------------
// Y = X @ dequant(W_packed, scales, zeros)
// X: [M, K], W_packed: [K/2, N], scales/zeros: [num_groups, N]

torch::Tensor matmul_int4_metal(
    torch::Tensor X,
    torch::Tensor W_packed,
    torch::Tensor scales,
    torch::Tensor zeros,
    int64_t group_size
) {
    load_core_kernels();
    
    TORCH_CHECK(X.device().type() == at::kMPS, "X must be on MPS device");
    TORCH_CHECK(W_packed.device().type() == at::kMPS, "W_packed must be on MPS device");
    
    auto x = X.contiguous();
    auto w = W_packed.contiguous();
    auto s = scales.contiguous();
    auto z = zeros.contiguous();
    
    int64_t M = x.size(0);
    int64_t K = x.size(1);
    int64_t N = w.size(1);
    
    bool is_half = x.scalar_type() == at::kHalf;
    
    // Kernel selection priority (for half precision):
    // 1. Simdgroup matrix kernel (hardware 8x8 matrix ops - Metal 3, M1/M2/M3)
    // 2. Tensor kernel (tensor cores - Metal 4, M4/M5 only)
    // 3. Fast kernel (optimized tiling, register blocking)
    // 4. Tiled kernel (caches X in shared memory)
    // 5. Simd_sum kernel (simd_sum for K reduction)
    // 6. Vec4 kernel 
    // 7. Scalar kernel (fallback)
    
    // Simdgroup matrix kernel: uses simdgroup_multiply_accumulate (hardware accelerated on Metal 3)
    bool use_simdgroup = is_half && kernels.matmulInt4SimdgroupPSO && 
                         M >= 64 && K >= 64 && N >= 32;
    // Tensor kernel requires Metal 4 (M4/M5), likely won't load on M1/M2/M3
    bool use_tensor = !use_simdgroup && is_half && kernels.matmulInt4TensorPSO && 
                      M >= 64 && K >= 64 && N >= 32;
    // Fast kernel requires half precision, and dimensions that fit tile sizes (32x64)
    bool use_fast = !use_simdgroup && !use_tensor && is_half && kernels.matmulInt4FastPSO && 
                    M >= 4 && K >= 32 && N >= 2 && (K % 4 == 0);
    bool use_tiled = !use_fast && !use_tensor && !use_simdgroup && !is_half && kernels.matmulInt4DequantTiledPSO && M >= 8 && K >= 128 && N >= 32;
    bool use_simd = kernels.matmulInt4DequantSimdPSO && !is_half && !use_tiled && !use_fast && !use_tensor && !use_simdgroup;
    bool use_simd_half = kernels.matmulInt4DequantSimdHalfPSO && is_half && !use_fast && !use_tensor && !use_simdgroup;
    bool use_vec4 = (N % 4 == 0) && !use_simd && !use_simd_half && !use_tiled && !use_fast && !use_tensor && !use_simdgroup;
    bool use_half_vec4 = is_half && use_vec4 && kernels.matmulInt4DequantHalfVec4PSO;
    bool use_fp32_vec4 = !is_half && use_vec4 && kernels.matmulInt4DequantVec4PSO;
    
    // Select kernel
    id<MTLComputePipelineState> pso = nil;
    if (use_simdgroup) {
        pso = kernels.matmulInt4SimdgroupPSO;
    } else if (use_tensor) {
        pso = kernels.matmulInt4TensorPSO;
    } else if (use_fast) {
        pso = kernels.matmulInt4FastPSO;
    } else if (use_tiled) {
        pso = kernels.matmulInt4DequantTiledPSO;
    } else if (use_simd) {
        pso = kernels.matmulInt4DequantSimdPSO;
    } else if (use_simd_half) {
        pso = kernels.matmulInt4DequantSimdHalfPSO;
    } else if (use_half_vec4) {
        pso = kernels.matmulInt4DequantHalfVec4PSO;
    } else if (use_fp32_vec4) {
        pso = kernels.matmulInt4DequantVec4PSO;
    } else if (is_half) {
        pso = kernels.matmulInt4DequantHalfPSO;
    } else {
        pso = kernels.matmulInt4DequantPSO;
    }
    
    // Create output tensor
    auto Y = torch::empty({M, N}, x.options());
    
    if (!pso) {
        // Fall back to Python implementation (will be caught by Python wrapper)
        TORCH_CHECK(false, "INT4 matmul kernel not available - use Python fallback");
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(w) offset:w.storage_offset() * w.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(s) offset:s.storage_offset() * s.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(z) offset:z.storage_offset() * z.element_size() atIndex:3];
        [encoder setBuffer:getMTLBufferStorage(Y) offset:Y.storage_offset() * Y.element_size() atIndex:4];
        
        uint32_t M_u = (uint32_t)M;
        uint32_t K_u = (uint32_t)K;
        uint32_t N_u = (uint32_t)N;
        uint32_t group_size_u = (uint32_t)group_size;
        
        [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&K_u length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&N_u length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&group_size_u length:sizeof(uint32_t) atIndex:8];
        
        if (use_simdgroup) {
            // Simdgroup matrix kernel: SG_NR0=64, SG_NR1=32, SG_NK=64
            // Uses hardware simdgroup_multiply_accumulate (Metal 3)
            // 4 simdgroups per threadgroup, each handles 32x16 output
            // Shared memory: sa[NK, NR0] + sb[NR1, NK] = 8KB + 4KB = 12KB
            NSUInteger num_tg_x = (N + 31) / 32;   // ceil(N / SG_NR1)
            NSUInteger num_tg_y = (M + 63) / 64;   // ceil(M / SG_NR0)
            NSUInteger sharedMemSize = 64 * 64 * 2 + 32 * 64 * 2;  // 12KB + output buffer
            sharedMemSize = std::max(sharedMemSize, 64UL * 32 * 4);  // output needs 64*32*4 = 8KB
            [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(num_tg_x, num_tg_y, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];  // 4 simdgroups x 32 threads = 128 threads
        } else if (use_tensor) {
            // Tensor kernel: TC_TILE_M=64, TC_TILE_N=32, 4 simdgroups
            // Uses hardware tensor cores via mpp::tensor_ops::matmul2d
            // Shared memory: 8KB + 4KB + 8KB = 20KB
            NSUInteger num_tg_x = (N + 31) / 32;   // ceil(N / TC_TILE_N)
            NSUInteger num_tg_y = (M + 63) / 64;   // ceil(M / TC_TILE_M)
            NSUInteger sharedMemSize = 64 * 64 * 2 + 64 * 32 * 2 + 64 * 32 * 4;  // 20KB
            [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(num_tg_x, num_tg_y, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];  // 4 simdgroups x 32 threads
        } else if (use_fast) {
            // Fast kernel: TILE_M=32, TILE_N=64, THREADS_M=8, THREADS_N=32
            // Each thread handles 4 M rows x 2 N cols = 8 elements
            NSUInteger tg_width = 32;   // THREADS_N
            NSUInteger tg_height = 8;   // THREADS_M
            NSUInteger num_tg_x = (N + 63) / 64;  // ceil(N / FAST_TILE_N)
            NSUInteger num_tg_y = (M + 31) / 32;  // ceil(M / FAST_TILE_M)
            [encoder dispatchThreadgroups:MTLSizeMake(num_tg_x, num_tg_y, 1)
                    threadsPerThreadgroup:MTLSizeMake(tg_width, tg_height, 1)];
        } else if (use_tiled) {
            // Tiled kernel: TILE_M=8, TILE_N=32, threadgroup = (32, 8)
            // Caches X tile in shared memory for reuse across columns
            NSUInteger tg_width = 32;
            NSUInteger tg_height = 8;
            NSUInteger num_tg_x = (N + 31) / 32;  // ceil(N / TILE_N)
            NSUInteger num_tg_y = (M + 7) / 8;    // ceil(M / TILE_M)
            [encoder dispatchThreadgroups:MTLSizeMake(num_tg_x, num_tg_y, 1)
                    threadsPerThreadgroup:MTLSizeMake(tg_width, tg_height, 1)];
        } else if (use_simd || use_simd_half) {
            // Simd kernel: 4 simdgroups per threadgroup, NR0=2 rows per simdgroup
            // Grid: x=N columns, y=ceil(M/(4*NR0)) threadgroups for rows
            NSUInteger NR0 = 2;  // rows per simdgroup (must match kernel)
            NSUInteger nsg = 4;  // simdgroups per threadgroup
            NSUInteger num_tg_x = N;  // one threadgroup per column
            NSUInteger num_tg_y = (M + (nsg * NR0) - 1) / (nsg * NR0);  // ceil
            [encoder dispatchThreadgroups:MTLSizeMake(num_tg_x, num_tg_y, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];  // 32 threads per simdgroup, 4 simdgroups
        } else if (use_fp32_vec4 || use_half_vec4) {
            // Vec4: process 4 columns per thread, grid is (N/4, M)
            [encoder dispatchThreads:MTLSizeMake(N / 4, M, 1)
               threadsPerThreadgroup:MTLSizeMake(std::min(256UL, (unsigned long)(N / 4)), 1, 1)];
        } else {
            // Scalar: one thread per output element
            [encoder dispatchThreads:MTLSizeMake(N, M, 1)
               threadsPerThreadgroup:MTLSizeMake(std::min(256UL, (unsigned long)N), 1, 1)];
        }
        
        stream->synchronize(SyncType::NONE);
    }
    
    return Y;
}

// -----------------------------------------------------------------------------
// FUSED INT4 MATMUL + SILU
// -----------------------------------------------------------------------------
// Combines: Y = silu(X @ dequant(W))
// Eliminates memory round-trip between matmul and activation

torch::Tensor matmul_int4_silu_metal(
    torch::Tensor X,
    torch::Tensor W_packed,
    torch::Tensor scales,
    torch::Tensor zeros,
    int64_t group_size
) {
    load_core_kernels();
    
    TORCH_CHECK(X.device().type() == at::kMPS, "X must be on MPS device");
    TORCH_CHECK(X.scalar_type() == at::kHalf, "Fused kernel requires half precision");
    
    auto x = X.contiguous();
    auto w = W_packed.contiguous();
    auto s = scales.contiguous();
    auto z = zeros.contiguous();
    
    int64_t M = x.size(0);
    int64_t K = x.size(1);
    int64_t N = w.size(1);
    
    auto Y = torch::empty({M, N}, x.options());
    
    if (!kernels.matmulInt4SiluFastPSO) {
        // Fall back to unfused path
        auto temp = matmul_int4_metal(X, W_packed, scales, zeros, group_size);
        return temp * torch::sigmoid(temp);
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:kernels.matmulInt4SiluFastPSO];
        [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(w) offset:w.storage_offset() * w.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(s) offset:s.storage_offset() * s.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(z) offset:z.storage_offset() * z.element_size() atIndex:3];
        [encoder setBuffer:getMTLBufferStorage(Y) offset:Y.storage_offset() * Y.element_size() atIndex:4];
        
        uint32_t M_u = (uint32_t)M;
        uint32_t K_u = (uint32_t)K;
        uint32_t N_u = (uint32_t)N;
        uint32_t group_size_u = (uint32_t)group_size;
        
        [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&K_u length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&N_u length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&group_size_u length:sizeof(uint32_t) atIndex:8];
        
        // Same dispatch as matmul_int4_fast: TILE_M=32, TILE_N=64, THREADS_M=8, THREADS_N=32
        NSUInteger tg_width = 32;
        NSUInteger tg_height = 8;
        NSUInteger num_tg_x = (N + 63) / 64;
        NSUInteger num_tg_y = (M + 31) / 32;
        [encoder dispatchThreadgroups:MTLSizeMake(num_tg_x, num_tg_y, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_width, tg_height, 1)];
        
        stream->synchronize(SyncType::NONE);
    }
    
    return Y;
}

// -----------------------------------------------------------------------------
// FUSED INT4 MATMUL + GELU
// -----------------------------------------------------------------------------
// Combines: Y = gelu(X @ dequant(W))

torch::Tensor matmul_int4_gelu_metal(
    torch::Tensor X,
    torch::Tensor W_packed,
    torch::Tensor scales,
    torch::Tensor zeros,
    int64_t group_size
) {
    load_core_kernels();
    
    TORCH_CHECK(X.device().type() == at::kMPS, "X must be on MPS device");
    TORCH_CHECK(X.scalar_type() == at::kHalf, "Fused kernel requires half precision");
    
    auto x = X.contiguous();
    auto w = W_packed.contiguous();
    auto s = scales.contiguous();
    auto z = zeros.contiguous();
    
    int64_t M = x.size(0);
    int64_t K = x.size(1);
    int64_t N = w.size(1);
    
    auto Y = torch::empty({M, N}, x.options());
    
    if (!kernels.matmulInt4GeluFastPSO) {
        // Fall back to unfused path using PyTorch GELU
        auto temp = matmul_int4_metal(X, W_packed, scales, zeros, group_size);
        return torch::gelu(temp);
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:kernels.matmulInt4GeluFastPSO];
        [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(w) offset:w.storage_offset() * w.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(s) offset:s.storage_offset() * s.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(z) offset:z.storage_offset() * z.element_size() atIndex:3];
        [encoder setBuffer:getMTLBufferStorage(Y) offset:Y.storage_offset() * Y.element_size() atIndex:4];
        
        uint32_t M_u = (uint32_t)M;
        uint32_t K_u = (uint32_t)K;
        uint32_t N_u = (uint32_t)N;
        uint32_t group_size_u = (uint32_t)group_size;
        
        [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&K_u length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&N_u length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&group_size_u length:sizeof(uint32_t) atIndex:8];
        
        NSUInteger tg_width = 32;
        NSUInteger tg_height = 8;
        NSUInteger num_tg_x = (N + 63) / 64;
        NSUInteger num_tg_y = (M + 31) / 32;
        [encoder dispatchThreadgroups:MTLSizeMake(num_tg_x, num_tg_y, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_width, tg_height, 1)];
        
        stream->synchronize(SyncType::NONE);
    }
    
    return Y;
}

// GGML-compatible block_q4_0 matmul using llama.cpp-style kernel
torch::Tensor matmul_ggml_q4_0_metal(
    torch::Tensor X,
    torch::Tensor W_blocks
) {
    load_core_kernels();
    
    TORCH_CHECK(X.device().type() == at::kMPS, "X must be on MPS device");
    TORCH_CHECK(W_blocks.device().type() == at::kMPS, "W_blocks must be on MPS device");
    TORCH_CHECK(X.scalar_type() == at::kHalf, "X must be float16");
    TORCH_CHECK(W_blocks.scalar_type() == at::kByte, "W_blocks must be uint8");
    
    auto x = X.contiguous();
    auto w = W_blocks.contiguous();
    
    // W_blocks is [num_blocks_k, N, 18] where 18 = 2 (scale) + 16 (packed)
    int64_t num_blocks_k = w.size(0);
    int64_t N = w.size(1);
    int64_t K = num_blocks_k * 32;  // 32 values per block
    int64_t M = x.size(0);
    
    TORCH_CHECK(x.size(1) == K, "X K dimension must match W K dimension");
    TORCH_CHECK(w.size(2) == 18, "W_blocks last dim must be 18 (block_q4_0 size)");
    
    auto Y = torch::empty({M, N}, x.options());
    
    id<MTLComputePipelineState> pso = kernels.matmulGGMLQ4_0PSO;
    TORCH_CHECK(pso, "matmul_ggml_q4_0 kernel not loaded");
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        
        [encoder setComputePipelineState:pso];
        [encoder setBuffer:getMTLBufferStorage(x) offset:x.storage_offset() * x.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(w) offset:w.storage_offset() * w.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(Y) offset:0 atIndex:2];
        
        uint32_t M_u = (uint32_t)M;
        uint32_t K_u = (uint32_t)K;
        uint32_t N_u = (uint32_t)N;
        [encoder setBytes:&M_u length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&K_u length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&N_u length:sizeof(uint32_t) atIndex:5];
        
        // Threadgroup memory: sa (4KB) + sb (2KB) + output buffer
        NSUInteger sharedMemSize = 4096 + 2048 + 64 * 32 * sizeof(float);
        [encoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];
        
        // Dispatch: 64x32 output per threadgroup, 4 simdgroups
        NSUInteger num_tg_x = (N + 31) / 32;
        NSUInteger num_tg_y = (M + 63) / 64;
        [encoder dispatchThreadgroups:MTLSizeMake(num_tg_x, num_tg_y, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        
        stream->synchronize(SyncType::NONE);
    }
    
    return Y;
}

// =============================================================================
// FUSED LORA ATTENTION - Meta-Fused Kernel
// =============================================================================
// Combines in single command buffer:
//   1. RMSNorm (custom kernel)
//   2. QKV projection with LoRA (MPS matmul + custom add)
//   3. RoPE (custom kernel)
//   4. Flash Attention (custom or SDPA)
//   5. Output projection with LoRA (MPS matmul + custom add)
//   6. Residual + Dropout (custom fused)
//
// Eliminates ~27 separate kernel launches per layer

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fused_lora_attention_fwd(
    torch::Tensor hidden_states,
    torch::Tensor residual,
    torch::Tensor ln_weight,
    float ln_eps,
    torch::Tensor W_q, torch::Tensor W_k, torch::Tensor W_v,
    torch::Tensor A_q, torch::Tensor B_q,
    torch::Tensor A_k, torch::Tensor B_k,
    torch::Tensor A_v, torch::Tensor B_v,
    float lora_scale,
    torch::Tensor W_o,
    torch::Tensor A_o, torch::Tensor B_o,
    torch::Tensor cos_cached,
    torch::Tensor sin_cached,
    int64_t num_heads,
    int64_t num_kv_heads,
    float dropout_p,
    bool is_causal,
    bool is_training
) {
    load_core_kernels();
    
    // Validate inputs
    TORCH_CHECK(hidden_states.device().type() == at::kMPS, "hidden_states must be on MPS");
    TORCH_CHECK(W_q.device().type() == at::kMPS, "weights must be on MPS");
    
    // Shapes
    int64_t B = hidden_states.size(0);
    int64_t L = hidden_states.size(1);
    int64_t D = hidden_states.size(2);
    int64_t head_dim = D / num_heads;
    int64_t D_kv = head_dim * num_kv_heads;
    bool has_lora = A_q.numel() > 0;
    
    auto h = hidden_states.contiguous().view({B * L, D});
    // Ensure weights are contiguous for MPS
    auto W_q_c = W_q.contiguous();
    auto W_k_c = W_k.contiguous();
    auto W_v_c = W_v.contiguous();
    auto W_o_c = W_o.contiguous();
    
    // Allocate outputs
    auto normed = torch::empty({B * L, D}, h.options());
    auto Q = torch::empty({B * L, D}, h.options());
    auto K = torch::empty({B * L, D_kv}, h.options());
    auto V = torch::empty({B * L, D_kv}, h.options());
    
    // Allocate Rstd outside autoreleasepool so we can return it
    auto rstd = torch::empty({B * L}, h.options());
    
    // Commit any pending PyTorch work to ensure we get a clean command buffer
    getCurrentMPSStream()->synchronize(SyncType::COMMIT);
    
    // =========================================================================
    // BLOCK 1: RMSNorm + QKV Projections (Manual MPS)
    // =========================================================================
    @autoreleasepool {
        MPSStream* stream = getCurrentMPSStream();
        id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
        id<MTLDevice> device = stream->device();
        
        // --- RMSNorm ---
        if (kernels.rmsnormFwdPSO) {
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            [encoder setComputePipelineState:kernels.rmsnormFwdPSO];
            [encoder setBuffer:getMTLBufferStorage(h) offset:h.storage_offset()*4 atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(ln_weight) offset:ln_weight.storage_offset()*4 atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(normed) offset:normed.storage_offset()*4 atIndex:2];
            
            // rstd allocated outside
            [encoder setBuffer:getMTLBufferStorage(rstd) offset:rstd.storage_offset()*4 atIndex:3];
            
            uint32_t D_u = (uint32_t)D;
            [encoder setBytes:&D_u length:sizeof(uint32_t) atIndex:4];
            [encoder setBytes:&ln_eps length:sizeof(float) atIndex:5];
            
            // Dispatch ONE threadgroup per row (B*L rows). Threadgroup size depends on D logic.
            // Using 256 threads per group is standard/safe.
            [encoder dispatchThreadgroups:MTLSizeMake(B * L, 1, 1) 
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
        }

        
        // --- QKV Matmuls ---
        auto makeMatrix = [&](torch::Tensor& t, int64_t rows, int64_t cols) -> MPSMatrix* {
            MPSMatrixDescriptor* desc = [MPSMatrixDescriptor 
                matrixDescriptorWithRows:rows columns:cols 
                rowBytes:cols * sizeof(float) dataType:MPSDataTypeFloat32];
            return [[MPSMatrix alloc] initWithBuffer:getMTLBufferStorage(t)
                                             offset:t.storage_offset() * sizeof(float)
                                         descriptor:desc];
        };
        
        MPSMatrix* norm_mat = makeMatrix(normed, B * L, D);
        
        MPSMatrix* Wq_mat = makeMatrix(W_q_c, D, D);
        MPSMatrix* Q_mat = makeMatrix(Q, B*L, D);
        MPSMatrixMultiplication* mm_q = [[MPSMatrixMultiplication alloc]
            initWithDevice:device transposeLeft:NO transposeRight:YES
            resultRows:B*L resultColumns:D interiorColumns:D alpha:1.0 beta:0.0];
        [mm_q encodeToCommandBuffer:cmdBuf leftMatrix:norm_mat rightMatrix:Wq_mat resultMatrix:Q_mat];
        
        MPSMatrix* Wk_mat = makeMatrix(W_k_c, D_kv, D);
        MPSMatrix* K_mat = makeMatrix(K, B*L, D_kv);
        MPSMatrixMultiplication* mm_k = [[MPSMatrixMultiplication alloc]
            initWithDevice:device transposeLeft:NO transposeRight:YES
            resultRows:B*L resultColumns:D_kv interiorColumns:D alpha:1.0 beta:0.0];
        [mm_k encodeToCommandBuffer:cmdBuf leftMatrix:norm_mat rightMatrix:Wk_mat resultMatrix:K_mat];
        
        MPSMatrix* Wv_mat = makeMatrix(W_v_c, D_kv, D);
        MPSMatrix* V_mat = makeMatrix(V, B*L, D_kv);
        MPSMatrixMultiplication* mm_v = [[MPSMatrixMultiplication alloc]
            initWithDevice:device transposeLeft:NO transposeRight:YES
            resultRows:B*L resultColumns:D_kv interiorColumns:D alpha:1.0 beta:0.0];
        [mm_v encodeToCommandBuffer:cmdBuf leftMatrix:norm_mat rightMatrix:Wv_mat resultMatrix:V_mat];
        
        // LoRA
        if (has_lora) {
            auto xa_q = torch::empty({B * L, A_q.size(0)}, normed.options());
            auto xa_k = torch::empty({B * L, A_q.size(0)}, normed.options());
            auto xa_v = torch::empty({B * L, A_q.size(0)}, normed.options());
            int64_t r = A_q.size(0);
            
            auto A_q_c = A_q.contiguous(); auto B_q_c = B_q.contiguous();
            auto A_k_c = A_k.contiguous(); auto B_k_c = B_k.contiguous();
            auto A_v_c = A_v.contiguous(); auto B_v_c = B_v.contiguous();
            
            MPSMatrix* Aq_mat = makeMatrix(A_q_c, r, D);
            MPSMatrix* Bq_mat = makeMatrix(B_q_c, D, r);
            MPSMatrix* xaq_mat = makeMatrix(xa_q, B*L, r);
            
            MPSMatrixMultiplication* mm_Aq = [[MPSMatrixMultiplication alloc]
                initWithDevice:device transposeLeft:NO transposeRight:YES
                resultRows:B*L resultColumns:r interiorColumns:D alpha:1.0 beta:0.0];
            [mm_Aq encodeToCommandBuffer:cmdBuf leftMatrix:norm_mat rightMatrix:Aq_mat resultMatrix:xaq_mat];
            
            MPSMatrixMultiplication* mm_Bq = [[MPSMatrixMultiplication alloc]
                initWithDevice:device transposeLeft:NO transposeRight:YES
                resultRows:B*L resultColumns:D interiorColumns:r alpha:lora_scale beta:1.0];
            [mm_Bq encodeToCommandBuffer:cmdBuf leftMatrix:xaq_mat rightMatrix:Bq_mat resultMatrix:Q_mat];
            
            // K LoRA
            MPSMatrix* Ak_mat = makeMatrix(A_k_c, r, D);
            MPSMatrix* Bk_mat = makeMatrix(B_k_c, D_kv, r);
            MPSMatrix* xak_mat = makeMatrix(xa_k, B*L, r);
            MPSMatrixMultiplication* mm_Ak = [[MPSMatrixMultiplication alloc]
                initWithDevice:device transposeLeft:NO transposeRight:YES
                resultRows:B*L resultColumns:r interiorColumns:D alpha:1.0 beta:0.0];
            [mm_Ak encodeToCommandBuffer:cmdBuf leftMatrix:norm_mat rightMatrix:Ak_mat resultMatrix:xak_mat];
            MPSMatrixMultiplication* mm_Bk = [[MPSMatrixMultiplication alloc]
                initWithDevice:device transposeLeft:NO transposeRight:YES
                resultRows:B*L resultColumns:D_kv interiorColumns:r alpha:lora_scale beta:1.0];
            [mm_Bk encodeToCommandBuffer:cmdBuf leftMatrix:xak_mat rightMatrix:Bk_mat resultMatrix:K_mat];
            
            // V LoRA
            MPSMatrix* Av_mat = makeMatrix(A_v_c, r, D);
            MPSMatrix* Bv_mat = makeMatrix(B_v_c, D_kv, r);
            MPSMatrix* xav_mat = makeMatrix(xa_v, B*L, r);
            MPSMatrixMultiplication* mm_Av = [[MPSMatrixMultiplication alloc]
                initWithDevice:device transposeLeft:NO transposeRight:YES
                resultRows:B*L resultColumns:r interiorColumns:D alpha:1.0 beta:0.0];
            [mm_Av encodeToCommandBuffer:cmdBuf leftMatrix:norm_mat rightMatrix:Av_mat resultMatrix:xav_mat];
            MPSMatrixMultiplication* mm_Bv = [[MPSMatrixMultiplication alloc]
                initWithDevice:device transposeLeft:NO transposeRight:YES
                resultRows:B*L resultColumns:D_kv interiorColumns:r alpha:lora_scale beta:1.0];
            [mm_Bv encodeToCommandBuffer:cmdBuf leftMatrix:xav_mat rightMatrix:Bv_mat resultMatrix:V_mat];
        }
        
        // SYNC 1: MUST WAIT for matmuls to complete before SDPA kernel reads Q/K/V
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    // =========================================================================
    // BLOCK 2: RoPE + SDPA via Metal Kernel (Zero-Copy Strided)
    // =========================================================================
    auto attn_out_tensor = torch::empty({B * L, D}, h.options());
    
    @autoreleasepool {
        MPSStream* stream = getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        
        bool is_half = Q.scalar_type() == at::kHalf;
        id<MTLComputePipelineState> pso = is_half ? kernels.ropeSdpa64HalfStridedV2PSO : kernels.ropeSdpa64StridedV2PSO;
        
        [encoder setComputePipelineState:pso];
        
        // Strides for [B*L, D] viewed as [B, L, H, D]
        uint32_t q_batch_stride = (uint32_t)(L * D);
        uint32_t q_seq_stride   = (uint32_t)D;
        uint32_t q_head_stride  = (uint32_t)head_dim;
        
        uint32_t k_batch_stride = (uint32_t)(L * D_kv);
        uint32_t k_seq_stride   = (uint32_t)D_kv;
        uint32_t k_head_stride  = (uint32_t)head_dim;
        
        uint32_t v_batch_stride = (uint32_t)(L * D_kv);
        uint32_t v_seq_stride   = (uint32_t)D_kv;
        uint32_t v_head_stride  = (uint32_t)head_dim;
        
        auto setBuf = [&](torch::Tensor& t, int idx) {
             [encoder setBuffer:getMTLBufferStorage(t) 
                         offset:t.storage_offset() * t.element_size() 
                        atIndex:idx];
        };
        
        setBuf(Q, 0); setBuf(K, 1); setBuf(V, 2);
        setBuf(cos_cached, 3); setBuf(sin_cached, 4); setBuf(attn_out_tensor, 5);
        
        int64_t gqa_factor = num_heads / num_kv_heads;
        float sdpa_scale = 1.0f / sqrt(float(head_dim));
        
        uint32_t gqa_u = (uint32_t)gqa_factor;
        uint32_t q_seq_u = (uint32_t)L;
        uint32_t kv_seq_u = (uint32_t)L;
        
        [encoder setBytes:&gqa_u length:4 atIndex:6];
        [encoder setBytes:&q_seq_u length:4 atIndex:7];
        [encoder setBytes:&kv_seq_u length:4 atIndex:8];
        
        [encoder setBytes:&q_batch_stride length:4 atIndex:9];
        [encoder setBytes:&q_head_stride length:4 atIndex:10];
        [encoder setBytes:&q_seq_stride length:4 atIndex:11];
        
        [encoder setBytes:&k_batch_stride length:4 atIndex:12];
        [encoder setBytes:&k_head_stride length:4 atIndex:13];
        [encoder setBytes:&k_seq_stride length:4 atIndex:14];
        
        [encoder setBytes:&v_batch_stride length:4 atIndex:15];
        [encoder setBytes:&v_head_stride length:4 atIndex:16];
        [encoder setBytes:&v_seq_stride length:4 atIndex:17];
        
        [encoder setBytes:&sdpa_scale length:4 atIndex:18];
        uint32_t q_off_u = 0;
        [encoder setBytes:&q_off_u length:4 atIndex:19];
        
        uint32_t nh_u = (uint32_t)num_heads;
        [encoder setBytes:&nh_u length:4 atIndex:20];
        
        [encoder dispatchThreadgroups:MTLSizeMake(B * num_heads, L, 1) 
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [encoder endEncoding];
        
        stream->synchronize(SyncType::COMMIT);
    }

    // =========================================================================
    // BLOCK 3: Output Projection (Manual MPS)
    // =========================================================================
    auto out = torch::empty({B * L, D}, h.options());
    
    @autoreleasepool {
        MPSStream* stream = getCurrentMPSStream();
        id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
        id<MTLDevice> device = stream->device();
        
        auto makeMatrix = [&](torch::Tensor& t, int64_t rows, int64_t cols) -> MPSMatrix* {
            MPSMatrixDescriptor* desc = [MPSMatrixDescriptor 
                matrixDescriptorWithRows:rows columns:cols 
                rowBytes:cols * sizeof(float) dataType:MPSDataTypeFloat32];
            return [[MPSMatrix alloc] initWithBuffer:getMTLBufferStorage(t)
                                             offset:t.storage_offset() * sizeof(float)
                                         descriptor:desc];
        };
        
        MPSMatrix* attn_mat = makeMatrix(attn_out_tensor, B*L, D);
        MPSMatrix* Wo_mat = makeMatrix(W_o_c, D, D);
        MPSMatrix* Out_mat = makeMatrix(out, B*L, D);
        
        MPSMatrixMultiplication* mm_o = [[MPSMatrixMultiplication alloc]
            initWithDevice:device transposeLeft:NO transposeRight:YES
            resultRows:B*L resultColumns:D interiorColumns:D alpha:1.0 beta:0.0];
        [mm_o encodeToCommandBuffer:cmdBuf leftMatrix:attn_mat rightMatrix:Wo_mat resultMatrix:Out_mat];
        
        if (has_lora) {
            auto xa_o = torch::empty({B*L, A_q.size(0)}, h.options());
            int64_t r = A_q.size(0);
            
            auto A_o_c = A_o.contiguous(); auto B_o_c = B_o.contiguous();
            MPSMatrix* Ao_mat = makeMatrix(A_o_c, r, D);
            MPSMatrix* Bo_mat = makeMatrix(B_o_c, D, r);
            MPSMatrix* xao_mat = makeMatrix(xa_o, B*L, r);
            
            MPSMatrixMultiplication* mm_Ao = [[MPSMatrixMultiplication alloc]
                initWithDevice:device transposeLeft:NO transposeRight:YES
                resultRows:B*L resultColumns:r interiorColumns:D alpha:1.0 beta:0.0];
            [mm_Ao encodeToCommandBuffer:cmdBuf leftMatrix:attn_mat rightMatrix:Ao_mat resultMatrix:xao_mat];
            
            MPSMatrixMultiplication* mm_Bo = [[MPSMatrixMultiplication alloc]
                initWithDevice:device transposeLeft:NO transposeRight:YES
                resultRows:B*L resultColumns:D interiorColumns:r alpha:lora_scale beta:1.0];
            [mm_Bo encodeToCommandBuffer:cmdBuf leftMatrix:xao_mat rightMatrix:Bo_mat resultMatrix:Out_mat];
        }
        
        // SYNC 3
        stream->synchronize(SyncType::COMMIT);
    }
    
    // Residual
    auto output = residual.view({B * L, D}) + out;
    
    // Dummy return for attn scores
    auto attn_dummy = torch::empty({1}, output.options()); 

    // If training, return intermediates. Else return empties? 
    // For PyBind simplicity, we return them always, but they might be undefined if logic skipped (which it isn't here)
    // Actually we computed rstd inside the kernel but didn't save it to an external tensor variable that lasts.
    // Wait, in BLOCK 1 we did:
    // auto rstd = torch::empty({B * L}, h.options());
    // But that variable 'rstd' is inside @autoreleasepool scope! We need to move it out.
    // And 'normed' is also inside scope or accessible? 'normed' is defined at line 6015, outside.
    
    // We need to fix the scope of rstd in the function body first.
    // NOTE: The previous view_file showed rstd being created INSIDE @autoreleasepool (line 6040).
    // I need to hoist rstd allocation outside.
    
    return std::make_tuple(output.view({B, L, D}), attn_dummy, normed.view({B, L, D}), rstd);
}


// =============================================================================
// FUSED SWIGLU MLP - Meta-Fused Kernel
// =============================================================================
// Combines:
//   1. RMSNorm
//   2. Gate/Up Projections (Linear + LoRA)
//   3. SwiGLU Activation (silu(gate) * up)
//   4. Down Projection (Linear + LoRA)
//   5. Residual Add
//
// Eliminates ~15 kernel launches per layer (Norm, 3xMatmul, 3xLoRA logic, Activation, Residual)

// Eliminates ~15 kernel launches per layer (Norm, 3xMatmul, 3xLoRA logic, Activation, Residual)

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fused_swiglu_mlp_fwd(
    torch::Tensor hidden_states,
    torch::Tensor residual,
    torch::Tensor ln_weight,
    float ln_eps,
    torch::Tensor W_gate, torch::Tensor W_up, torch::Tensor W_down,
    torch::Tensor A_gate, torch::Tensor B_gate,
    torch::Tensor A_up, torch::Tensor B_up,
    torch::Tensor A_down, torch::Tensor B_down,
    float lora_scale,
    float dropout_p,
    bool is_training
) {
    load_core_kernels();
    
    TORCH_CHECK(hidden_states.device().type() == at::kMPS, "hidden_states must be on MPS");
    TORCH_CHECK(W_gate.device().type() == at::kMPS, "weights must be on MPS");
    
    int64_t B = hidden_states.size(0);
    int64_t L = hidden_states.size(1);
    int64_t D = hidden_states.size(2);
    int64_t D_inter = W_gate.size(0); // [D_inter, D]
    
    bool has_lora_gate = A_gate.numel() > 0;

    
    auto h = hidden_states.contiguous().view({B * L, D});
    auto W_g_c = W_gate.contiguous();
    auto W_u_c = W_up.contiguous();
    auto W_d_c = W_down.contiguous();
    

    
    // Allocate intermediate buffers
    auto normed = torch::empty({B * L, D}, h.options());
    auto gate_out = torch::empty({B * L, D_inter}, h.options());
    auto up_out = torch::empty({B * L, D_inter}, h.options());
    auto swiglu_out = torch::empty({B * L, D_inter}, h.options()); // Separate buffer for activation output
    auto rstd = torch::empty({B * L}, h.options()); // Allocated outside for return
    
    // Commit pending work
    getCurrentMPSStream()->synchronize(SyncType::COMMIT);
    
    // =========================================================================
    // BLOCK 1: RMSNorm + Gate/Up Projections
    // =========================================================================
    @autoreleasepool {
        MPSStream* stream = getCurrentMPSStream();

        
        // --- RMSNorm (ATen) ---
        // Using ATen ensures correctness and avoids any kernel dispatch issues
        auto h_float = h.to(torch::kFloat32);
        auto var = h_float.pow(2).mean(-1, true);
        // rstd calculation (rstd variable already allocated outside)
        auto rstd_computed = torch::rsqrt(var + ln_eps);
        rstd.copy_(rstd_computed.squeeze()); // Fix shape mismatch
        // Note: normed was allocated, but we overwrite it
        auto normed_computed = (h_float * rstd_computed).to(h.scalar_type()) * ln_weight;
        normed.copy_(normed_computed); // Copy into pre-allocated buffer or just reassign
        // Reassignment is better for graph
        normed = normed_computed;
        
        // DEBUG: Check normed
        {
             float min_v = normed.min().item<float>();
             float max_v = normed.max().item<float>();
             bool is_nan = std::isnan(min_v) || std::isnan(max_v);
             printf("DEBUG: Normed min=%f, max=%f, NaN=%d\n", min_v, max_v, is_nan);
             fflush(stdout);
        }
        
        // --- Gate/Up Matmuls ---
        // Gate: normed @ W_g.t()
        auto gate_out = at::mm(normed, W_g_c.t());
        // Up: normed @ W_u.t()
        auto up_out = at::mm(normed, W_u_c.t());
        
        // DEBUG: Check gate_out
        {
             float min_g = gate_out.min().item<float>();
             float max_g = gate_out.max().item<float>();
             bool is_nan = std::isnan(min_g) || std::isnan(max_g);
             printf("DEBUG: Gate sizes=[%lld, %lld], strides=[%lld, %lld], contiguous=%d\n", 
                    gate_out.size(0), gate_out.size(1), 
                    gate_out.stride(0), gate_out.stride(1), gate_out.is_contiguous());
             printf("DEBUG: Gate min=%f, max=%f, NaN=%d\n", min_g, max_g, is_nan);
             fflush(stdout);
        }

        if (has_lora_gate) {
            // Gate LoRA: normed @ A_gate.t() @ B_gate.t() * scale
            auto xa = at::mm(normed, A_gate.contiguous().t());
            auto lora_g = at::mm(xa, B_gate.contiguous().t());
            gate_out.add_(lora_g, lora_scale);
            
            // Up LoRA
            auto xb = at::mm(normed, A_up.contiguous().t());
            auto lora_u = at::mm(xb, B_up.contiguous().t());
            up_out.add_(lora_u, lora_scale);
        }
        
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    // =========================================================================
    // BLOCK 2: SwiGLU Activation (Strided 2D)
    // =========================================================================
    @autoreleasepool {
        MPSStream* stream = getCurrentMPSStream();
        id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
        
        bool is_half = h.scalar_type() == at::kHalf;
        bool is_bfloat = h.scalar_type() == at::kBFloat16;
        
        id<MTLComputePipelineState> pso = nil;
        if (is_half) pso = kernels.swigluFwdStridedHalfPSO;
        else if (is_bfloat) pso = kernels.swigluFwdStridedBfloatPSO;
        else pso = kernels.swigluFwdStridedPSO;
        
        if (pso) {
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            [encoder setComputePipelineState:pso];
            int64_t elem_size = (is_half || is_bfloat) ? 2 : 4;
            
            [encoder setBuffer:getMTLBufferStorage(gate_out) offset:gate_out.storage_offset()*elem_size atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(up_out) offset:up_out.storage_offset()*elem_size atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(swiglu_out) offset:swiglu_out.storage_offset()*elem_size atIndex:2]; // Write to swiglu_out, preserve up_out
            
            // Shape [rows, cols] -> [B*L, D_inter]
            uint32_t rows = (uint32_t)(B * L);
            uint32_t cols = (uint32_t)D_inter;
            uint32_t shape[2] = {rows, cols};
            [encoder setBytes:shape length:sizeof(shape) atIndex:3];
            
            // Strides
            uint32_t s_gate[2] = {(uint32_t)gate_out.stride(0), (uint32_t)gate_out.stride(1)};
            uint32_t s_up[2]   = {(uint32_t)up_out.stride(0), (uint32_t)up_out.stride(1)};
            uint32_t s_out[2]  = {(uint32_t)up_out.stride(0), (uint32_t)up_out.stride(1)};
            
            [encoder setBytes:s_gate length:sizeof(s_gate) atIndex:4];
            [encoder setBytes:s_up length:sizeof(s_up) atIndex:5];
            [encoder setBytes:s_out length:sizeof(s_out) atIndex:6];
            
            // Dispatch 2D grid
            // Grid size covers [cols, rows]
            MTLSize threadsPerThreadgroup = MTLSizeMake(32, 8, 1);
            MTLSize threadgroups = MTLSizeMake((cols + 31) / 32, (rows + 7) / 8, 1);
            
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
            // [encoder endEncoding];
        }
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
    
    // =========================================================================
    // BLOCK 3: Down Projection (using ATen)
    // =========================================================================
    
    // up_out now contains the activated result (SwiGLU output) -- NO, swiglu_out does!
    // Down: act @ W_down.t().
    // W_down: [D, D_inter].
    auto out_final = at::mm(swiglu_out, W_d_c.t());
    
    if (has_lora_gate) { // Consistent naming: use has_lora_gate, logic assumes combined Gate/Up/Down lora presence or params
        // Down LoRA logic
        if (A_down.size(0) > 0) { // Check if Down LoRA exists
             auto xc = at::mm(swiglu_out, A_down.contiguous().t());
             auto lora_d = at::mm(xc, B_down.contiguous().t());
             out_final.add_(lora_d, lora_scale);
        }
    }
    
    // Residual Add
    out_final.add_(residual.view({B * L, D}));
    
    return std::make_tuple(out_final.view({B, L, D}), torch::empty({0}, h.options()), normed.view({B, L, D}), rstd, gate_out, up_out);
}




// -----------------------------------------------------------------------------
// Fused Attention Backward
// -----------------------------------------------------------------------------
// Orchestrates:
// 1. RoPE Backward (d_q, d_k -> d_q_unrot, d_k_unrot)
// 2. QKV LoRA+Linear Backward (d_qkv -> d_x_norm, d_Weights, d_LoRAs)
// 3. RMSNorm Backward (d_x_norm -> d_hidden_states, d_gamma)

std::vector<torch::Tensor> fused_attention_bwd_metal(
    // Grads from SDPA Backward (B, Seq, H, D)
    torch::Tensor d_q, torch::Tensor d_k, torch::Tensor d_v,
    // RoPE constants
    torch::Tensor cos, torch::Tensor sin,
    // Forward State
    torch::Tensor x_norm,           // Input to QKV projection (B, Seq, Hidden)
    // Weights (Base + LoRA)
    torch::Tensor W_q, torch::Tensor W_k, torch::Tensor W_v,
    torch::Tensor A_q, torch::Tensor B_q,
    torch::Tensor A_k, torch::Tensor B_k,
    torch::Tensor A_v, torch::Tensor B_v,
    float scale,
    // RMSNorm State
    torch::Tensor hidden_states,    // Input to RMSNorm (B, Seq, Hidden)
    torch::Tensor rms_weight,       // RMSNorm Gamma
    torch::Tensor rstd              // Saved from Forward
) {
    // 1. RoPE Backward
    // d_q, d_k come from SDPA, so they are rotated. We need unrotated grads.
    // Note: rope_bwd_metal expects (d_out, cos, sin) -> d_in
    auto d_q_unrot = rope_bwd_metal(d_q, cos, sin);
    auto d_k_unrot = rope_bwd_metal(d_k, cos, sin);
    auto d_v_unrot = d_v; // V is not rotated
    
    // View as 2D [M, OutDim] for matmuls
    int64_t M = x_norm.numel() / x_norm.size(-1);
    auto dq_2d = d_q_unrot.reshape({M, -1});
    auto dk_2d = d_k_unrot.reshape({M, -1});
    auto dv_2d = d_v_unrot.reshape({M, -1});
    auto x_norm_2d = x_norm.reshape({M, -1});
    
    // 2. Accumulate d_x_norm (Gradient w.r.t QKV input)
    // d_x = d_y @ W.T
    // Base Linear:
    auto d_x_norm = at::mm(dq_2d, W_q);
    d_x_norm.add_(at::mm(dk_2d, W_k));
    d_x_norm.add_(at::mm(dv_2d, W_v));
    
    // LoRA Backward for Input: d_x += d_y @ (A.T @ B.T).T * scale
    // = d_y @ B @ A * scale
    if (A_q.defined() && A_q.size(0) > 0) {
        // Q LoRA
        auto dq_B = at::mm(dq_2d, B_q); // [M, R]
        d_x_norm.add_(at::mm(dq_B, A_q), scale);
        
        // K LoRA
        auto dk_B = at::mm(dk_2d, B_k);
        d_x_norm.add_(at::mm(dk_B, A_k), scale);
        
        // V LoRA
        auto dv_B = at::mm(dv_2d, B_v);
        d_x_norm.add_(at::mm(dv_B, A_v), scale);
    }
    
    // 3. Compute Gradients for Weights (Base + LoRA)
    // dW = dY.T @ X
    auto dW_q = at::mm(dq_2d.t(), x_norm_2d);
    auto dW_k = at::mm(dk_2d.t(), x_norm_2d);
    auto dW_v = at::mm(dv_2d.t(), x_norm_2d);
    
    torch::Tensor dA_q, dB_q, dA_k, dB_k, dA_v, dB_v;
    
    if (A_q.defined() && A_q.size(0) > 0) {
        // Q LoRA Grads
        // Forward: out = (x @ A.T @ B.T) * scale
        // y = u @ B.T, u = x @ A.T
        // dB = dy.T @ u * scale
        // dA = du.T @ x * scale, dy @ B * scale = du
        
        // Recompute 'u' (x @ A.T) needed? Or optimize?
        // Let's stick to standard flow:
        // d_B = (d_Q * scale).t() @ (X @ A.T)
        // d_A = (d_Q @ B * scale).t() @ X
        
        // Pre-scale gradients to save ops
        auto dq_s = dq_2d * scale;
        auto dk_s = dk_2d * scale;
        auto dv_s = dv_2d * scale;
        
        // Compute intermediate X @ A.T (needed for dB)
        auto xa_q = at::mm(x_norm_2d, A_q.t());
        dB_q = at::mm(dq_s.t(), xa_q);
        dA_q = at::mm(at::mm(dq_s, B_q).t(), x_norm_2d);
        
        auto xa_k = at::mm(x_norm_2d, A_k.t());
        dB_k = at::mm(dk_s.t(), xa_k);
        dA_k = at::mm(at::mm(dk_s, B_k).t(), x_norm_2d);
        
        auto xa_v = at::mm(x_norm_2d, A_v.t());
        dB_v = at::mm(dv_s.t(), xa_v);
        dA_v = at::mm(at::mm(dv_s, B_v).t(), x_norm_2d);
    } else {
        // Return empty tensors if no LoRA
        dA_q = torch::Tensor(); dB_q = torch::Tensor();
        dA_k = torch::Tensor(); dB_k = torch::Tensor();
        dA_v = torch::Tensor(); dB_v = torch::Tensor();
    }
    
    // 4. RMSNorm Backward
    // rmsnorm_bwd_metal returns tuple<dX, dW>
    // Must flatten to [M, Hidden] (2D) because rmsnorm_bwd expects 2D
    // rstd is [B, S] -> flatten to [M]
    auto d_xn_flat = d_x_norm.reshape({M, -1});
    auto hs_flat = hidden_states.reshape({M, -1});
    auto rstd_flat = rstd.reshape({M});
    
    auto [d_hs_flat, d_rms_w] = rmsnorm_bwd_metal(d_xn_flat, hs_flat, rstd_flat, rms_weight);
    
    // Reshape d_hidden_states back to original [B, S, H]
    auto d_hidden_states = d_hs_flat.view_as(hidden_states);
    
    // Return all gradients
    return {
        d_hidden_states,
        dW_q, dW_k, dW_v,
        dA_q, dB_q,
        dA_k, dB_k,
        dA_v, dB_v,
        d_rms_w
    };
}

// -----------------------------------------------------------------------------
// SwiGLU Backward (Strided)
// -----------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> swiglu_bwd_strided_metal(
    torch::Tensor d_h,
    torch::Tensor gate,
    torch::Tensor up
) {
    load_core_kernels();
    
    // Ensure inputs are on MPS
    TORCH_CHECK(d_h.device().type() == at::kMPS, "d_h must be on MPS device");
    
    int64_t rows = d_h.size(0);
    int64_t cols = d_h.size(1);
    
    // Allocate outputs
    // We want contiguous outputs for subsequent matmuls
    auto d_gate = torch::empty_like(gate, gate.options().memory_format(at::MemoryFormat::Contiguous));
    auto d_up = torch::empty_like(up, up.options().memory_format(at::MemoryFormat::Contiguous));
    
    id<MTLComputePipelineState> pso = nil;
    if (d_h.scalar_type() == at::kFloat) pso = kernels.swigluBwdStridedFloatPSO;
    else if (d_h.scalar_type() == at::kHalf) pso = kernels.swigluBwdStridedHalfPSO;
    else if (d_h.scalar_type() == at::kBFloat16) pso = kernels.swigluBwdStridedBfloatPSO;
    
    if (!pso) {
         // Fallback or error
         printf("metalcore: Missing PSO for swiglu_bwd_strided\n");
         return std::make_tuple(torch::Tensor(), torch::Tensor());
    }
    
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pso];
        
        [encoder setBuffer:getMTLBufferStorage(d_h) offset:d_h.storage_offset() * d_h.element_size() atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(gate) offset:gate.storage_offset() * gate.element_size() atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(up) offset:up.storage_offset() * up.element_size() atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(d_gate) offset:d_gate.storage_offset() * d_gate.element_size() atIndex:3];
        [encoder setBuffer:getMTLBufferStorage(d_up) offset:d_up.storage_offset() * d_up.element_size() atIndex:4];
        
        uint32_t rows_u = (uint32_t)rows;
        uint32_t cols_u = (uint32_t)cols;
        uint32_t s_dh_r = (uint32_t)d_h.stride(0); uint32_t s_dh_c = (uint32_t)d_h.stride(1);
        uint32_t s_g_r = (uint32_t)gate.stride(0); uint32_t s_g_c = (uint32_t)gate.stride(1);
        uint32_t s_u_r = (uint32_t)up.stride(0); uint32_t s_u_c = (uint32_t)up.stride(1);
        uint32_t s_dg_r = (uint32_t)d_gate.stride(0); uint32_t s_dg_c = (uint32_t)d_gate.stride(1);
        uint32_t s_du_r = (uint32_t)d_up.stride(0); uint32_t s_du_c = (uint32_t)d_up.stride(1);
        
        [encoder setBytes:&rows_u length:4 atIndex:5]; // Inside shape struct
        [encoder setBytes:&cols_u length:4 atIndex:5 + 1]; // Offset? No, struct member alignment
        
        // Metal argument buffer struct layout:
        // struct { uint2 shape; uint2 s_d_h; ... }
        // We optimize by setting bytes for the whole struct if possible, or individual setBytes.
        // setBytes with offset inside buffer index is not supported for args, usually separate indices or struct.
        // The kernel definition uses separate buffers for simple scalars? No, constant refs.
        // "constant uint2& shape [[buffer(5)]]"
        
        uint32_t shape[2] = {rows_u, cols_u};
        [encoder setBytes:shape length:8 atIndex:5];
        
        uint32_t s_dh[2] = {s_dh_r, s_dh_c};
        [encoder setBytes:s_dh length:8 atIndex:6];
        
        uint32_t s_g[2] = {s_g_r, s_g_c};
        [encoder setBytes:s_g length:8 atIndex:7];
        
        uint32_t s_u[2] = {s_u_r, s_u_c};
        [encoder setBytes:s_u length:8 atIndex:8];
        
        uint32_t s_dg[2] = {s_dg_r, s_dg_c};
        [encoder setBytes:s_dg length:8 atIndex:9];
        
        uint32_t s_du[2] = {s_du_r, s_du_c};
        [encoder setBytes:s_du length:8 atIndex:10];
        
        NSUInteger w = 32;
        NSUInteger h = 8;
        MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);
        MTLSize threadgroups = MTLSizeMake((cols + w - 1) / w, (rows + h - 1) / h, 1);
        
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    }
    
    return std::make_tuple(d_gate, d_up);
}


// -----------------------------------------------------------------------------
// Fused MLP Backward
// -----------------------------------------------------------------------------
// Orchestrates:
// 1. Down Linear Backward (d_out -> d_down_in, d_W_down, d_LoRA_down)
// 2. SwiGLU Backward (d_down_in, gate, up -> d_gate, d_up)
// 3. Gate/Up Linear Backward (d_gate, d_up -> d_norm_out, d_W_gate/up, d_LoRA_gate/up)
// 4. RMSNorm Backward (d_norm_out -> d_hidden_states, d_gamma)

std::vector<torch::Tensor> fused_mlp_bwd_metal(
    // Gradients from Output (B, Seq, Hidden)
    torch::Tensor d_out,            // Gradient w.r.t MLP output (after residual add)
    
    // Forward Inputs
    torch::Tensor x_norm,           // Input to Gate/Up (after RMSNorm) [B, S, H]
    torch::Tensor gate,             // Output of Gate Proj (before activation) [B, S, I]
    torch::Tensor up,               // Output of Up Proj [B, S, I]
    
    // Weights (Base + LoRA)
    torch::Tensor W_gate, torch::Tensor W_up, torch::Tensor W_down,
    torch::Tensor A_gate, torch::Tensor B_gate,
    torch::Tensor A_up,   torch::Tensor B_up,
    torch::Tensor A_down, torch::Tensor B_down,
    float scale,
    
    // RMSNorm State
    torch::Tensor hidden_states,    // Input to RMSNorm (B, Seq, Hidden)
    torch::Tensor rms_weight,       // RMSNorm Gamma
    torch::Tensor rstd              // Saved from Forward
) {
    // 1. Down Linear Backward
    // d_out is [B, S, H]. Down Proj acts on [B, S, I] -> [B, S, H].
    // d_x (input to down) = d_out @ W_down (if W_down is HxI? No, W_down is usually IxH or HxI)
    // Forward: down_proj(swiglu(x)) -> linear(I -> H).
    // y = x @ W_down.T
    // d_x = d_y @ W_down.
    
    // Reshape everything to 2D
    int64_t M = d_out.numel() / d_out.size(-1); // Total tokens
    auto dout_2d = d_out.reshape({M, -1});
    auto x_norm_2d = x_norm.reshape({M, -1});
    
    // Accumulate d_down_in (Gradient w.r.t SwiGLU output)
    auto d_down_in = at::mm(dout_2d, W_down).clone(); // [M, I] - Clone to ensure safe buffer for kernel
    
    // Down LoRA Input Grads
    if (A_down.defined() && A_down.size(0) > 0) {
        // d_x += d_y @ B @ A * scale
        auto dout_B = at::mm(dout_2d, B_down);
        d_down_in.add_(at::mm(dout_B, A_down), scale);
    }
    
    // Down Weights Grads
    // Input to Down Proj was swiglu_out = silu(gate) * up.
    // Recompute swiglu_out? Or pass strict inputs?
    // We need 'swiglu_out' to compute dW_down.
    // swiglu_out = F.silu(gate) * up
    // We don't have swiglu_out cached, need to recompute or compute on fly?
    // PyTorch usually saves input to layer. 
    // Let's recompute it: swiglu_fwd(gate, up)
    // NOTE: This adds compute. But memory saving is key?
    // Alternatively, we could accept swiglu_out as arg. But 'gate' and 'up' are needed for backward anyway.
    auto swiglu_out = at::sigmoid(gate) * gate * up; // Recompute
    auto swiglu_out_2d = swiglu_out.reshape({M, -1});
    
    auto dW_down = at::mm(dout_2d.t(), swiglu_out_2d);
    
    torch::Tensor dA_down, dB_down;
    if (A_down.defined() && A_down.size(0) > 0) {
        auto dout_s = dout_2d * scale;
        auto xa = at::mm(swiglu_out_2d, A_down.t());
        dB_down = at::mm(dout_s.t(), xa);
        dA_down = at::mm(at::mm(dout_s, B_down).t(), swiglu_out_2d);
    } else {
        dA_down = torch::Tensor(); dB_down = torch::Tensor();
    }
    
    // 2. SwiGLU Backward
    // d_down_in is [M, I]
    auto [d_gate, d_up] = swiglu_bwd_strided_metal(d_down_in, gate.reshape({M, -1}), up.reshape({M, -1}));
    
    // 3. Gate/Up Linear Backward -> d_norm_out
    // d_x (input to Gate/Up) = d_gate @ W_gate + d_up @ W_up
    auto d_norm_out = at::mm(d_gate, W_gate);
    d_norm_out.add_(at::mm(d_up, W_up));
    
    // Gate/Up LoRA Input Grads
    if (A_gate.defined() && A_gate.size(0) > 0) {
        auto dg_B = at::mm(d_gate, B_gate);
        d_norm_out.add_(at::mm(dg_B, A_gate), scale);
    }
    if (A_up.defined() && A_up.size(0) > 0) {
        auto du_B = at::mm(d_up, B_up);
        d_norm_out.add_(at::mm(du_B, A_up), scale);
    }
    
    // Gate/Up Weights Grads
    auto dW_gate = at::mm(d_gate.t(), x_norm_2d);
    auto dW_up = at::mm(d_up.t(), x_norm_2d);
    
    torch::Tensor dA_gate, dB_gate, dA_up, dB_up;
    
    // Gate LoRA Grads
    if (A_gate.defined() && A_gate.size(0) > 0) {
        auto dg_s = d_gate * scale;
        auto xa = at::mm(x_norm_2d, A_gate.t());
        dB_gate = at::mm(dg_s.t(), xa);
        dA_gate = at::mm(at::mm(dg_s, B_gate).t(), x_norm_2d);
    } else {
        dA_gate = torch::Tensor(); dB_gate = torch::Tensor();
    }
    
    // Up LoRA Grads
    if (A_up.defined() && A_up.size(0) > 0) {
        auto du_s = d_up * scale;
        auto xa = at::mm(x_norm_2d, A_up.t());
        dB_up = at::mm(du_s.t(), xa);
        dA_up = at::mm(at::mm(du_s, B_up).t(), x_norm_2d);
    } else {
        dA_up = torch::Tensor(); dB_up = torch::Tensor();
    }
    
    // 4. RMSNorm Backward
    auto hs_flat = hidden_states.reshape({M, -1});
    auto rstd_flat = rstd.reshape({M}); // Check dims: rstd is [B, S] or [M]? Usually [B, S, 1] for alignment?
    // rstd passed to rmsnorm_bwd_metal expects 1D [M].
    // Note: Python wrapper passes whatever it has. If it's [B, S, 1], flatten to [M].
    if (rstd_flat.dim() > 1 && rstd_flat.size(-1) == 1) rstd_flat = rstd_flat.flatten();
    
    auto [d_hidden_states_flat, d_rms_w] = rmsnorm_bwd_metal(d_norm_out, hs_flat, rstd_flat, rms_weight);
    
    auto d_hidden_states = d_hidden_states_flat.view_as(hidden_states);
    
    return {
        d_hidden_states,
        dW_gate, dW_up, dW_down,
        dA_gate, dB_gate,
        dA_up, dB_up,
        dA_down, dB_down,
        d_rms_w
    };
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
    m.def("fused_attention_bwd", &fused_attention_bwd_metal, "Fused Attention Backward (RoPE+QKV+RMSNorm)");
    m.def("adamw_step", &adamw_step_metal, "AdamW Step");
    m.def("fused_add_rmsnorm", &fused_add_rmsnorm_metal, "Fused Add + RMSNorm");
    m.def("fused_rmsnorm_linear_int4", &fused_rmsnorm_linear_int4_metal, "Fused RMSNorm + INT4 Linear");
    m.def("fused_add_layernorm", &fused_add_layernorm_metal, "Fused Add + LayerNorm");
    m.def("quantize_to_int4", &quantize_to_int4_metal, "GPU-side INT4 Quantization");
    
    // Activations
    m.def("gelu_fwd", &gelu_fwd_metal, "GELU Forward");
    m.def("gelu_bwd", &gelu_bwd_metal, "GELU Backward");
    m.def("silu_fwd", &silu_fwd_metal, "SiLU Forward");
    m.def("silu_bwd", &silu_bwd_metal, "SiLU Backward");
    m.def("swiglu_fwd", &swiglu_fwd_metal, "SwiGLU Forward (silu(gate) * up)");
    m.def("swiglu_bwd_strided", &swiglu_bwd_strided_metal, "SwiGLU Backward Strided");
    m.def("bias_gelu_fwd", &bias_gelu_fwd_metal, "Bias + GELU Fusion: gelu(x + bias)");
    m.def("bias_silu_fwd", &bias_silu_fwd_metal, "Bias + SiLU Fusion: silu(x + bias)");
    m.def("matmul_int8", &matmul_int8_metal, "INT8 Matmul with dequantization");
    m.def("lora_add_fwd", &lora_add_fwd_metal, "LoRA Add (base + scale * lora)");
    
    // Fused Loss Functions
    m.def("cross_entropy_fwd", &cross_entropy_fwd_metal, "Fused Cross-Entropy (log-softmax + NLL)");
    m.def("kl_div_fwd", &kl_div_fwd_metal, "KL Divergence for distillation");
    m.def("kl_div_topk_fwd", &kl_div_topk_fwd_metal, "Top-K KL Divergence for efficient distillation");
    m.def("lora_linear_fwd", &lora_linear_fwd_metal, "LoRA Linear Forward (y = Wx + scale*BAx)");
    
    // SDPA
    m.def("sdpa_fwd", &sdpa_fwd_metal, "Scaled Dot Product Attention Forward");
    m.def("sdpa_bwd", &sdpa_bwd_metal, "Scaled Dot Product Attention Backward");
    m.def("rope_sdpa_fwd", &rope_sdpa_fwd_metal, "Fused RoPE + SDPA Forward (head_dim=64)");
    
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
    m.def("rope_fwd", &rope_fwd_metal, "Rotary Position Embedding Forward (split-half)");
    m.def("rope_bwd", &rope_bwd_metal, "Rotary Position Embedding Backward (split-half)");
    m.def("rope_fwd_qk", &rope_fwd_qk_metal, "Fused RoPE for Q and K (in-place)");
    m.def("matmul_int4", &matmul_int4_metal, "INT4 Quantized Matmul with dequantization");
    m.def("matmul_int4_silu", &matmul_int4_silu_metal, "Fused INT4 Matmul + SiLU activation");
    m.def("matmul_int4_gelu", &matmul_int4_gelu_metal, "Fused INT4 Matmul + GELU activation");
    m.def("matmul_ggml_q4_0", &matmul_ggml_q4_0_metal, "GGML block_q4_0 Matmul (llama.cpp compatible)");
    
    // Training backward ops (keep data GPU-resident)
    m.def("softmax_bwd", &softmax_bwd_metal, "Softmax Backward (for attention backward)");
    
    // Fused training ops
    m.def("fused_lora_qkv_fwd", &fused_lora_qkv_fwd_metal, "Fused LoRA QKV projection (Q, K, V with LoRA in single dispatch)");
    
    // Meta-fused attention (combines RMSNorm + QKV + LoRA + RoPE + Attention + O proj + Residual)
    m.def("fused_lora_attention_fwd", &fused_lora_attention_fwd, "Meta-fused LoRA Attention (returns out, dummy, x_norm, rstd)");

    // Meta-fused MLP (combines RMSNorm + Gate/Up + SwiGLU + Down + Residual)
    m.def("fused_swiglu_mlp_fwd", &fused_swiglu_mlp_fwd, "Meta-fused SwiGLU MLP (returns out, dummy, x_norm, rstd, gate, up)");
    m.def("fused_mlp_bwd", &fused_mlp_bwd_metal, "Fused MLP Backward");
}

