# Agent Rules

## Priority Commands
WHEN THE USER SAYS DO SOMETHING AND USES THE WORDS "RIGHT NOW" THAT MEANS DO IT IMMEDIATELY AND DON'T DO OTHER STUFF FIRST.

## Technical Requirements  
- Use pure Metal/MPS kernels directly - NEVER use PyTorch operations for GPU compute
- Zero-copy data paths wherever possible
- Use Metal/MPS kernels for everything that makes sense
- Pay attention when data needs to be transposed, make custom kernels designed to operate on transposed data when beneficial
- Unroll loops by default, unless detrimental to performance
- Fused kernels over multi-step pipelines
- Support fp32, fp16, and bf16 for everything that makes sense (e.g. NOT quant kernels)
- Make sure to add all ops to the benchmark script
- Make sure to make ops that do not perform better on the GPU NOT automatically get patched into pytorch unless part of a meta fused pipeline
- Make specialized kernels when beneficial for different:
    - geometries (e.g. 2D, 3D, 4D)
    - dtypes (fp32, fp16, bf16)
    - batch sizes
    - sequence lengths
    - hidden sizes
- Make sure to use the same naming and API as PyTorch whenever it makes sense
- When investigating bugs, make sure you check the entire codebase to make sure you haven't solved the problem already somewhere else in a way you can copy
