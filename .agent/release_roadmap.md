# Release Readiness & Roadmap

To make this library "Complete" and ready for public GitHub release, the following items are recommended.

## Critical for PyTorch Users
### 1. Autograd Support (Backward Pass)
- **Why**: Currently, the library supports `forward` only. Users cannot use `mpssvd.svd` inside a neural network training loop (e.g., for spectral normalization or SVD-based layers) because gradients won't propagate.
- **Solution**: Implement a `torch.autograd.Function` wrapper (`MPS_SVD_Function`).
- **Implementation**: The gradient formula for SVD is well-known. We can compute it in pure Python using PyTorch ops ($U, S, V$ are already on GPU, so the backward math runs on GPU automatically). No new Metal kernel needed.

### 2. Odd Dimension Handling
- **Why**: Creating a matrix with an odd number of columns (e.g., 129) currently might fail or produce undefined behavior because the pairing strategy assumes even/padding.
- **Solution**: Auto-pad inputs in `mpssvd.svd` to nearest even number, then slice the output.

## High Value "Nice-to-Haves"
### 3. Half Precision (FP16/BF16)
- **Why**: MPS is extremely fast with FP16. SVD is sensitive to precision, but for many ML tasks (like LoRA), FP16 is acceptable.
- **Solution**: Template the Metal kernel for `half` or `bfloat16`.

### 4. Input Validation
- **Why**: Public code needs safe guards.
- **Solution**: Check for `NaN`s, ensure contiguous memory layouts (add `.contiguous()` calls if needed), and validate device compatibility.

## Packaging
### 5. `pyproject.toml`
- Modern Python packaging standard.

### 6. CI/CD
- A basic GitHub Action to compile and run tests (though running Metal kernels on GitHub Runners is usually not possible, you can build-test).

## 3. Production Readiness (The "Last Mile" for Public Release)
To make this a standard library that people rely on, you need:

- [ ] **Half Precision (FP16/BF16) Support**: 
    - **Why**: MPS is optimized for FP16. LLMs and diffusion models run in FP16. Currently, users would have to cast to FP32 to use our SVD, losing performance and memory.
    - **How**: Templatize the Metal kernel to support `half` and `bfloat16`.

- [ ] **Zero-Code Integration (Monkeypatching)**:
    - **Why**: Users shouldn't have to rewrite imports. They should just import your library, and `torch.linalg.svd` should magically work on MPS.
    - **How**: Provide a `mpssvd.patch_torch()` function.

- [ ] **Complex Number Support**:
    - **Why**: Essential for Quantum Computing, Signal Processing (STFT), and Audio.
    - **How**: Complex Jacobi rotations are significantly improved in Metal but require a different kernel arithmetic ($2 \times 2$ unitary rotations).

- [ ] **Binary Wheels (CI/CD)**:
    - **Why**: Compiling C++/Metal extensions locally is failure-prone (missing Xcode, wrong header versions).
    - **How**: Set up GitHub Actions to build `.whl` files for `cp39`–`cp312` on `macosx_arm64`.

## Recommendation for Immediate Next Step
I recommend implementing **FP16 Support** and **Monkeypatching** next. These provide the highest value for typical Deep Learning workflows.

