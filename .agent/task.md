# Tasks

[x] Project Setup
  [x] Create project directory structure
  [x] Create `setup.py`

[x] Core Implementation
  [x] Metal Kernels (Jacobi, Transpose, Norms)
  [x] Host Orchestrator (C++)
  [x] Python Bindings

[x] Large Matrix Support (Randomized SVD)
  [x] Implementation
  [x] Verification 10k x 10k

[x] Optimization
  [x] Threadgroup Reduction (20x speedup)
  [x] Fused Block-Jacobi Kernel (Reduce Async Overhead)
    [x] Implement Kernel
    [x] Implement Host Dispatch
    [x] Verify Speedup (Stable! Optimized for small N)

[x] Release Polish
  [x] Autograd
  [x] Odd Dimensions robustness

[x] Advanced Algorithms ("Harder but Better")
  [x] Implement Golub-Kahan-Lanczos Bidiagonalization
  [x] Implement Full Re-orthogonalization
  [x] Benchmark Lanczos vs rSVD

[x] Verification & Robustness
  [x] Stress Testing (120 edge cases)
  [x] Verify Correctness (Valid inputs -> Valid Valid outputs)
    [x] Square, Tall, Odd, Small, Batched
    [x] Handle Wide Matrix (Explicit Error -> Pass)

[x] Production Readiness
  [x] Implement Monkeypatching (`patch_torch()`)
  [x] Implement FP16 Support (`svd.metal` templating)
    [x] Modify Metal Kernels (Templated)
    [x] Modify Host Code (Dispatch / Durability)
    [x] Verify FP16 Accuracy (Works!)
  [x] Benchmark FP16 vs FP32
  [x] Update Documentation (Universal Shape, Monkeypatching, usage, Install)
  [x] Scikit-Learn Support (`MetalTruncatedSVD`)

[x] Cleanup & Release
  [x] Identify cruft files
  [x] Move cruft to `unneeded/`
  [x] Update `.gitignore`
  [x] Add LICENSE (MIT, Kris Bailey via Antigravity)
  [x] Update Metadata (setup.py)
  [x] Init Git and Push to GitHub
  [x] Rename to `metalsvd` (Code, Docs, Repo)
  [x] Verify Autograd (Forward/Backward)
  [x] Create `benchmark_suite.py` (Heavy verification)
  [x] Stress Breaker (Fuzzing & Edge Cases: NaN, Zero, Rank-1, Ill-cond)
  [x] Publish to PyPI (Wheels built)
  [x] Build Wheels for Python 3.10, 3.11, 3.12, 3.13, 3.14 (Multi-version Support)
  [x] Refactor to Src-Layout (PyPI standard compliance)
