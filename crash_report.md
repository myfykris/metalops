# Crash Investigation Findings

## 1. Python 3.9 (`.venv39`) - Import Error (Not a Crash)
- **Status**: Failed immediately.
- **Error**: `AttributeError: module 'metalcore_backend' has no attribute 'lu_batched'`
- **Cause**: The compiled C++ extension in `.venv39` is outdated and missing the `lu_batched` export, while the Python `__init__.py` wrapper expects it.
- **Fix**: Needs a rebuild/reinstall in that environment.

## 2. Python 3.10 (`.venv310`) - The "Hard Crash"
- **Status**: Interrupted during execution.
- **Last Action**: Running SVD benchmark on **Llama-7B Attention matrix (4096x4096)**.
- **Symptoms**: The log ends abruptly during the CPU comparison phase (`torch.linalg.svd(A_cpu)`).
- **Diagnosis**: This operation is extremely heavy. 
    - **Memory**: A 4096² float32 matrix is 64MB, but SVD workspace can be 10-100x that. Running this locally might trigger an OOM (Out Of Memory) killer if the machine is under load, potentially taking down the IDE's extension host with it.
    - **GPU/Metal**: If the Metal kernel messed up GPU state previously, the subsequent CPU or GPU ops might hang the window server.
    - **Timeout**: It might have simply hung, and the "crash" was the result of the system terminating the unresponsive process group.

## Conclusion
The "crash" is likely **Resource Exhaustion** (Memory/CPU hang) on the Large Matrix SVD tests in `.venv310`. The `.venv39` issue is a separate build artifact issue.
