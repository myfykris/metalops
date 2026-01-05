#!/bin/bash
# Build wheels for all python versions

echo "=== Building Wheels for MetalOps ==="

# Clean previous builds
rm -rf dist build packages/metalcore/build packages/metalcore/dist packages/metalcore/metalcore.egg-info

environments=(.venv .venv39 .venv310 .venv311 .venv312 .venv313)

for env in "${environments[@]}"; do
    if [ -f "./$env/bin/python" ]; then
        py_ver=$(./$env/bin/python --version)
        echo "------------------------------------------------"
        echo "Building for $env ($py_ver)..."
        
        # Build binary wheel
        abs_python=$(realpath ./$env/bin/python)
        
        # Try installing with --break-system-packages for newer python versions that enforce PEP 668
        $abs_python -m pip install wheel setuptools torch numpy --break-system-packages || $abs_python -m pip install wheel setuptools torch numpy
        (cd packages/metalcore && $abs_python setup.py bdist_wheel)
        
        if [ $? -eq 0 ]; then
             echo "✅ Built wheel for $env"
        else
             echo "❌ Failed to build for $env"
        fi
    else
        echo "⚠️  Skipping $env (Python not found)"
    fi
done

echo "------------------------------------------------"
echo "All Builds Complete. Wheels in dist/:"
ls -lh dist/
echo "Verifying .metallib in wheels..."
for w in dist/*.whl; do unzip -l "$w" | grep .metallib && echo "  - OK: $w"; done
