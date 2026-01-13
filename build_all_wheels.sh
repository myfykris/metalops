#!/bin/bash
# Build wheels for all python versions with validation
# Exits immediately on any error

set -e  # Exit on first error

echo "=== Building Wheels for MetalOps ==="

# Clean previous builds
rm -rf dist build packages/metalcore/build packages/metalcore/dist packages/metalcore/metalcore.egg-info

# Use every directory in envs/ as a venv (includes hidden .venv* dirs)
environments=($(ls -a envs/ | grep -E '^\.' | grep -v '^\.\.$' | grep -v '^\.$'))

echo "Found environments: ${environments[*]}"

# First pass: uninstall metalcore from all venvs
echo "=== Uninstalling metalcore from all venvs ==="
for env in "${environments[@]}"; do
    if [ -f "./envs/$env/bin/python" ]; then
        py_ver=$(./envs/$env/bin/python --version)
        echo "Uninstalling metalcore from $env ($py_ver)..."
        ./envs/$env/bin/pip uninstall metalcore -y 2>/dev/null || true
    fi
done

# Second pass: build wheels and run benchmarks
echo "=== Building wheels and running benchmarks ==="
for env in "${environments[@]}"; do
    if [ -f "./envs/$env/bin/python" ]; then
        py_ver=$(./envs/$env/bin/python --version)
        echo "================================================"
        echo "Building for $env ($py_ver)..."
        echo "================================================"
        
        abs_python=$(realpath ./envs/$env/bin/python)
        
        # Install dependencies
        echo "Installing dependencies..."
        $abs_python -m pip install wheel setuptools torch numpy --break-system-packages 2>/dev/null || $abs_python -m pip install wheel setuptools torch numpy
        
        # Build wheel
        echo "Building wheel..."
        (cd packages/metalcore && $abs_python setup.py bdist_wheel)
        
        echo "✅ Built wheel for $env"
        
        # Install in editable mode for testing
        echo "Installing metalcore for testing..."
        $abs_python -m pip install -e packages/metalcore --break-system-packages 2>/dev/null || $abs_python -m pip install -e packages/metalcore
        
        # Run lite benchmark (no long ops)
        echo "Running lite benchmark for $env..."
        $abs_python benchmark.py --lite --nolongops
        
        echo "✅ Benchmark passed for $env"
        
    else
        echo "⚠️  Skipping $env (Python not found)"
    fi
done

echo "================================================"
echo "All Builds and Benchmarks Complete!"
echo "================================================"

# Fix License-File metadata issue for PyPI compatibility
echo "Fixing wheel metadata..."
for w in packages/metalcore/dist/*.whl; do
    echo "Fixing: $w"
    tmpdir=$(mktemp -d)
    unzip -q "$w" -d "$tmpdir"
    # Remove ONLY the License-File line from METADATA
    metafile=$(ls "$tmpdir"/metalcore-*.dist-info/METADATA)
    grep -v "^License-File:" "$metafile" | grep -v "^Dynamic: license-file" > "$metafile.tmp"
    mv "$metafile.tmp" "$metafile"
    # Repack wheel
    rm "$w"
    (cd "$tmpdir" && zip -rq "../fixed.whl" .)
    mv "$(dirname $tmpdir)/fixed.whl" "$w"
    rm -rf "$tmpdir"
done

echo ""
echo "=== Final Summary ==="
echo "Wheels in packages/metalcore/dist/:"
ls -lh packages/metalcore/dist/
echo ""
echo "Verifying .metallib in wheels..."
for w in packages/metalcore/dist/*.whl; do 
    if unzip -l "$w" | grep -q .metallib; then
        echo "  ✅ $w"
    else
        echo "  ❌ Missing .metallib: $w"
        exit 1
    fi
done

echo ""
echo "🎉 All builds complete and validated!"
