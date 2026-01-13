---
description: How to release a new version of metalcore to PyPI
---

## Release Workflow for metalcore

### 1. Update Version Number (SINGLE LOCATION)
// turbo
Bump version in **only one file**:
```
packages/metalcore/src/metalcore/__init__.py  # Line 105: __version__ = "X.Y.Z"
```
setup.py and pyproject.toml read from this automatically.

### 2. Test in All Python Venvs
Test builds work in all supported Python versions.

**Venv Locations** (in `/Users/kris/localprojects/metalops/envs/`):
- `.venv39` - Python 3.9
- `.venv310` - Python 3.10  
- `.venv311` - Python 3.11
- `.venv312` - Python 3.12
- `.venv313` - Python 3.13
- `.venv314` - Python 3.14

// turbo
Quick test command:
```bash
cd /Users/kris/localprojects/metalops
for ver in 310 311 312 313; do
  envs/.venv$ver/bin/pip install -e packages/metalcore -q
  envs/.venv$ver/bin/python -c "import metalcore; print(f'Python 3.{str(ver)[1:]}: {metalcore.__version__}')"
done
```

### 3. Build Wheels
// turbo
Run the build script (handles License-File metadata issue automatically):
```bash
cd /Users/kris/localprojects/metalops && bash build_all_wheels.sh
```

This script:
- Builds wheels for all Python venvs in `envs/`
- Fixes the License-File metadata issue that causes PyPI rejection
- Outputs wheels to `packages/metalcore/dist/`

### 4. Push to GitHub
```bash
git add -A && git commit -m "vX.Y.Z: description" && git push origin main
```

### 5. Upload to PyPI
User must run manually (requires API token interactively):
```bash
cd /Users/kris/localprojects/metalops/packages/metalcore && twine upload dist/*.whl
```

**Note:** Twine requires interactive API token entry. Cannot be automated by agent.

