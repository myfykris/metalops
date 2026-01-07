---
description: How to release a new version of metalcore to PyPI
---

## Release Workflow for metalcore

### 1. Update Version Number (SINGLE LOCATION)
// turbo
Bump version in **only one file**:
```
packages/metalcore/src/metalcore/__init__.py  # Line 87: __version__ = "X.Y.Z"
```
setup.py and pyproject.toml read from this automatically.

### 2. Build Wheels
// turbo
Run the build script (handles License-File metadata issue automatically):
```bash
cd /Users/kris/localprojects/metalops && bash build_all_wheels.sh
```

This script:
- Builds wheels for all Python venvs (.venv, .venv310, etc.)
- Fixes the License-File metadata issue that causes PyPI rejection
- Outputs wheels to `packages/metalcore/dist/`

### 3. Push to GitHub
```bash
git add -A && git commit -m "vX.Y.Z: description" && git push origin main
```

### 4. Upload to PyPI
User must run manually (requires API token interactively):
```bash
cd /Users/kris/localprojects/metalops/packages/metalcore && twine upload dist/*.whl
```

**Note:** Twine requires interactive API token entry. Cannot be automated by agent.
