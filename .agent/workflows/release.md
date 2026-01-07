---
description: How to release a new version of metalcore to PyPI
---

## Release Workflow for metalcore

### 1. Update Version Numbers
Bump version in these 3 files (do NOT use global search/replace):
- `packages/metalcore/setup.py` (line ~43)
- `packages/metalcore/pyproject.toml` (line ~7)
- `packages/metalcore/src/metalcore/__init__.py` (line ~87)

### 2. Build Wheels
// turbo
Run the build script (handles License-File metadata issue automatically):
```bash
cd /Users/kris/localprojects/metalops && bash build_all_wheels.sh
```

This script:
- Builds wheels for all Python venvs (.venv, .venv39, .venv310, etc.)
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
