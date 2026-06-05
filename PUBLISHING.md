# Publishing to PyPI

Releases are automated by GitHub Actions (`.github/workflows/release.yml`):
pushing a version tag builds portable `manylinux` wheels (with a minimal static
OpenCV linked in — see `ci/build-opencv.sh`) plus an sdist, and uploads them to
PyPI.

## One-time setup: PyPI token as a repo secret

The release workflow authenticates with a PyPI API token stored as the GitHub
Actions secret **`PYPI_API_TOKEN`**.

- GitHub → repo → Settings → Secrets and variables → Actions → New repository secret
- Name: `PYPI_API_TOKEN`
- Value: a token from https://pypi.org/manage/account/token/ (starts with `pypi-`)

(Or, instead of `gh`'s web UI: `gh secret set PYPI_API_TOKEN`.)

## Cut a release

```bash
# bump version in pyproject.toml first if needed, commit, then:
git tag v0.1.0
git push origin main
git push origin v0.1.0     # this triggers the release workflow -> PyPI
```

Each release needs a **new version** — PyPI rejects re-uploading an existing one.
Pushing to `main` only runs the fast `ci` checks; it does **not** publish.

## Manual / local publish (fallback)

`.pypirc` at the repo root (git-ignored) holds your token for manual uploads:

```bash
.venv/bin/python -m pip install build twine
.venv/bin/python -m build
.venv/bin/python -m twine upload --config-file .pypirc -r pypi dist/*
```

Note: a locally built wheel links your **system OpenCV** and is not portable —
use it only for a TestPyPI smoke test. The CI wheels are the portable ones.
