# Publishing to PyPI

Releases are automated by GitHub Actions (`.github/workflows/release.yml`):
pushing a version tag builds the wheels — Linux x86_64 / aarch64 (manylinux,
static minimal OpenCV from `ci/build-opencv.sh`) and Windows AMD64 (prebuilt
OpenCV 4.10, bundled with `delvewheel`) — plus an sdist, runs the test-suite
against every wheel, and uploads them to PyPI.

## Authentication: trusted publishing (OIDC)

No token is stored. The `nanofractal` project on PyPI has a trusted publisher
configured (Publishing → GitHub): owner `Onikore`, repository `nanofractal`,
workflow `release.yml`, environment `pypi`. It must be attached to the
**existing project** — a *pending* publisher fails with
`invalid-pending-publisher: valid token, but project already exists`.

## Cut a release

```bash
# bump version in pyproject.toml and add a CHANGELOG entry, commit, then:
git push origin main
git tag v0.6.0
git push origin v0.6.0     # triggers the release workflow -> PyPI
```

Each release needs a **new version** — PyPI rejects re-uploading an existing one.
Pushing to `main` only runs the CI checks; it does **not** publish.
`workflow_dispatch` on any branch builds and tests all wheels without
publishing (the publish job only runs for `v*` tags).

## Manual / local publish (fallback)

```bash
python -m pip install build twine
python -m build
python -m twine upload dist/*
```

A locally built wheel links your **system OpenCV** and is not portable — use
it only for a TestPyPI smoke test. The CI wheels are the portable ones.
