# Publishing to PyPI

## 1. Put your token in `.pypirc`

`.pypirc` already exists at the repo root and is git-ignored (your token is never
committed). Open it and replace `pypi-PASTE_YOUR_..._TOKEN_HERE` with the token
from https://pypi.org/manage/account/token/ (and test.pypi.org for TestPyPI).

## 2. Build the distribution

```bash
env -u PYTHONPATH .venv/bin/python -m pip install build twine
env -u PYTHONPATH .venv/bin/python -m build
```

This produces `dist/*.whl` and `dist/*.tar.gz`.

## 3. Upload

Test on TestPyPI first, then the real index:

```bash
# TestPyPI
env -u PYTHONPATH .venv/bin/python -m twine upload --config-file .pypirc -r testpypi dist/*

# PyPI
env -u PYTHONPATH .venv/bin/python -m twine upload --config-file .pypirc -r pypi dist/*
```

## ⚠️ Important: the local wheel is NOT portable yet

The wheel built here links dynamically against this machine's **system OpenCV** and
is tagged for this exact Python/platform. It will only install where a compatible
OpenCV is present. That is fine for a TestPyPI smoke test, but **not** for public
distribution.

Portable, self-contained `manylinux` wheels (minimal OpenCV statically linked,
built across CPython versions via `cibuildwheel`) are the subject of **Plan B**
(`docs/superpowers/plans/`), to be implemented after the core library is complete.
Publish real releases only after Plan B.
