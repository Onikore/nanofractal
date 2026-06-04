# Development notes

## Environment

Use the project virtualenv at `.venv` for all builds and tests:

```bash
.venv/bin/python -m pip install -e ".[test]"
```

### IMPORTANT: strip PYTHONPATH when running Python/pytest

This machine sources a ROS (Jazzy) + Livox workspace in the shell, which sets a
`PYTHONPATH` pointing at `/opt/ros/jazzy/...` and a `ws_livox` workspace. Those
directories contain pytest plugins (ament/launch_testing_ros) that get
autoloaded and crash during collection (e.g. `ModuleNotFoundError: No module
named 'yaml'`). The fix is to unset `PYTHONPATH` for our commands:

```bash
env -u PYTHONPATH .venv/bin/python -m pytest tests/ -v
env -u PYTHONPATH .venv/bin/python -m pip install -e ".[test]"
```

Always prefix Python/pytest invocations with `env -u PYTHONPATH`.

## Build

The extension is built by scikit-build-core + CMake (nanobind). A normal
editable install recompiles after C++ changes:

```bash
env -u PYTHONPATH .venv/bin/python -m pip install -e ".[test]"
```
