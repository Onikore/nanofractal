"""The speed regression gate of benchmarks/fractal_bench.py (the CI bench job)."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fractal_bench", Path(__file__).parent.parent / "benchmarks" / "fractal_bench.py"
)
fb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fb)


def _runs(tmp_path, name, rounds):
    paths = []
    for i, cells in enumerate(rounds):
        p = tmp_path / f"{name}{i}.json"
        rows = [{"key": k, "median_ms": v} for k, v in cells.items()]
        rows.append({"key": "batch_1080p_fps", "fps": 100.0, "median_ms": 10.0})
        p.write_text(json.dumps({"opencv": "x", "rows": rows}))
        paths.append(str(p))
    return paths


def _gate(base, new, max_mean=1.10, max_cell=1.25):
    fb.cmd_gate(SimpleNamespace(base=base, new=new, max_mean=max_mean, max_cell=max_cell))


def test_gate_takes_the_fastest_round(tmp_path, capsys):
    base = _runs(tmp_path, "b", [{"a": 1.0, "b": 2.0}, {"a": 1.3, "b": 2.6}])
    # the new build drifted slower in its first round only
    new = _runs(tmp_path, "n", [{"a": 1.4, "b": 2.8}, {"a": 1.02, "b": 2.04}])
    _gate(base, new)
    assert "x1.020" in capsys.readouterr().out


def test_gate_fails_on_a_mean_slowdown(tmp_path):
    base = _runs(tmp_path, "b", [{"a": 1.0, "b": 2.0, "c": 3.0}])
    new = _runs(tmp_path, "n", [{"a": 1.15, "b": 2.3, "c": 3.45}])
    with pytest.raises(SystemExit) as e:
        _gate(base, new)
    assert e.value.code == 1


def test_gate_fails_on_one_slow_cell(tmp_path):
    base = _runs(tmp_path, "b", [{"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}])
    new = _runs(tmp_path, "n", [{"a": 1.3, "b": 2.0, "c": 3.0, "d": 4.0}])
    with pytest.raises(SystemExit) as e:
        _gate(base, new)
    assert e.value.code == 1


def test_gate_ignores_cells_of_one_build_only(tmp_path):
    base = _runs(tmp_path, "b", [{"a": 1.0, "old": 9.0}])
    new = _runs(tmp_path, "n", [{"a": 0.9, "new": 0.1}])
    _gate(base, new)
