"""Tests for PoseSmoother — written BEFORE implementation (TDD)."""

from __future__ import annotations

import numpy as np
import pytest

from nanofractal import PoseSmoother

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rvec(vals=(0.1, 0.2, 0.3)):
    return np.array(vals, dtype=np.float64)


def _tvec(vals=(1.0, 2.0, 3.0)):
    return np.array(vals, dtype=np.float64)


# ---------------------------------------------------------------------------
# Basic output shape / dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["kalman", "ema"])
def test_update_returns_float64_shape3(mode):
    s = PoseSmoother(mode=mode)
    rv_s, tv_s = s.update(_rvec(), _tvec())
    assert rv_s.shape == (3,)
    assert tv_s.shape == (3,)
    assert rv_s.dtype == np.float64
    assert tv_s.dtype == np.float64


# ---------------------------------------------------------------------------
# First update returns input unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["kalman", "ema"])
def test_first_update_returns_input_unchanged(mode):
    s = PoseSmoother(mode=mode)
    rv = _rvec((0.5, -0.3, 1.1))
    tv = _tvec((10.0, -5.0, 2.7))
    rv_s, tv_s = s.update(rv, tv)
    np.testing.assert_array_equal(rv_s, rv)
    np.testing.assert_array_equal(tv_s, tv)


# ---------------------------------------------------------------------------
# Convergence: constant pose → smoothed output converges to it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["kalman", "ema"])
def test_constant_pose_convergence(mode):
    s = PoseSmoother(process_noise=1e-4, measurement_noise=1e-2, mode=mode)
    rv = _rvec((0.1, 0.2, 0.3))
    tv = _tvec((1.0, 2.0, 3.0))
    for _ in range(50):
        rv_s, tv_s = s.update(rv, tv)
    np.testing.assert_allclose(rv_s, rv, atol=1e-6)
    np.testing.assert_allclose(tv_s, tv, atol=1e-6)


# ---------------------------------------------------------------------------
# Noise reduction: smoothed std must be clearly less than raw std
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["kalman", "ema"])
def test_noise_reduction(mode):
    rng = np.random.default_rng(42)
    n_steps = 500
    burn_in = 50
    true_rvec = np.array([0.1, 0.2, 0.3])
    true_tvec = np.array([1.0, 2.0, 3.0])
    meas_noise_std = 0.05

    s = PoseSmoother(process_noise=1e-4, measurement_noise=meas_noise_std**2, mode=mode)

    smoothed_rv = []
    smoothed_tv = []
    raw_rv = []
    raw_tv = []

    for i in range(n_steps):
        rv_noisy = true_rvec + rng.normal(0, meas_noise_std, 3)
        tv_noisy = true_tvec + rng.normal(0, meas_noise_std, 3)
        raw_rv.append(rv_noisy.copy())
        raw_tv.append(tv_noisy.copy())
        rv_s, tv_s = s.update(rv_noisy, tv_noisy)
        if i >= burn_in:
            smoothed_rv.append(rv_s.copy())
            smoothed_tv.append(tv_s.copy())

    raw_rv = np.array(raw_rv[burn_in:])
    raw_tv = np.array(raw_tv[burn_in:])
    smoothed_rv = np.array(smoothed_rv)
    smoothed_tv = np.array(smoothed_tv)

    # Use per-component std (averaged) so that the different mean values of
    # rvec components (0.1, 0.2, 0.3) don't inflate the across-all-elements
    # std and mask the actual noise reduction.
    raw_std = np.mean(np.std(raw_rv, axis=0))
    smoothed_std = np.mean(np.std(smoothed_rv, axis=0))
    # Smoothed std must be at least 3× smaller
    assert smoothed_std < raw_std / 3.0, (
        f"[{mode}] rvec: smoothed_std={smoothed_std:.4f} not < raw_std/3={raw_std / 3:.4f}"
    )

    raw_std_t = np.mean(np.std(raw_tv, axis=0))
    smoothed_std_t = np.mean(np.std(smoothed_tv, axis=0))
    assert smoothed_std_t < raw_std_t / 3.0, (
        f"[{mode}] tvec: smoothed_std={smoothed_std_t:.4f} not < raw_std/3={raw_std_t / 3:.4f}"
    )


# ---------------------------------------------------------------------------
# reset() → next update re-initialises from new measurement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["kalman", "ema"])
def test_reset_reinitialises(mode):
    s = PoseSmoother(mode=mode)
    # Warm up the filter
    for _ in range(20):
        s.update(_rvec((0.0, 0.0, 0.0)), _tvec((0.0, 0.0, 0.0)))

    # Reset and feed a very different measurement
    s.reset()
    rv_new = _rvec((9.0, 8.0, 7.0))
    tv_new = _tvec((100.0, 200.0, 300.0))
    rv_s, tv_s = s.update(rv_new, tv_new)

    # Should return the new measurement exactly (first sample after reset)
    np.testing.assert_array_equal(rv_s, rv_new)
    np.testing.assert_array_equal(tv_s, tv_new)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        PoseSmoother(mode="bogus")


@pytest.mark.parametrize("mode", ["kalman", "ema"])
def test_wrong_shape_rvec_raises(mode):
    s = PoseSmoother(mode=mode)
    with pytest.raises(ValueError):
        s.update(np.array([1.0, 2.0]), _tvec())  # shape (2,) — wrong


@pytest.mark.parametrize("mode", ["kalman", "ema"])
def test_wrong_shape_tvec_raises(mode):
    s = PoseSmoother(mode=mode)
    with pytest.raises(ValueError):
        s.update(_rvec(), np.array([[1.0, 2.0, 3.0]]))  # shape (1,3) — wrong


# ---------------------------------------------------------------------------
# Importable from package root
# ---------------------------------------------------------------------------


def test_importable_from_package():
    import nanofractal

    assert hasattr(nanofractal, "PoseSmoother")
    assert "PoseSmoother" in nanofractal.__all__


# ---------------------------------------------------------------------------
# R13: rotation near pi, timestamps (constant velocity), gating, noise scaling
# ---------------------------------------------------------------------------


def _rot_dist(a, b):
    """Angle (rad) between two rotations given as rotation vectors."""
    from nanofractal.smoother import _boxminus, _quat_from_rotvec

    return float(np.linalg.norm(_boxminus(_quat_from_rotvec(a), _quat_from_rotvec(b))))


def _canonical(axis, angle):
    """Rodrigues vector as OpenCV returns it: angle in [0, pi]."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    angle = float(angle) % (2 * np.pi)
    if angle > np.pi:
        angle, axis = 2 * np.pi - angle, -axis
    return axis * angle


@pytest.mark.parametrize("mode", ["kalman", "ema"])
def test_rotation_near_pi_does_not_jump(mode):
    # camera looking down at the marker: rotation ~pi about x; noise makes the
    # canonical Rodrigues vector flip between ~(pi,0,0) and ~(-pi,0,0)
    rng = np.random.default_rng(1)
    s = PoseSmoother(process_noise=1e-4, measurement_noise=1e-2, mode=mode)
    raw, out = [], []
    for _ in range(300):
        axis = np.array([1.0, 0, 0]) + rng.normal(0, 0.01, 3)
        rv = _canonical(axis, np.pi + rng.normal(0, 0.01))
        raw.append(rv)
        out.append(s.update(rv, _tvec())[0])
    raw, out = np.array(raw), np.array(out)
    assert np.abs(np.diff(raw[:, 0])).max() > 6  # the raw stream does flip
    assert np.abs(np.diff(out, axis=0)).max() < 0.05
    assert abs(np.linalg.norm(out[-1]) - np.pi) < 0.05


def test_constant_velocity_has_no_lag():
    s = PoseSmoother(process_noise=1e-3, measurement_noise=1e-4)
    w = 0.5  # rad/s about z
    v = np.array([1.0, -0.5, 0.2])  # m/s
    for i in range(300):
        t = i / 30
        rv_s, tv_s = s.update(_canonical([0, 0, 1], 0.3 + w * t), _tvec() + v * t, t=t)
    np.testing.assert_allclose(tv_s, _tvec() + v * t, atol=1e-3)
    assert _rot_dist(rv_s, _canonical([0, 0, 1], 0.3 + w * t)) < 1e-3


def test_random_walk_lags_without_timestamps():
    # the old model (no t) trails a moving target -- documents why t matters
    s = PoseSmoother(process_noise=1e-3, measurement_noise=1e-4)
    for i in range(300):
        _, tv_s = s.update(_rvec(), _tvec() + np.array([1.0, 0, 0]) * i / 30)
    assert abs(tv_s[0] - (1.0 + 299 / 30)) > 1e-3


def test_irregular_timestamps():
    s = PoseSmoother(process_noise=1e-3, measurement_noise=1e-4)
    rng = np.random.default_rng(3)
    t = 0.0
    for _ in range(300):
        t += rng.uniform(0.01, 0.1)  # dropped frames
        _, tv_s = s.update(_rvec(), _tvec() * (1 + t), t=t)
    np.testing.assert_allclose(tv_s, _tvec() * (1 + t), atol=1e-3)


def test_gate_rejects_single_outlier():
    rng = np.random.default_rng(2)
    s = PoseSmoother(process_noise=1e-4, measurement_noise=1e-4, gate=4.7)
    for _ in range(50):
        s.update(_rvec() + rng.normal(0, 0.01, 3), _tvec() + rng.normal(0, 0.01, 3))
    _, tv_s = s.update(_rvec(), _tvec() + np.array([0, 0, 1.0]))  # 1 m spike
    assert s.last_rejected
    np.testing.assert_allclose(tv_s, _tvec(), atol=0.02)
    s.update(_rvec(), _tvec())
    assert not s.last_rejected


def test_gate_restarts_after_real_jump():
    s = PoseSmoother(process_noise=1e-4, measurement_noise=1e-4, gate=4.7, max_rejects=3)
    for _ in range(50):
        s.update(_rvec(), _tvec())
    far = _tvec() + 5.0
    outs = [s.update(_rvec(), far)[1] for _ in range(4)]
    np.testing.assert_allclose(outs[2], _tvec())  # still rejected
    np.testing.assert_allclose(outs[3], far)  # 4th: restart from the measurement


def test_reproj_err_scales_measurement_noise():
    def step(**kw):
        s = PoseSmoother(process_noise=1e-4, measurement_noise=1e-2)
        s.update(_rvec(), _tvec())
        return s.update(_rvec(), _tvec() + 1.0, **kw)[1][0] - 1.0

    assert step(reproj_err=0.2) > step() > step(reproj_err=3.0)
    assert step(n_points=100) > step(n_points=4)


def test_timestamps_must_be_consistent():
    s = PoseSmoother()
    s.update(_rvec(), _tvec(), t=0.0)
    with pytest.raises(ValueError):
        s.update(_rvec(), _tvec())
    with pytest.raises(ValueError):
        s.update(_rvec(), _tvec(), t=0.0)
