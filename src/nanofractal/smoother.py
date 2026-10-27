"""Pure-Python 6-DOF pose smoother for nanofractal.

Provides :class:`PoseSmoother`, which smooths a pose stream (``rvec`` +
``tvec``). The orientation is kept as a unit quaternion and filtered in its
tangent space (the innovation is the small rotation vector between the
measurement and the estimate), so rotations near pi -- a camera looking down
at a marker -- do not jump when the Rodrigues vector flips sign. The
translation is filtered per axis.

Without timestamps the model is the original constant-pose random walk, one
step per call. With timestamps (``update(..., t=seconds)``) the Kalman mode
uses a constant-velocity model for both rotation and translation and real
time steps, so steady motion is followed without lag.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _quat_from_rotvec(r: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation vector."""
    a = float(np.linalg.norm(r))
    if a < 1e-12:
        return np.array([1.0, 0.5 * r[0], 0.5 * r[1], 0.5 * r[2]]) / math.sqrt(1.0 + 0.25 * a * a)
    s = math.sin(0.5 * a) / a
    return np.array([math.cos(0.5 * a), s * r[0], s * r[1], s * r[2]])


def _rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    """Rotation vector (angle in [0, pi]) of a unit quaternion."""
    if q[0] < 0:
        q = -q
    v = q[1:]
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return 2.0 * v
    return (2.0 * math.atan2(n, q[0]) / n) * v


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _boxminus(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Small rotation vector d with qa = exp(d) * qb (shortest way)."""
    return _rotvec_from_quat(_quat_mul(qa, _quat_conj(qb)))


def _boxplus(d: np.ndarray, q: np.ndarray) -> np.ndarray:
    out = _quat_mul(_quat_from_rotvec(d), q)
    return out / np.linalg.norm(out)


def _continuous_rotvec(q: np.ndarray, prev: Optional[np.ndarray]) -> np.ndarray:
    """Rotation vector of *q* on the branch closest to *prev*.

    ``r`` and ``r - 2*pi*r/|r|`` describe the same rotation; near pi the
    canonical vector flips direction, so pick the one nearer the previous
    output to keep the output stream continuous.
    """
    r = _rotvec_from_quat(q)
    if prev is None:
        return r
    a = float(np.linalg.norm(r))
    if a < 1e-12:
        return r
    alt = r * (1.0 - 2.0 * math.pi / a)
    return alt if np.linalg.norm(alt - prev) < np.linalg.norm(r - prev) else r


class PoseSmoother:
    """Smooth a 6-DOF pose (rvec + tvec) stream.

    Parameters
    ----------
    process_noise : float
        Process noise Q. Without timestamps: variance added per call
        (random-walk model). With timestamps: white-acceleration spectral
        density of the constant-velocity model, per second. Larger values
        follow faster motion with less smoothing.
    measurement_noise : float
        Measurement noise variance R, shared by the rotation (rad) and the
        translation (your units). With ``reproj_err`` / ``n_points`` passed to
        :meth:`update` it is the variance of a 4-corner pose at 1 px RMS.
    mode : {"kalman", "ema"}
        ``"kalman"``: Kalman filter with time-varying gain. ``"ema"``: fixed
        gain equal to the steady-state Kalman gain of the random-walk model,
        ``alpha = (-r + sqrt(r**2 + 4*r)) / 2`` with ``r = Q / R``; it ignores
        timestamps.
    gate : float or None
        Reject a measurement whose Mahalanobis distance from the prediction
        (over the 6 pose components) exceeds this value; the prediction is
        returned instead. ``None`` (default) accepts everything. 4.7 is the
        99.9 % point of the chi distribution with 6 degrees of freedom.
    max_rejects : int
        After this many consecutive rejections the filter restarts from the
        measurement (the marker really moved, e.g. after a dropout).
    """

    _VALID_MODES = frozenset({"kalman", "ema"})

    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2,
        mode: str = "kalman",
        gate: Optional[float] = None,
        max_rejects: int = 3,
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self._VALID_MODES)!r}, got {mode!r}")
        if gate is not None and not gate > 0:
            raise ValueError(f"gate must be positive or None, got {gate!r}")
        self._mode = mode
        self._Q = float(process_noise)
        self._R = float(measurement_noise)
        self._gate = None if gate is None else float(gate)
        self._max_rejects = int(max_rejects)

        r = self._Q / self._R
        alpha = (-r + math.sqrt(r * r + 4.0 * r)) / 2.0
        self._alpha = max(1e-15, min(1.0, alpha))  # clamp to (0, 1]

        self._q: Optional[np.ndarray] = None  # orientation, unit quaternion
        self._x: np.ndarray = np.zeros(3)  # position
        self._w: np.ndarray = np.zeros(3)  # angular velocity (rad/s, world frame)
        self._v: np.ndarray = np.zeros(3)  # velocity
        # Error covariance of one axis: [[pose, pose-rate], [rate-pose, rate]]
        # (identical for all six axes since they share Q, R and the gating).
        self._P: np.ndarray = np.zeros((2, 2))
        self._t: Optional[float] = None
        self._prev_rvec: Optional[np.ndarray] = None
        self._rejects = 0
        self._last_rejected = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def last_rejected(self) -> bool:
        """True when the last :meth:`update` measurement was gated out."""
        return self._last_rejected

    def update(
        self,
        rvec: "array-like",  # type: ignore[valid-type]  # noqa: F821
        tvec: "array-like",  # type: ignore[valid-type]  # noqa: F821
        t: Optional[float] = None,
        reproj_err: Optional[float] = None,
        n_points: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Feed one pose measurement and return the smoothed estimate.

        Parameters
        ----------
        rvec, tvec : array-like, shape (3,)
            Rodrigues rotation vector and translation vector.
        t : float, optional
            Frame time in seconds. Either always or never pass it between
            resets; with it the Kalman mode uses the real time step and a
            constant-velocity model.
        reproj_err, n_points : optional
            ``estimate_pose``'s RMS error (px) and the number of
            correspondences; they scale the measurement noise as
            ``R * reproj_err**2 * 4 / n_points``.

        Returns
        -------
        rvec_s, tvec_s : np.ndarray, dtype=float64, shape (3,)
            Smoothed rotation and translation vectors. The rotation vector is
            kept continuous with the previous output (no flips near pi).
        """
        rv = np.asarray(rvec, dtype=np.float64)
        tv = np.asarray(tvec, dtype=np.float64)
        if rv.shape != (3,):
            raise ValueError(f"rvec must have shape (3,), got {rv.shape}")
        if tv.shape != (3,):
            raise ValueError(f"tvec must have shape (3,), got {tv.shape}")

        R = self._R
        if reproj_err is not None:
            R *= max(float(reproj_err), 0.05) ** 2
        if n_points is not None:
            R *= 4.0 / max(int(n_points), 4)

        q = _quat_from_rotvec(rv)
        self._last_rejected = False
        if self._q is None:
            return self._init(rv, tv, q, t)

        if (t is None) != (self._t is None):
            raise ValueError("pass t on every update or on none (call reset() to switch)")
        if t is not None and self._t is not None:
            dt = float(t) - self._t
            if dt <= 0:
                raise ValueError(f"t must increase, got dt={dt}")
            if self._mode == "kalman":
                self._predict_cv(dt)
            self._t = float(t)
        elif self._mode == "kalman":
            self._P[0, 0] += self._Q

        d_rot = _boxminus(q, self._q)
        d_pos = tv - self._x
        if self._gate is not None and self._mode == "kalman":
            S = self._P[0, 0] + R
            d2 = (float(d_rot @ d_rot) + float(d_pos @ d_pos)) / S
            if d2 > self._gate**2:
                self._rejects += 1
                if self._rejects > self._max_rejects:
                    return self._init(rv, tv, q, t)
                self._last_rejected = True
                return self._output()
        self._rejects = 0

        if self._mode == "ema":
            k0, k1 = self._alpha, 0.0
        else:
            P = self._P
            S = P[0, 0] + R
            k0, k1 = P[0, 0] / S, P[1, 0] / S
            self._P = P - np.outer([k0, k1], P[0])
        self._q = _boxplus(k0 * d_rot, self._q)
        self._x = self._x + k0 * d_pos
        self._w = self._w + k1 * d_rot
        self._v = self._v + k1 * d_pos
        return self._output()

    def reset(self) -> None:
        """Clear all filter state.

        The next call to :meth:`update` re-initialises from its measurement
        (identical to the very first call after construction).
        """
        self._q = None
        self._x = np.zeros(3)
        self._w = np.zeros(3)
        self._v = np.zeros(3)
        self._P = np.zeros((2, 2))
        self._t = None
        self._prev_rvec = None
        self._rejects = 0
        self._last_rejected = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init(self, rv, tv, q, t) -> tuple[np.ndarray, np.ndarray]:
        self.reset()
        self._q = q
        self._x = tv.copy()
        self._t = None if t is None else float(t)
        if self._t is not None and self._mode == "kalman":
            # unknown velocity: large prior so it is learnt from the next steps
            self._P = np.array([[self._R, 0.0], [0.0, 1e3 * self._R]])
        else:
            self._P[0, 0] = self._Q
        self._prev_rvec = rv.copy()
        return rv.copy(), tv.copy()

    def _predict_cv(self, dt: float) -> None:
        self._q = _boxplus(self._w * dt, self._q)  # type: ignore[arg-type]
        self._x = self._x + self._v * dt
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Qd = self._Q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        self._P = F @ self._P @ F.T + Qd

    def _output(self) -> tuple[np.ndarray, np.ndarray]:
        r = _continuous_rotvec(self._q, self._prev_rvec)  # type: ignore[arg-type]
        self._prev_rvec = r
        return r.copy(), self._x.copy()
