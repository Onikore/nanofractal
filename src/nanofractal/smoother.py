"""Pure-Python 6-DOF pose smoother for nanofractal.

Provides :class:`PoseSmoother`, which smooths a 6-DOF pose stream
(``rvec`` + ``tvec``, each a 3-component vector) by treating every
component as an independent scalar signal and applying either a
constant-state (random-walk) Kalman filter or an EMA with the same
steady-state gain.

.. note::
    ``rvec`` (Rodrigues rotation vector) components are smoothed
    **component-wise** in Euclidean space.  This is adequate for small
    frame-to-frame rotations (the drone/marker tracking use-case) but
    is **not** a geometrically rigorous rotation average.  For large
    inter-frame rotations a quaternion / SLERP filter would be
    preferable.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class PoseSmoother:
    """Smooth a 6-DOF pose (rvec + tvec) stream per-DOF.

    Parameters
    ----------
    process_noise : float
        Process noise variance Q for the random-walk model.  Larger
        values allow the filter to track faster motion at the cost of
        less smoothing.
    measurement_noise : float
        Measurement noise variance R.  Larger values apply more
        smoothing.
    mode : {"kalman", "ema"}
        Filter type.  ``"kalman"`` runs a per-DOF scalar Kalman filter
        with time-varying gain.  ``"ema"`` uses a fixed exponential
        moving average with a gain equal to the steady-state Kalman
        gain for the random-walk model:

        .. code-block:: python

            r = process_noise / measurement_noise
            alpha = (-r + sqrt(r**2 + 4*r)) / 2   # ∈ (0, 1]

    Notes
    -----
    ``rvec`` components are smoothed in Euclidean space
    (component-wise), which is valid only for small inter-frame
    rotation changes.  For large rotations a quaternion/SLERP filter
    is preferable.
    """

    _VALID_MODES = frozenset({"kalman", "ema"})

    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2,
        mode: str = "kalman",
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self._VALID_MODES)!r}, got {mode!r}")
        self._mode = mode
        self._Q = float(process_noise)
        self._R = float(measurement_noise)

        # Steady-state EMA gain (used only for mode="ema")
        r = self._Q / self._R
        alpha = (-r + math.sqrt(r * r + 4.0 * r)) / 2.0
        self._alpha = max(1e-15, min(1.0, alpha))  # clamp to (0, 1]

        # Filter state — None until first measurement.
        # For mode="kalman":  _x (6,), _P (6,) per-DOF error covariance
        # For mode="ema":     _x (6,)  per-DOF estimate
        self._x: Optional[np.ndarray] = None  # shape (6,) float64
        self._P: Optional[np.ndarray] = None  # shape (6,) float64 [kalman only]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        rvec: "array-like",  # type: ignore[valid-type]  # noqa: F821
        tvec: "array-like",  # type: ignore[valid-type]  # noqa: F821
    ) -> tuple[np.ndarray, np.ndarray]:
        """Feed one pose measurement and return the smoothed estimate.

        Parameters
        ----------
        rvec, tvec : array-like, shape (3,)
            Rodrigues rotation vector and translation vector.

        Returns
        -------
        rvec_s, tvec_s : np.ndarray, dtype=float64, shape (3,)
            Smoothed rotation and translation vectors.

        Raises
        ------
        ValueError
            If ``rvec`` or ``tvec`` does not have shape ``(3,)``.
        """
        rv = np.asarray(rvec, dtype=np.float64)
        tv = np.asarray(tvec, dtype=np.float64)
        if rv.shape != (3,):
            raise ValueError(f"rvec must have shape (3,), got {rv.shape}")
        if tv.shape != (3,):
            raise ValueError(f"tvec must have shape (3,), got {tv.shape}")

        z = np.concatenate([rv, tv])  # measurement, shape (6,)

        if self._x is None:
            # First call: initialise state to measurement, return unchanged.
            self._x = z.copy()
            if self._mode == "kalman":
                self._P = np.full(6, self._Q, dtype=np.float64)
            smoothed = z.copy()
        elif self._mode == "kalman":
            smoothed = self._kalman_step(z)
        else:
            smoothed = self._ema_step(z)

        return smoothed[:3].copy(), smoothed[3:].copy()

    def reset(self) -> None:
        """Clear all filter state.

        The next call to :meth:`update` will re-initialise from its
        measurement (identical to the very first call after construction).
        """
        self._x = None
        self._P = None

    # ------------------------------------------------------------------
    # Internal filter steps
    # ------------------------------------------------------------------

    def _kalman_step(self, z: np.ndarray) -> np.ndarray:
        """One Kalman predict+update step; returns smoothed 6-vec."""
        assert self._x is not None and self._P is not None

        # Predict: state unchanged, covariance grows by process noise
        self._P += self._Q

        # Update
        K = self._P / (self._P + self._R)  # Kalman gain, shape (6,)
        self._x = self._x + K * (z - self._x)
        self._P = self._P * (1.0 - K)

        return self._x.copy()

    def _ema_step(self, z: np.ndarray) -> np.ndarray:
        """One EMA update step; returns smoothed 6-vec."""
        assert self._x is not None

        self._x = self._x + self._alpha * (z - self._x)
        return self._x.copy()
