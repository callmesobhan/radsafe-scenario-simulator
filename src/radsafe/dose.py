"""Deterministic point-source dose-rate and dose arithmetic.

All functions are vectorised: scalars or arrays in, ``numpy`` arrays out.
Internal dose unit is the microsievert (uSv); dose rate is uSv/h.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

#: Nominal distances below this trigger a scenario warning: real sources have
#: finite size, so 1/d^2 is unreliable at close range.
MIN_RECOMMENDED_DISTANCE_M = 0.1


def unshielded_dose_rate(
    activity_mbq: ArrayLike, distance_m: ArrayLike, gamma_constant: ArrayLike
) -> np.ndarray:
    """Dose rate [uSv/h] from an unshielded point source (inverse-square law).

    ``rate = gamma_constant * activity / distance**2`` with ``gamma_constant``
    in uSv*m^2/(MBq*h).
    """
    activity = np.asarray(activity_mbq, dtype=float)
    distance = np.asarray(distance_m, dtype=float)
    gamma = np.asarray(gamma_constant, dtype=float)
    if np.any(activity < 0):
        raise ValueError("activity must be >= 0")
    if np.any(distance <= 0):
        raise ValueError("distance must be > 0")
    if np.any(gamma <= 0):
        raise ValueError("gamma constant must be > 0")
    return gamma * activity / distance**2


def dose_from_rate(rate_usv_h: ArrayLike, duration_h: ArrayLike) -> np.ndarray:
    """Dose [uSv] accumulated at a constant rate over ``duration_h`` hours."""
    duration = np.asarray(duration_h, dtype=float)
    if np.any(duration < 0):
        raise ValueError("duration must be >= 0")
    return np.asarray(rate_usv_h, dtype=float) * duration
