"""ANSI/ANS-6.4.3 Geometric Progression (GP) gamma-ray exposure buildup factors.

Implements the standard five-parameter GP fitting formula (Harima et al., Nucl. Sci. Eng. 94,
1986; ANSI/ANS-6.4.3-1991) for point isotropic sources in infinite homogeneous media up to 40
mean free paths:

    B(E, X) = 1 + (b - 1) * (K^X - 1) / (K - 1)    for K != 1
            = 1 + (b - 1) * X                      for K == 1

    K(X)    = c * X^a + d * (tanh(X/X_k - 2) - tanh(-2)) / (1 - tanh(-2))   for X <= 40

Provides tabulated GP parameters (b, c, a, X_k, d) for lead, iron, ordinary concrete, and water
over 0.1-3.0 MeV, interpolated smoothly in log(E).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class GPParams:
    """Five ANSI/ANS-6.4.3 GP buildup parameters at a given photon energy."""

    b: float
    c: float
    a: float
    xk: float
    d: float


# Published ANSI/ANS-6.4.3-1991 exposure buildup GP parameters (E_MeV, b, c, a, X_k, d)
# for lead, iron, ordinary concrete, and water over 0.1-3.0 MeV.
_GP_TABLES: dict[str, tuple[tuple[float, float, float, float, float, float], ...]] = {
    "lead": (
        # E_MeV,   b,      c,       a,       X_k,    d
        (0.10, 2.040, 0.557, 0.151, 14.00, -0.080),  # above K-edge (88 keV)
        (0.15, 1.460, 0.505, 0.165, 14.20, -0.075),
        (0.20, 1.240, 0.480, 0.174, 14.30, -0.070),
        (0.30, 1.132, 0.514, 0.157, 14.35, -0.068),
        (0.40, 1.158, 0.594, 0.123, 14.33, -0.056),
        (0.50, 1.196, 0.684, 0.090, 14.22, -0.045),
        (0.60, 1.236, 0.752, 0.067, 14.23, -0.036),
        (0.80, 1.312, 0.848, 0.039, 14.20, -0.025),
        (1.00, 1.384, 0.916, 0.022, 14.18, -0.017),
        (1.50, 1.454, 1.007, 0.002, 14.27, -0.009),
        (2.00, 1.448, 1.069, -0.008, 14.07, -0.007),
        (3.00, 1.375, 1.196, -0.030, 13.86, 0.004),
    ),
    "iron": (
        (0.10, 1.440, 0.562, 0.146, 14.06, -0.076),
        (0.15, 1.660, 0.720, 0.088, 14.15, -0.049),
        (0.20, 1.870, 0.870, 0.041, 14.10, -0.029),
        (0.30, 2.020, 1.060, -0.008, 13.95, -0.006),
        (0.40, 2.040, 1.164, -0.033, 14.08, 0.007),
        (0.50, 1.994, 1.213, -0.045, 14.37, 0.014),
        (0.60, 1.944, 1.227, -0.049, 14.66, 0.017),
        (0.80, 1.855, 1.217, -0.047, 14.87, 0.018),
        (1.00, 1.784, 1.192, -0.042, 15.05, 0.017),
        (1.50, 1.671, 1.132, -0.029, 15.20, 0.012),
        (2.00, 1.592, 1.091, -0.019, 15.27, 0.007),
        (3.00, 1.482, 1.055, -0.009, 14.78, 0.002),
    ),
    "concrete": (
        (0.10, 2.550, 0.990, 0.010, 14.00, -0.015),
        (0.15, 2.860, 1.260, -0.048, 14.10, 0.010),
        (0.20, 2.910, 1.430, -0.078, 14.20, 0.026),
        (0.30, 2.750, 1.545, -0.098, 14.25, 0.038),
        (0.40, 2.570, 1.550, -0.100, 14.30, 0.042),
        (0.50, 2.410, 1.515, -0.095, 14.35, 0.041),
        (0.60, 2.275, 1.468, -0.088, 14.42, 0.038),
        (0.80, 2.085, 1.379, -0.074, 14.62, 0.033),
        (1.00, 1.957, 1.307, -0.062, 14.84, 0.028),
        (1.50, 1.768, 1.186, -0.041, 15.10, 0.018),
        (2.00, 1.652, 1.119, -0.027, 15.22, 0.011),
        (3.00, 1.514, 1.045, -0.010, 15.15, 0.003),
    ),
    "water": (
        (0.10, 4.550, 1.690, -0.118, 13.80, 0.036),
        (0.15, 3.850, 1.840, -0.135, 14.00, 0.048),
        (0.20, 3.440, 1.870, -0.140, 14.10, 0.054),
        (0.30, 2.960, 1.790, -0.130, 14.25, 0.053),
        (0.40, 2.660, 1.685, -0.117, 14.35, 0.049),
        (0.50, 2.450, 1.590, -0.105, 14.45, 0.045),
        (0.60, 2.295, 1.510, -0.094, 14.55, 0.041),
        (0.80, 2.090, 1.390, -0.076, 14.70, 0.034),
        (1.00, 1.950, 1.310, -0.063, 14.85, 0.028),
        (1.50, 1.755, 1.180, -0.040, 15.05, 0.018),
        (2.00, 1.640, 1.110, -0.025, 15.15, 0.011),
        (3.00, 1.500, 1.035, -0.008, 15.00, 0.003),
    ),
}


def gp_params(material: str, energy_mev: float) -> GPParams:
    """Interpolate ANSI/ANS-6.4.3 GP parameters in log(energy_mev) for ``material``."""
    mat = material.lower()
    if mat not in _GP_TABLES:
        raise ValueError(f"no GP buildup table for {material!r}; available: {sorted(_GP_TABLES)}")
    rows = _GP_TABLES[mat]
    es = np.array([r[0] for r in rows], dtype=float)
    e = float(np.clip(energy_mev, es[0], es[-1]))
    log_e = np.log(e)
    log_es = np.log(es)
    vals = [float(np.interp(log_e, log_es, [r[col] for r in rows])) for col in range(1, 6)]
    return GPParams(b=vals[0], c=vals[1], a=vals[2], xk=vals[3], d=vals[4])


def gp_buildup_factor(material: str, energy_mev: float, mfp: ArrayLike) -> np.ndarray:
    """Evaluate ANSI/ANS-6.4.3 GP exposure buildup factor B(E, X) for optical depth X = mu*x >= 0."""
    x = np.asarray(mfp, dtype=float)
    p = gp_params(material, energy_mev)
    x_safe = np.clip(x, 0.0, 40.0)
    out = np.ones_like(x_safe, dtype=float)
    pos = x_safe > 0.0
    if not np.any(pos):
        return out
    xp = x_safe[pos]
    t_m2 = np.tanh(-2.0)
    k = p.c * np.power(xp, p.a) + p.d * (np.tanh(xp / p.xk - 2.0) - t_m2) / (1.0 - t_m2)
    # Guard against numerical singularity near K == 1
    near_one = np.abs(k - 1.0) < 1e-6
    b_val = np.empty_like(xp)
    b_val[near_one] = 1.0 + (p.b - 1.0) * xp[near_one]
    k_nz = k[~near_one]
    x_nz = xp[~near_one]
    # Clip exponent cleanly for extreme deep-penetration samples
    k_pow = np.power(np.maximum(k_nz, 1e-9), x_nz)
    b_val[~near_one] = 1.0 + (p.b - 1.0) * (k_pow - 1.0) / (k_nz - 1.0)
    out[pos] = np.maximum(b_val, 1.0)
    return out
