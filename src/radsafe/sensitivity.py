"""Which uncertain inputs drive the dose uncertainty?

Uses Spearman rank correlation between each input and the dose. It is
cheap (no extra model runs), robust to the heavy-tailed, multiplicative
shape of dose distributions, and invariant to monotone transforms, so it
needs no log-transform. The ``share`` column normalises rho^2 across inputs
so the values read as "fraction of explained rank variance"; treat it as a
ranking aid, not an exact variance decomposition (that needs Sobol indices -
see docs/roadmap.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from radsafe.montecarlo import SimulationResult


@dataclass(frozen=True)
class Sensitivity:
    input: str
    rho: float
    share: float


def _ranks(x: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(x), dtype=float)
    ranks[np.argsort(x, kind="stable")] = np.arange(len(x), dtype=float)
    return ranks


def spearman_sensitivity(result: SimulationResult, series: str) -> list[Sensitivity]:
    """Rank inputs by influence on ``result.dose_usv[series]`` (largest first).

    Inputs with no variation (constants) are omitted.
    """
    dose = result.dose_usv[series]
    if np.ptp(dose) == 0:
        return []
    y = _ranks(dose)
    rhos = {}
    for name, x in result.inputs.items():
        if np.ptp(x) == 0:
            continue
        rhos[name] = float(np.corrcoef(_ranks(x), y)[0, 1])
    total = sum(r * r for r in rhos.values())
    out = [Sensitivity(k, r, (r * r / total) if total else 0.0) for k, r in rhos.items()]
    return sorted(out, key=lambda s: s.share, reverse=True)
