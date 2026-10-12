"""Dose summaries, limit exceedance, and nominal stochastic-risk conversion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

#: ICRP Publication 103 (2007) recommended dose limits, mSv per year.
#: Occupational: 20 mSv/y averaged over 5 years, no more than 50 mSv in any one year.
ICRP103_PUBLIC_LIMIT_MSV = 1.0
ICRP103_WORKER_LIMIT_MSV = 20.0

#: ICRP 103 detriment-adjusted nominal risk coefficients for stochastic
#: effects (cancer + heritable), per Sv. Population-level averages for low
#: doses and low dose rates - not a prediction for any individual.
NOMINAL_DETRIMENT_PER_SV = {"population": 5.7e-2, "adult_worker": 4.2e-2}


@dataclass(frozen=True)
class DoseSummary:
    """Summary statistics of a dose distribution, in mSv."""

    mean: float
    median: float
    p5: float
    p95: float
    p99: float


@dataclass(frozen=True)
class Exceedance:
    probability: float
    ci_low: float
    ci_high: float
    n_exceed: int
    n: int


def summarize(dose_usv: ArrayLike) -> DoseSummary:
    msv = np.asarray(dose_usv, dtype=float) / 1000.0
    p5, p50, p95, p99 = np.percentile(msv, [5, 50, 95, 99])
    return DoseSummary(float(msv.mean()), float(p50), float(p5), float(p95), float(p99))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (well-behaved at k=0 and k=n)."""
    if n <= 0:
        raise ValueError("n must be > 0")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def exceedance(dose_usv: ArrayLike, limit_msv: float) -> Exceedance:
    """P(dose > limit) estimated from samples, with a 95% Wilson interval for
    the Monte Carlo sampling error (not for input or model uncertainty)."""
    doses = np.asarray(dose_usv, dtype=float)
    k = int(np.count_nonzero(doses > limit_msv * 1000.0))
    lo, hi = wilson_interval(k, doses.size)
    return Exceedance(k / doses.size, lo, hi, k, doses.size)


def nominal_detriment(dose_msv: float, population: str = "adult_worker") -> float:
    """Nominal stochastic detriment for a dose, using ICRP 103 linear coefficients."""
    return dose_msv / 1000.0 * NOMINAL_DETRIMENT_PER_SV[population]
