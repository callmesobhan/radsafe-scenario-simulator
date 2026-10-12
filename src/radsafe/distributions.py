"""Input-uncertainty distributions.

Each distribution is an immutable dataclass with

* ``sample(n, rng)`` - draw ``n`` values using a ``numpy.random.Generator``;
* ``nominal`` - the point estimate used for deterministic (nominal) runs.

Scenario files describe distributions as plain numbers (constants) or
mappings such as ``{dist: lognormal, median: 3700, gsd: 1.05}``; see
:func:`from_spec`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


class DistributionError(ValueError):
    """Invalid distribution parameters or specification."""


class Distribution(ABC):
    """Base class: something that can be sampled and has a nominal value."""

    @property
    @abstractmethod
    def nominal(self) -> float: ...

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray: ...


@dataclass(frozen=True)
class Constant(Distribution):
    value: float

    @property
    def nominal(self) -> float:
        return float(self.value)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.full(n, float(self.value))


@dataclass(frozen=True)
class Uniform(Distribution):
    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.low <= self.high:
            raise DistributionError(f"uniform: low ({self.low}) must be <= high ({self.high})")

    @property
    def nominal(self) -> float:
        return 0.5 * (self.low + self.high)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, n)


@dataclass(frozen=True)
class Normal(Distribution):
    """Normal distribution, optionally truncated to ``[lower, upper]``.

    Truncation is done by rejection sampling, so it is exact but becomes slow
    if the bounds exclude most of the probability mass (an error is raised
    rather than looping forever).
    """

    mean: float
    sd: float
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if self.sd < 0:
            raise DistributionError(f"normal: sd must be >= 0, got {self.sd}")
        lo = -math.inf if self.lower is None else self.lower
        hi = math.inf if self.upper is None else self.upper
        if lo >= hi:
            raise DistributionError(f"normal: lower ({lo}) must be < upper ({hi})")

    @property
    def nominal(self) -> float:
        lo = -math.inf if self.lower is None else self.lower
        hi = math.inf if self.upper is None else self.upper
        return float(min(max(self.mean, lo), hi))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.sd == 0:
            return np.full(n, self.nominal)
        lo = -math.inf if self.lower is None else self.lower
        hi = math.inf if self.upper is None else self.upper
        out = rng.normal(self.mean, self.sd, n)
        for _ in range(100):
            bad = (out < lo) | (out > hi)
            k = int(bad.sum())
            if k == 0:
                return out
            out[bad] = rng.normal(self.mean, self.sd, k)
        raise DistributionError(
            f"normal(mean={self.mean}, sd={self.sd}): truncation bounds "
            f"[{lo}, {hi}] exclude almost all of the distribution"
        )


@dataclass(frozen=True)
class LogNormal(Distribution):
    """Log-normal parameterised by its median and geometric standard deviation.

    ``ln X ~ Normal(ln median, ln gsd)``. About 68% of samples fall in
    ``[median / gsd, median * gsd]``.
    """

    median: float
    gsd: float

    def __post_init__(self) -> None:
        if self.median <= 0:
            raise DistributionError(f"lognormal: median must be > 0, got {self.median}")
        if self.gsd < 1:
            raise DistributionError(f"lognormal: gsd must be >= 1, got {self.gsd}")

    @classmethod
    def from_rel_sd(cls, median: float, rel_sd: float) -> Distribution:
        """Log-normal whose coefficient of variation is ``rel_sd``."""
        if rel_sd < 0:
            raise DistributionError(f"rel_sd must be >= 0, got {rel_sd}")
        if rel_sd == 0:
            return Constant(median)
        sigma = math.sqrt(math.log1p(rel_sd**2))
        return cls(median=median, gsd=math.exp(sigma))

    @property
    def nominal(self) -> float:
        return float(self.median)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.median * np.exp(rng.normal(0.0, math.log(self.gsd), n))


@dataclass(frozen=True)
class Triangular(Distribution):
    low: float
    mode: float
    high: float

    def __post_init__(self) -> None:
        if not self.low <= self.mode <= self.high:
            raise DistributionError(
                f"triangular: need low <= mode <= high, got {self.low}, {self.mode}, {self.high}"
            )
        if self.low == self.high:
            raise DistributionError("triangular: low and high must differ (use a constant)")

    @property
    def nominal(self) -> float:
        return float(self.mode)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.triangular(self.low, self.mode, self.high, n)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_ALLOWED_KEYS: dict[str, set[str]] = {
    "constant": {"value"},
    "uniform": {"low", "high"},
    "normal": {"mean", "sd", "rel_sd", "lower", "upper"},
    "lognormal": {"median", "gsd", "rel_sd"},
    "triangular": {"low", "mode", "high"},
}


def _as_number(value: Any) -> Any:
    """Accept numeric strings. PyYAML follows YAML 1.1, which reads ``3.7e6``
    (no sign on the exponent) as a *string*, a trap in scientific input files."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _number(spec: Mapping[str, Any], key: str, where: str) -> float:
    if key not in spec:
        raise DistributionError(f"{where}: missing required key '{key}'")
    value = _as_number(spec[key])
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DistributionError(f"{where}: '{key}' must be a number, got {value!r}")
    if not math.isfinite(value):
        raise DistributionError(f"{where}: '{key}' must be finite, got {value!r}")
    return float(value)


def from_spec(spec: Any, *, where: str = "value", nonnegative: bool = False) -> Distribution:
    """Build a :class:`Distribution` from a number or a ``{dist: ...}`` mapping.

    ``nonnegative`` gives unbounded normals a lower bound of 0 (appropriate
    for distances, durations, thicknesses, activities).
    """
    if isinstance(spec, Distribution):
        return spec
    spec = _as_number(spec)
    if isinstance(spec, bool):
        raise DistributionError(f"{where}: expected a number or mapping, got {spec!r}")
    if isinstance(spec, int | float):
        if not math.isfinite(spec):
            raise DistributionError(f"{where}: must be finite, got {spec!r}")
        return Constant(float(spec))
    if not isinstance(spec, Mapping):
        raise DistributionError(f"{where}: expected a number or mapping, got {spec!r}")

    kind = spec.get("dist")
    if kind not in _ALLOWED_KEYS:
        raise DistributionError(f"{where}: 'dist' must be one of {sorted(_ALLOWED_KEYS)}, got {kind!r}")
    unknown = set(spec) - _ALLOWED_KEYS[kind] - {"dist"}
    if unknown:
        raise DistributionError(
            f"{where}: unknown key(s) {sorted(unknown)} for dist '{kind}' "
            f"(allowed: {sorted(_ALLOWED_KEYS[kind])})"
        )

    if kind == "constant":
        return Constant(_number(spec, "value", where))
    if kind == "uniform":
        return Uniform(_number(spec, "low", where), _number(spec, "high", where))
    if kind == "triangular":
        return Triangular(
            _number(spec, "low", where),
            _number(spec, "mode", where),
            _number(spec, "high", where),
        )
    if kind == "normal":
        mean = _number(spec, "mean", where)
        if ("sd" in spec) == ("rel_sd" in spec):
            raise DistributionError(f"{where}: give exactly one of 'sd' or 'rel_sd'")
        sd = _number(spec, "sd", where) if "sd" in spec else abs(mean) * _number(spec, "rel_sd", where)
        lower = _number(spec, "lower", where) if "lower" in spec else (0.0 if nonnegative else None)
        upper = _number(spec, "upper", where) if "upper" in spec else None
        return Normal(mean, sd, lower, upper)
    # lognormal
    median = _number(spec, "median", where)
    if ("gsd" in spec) == ("rel_sd" in spec):
        raise DistributionError(f"{where}: give exactly one of 'gsd' or 'rel_sd'")
    if "gsd" in spec:
        return LogNormal(median, _number(spec, "gsd", where))
    return LogNormal.from_rel_sd(median, _number(spec, "rel_sd", where))
