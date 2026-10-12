"""Simplified shielding models.

Each model maps a stack of shield layers to a dimensionless transmission
``T = dose behind shield / dose without shield`` (0 < T <= 1). They are
deliberately simple and *disagree* with each other - that disagreement is the
model-form uncertainty the simulator is designed to expose.

========================  ==================================================
``narrow_beam``           ``T = exp(-sum mu_i x_i)``. Counts only unscattered
                          photons. Optimistic (a lower bound on T) for any
                          real, extended geometry.
``linear_buildup``        ``T = (1 + sum mu_i x_i) * exp(-sum mu_i x_i)``.
                          Crude broad-beam correction. Typically overstates
                          scatter buildup in high-Z shields (lead) and can
                          understate it in low-Z shields (water, concrete)
                          at many mean free paths.
``practical_hvl``         ``T = 2 ** -(sum x_i / HVL_i)`` with user-supplied
                          *practical* (broad-beam) half-value layers, e.g.
                          from NCRP 151 or vendor data.
========================  ==================================================

Multi-layer stacks are treated by multiplying per-layer transmissions
(equivalently, summing optical depths). That ignores layer-ordering effects
such as scatter and spectrum hardening - see docs/physics.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike


class ShieldingError(ValueError):
    """A shielding model cannot be evaluated with the data supplied."""


@dataclass(frozen=True)
class LayerSample:
    """One shield layer with (possibly sampled) parameters.

    ``thickness_cm`` and ``mu_per_cm`` are scalars or equal-length arrays.
    ``hvl_cm`` is only needed by :class:`PracticalHVL`.
    """

    material: str
    thickness_cm: ArrayLike
    mu_per_cm: ArrayLike
    hvl_cm: ArrayLike | None = None

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.thickness_cm) < 0):
            raise ShieldingError(f"{self.material}: thickness must be >= 0")
        if np.any(np.asarray(self.mu_per_cm) < 0):
            raise ShieldingError(f"{self.material}: mu must be >= 0")
        if self.hvl_cm is not None and np.any(np.asarray(self.hvl_cm) <= 0):
            raise ShieldingError(f"{self.material}: hvl_cm must be > 0")


def optical_depth(layers: Sequence[LayerSample]) -> np.ndarray:
    """Total number of mean free paths, ``sum mu_i * x_i``."""
    total: np.ndarray | float = 0.0
    for layer in layers:
        total = total + np.asarray(layer.mu_per_cm, dtype=float) * np.asarray(layer.thickness_cm, dtype=float)
    return np.asarray(total, dtype=float)


class ShieldingModel(ABC):
    name: ClassVar[str]
    summary: ClassVar[str]

    @abstractmethod
    def transmission(self, layers: Sequence[LayerSample]) -> np.ndarray: ...


class NarrowBeam(ShieldingModel):
    name = "narrow_beam"
    summary = "exp(-mu x): unscattered photons only (optimistic)"

    def transmission(self, layers: Sequence[LayerSample]) -> np.ndarray:
        return np.exp(-optical_depth(layers))


class LinearBuildup(ShieldingModel):
    name = "linear_buildup"
    summary = "(1 + mu x) exp(-mu x): crude broad-beam correction"

    def transmission(self, layers: Sequence[LayerSample]) -> np.ndarray:
        mfp = optical_depth(layers)
        return (1.0 + mfp) * np.exp(-mfp)


class PracticalHVL(ShieldingModel):
    name = "practical_hvl"
    summary = "2^-(x/HVL) with user-supplied broad-beam half-value layers"

    def transmission(self, layers: Sequence[LayerSample]) -> np.ndarray:
        halvings: np.ndarray | float = 0.0
        for layer in layers:
            if layer.hvl_cm is None:
                raise ShieldingError(
                    f"practical_hvl needs 'hvl_cm' for every layer (missing for {layer.material})"
                )
            halvings = halvings + np.asarray(layer.thickness_cm, dtype=float) / np.asarray(
                layer.hvl_cm, dtype=float
            )
        return np.exp2(-np.asarray(halvings, dtype=float))


MODELS: dict[str, ShieldingModel] = {m.name: m for m in (NarrowBeam(), LinearBuildup(), PracticalHVL())}


def get_model(name: str) -> ShieldingModel:
    try:
        return MODELS[name]
    except KeyError:
        raise ShieldingError(f"unknown shielding model {name!r}; choose from {sorted(MODELS)}") from None
