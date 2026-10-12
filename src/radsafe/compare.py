"""Deterministic comparison of shielding models across shield thickness."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from radsafe.materials import get_material
from radsafe.shielding import LayerSample, get_model
from radsafe.sources import get_source


def transmission_curve(
    nuclide: str,
    material: str,
    thickness_cm: ArrayLike,
    models: Sequence[str] = ("narrow_beam", "linear_buildup"),
    hvl_cm: float | None = None,
) -> dict[str, np.ndarray]:
    """Transmission of a single-layer shield versus thickness, per model.

    Uses nominal attenuation coefficients (no uncertainty). ``hvl_cm`` is
    required if ``practical_hvl`` is requested.
    """
    mu = get_material(material).linear_attenuation(get_source(nuclide).effective_energy_mev)
    x = np.asarray(thickness_cm, dtype=float)
    layer = [LayerSample(material, x, mu, hvl_cm)]
    return {name: get_model(name).transmission(layer) for name in models}
