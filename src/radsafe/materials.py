"""Shielding materials and photon attenuation coefficients (NIST XCOM)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import yaml

from radsafe.sources import DataError


@dataclass(frozen=True)
class Material:
    name: str
    density_g_cm3: float
    energies_mev: tuple[float, ...]
    mu_over_rho_cm2_g: tuple[float, ...]

    def mass_attenuation(self, energy_mev: float) -> float:
        """Total mass attenuation coefficient mu/rho [cm^2/g] (log-log interpolation)."""
        lo, hi = self.energies_mev[0], self.energies_mev[-1]
        if not lo <= energy_mev <= hi:
            raise DataError(
                f"{self.name}: no attenuation data at {energy_mev} MeV "
                f"(table covers {lo}-{hi} MeV; extend data/materials.yaml)"
            )
        log_mu = np.interp(np.log(energy_mev), np.log(self.energies_mev), np.log(self.mu_over_rho_cm2_g))
        return float(np.exp(log_mu))

    def linear_attenuation(self, energy_mev: float) -> float:
        """Linear attenuation coefficient mu [1/cm] at the nominal density."""
        return self.mass_attenuation(energy_mev) * self.density_g_cm3


@lru_cache(maxsize=1)
def _load() -> dict[str, Material]:
    text = (resources.files("radsafe") / "data" / "materials.yaml").read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    energies = tuple(float(e) for e in raw["energies_mev"])
    out: dict[str, Material] = {}
    for name, row in raw["materials"].items():
        mu = tuple(float(v) for v in row["mu_over_rho_cm2_g"])
        if len(mu) != len(energies):
            raise DataError(f"{name}: {len(mu)} mu/rho values for {len(energies)} energies")
        out[name.lower()] = Material(name.lower(), float(row["density_g_cm3"]), energies, mu)
    return out


def list_materials() -> list[Material]:
    return sorted(_load().values(), key=lambda m: m.name)


def get_material(name: str) -> Material:
    try:
        return _load()[name.lower()]
    except KeyError:
        known = ", ".join(m.name for m in list_materials())
        raise DataError(f"unknown material {name!r}; built-in materials: {known}") from None
