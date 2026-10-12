"""Radiation sources: nuclide data used for the unshielded dose rate."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import yaml


class DataError(ValueError):
    """Reference data is missing, malformed, or out of range."""


@dataclass(frozen=True)
class Source:
    """A point gamma source.

    Attributes
    ----------
    nuclide
        Label, e.g. ``"Cs-137"``.
    gamma_constant
        Ambient dose equivalent rate constant, uSv*m^2/(MBq*h).
    gamma_rel_unc
        1-sigma relative uncertainty on ``gamma_constant`` (log-normal).
    effective_energy_mev
        Single photon energy used for attenuation-coefficient lookup.
    """

    nuclide: str
    gamma_constant: float
    gamma_rel_unc: float
    effective_energy_mev: float
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.gamma_constant > 0:
            raise DataError(f"{self.nuclide}: gamma_constant must be > 0")
        if self.gamma_rel_unc < 0:
            raise DataError(f"{self.nuclide}: gamma_rel_unc must be >= 0")
        if not self.effective_energy_mev > 0:
            raise DataError(f"{self.nuclide}: effective_energy_mev must be > 0")


def _key(name: str) -> str:
    """'Cs-137', 'cs137', 'CS 137' -> 'cs137'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


@lru_cache(maxsize=1)
def _load() -> dict[str, Source]:
    text = (resources.files("radsafe") / "data" / "sources.yaml").read_text(encoding="utf-8")
    raw = yaml.safe_load(text)["sources"]
    out: dict[str, Source] = {}
    for label, row in raw.items():
        out[_key(label)] = Source(
            nuclide=label,
            gamma_constant=float(row["gamma_constant_usv_m2_per_mbq_h"]),
            gamma_rel_unc=float(row["gamma_rel_unc"]),
            effective_energy_mev=float(row["effective_energy_mev"]),
            notes=" ".join(str(row.get("notes", "")).split()),
        )
    return out


def list_sources() -> list[Source]:
    return sorted(_load().values(), key=lambda s: s.nuclide)


def get_source(name: str) -> Source:
    try:
        return _load()[_key(name)]
    except KeyError:
        known = ", ".join(s.nuclide for s in list_sources())
        raise DataError(f"unknown nuclide {name!r}; built-in sources: {known}") from None
