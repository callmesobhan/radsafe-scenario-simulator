"""Scenario definition and YAML loading.

A scenario is a frozen, validated description of one controlled-source
exposure: the source, the geometry, the exposure time, the shield stack, the
shielding models to compare, the dose limits to test against, and the Monte
Carlo settings. See docs/scenario-schema.md for the file format.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from radsafe.distributions import Distribution, DistributionError, from_spec
from radsafe.dose import MIN_RECOMMENDED_DISTANCE_M
from radsafe.materials import get_material
from radsafe.shielding import MODELS
from radsafe.sources import DataError, Source, get_source

DEFAULT_MU_REL_UNC = 0.05
DEFAULT_MODELS = ("narrow_beam", "linear_buildup")
DEFAULT_N_SAMPLES = 100_000
DEFAULT_SEED = 12345


class ScenarioError(ValueError):
    """The scenario file is malformed or physically inconsistent."""


@dataclass(frozen=True)
class LayerSpec:
    material: str
    thickness_cm: Distribution
    #: 1-sigma relative uncertainty on mu (log-normal), independent per layer.
    mu_rel_unc: float = DEFAULT_MU_REL_UNC
    #: Practical half-value layer; required only by the ``practical_hvl`` model.
    hvl_cm: Distribution | None = None


@dataclass(frozen=True)
class Limit:
    name: str
    value_msv: float


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    source: Source
    activity_mbq: Distribution
    distance_m: Distribution
    duration_h: Distribution
    layers: tuple[LayerSpec, ...] = ()
    models: tuple[str, ...] = ()
    limits: tuple[Limit, ...] = ()
    n_samples: int = DEFAULT_N_SAMPLES
    seed: int | None = DEFAULT_SEED
    warnings: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScenarioError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _check_keys(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ScenarioError(f"{where}: unknown key(s) {sorted(unknown)} (allowed: {sorted(allowed)})")


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ScenarioError(f"{where}: missing required key '{key}'")
    return mapping[key]


def _dist(spec: Any, where: str) -> Distribution:
    try:
        return from_spec(spec, where=where, nonnegative=True)
    except DistributionError as exc:
        raise ScenarioError(str(exc)) from None


def _positive_support(dist: Distribution, where: str, *, strict: bool) -> None:
    """Reject distributions whose nominal value is non-physical."""
    nominal = dist.nominal
    if (nominal <= 0) if strict else (nominal < 0):
        raise ScenarioError(f"{where}: nominal value must be {'> 0' if strict else '>= 0'}")


def _build_source(raw: Mapping[str, Any]) -> tuple[Source, Distribution]:
    allowed = {
        "nuclide",
        "activity_mbq",
        "gamma_constant_usv_m2_per_mbq_h",
        "gamma_rel_unc",
        "effective_energy_mev",
    }
    _check_keys(raw, allowed, "source")
    nuclide = str(_require(raw, "nuclide", "source"))
    activity = _dist(_require(raw, "activity_mbq", "source"), "source.activity_mbq")
    _positive_support(activity, "source.activity_mbq", strict=True)

    overrides = {
        "gamma_constant": raw.get("gamma_constant_usv_m2_per_mbq_h"),
        "gamma_rel_unc": raw.get("gamma_rel_unc"),
        "effective_energy_mev": raw.get("effective_energy_mev"),
    }
    overrides = {k: float(v) for k, v in overrides.items() if v is not None}
    try:
        try:
            source = get_source(nuclide)
        except DataError:
            if len(overrides) != 3:
                raise ScenarioError(
                    f"source.nuclide {nuclide!r} is not built in; for a custom source give all of "
                    "gamma_constant_usv_m2_per_mbq_h, gamma_rel_unc and effective_energy_mev"
                ) from None
            source = Source(nuclide, **overrides)  # type: ignore[arg-type]
        else:
            if overrides:
                source = dataclasses.replace(source, **overrides)
    except DataError as exc:
        raise ScenarioError(f"source: {exc}") from None
    return source, activity


def _build_layer(raw: Any, index: int, energy_mev: float) -> LayerSpec:
    where = f"shielding.layers[{index}]"
    raw = _mapping(raw, where)
    _check_keys(raw, {"material", "thickness_cm", "mu_rel_unc", "hvl_cm"}, where)
    material = str(_require(raw, "material", where))
    try:
        mat = get_material(material)
        mat.linear_attenuation(energy_mev)  # fail early if the energy is out of range
    except DataError as exc:
        raise ScenarioError(f"{where}: {exc}") from None
    thickness = _dist(_require(raw, "thickness_cm", where), f"{where}.thickness_cm")
    _positive_support(thickness, f"{where}.thickness_cm", strict=False)
    mu_rel_unc = float(raw.get("mu_rel_unc", DEFAULT_MU_REL_UNC))
    if mu_rel_unc < 0:
        raise ScenarioError(f"{where}.mu_rel_unc: must be >= 0")
    hvl = None
    if raw.get("hvl_cm") is not None:
        hvl = _dist(raw["hvl_cm"], f"{where}.hvl_cm")
        _positive_support(hvl, f"{where}.hvl_cm", strict=True)
    return LayerSpec(mat.name, thickness, mu_rel_unc, hvl)


def scenario_from_dict(data: Mapping[str, Any]) -> Scenario:
    data = _mapping(data, "scenario")
    _check_keys(
        data,
        {"name", "description", "source", "geometry", "exposure", "shielding", "limits", "simulation"},
        "scenario",
    )
    name = str(_require(data, "name", "scenario"))
    source, activity = _build_source(_mapping(_require(data, "source", "scenario"), "source"))

    geometry = _mapping(_require(data, "geometry", "scenario"), "geometry")
    _check_keys(geometry, {"distance_m"}, "geometry")
    distance = _dist(_require(geometry, "distance_m", "geometry"), "geometry.distance_m")
    _positive_support(distance, "geometry.distance_m", strict=True)

    exposure = _mapping(_require(data, "exposure", "scenario"), "exposure")
    _check_keys(exposure, {"duration_h"}, "exposure")
    duration = _dist(_require(exposure, "duration_h", "exposure"), "exposure.duration_h")
    _positive_support(duration, "exposure.duration_h", strict=False)

    shielding = _mapping(data.get("shielding") or {}, "shielding")
    _check_keys(shielding, {"models", "layers"}, "shielding")
    raw_layers = shielding.get("layers") or []
    if not isinstance(raw_layers, list):
        raise ScenarioError("shielding.layers: expected a list")
    layers = tuple(_build_layer(layer, i, source.effective_energy_mev) for i, layer in enumerate(raw_layers))

    if layers:
        models = tuple(shielding.get("models") or DEFAULT_MODELS)
        for model in models:
            if model not in MODELS:
                raise ScenarioError(
                    f"shielding.models: unknown model {model!r}; choose from {sorted(MODELS)}"
                )
        if len(set(models)) != len(models):
            raise ScenarioError("shielding.models: duplicate entries")
        if "practical_hvl" in models and any(layer.hvl_cm is None for layer in layers):
            raise ScenarioError("shielding.models includes 'practical_hvl' but not every layer has 'hvl_cm'")
    else:
        if shielding.get("models"):
            raise ScenarioError("shielding.models given but there are no shielding.layers")
        models = ()

    limits = []
    for i, raw_limit in enumerate(data.get("limits") or []):
        where = f"limits[{i}]"
        raw_limit = _mapping(raw_limit, where)
        _check_keys(raw_limit, {"name", "msv"}, where)
        value = float(_require(raw_limit, "msv", where))
        if not value > 0:
            raise ScenarioError(f"{where}.msv: must be > 0")
        limits.append(Limit(str(_require(raw_limit, "name", where)), value))

    sim = _mapping(data.get("simulation") or {}, "simulation")
    _check_keys(sim, {"n_samples", "seed"}, "simulation")
    n_samples = int(sim.get("n_samples", DEFAULT_N_SAMPLES))
    if n_samples < 2:
        raise ScenarioError("simulation.n_samples: must be >= 2")
    seed = sim.get("seed", DEFAULT_SEED)
    seed = None if seed is None else int(seed)

    warnings = []
    if distance.nominal < MIN_RECOMMENDED_DISTANCE_M:
        warnings.append(
            f"nominal distance {distance.nominal} m is below {MIN_RECOMMENDED_DISTANCE_M} m; "
            "the point-source approximation is unreliable this close to a real source"
        )

    return Scenario(
        name=name,
        description=" ".join(str(data.get("description", "")).split()),
        source=source,
        activity_mbq=activity,
        distance_m=distance,
        duration_h=duration,
        layers=layers,
        models=models,
        limits=tuple(limits),
        n_samples=n_samples,
        seed=seed,
        warnings=tuple(warnings),
    )


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{path}: invalid YAML: {exc}") from None
    try:
        return scenario_from_dict(data)
    except ScenarioError as exc:
        raise ScenarioError(f"{path}: {exc}") from None


__all__ = [
    "Limit",
    "LayerSpec",
    "Scenario",
    "ScenarioError",
    "load_scenario",
    "scenario_from_dict",
]
