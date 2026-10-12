"""Monte Carlo propagation of input uncertainty to dose.

Every input is sampled once, and *the same samples are fed to every shielding
model* (common random numbers). Differences between models therefore reflect
the models, not sampling noise, and per-sample ratios are meaningful.

Random streams are spawned from one ``SeedSequence`` in a fixed order, so
adding a model or changing ``n_samples`` does not reshuffle the streams of
unrelated inputs, and results are reproducible for a given seed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from radsafe.distributions import Distribution, LogNormal
from radsafe.dose import dose_from_rate, unshielded_dose_rate
from radsafe.materials import get_material
from radsafe.scenario import Scenario
from radsafe.shielding import LayerSample, get_model

#: Key under which the no-shield reference dose is stored in result dicts.
UNSHIELDED = "unshielded"


def layer_key(index: int, material: str, field: str) -> str:
    """Name of a sampled shield-layer input, e.g. ``shield1_lead_thickness_cm``."""
    return f"shield{index + 1}_{material}_{field}"


@dataclass
class SimulationResult:
    """Everything produced by one Monte Carlo run. Doses are in uSv."""

    scenario: Scenario
    n_samples: int
    seed: int | None
    inputs: dict[str, np.ndarray]
    unshielded_rate_usv_h: np.ndarray
    transmission: dict[str, np.ndarray]
    dose_usv: dict[str, np.ndarray]
    nominal_dose_usv: dict[str, float]

    @property
    def series(self) -> tuple[str, ...]:
        """Result keys in display order: the unshielded reference, then each model."""
        return (UNSHIELDED, *self.scenario.models)


def _layer_mu(scenario: Scenario) -> list[float]:
    energy = scenario.source.effective_energy_mev
    return [get_material(layer.material).linear_attenuation(energy) for layer in scenario.layers]


def _evaluate(
    scenario: Scenario, values: Mapping[str, ArrayLike]
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Dose model for one set of input values (scalars or equal-length arrays)."""
    rate = unshielded_dose_rate(values["activity_mbq"], values["distance_m"], values["gamma_constant"])
    duration = values["duration_h"]
    dose = {UNSHIELDED: dose_from_rate(rate, duration)}
    transmission: dict[str, np.ndarray] = {}
    if scenario.layers:
        stack = [
            LayerSample(
                material=layer.material,
                thickness_cm=values[layer_key(i, layer.material, "thickness_cm")],
                mu_per_cm=values[layer_key(i, layer.material, "mu_per_cm")],
                hvl_cm=values.get(layer_key(i, layer.material, "hvl_cm")),
            )
            for i, layer in enumerate(scenario.layers)
        ]
        for name in scenario.models:
            t = get_model(name).transmission(stack)
            transmission[name] = t
            dose[name] = dose_from_rate(rate * t, duration)
    return rate, transmission, dose


def nominal_inputs(scenario: Scenario) -> dict[str, float]:
    values = {
        "activity_mbq": scenario.activity_mbq.nominal,
        "distance_m": scenario.distance_m.nominal,
        "duration_h": scenario.duration_h.nominal,
        "gamma_constant": scenario.source.gamma_constant,
    }
    for i, (layer, mu) in enumerate(zip(scenario.layers, _layer_mu(scenario), strict=False)):
        values[layer_key(i, layer.material, "thickness_cm")] = layer.thickness_cm.nominal
        values[layer_key(i, layer.material, "mu_per_cm")] = mu
        if layer.hvl_cm is not None:
            values[layer_key(i, layer.material, "hvl_cm")] = layer.hvl_cm.nominal
    return values


def nominal_run(scenario: Scenario) -> dict[str, float]:
    """Deterministic dose [uSv] per model with every input at its nominal value."""
    _, _, dose = _evaluate(scenario, nominal_inputs(scenario))
    return {k: float(v) for k, v in dose.items()}


def _draw(scenario: Scenario, n: int, seed: int | None) -> dict[str, np.ndarray]:
    """Sample every uncertain input. Streams are consumed in a fixed order."""
    n_streams = 4 + 3 * len(scenario.layers)
    streams = iter(np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n_streams))

    gamma = LogNormal.from_rel_sd(scenario.source.gamma_constant, scenario.source.gamma_rel_unc)
    inputs: dict[str, np.ndarray] = {
        "activity_mbq": scenario.activity_mbq.sample(n, next(streams)),
        "distance_m": scenario.distance_m.sample(n, next(streams)),
        "duration_h": scenario.duration_h.sample(n, next(streams)),
        "gamma_constant": gamma.sample(n, next(streams)),
    }
    for i, (layer, mu) in enumerate(zip(scenario.layers, _layer_mu(scenario), strict=False)):
        thickness_rng, mu_rng, hvl_rng = next(streams), next(streams), next(streams)
        mu_dist: Distribution = LogNormal.from_rel_sd(mu, layer.mu_rel_unc)
        inputs[layer_key(i, layer.material, "thickness_cm")] = layer.thickness_cm.sample(n, thickness_rng)
        inputs[layer_key(i, layer.material, "mu_per_cm")] = mu_dist.sample(n, mu_rng)
        if layer.hvl_cm is not None:
            inputs[layer_key(i, layer.material, "hvl_cm")] = layer.hvl_cm.sample(n, hvl_rng)
    return inputs


class _FromScenario:
    """Sentinel: use the seed stored in the scenario."""


FROM_SCENARIO = _FromScenario()


def run_simulation(
    scenario: Scenario, n_samples: int | None = None, seed: int | None | _FromScenario = FROM_SCENARIO
) -> SimulationResult:
    """Propagate the scenario's input uncertainty to dose.

    ``n_samples`` and ``seed`` default to the scenario's own settings; pass
    ``seed=None`` explicitly for a non-reproducible run.
    """
    n = scenario.n_samples if n_samples is None else int(n_samples)
    if n < 2:
        raise ValueError("n_samples must be >= 2")
    used_seed = scenario.seed if isinstance(seed, _FromScenario) else seed
    inputs = _draw(scenario, n, used_seed)
    rate, transmission, dose = _evaluate(scenario, inputs)
    return SimulationResult(
        scenario=scenario,
        n_samples=n,
        seed=used_seed,
        inputs=inputs,
        unshielded_rate_usv_h=rate,
        transmission=transmission,
        dose_usv=dose,
        nominal_dose_usv=nominal_run(scenario),
    )
