"""Inverse probabilistic NCRP/AAPM TG-108 barrier optimizer, Pareto frontier, and ALARA Value-of-Information.

Commercial point-kernel shielding tools (MicroShield) and deterministic NCRP spreadsheets solve the
*forward* deterministic problem for a single nominal thickness. That hides two major risks:
  1. Uncertain future workload growth, occupancy factor T, concrete density/mu tolerance, and
     source-to-barrier geometry can push a "nominally compliant" barrier to 35-55% exceedance probability.
  2. Deterministic tools cannot tell a Radiation Safety Officer (RSO) whether paying for an occupancy
     survey or pouring thicker concrete / adding lead-lined drywall is cheaper (ALARA Value-of-Information).

This module solves the *inverse* probabilistic design problem under ANSI/ANS-6.4.3 GP buildup and
epistemic model spread:
  - ``optimize_barrier``: finds minimum thickness x*(material, confidence) ensuring
    P(D > limit) <= 1 - confidence (with Wilson upper bound option) and contrasts deterministic,
    probabilistic, and epistemic-robust designs.
  - ``pareto_frontier``: sweeps multiple shielding materials (lead, concrete, iron) to extract the
    non-dominated Cost (USD) vs. Exceedance Probability Pareto frontier.
  - ``value_of_information``: quantifies exact construction dollars saved by eliminating uncertainty in
    each individual input (occupancy T, workload/duration, distance, material density/mu, gamma constant).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from radsafe.distributions import Distribution, LogNormal, Normal, Triangular, Uniform
from radsafe.dose import dose_from_rate, unshielded_dose_rate
from radsafe.materials import get_material
from radsafe.scenario import Scenario
from radsafe.shielding import LayerSample, get_model
from radsafe.sources import get_source


@dataclass(frozen=True)
class MaterialCostSpec:
    """Illustrative installed barrier cost & attenuation-tolerance metadata (user-overridable)."""

    material: str
    usd_per_m2_per_cm: float  # installed material + framing cost per m^2 per cm thickness
    fixed_install_usd_per_m2: float  # base mobilization / finish cost per m^2
    default_mu_rel_unc: float  # realistic field density/attenuation tolerance (concrete ~8%, lead ~3%)


# Illustrative North American / EU clinical shielding cost benchmarks (USD, 2026)
DEFAULT_COSTS: dict[str, MaterialCostSpec] = {
    "lead": MaterialCostSpec(
        "lead", usd_per_m2_per_cm=1450.0, fixed_install_usd_per_m2=120.0, default_mu_rel_unc=0.03
    ),
    "concrete": MaterialCostSpec(
        "concrete", usd_per_m2_per_cm=42.0, fixed_install_usd_per_m2=95.0, default_mu_rel_unc=0.08
    ),
    "iron": MaterialCostSpec(
        "iron", usd_per_m2_per_cm=460.0, fixed_install_usd_per_m2=135.0, default_mu_rel_unc=0.04
    ),
    "water": MaterialCostSpec(
        "water", usd_per_m2_per_cm=18.0, fixed_install_usd_per_m2=150.0, default_mu_rel_unc=0.02
    ),
}


@dataclass(frozen=True)
class BarrierProblem:
    """Uncertain NCRP 147 / 151 / 155 / AAPM TG-108 facility barrier exposure problem.

    Annual unshielded dose [uSv] = (Gamma * A / d^2) * duration_h * workload_scale * use_factor * occupancy_T.
    """

    name: str
    nuclide: str
    activity_mbq: Distribution
    distance_m: Distribution
    duration_h: Distribution
    workload_scale: Distribution  # future patient-volume / workload growth factor (nominal 1.0)
    use_factor: Distribution  # beam/source use factor U in (0, 1]
    occupancy_factor: Distribution  # NCRP occupancy factor T in (0, 1]
    limit_msv: float
    limit_name: str
    barrier_area_m2: float = 12.0
    reference: str = ""


# Built-in NCRP / AAPM TG-108 facility shielding presets
FACILITY_PRESETS: dict[str, BarrierProblem] = {
    "pet_f18_uptake": BarrierProblem(
        name="PET/CT F-18 Uptake Room -> Adjacent Office (AAPM TG-108)",
        nuclide="F-18",
        # Effective decay-averaged uptake activity (~280 MBq after 0.64 patient self-attenuation)
        activity_mbq=LogNormal.from_rel_sd(280.0, 0.12),
        distance_m=Normal(mean=2.5, sd=0.25, lower=1.2),
        # 2000 uptake hours/year (~8 patients/day * 1 h * 250 days)
        duration_h=Triangular(1600.0, 2000.0, 2600.0),
        workload_scale=LogNormal.from_rel_sd(1.10, 0.15),  # 5-yr clinical volume growth uncertainty
        use_factor=Uniform(1.0, 1.0),
        occupancy_factor=Triangular(0.25, 0.50, 1.0),  # uncertain adjacent-room occupancy T
        limit_msv=1.0,
        limit_name="ICRP 103 / NCRP public limit (1.0 mSv/yr)",
        barrier_area_m2=12.0,
        reference="AAPM TG-108 (Madsen et al. 2006) — illustrative",
    ),
    "hdr_ir192_suite": BarrierProblem(
        name="HDR Ir-192 Brachytherapy Suite -> Control Corridor (NCRP 151)",
        nuclide="Ir-192",
        activity_mbq=LogNormal.from_rel_sd(370_000.0, 0.10),  # 10 Ci Ir-192 source
        distance_m=Normal(mean=4.0, sd=0.30, lower=2.0),
        duration_h=Triangular(40.0, 60.0, 95.0),  # annual beam-on hours
        workload_scale=LogNormal.from_rel_sd(1.05, 0.15),
        use_factor=Uniform(1.0, 1.0),
        occupancy_factor=Triangular(0.125, 0.25, 0.50),  # corridor occupancy T ~ 1/5 to 1/2
        limit_msv=1.0,
        limit_name="Public corridor limit (1.0 mSv/yr)",
        barrier_area_m2=18.0,
        reference="NCRP Report No. 151 — illustrative",
    ),
    "i131_therapy_room": BarrierProblem(
        name="I-131 Theranostics Inpatient Room -> Nurse Station (NCRP 155)",
        nuclide="I-131",
        activity_mbq=LogNormal.from_rel_sd(3700.0, 0.15),  # 100 mCi retained effective activity
        distance_m=Normal(mean=3.0, sd=0.30, lower=1.5),
        duration_h=Triangular(900.0, 1300.0, 1800.0),
        workload_scale=LogNormal.from_rel_sd(1.15, 0.18),
        use_factor=Uniform(1.0, 1.0),
        occupancy_factor=Triangular(0.50, 0.75, 1.0),
        limit_msv=1.0,
        limit_name="Public / non-radiation worker limit (1.0 mSv/yr)",
        barrier_area_m2=14.0,
        reference="NCRP Report No. 155 — illustrative",
    ),
    "cs137_calibrator": BarrierProblem(
        name="Cs-137 Calibration Bench -> Adjacent Lab (ICRP 103)",
        nuclide="Cs-137",
        activity_mbq=LogNormal.from_rel_sd(3700.0, 0.05),
        distance_m=Normal(mean=2.0, sd=0.25, lower=0.6),
        duration_h=Triangular(100.0, 150.0, 220.0),
        workload_scale=LogNormal.from_rel_sd(1.0, 0.12),
        use_factor=Uniform(1.0, 1.0),
        occupancy_factor=Triangular(0.5, 0.75, 1.0),
        limit_msv=0.3,
        limit_name="Facility ALARA constraint (0.3 mSv/yr)",
        barrier_area_m2=6.0,
        reference="ICRP 103 constraint — illustrative",
    ),
}


def problem_from_scenario(
    scenario: Scenario,
    *,
    limit_msv: float | None = None,
    barrier_area_m2: float = 12.0,
) -> BarrierProblem:
    """Wrap any existing RadSafe YAML ``Scenario`` as an inverse barrier optimization problem."""
    if limit_msv is None:
        if scenario.limits:
            limit_msv = scenario.limits[0].value_msv
            limit_name = scenario.limits[0].name
        else:
            limit_msv = 1.0
            limit_name = "ICRP 103 public limit (1.0 mSv/yr)"
    else:
        limit_name = f"Target limit ({limit_msv:g} mSv)"
    return BarrierProblem(
        name=scenario.name,
        nuclide=scenario.source.nuclide,
        activity_mbq=scenario.activity_mbq,
        distance_m=scenario.distance_m,
        duration_h=scenario.duration_h,
        workload_scale=Uniform(1.0, 1.0),
        use_factor=Uniform(1.0, 1.0),
        occupancy_factor=Uniform(1.0, 1.0),
        limit_msv=limit_msv,
        limit_name=limit_name,
        barrier_area_m2=barrier_area_m2,
    )


def _barrier_cost_usd(material: str, thickness_cm: float, area_m2: float) -> float:
    spec = DEFAULT_COSTS[material.lower()]
    if thickness_cm <= 0.0:
        return 0.0
    return area_m2 * (spec.fixed_install_usd_per_m2 + spec.usd_per_m2_per_cm * thickness_cm)


def _barrier_mass_kg(material: str, thickness_cm: float, area_m2: float) -> float:
    mat = get_material(material)
    vol_cm3 = area_m2 * 1e4 * thickness_cm
    return vol_cm3 * mat.density_g_cm3 / 1000.0


def _wilson_upper(exceedances: int, n: int, z: float = 1.959963984540054) -> float:
    if n <= 0:
        return 1.0
    p = exceedances / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    return min(1.0, center + margin)


def _sample_problem(
    problem: BarrierProblem,
    material: str,
    n_samples: int,
    seed: int,
    *,
    mu_rel_unc: float | None = None,
    fix_inputs: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Sample all uncertain inputs using independent SeedSequence streams (CRN-safe)."""
    fixed = fix_inputs or set()
    source = get_source(problem.nuclide)
    mat = get_material(material)
    mu_nom = mat.linear_attenuation(source.effective_energy_mev)
    mu_unc = DEFAULT_COSTS[mat.name].default_mu_rel_unc if mu_rel_unc is None else mu_rel_unc

    streams = iter(np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(8))

    def _draw_or_fix(key: str, dist: Distribution) -> np.ndarray:
        rng = next(streams)
        if key in fixed:
            return np.full(n_samples, float(dist.nominal), dtype=float)
        return dist.sample(n_samples, rng)

    gamma_dist = LogNormal.from_rel_sd(source.gamma_constant, source.gamma_rel_unc)
    mu_dist = LogNormal.from_rel_sd(mu_nom, mu_unc) if mu_unc > 0 else Uniform(mu_nom, mu_nom)

    return {
        "activity_mbq": _draw_or_fix("activity_mbq", problem.activity_mbq),
        "distance_m": _draw_or_fix("distance_m", problem.distance_m),
        "duration_h": _draw_or_fix("duration_h", problem.duration_h),
        "workload_scale": _draw_or_fix("workload_scale", problem.workload_scale),
        "use_factor": _draw_or_fix("use_factor", problem.use_factor),
        "occupancy_factor": _draw_or_fix("occupancy_factor", problem.occupancy_factor),
        "gamma_constant": _draw_or_fix("gamma_constant", gamma_dist),
        "mu_per_cm": _draw_or_fix("mu_per_cm", mu_dist),
    }


def _unshielded_dose_msv(samples: dict[str, np.ndarray]) -> np.ndarray:
    rate = unshielded_dose_rate(samples["activity_mbq"], samples["distance_m"], samples["gamma_constant"])
    base_usv = dose_from_rate(rate, samples["duration_h"])
    return base_usv * samples["workload_scale"] * samples["use_factor"] * samples["occupancy_factor"] / 1000.0


def _solve_thickness(
    unshielded_msv: np.ndarray,
    mu_per_cm: np.ndarray,
    material: str,
    energy_mev: float,
    limit_msv: float,
    model_name: str,
    *,
    alpha: float,
    use_wilson_upper: bool = False,
    max_cm: float = 250.0,
) -> tuple[float, float]:
    """Bisect for the exact minimum barrier thickness [cm] achieving P(dose > limit) <= alpha."""
    model = get_model(model_name)
    n = int(unshielded_msv.size)

    def _p_exceed(x_cm: float) -> float:
        layer = LayerSample(material=material, thickness_cm=x_cm, mu_per_cm=mu_per_cm, energy_mev=energy_mev)
        dose = unshielded_msv * model.transmission([layer])
        k = int(np.sum(dose > limit_msv))
        return _wilson_upper(k, n) if use_wilson_upper else (k / n)

    p0 = _p_exceed(0.0)
    if p0 <= alpha:
        return 0.0, p0

    lo, hi = 0.0, 2.0
    while _p_exceed(hi) > alpha and hi < max_cm:
        hi = min(hi * 2.0, max_cm)
    if _p_exceed(hi) > alpha:
        return hi, _p_exceed(hi)

    for _ in range(36):
        mid = 0.5 * (lo + hi)
        if _p_exceed(mid) <= alpha:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-3:
            break
    return hi, _p_exceed(hi)


@dataclass(frozen=True)
class BarrierOptimizationResult:
    """Inverse probabilistic vs deterministic barrier design comparison for one material."""

    problem_name: str
    nuclide: str
    energy_mev: float
    material: str
    limit_msv: float
    confidence: float
    barrier_area_m2: float
    # Deterministic nominal (MicroShield/NCRP-style) thickness and its TRUE exceedance risk under uncertainty
    deterministic_cm: float
    deterministic_true_exceedance_prob: float
    deterministic_cost_usd: float
    # Probabilistic GP-buildup design guaranteeing P(D > limit) <= 1 - confidence
    probabilistic_gp_cm: float
    probabilistic_gp_exceedance_prob: float
    probabilistic_gp_cost_usd: float
    probabilistic_gp_mass_kg: float
    # Epistemic model spread across models at the same target confidence
    thickness_by_model_cm: dict[str, float]
    epistemic_envelope_cm: float
    epistemic_envelope_cost_usd: float
    # Hidden design safety margin gap (probabilistic minus deterministic)
    uncertainty_margin_cm: float
    uncertainty_cost_premium_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_name": self.problem_name,
            "nuclide": self.nuclide,
            "energy_mev": self.energy_mev,
            "material": self.material,
            "limit_msv": self.limit_msv,
            "confidence": self.confidence,
            "barrier_area_m2": self.barrier_area_m2,
            "deterministic_design": {
                "thickness_cm": round(self.deterministic_cm, 3),
                "true_exceedance_prob": round(self.deterministic_true_exceedance_prob, 4),
                "cost_usd": round(self.deterministic_cost_usd, 2),
            },
            "probabilistic_gp_design": {
                "thickness_cm": round(self.probabilistic_gp_cm, 3),
                "exceedance_prob": round(self.probabilistic_gp_exceedance_prob, 4),
                "cost_usd": round(self.probabilistic_gp_cost_usd, 2),
                "mass_kg": round(self.probabilistic_gp_mass_kg, 1),
            },
            "epistemic_models_cm": {k: round(v, 3) for k, v in self.thickness_by_model_cm.items()},
            "epistemic_envelope_cm": round(self.epistemic_envelope_cm, 3),
            "epistemic_envelope_cost_usd": round(self.epistemic_envelope_cost_usd, 2),
            "uncertainty_margin_cm": round(self.uncertainty_margin_cm, 3),
            "uncertainty_cost_premium_usd": round(self.uncertainty_cost_premium_usd, 2),
        }


def optimize_barrier(
    problem: BarrierProblem,
    material: str = "concrete",
    *,
    confidence: float = 0.95,
    n_samples: int = 20_000,
    seed: int = 42,
    mu_rel_unc: float | None = None,
    use_wilson_upper: bool = False,
) -> BarrierOptimizationResult:
    """Solve the inverse probabilistic shielding design problem under ANSI/ANS-6.4.3 GP buildup."""
    if not 0.5 <= confidence < 1.0:
        raise ValueError("confidence must be in [0.5, 1.0)")
    alpha = 1.0 - confidence
    source = get_source(problem.nuclide)
    mat_name = get_material(material).name

    # 1. Deterministic nominal thickness (what MicroShield / NCRP spreadsheets output)
    nom_samples = _sample_problem(
        problem,
        mat_name,
        n_samples=1,
        seed=seed,
        mu_rel_unc=0.0,
        fix_inputs={
            "activity_mbq",
            "distance_m",
            "duration_h",
            "workload_scale",
            "use_factor",
            "occupancy_factor",
            "gamma_constant",
            "mu_per_cm",
        },
    )
    unsh_nom = _unshielded_dose_msv(nom_samples)
    det_cm, _ = _solve_thickness(
        unsh_nom,
        nom_samples["mu_per_cm"],
        mat_name,
        source.effective_energy_mev,
        problem.limit_msv,
        "gp_buildup",
        alpha=0.0,
        use_wilson_upper=False,
    )

    # 2. Full Monte Carlo samples
    samples = _sample_problem(problem, mat_name, n_samples=n_samples, seed=seed, mu_rel_unc=mu_rel_unc)
    unsh_mc = _unshielded_dose_msv(samples)
    mu_mc = samples["mu_per_cm"]

    # True exceedance risk of the deterministic MicroShield/NCRP thickness
    gp_model = get_model("gp_buildup")
    det_dose_mc = unsh_mc * gp_model.transmission(
        [LayerSample(mat_name, det_cm, mu_mc, energy_mev=source.effective_energy_mev)]
    )
    det_true_p = float(np.mean(det_dose_mc > problem.limit_msv))

    # 3. Inverse probabilistic thickness across epistemic shielding models
    by_model: dict[str, float] = {}
    gp_p = 0.0
    for m_name in ("narrow_beam", "gp_buildup", "linear_buildup"):
        x_m, p_m = _solve_thickness(
            unsh_mc,
            mu_mc,
            mat_name,
            source.effective_energy_mev,
            problem.limit_msv,
            m_name,
            alpha=alpha,
            use_wilson_upper=use_wilson_upper,
        )
        by_model[m_name] = x_m
        if m_name == "gp_buildup":
            gp_p = p_m

    prob_cm = by_model["gp_buildup"]
    env_cm = max(by_model.values())

    det_cost = _barrier_cost_usd(mat_name, det_cm, problem.barrier_area_m2)
    prob_cost = _barrier_cost_usd(mat_name, prob_cm, problem.barrier_area_m2)
    env_cost = _barrier_cost_usd(mat_name, env_cm, problem.barrier_area_m2)

    return BarrierOptimizationResult(
        problem_name=problem.name,
        nuclide=problem.nuclide,
        energy_mev=source.effective_energy_mev,
        material=mat_name,
        limit_msv=problem.limit_msv,
        confidence=confidence,
        barrier_area_m2=problem.barrier_area_m2,
        deterministic_cm=det_cm,
        deterministic_true_exceedance_prob=det_true_p,
        deterministic_cost_usd=det_cost,
        probabilistic_gp_cm=prob_cm,
        probabilistic_gp_exceedance_prob=gp_p,
        probabilistic_gp_cost_usd=prob_cost,
        probabilistic_gp_mass_kg=_barrier_mass_kg(mat_name, prob_cm, problem.barrier_area_m2),
        thickness_by_model_cm=by_model,
        epistemic_envelope_cm=env_cm,
        epistemic_envelope_cost_usd=env_cost,
        uncertainty_margin_cm=max(0.0, prob_cm - det_cm),
        uncertainty_cost_premium_usd=max(0.0, prob_cost - det_cost),
    )


@dataclass(frozen=True)
class ValueOfInformationRow:
    """Construction dollars and thickness saved by eliminating uncertainty in one parameter."""

    parameter: str
    thickness_if_known_cm: float
    thickness_saved_cm: float
    cost_saved_usd: float
    share_of_uncertainty_premium: float


def value_of_information(
    problem: BarrierProblem,
    material: str = "concrete",
    *,
    confidence: float = 0.95,
    n_samples: int = 15_000,
    seed: int = 42,
) -> list[ValueOfInformationRow]:
    """ALARA Value-of-Information: exact USD and cm saved by measuring/fixing each uncertain input."""
    base = optimize_barrier(problem, material, confidence=confidence, n_samples=n_samples, seed=seed)
    source = get_source(problem.nuclide)
    mat_name = get_material(material).name
    alpha = 1.0 - confidence

    candidates = (
        "occupancy_factor",
        "workload_scale",
        "duration_h",
        "distance_m",
        "mu_per_cm",
        "activity_mbq",
        "gamma_constant",
    )
    rows: list[ValueOfInformationRow] = []
    prem = max(base.uncertainty_cost_premium_usd, 1e-9)
    for param in candidates:
        samp = _sample_problem(problem, mat_name, n_samples=n_samples, seed=seed, fix_inputs={param})
        unsh = _unshielded_dose_msv(samp)
        x_known, _ = _solve_thickness(
            unsh,
            samp["mu_per_cm"],
            mat_name,
            source.effective_energy_mev,
            problem.limit_msv,
            "gp_buildup",
            alpha=alpha,
        )
        cost_known = _barrier_cost_usd(mat_name, x_known, problem.barrier_area_m2)
        saved_cm = max(0.0, base.probabilistic_gp_cm - x_known)
        saved_usd = max(0.0, base.probabilistic_gp_cost_usd - cost_known)
        rows.append(
            ValueOfInformationRow(
                parameter=param,
                thickness_if_known_cm=round(x_known, 3),
                thickness_saved_cm=round(saved_cm, 3),
                cost_saved_usd=round(saved_usd, 2),
                share_of_uncertainty_premium=round(saved_usd / prem, 4)
                if base.uncertainty_cost_premium_usd > 0
                else 0.0,
            )
        )
    return sorted(rows, key=lambda r: -r.cost_saved_usd)


@dataclass(frozen=True)
class ParetoOption:
    material: str
    thickness_cm: float
    exceedance_prob: float
    cost_usd: float
    mass_kg: float
    compliant_95: bool


def pareto_frontier(
    problem: BarrierProblem,
    materials: Sequence[str] = ("concrete", "lead", "iron"),
    *,
    confidence: float = 0.95,
    n_samples: int = 12_000,
    seed: int = 42,
    n_grid: int = 18,
) -> dict[str, Any]:
    """Compute the multi-material Cost (USD) vs. Exceedance Probability non-dominated Pareto front."""
    source = get_source(problem.nuclide)
    gp = get_model("gp_buildup")
    all_pts: list[ParetoOption] = []
    best_per_material: dict[str, dict[str, Any]] = {}

    for mat in materials:
        opt = optimize_barrier(problem, mat, confidence=confidence, n_samples=n_samples, seed=seed)
        best_per_material[opt.material] = opt.to_dict()
        samples = _sample_problem(problem, opt.material, n_samples=n_samples, seed=seed)
        unsh = _unshielded_dose_msv(samples)
        mu_mc = samples["mu_per_cm"]
        x_max = max(opt.probabilistic_gp_cm * 1.35, 1.0)
        grid = sorted(
            {*np.linspace(0.0, x_max, n_grid).tolist(), opt.deterministic_cm, opt.probabilistic_gp_cm}
        )
        for x_cm in grid:
            dose = unsh * gp.transmission(
                [LayerSample(opt.material, x_cm, mu_mc, energy_mev=source.effective_energy_mev)]
            )
            p_ex = float(np.mean(dose > problem.limit_msv))
            all_pts.append(
                ParetoOption(
                    material=opt.material,
                    thickness_cm=round(x_cm, 3),
                    exceedance_prob=round(p_ex, 4),
                    cost_usd=round(_barrier_cost_usd(opt.material, x_cm, problem.barrier_area_m2), 2),
                    mass_kg=round(_barrier_mass_kg(opt.material, x_cm, problem.barrier_area_m2), 1),
                    compliant_95=p_ex <= (1.0 - confidence),
                )
            )

    # Extract non-dominated frontier: sort by ascending cost, keep points that strictly reduce exceedance_prob
    all_pts.sort(key=lambda o: (o.cost_usd, o.exceedance_prob))
    frontier: list[ParetoOption] = []
    best_p = float("inf")
    for pt in all_pts:
        if pt.exceedance_prob < best_p - 1e-4:
            frontier.append(pt)
            best_p = pt.exceedance_prob

    compliant = [p for p in frontier if p.compliant_95]
    cheapest_compliant = compliant[0] if compliant else None
    return {
        "problem_name": problem.name,
        "limit_msv": problem.limit_msv,
        "confidence": confidence,
        "barrier_area_m2": problem.barrier_area_m2,
        "cheapest_compliant": None
        if cheapest_compliant is None
        else {
            "material": cheapest_compliant.material,
            "thickness_cm": cheapest_compliant.thickness_cm,
            "cost_usd": cheapest_compliant.cost_usd,
            "mass_kg": cheapest_compliant.mass_kg,
            "exceedance_prob": cheapest_compliant.exceedance_prob,
        },
        "best_per_material": best_per_material,
        "pareto_front": [
            {
                "material": p.material,
                "thickness_cm": p.thickness_cm,
                "exceedance_prob": p.exceedance_prob,
                "cost_usd": p.cost_usd,
                "mass_kg": p.mass_kg,
                "compliant": p.compliant_95,
            }
            for p in frontier
        ],
    }


def format_optimization_text(
    opt: BarrierOptimizationResult,
    voi: list[ValueOfInformationRow] | None = None,
) -> str:
    det_p_pct = opt.deterministic_true_exceedance_prob * 100.0
    conf_pct = opt.confidence * 100.0
    lines = [
        f"RadSafe Inverse Probabilistic Barrier Design — {opt.problem_name}",
        f"  Source: {opt.nuclide} ({opt.energy_mev} MeV) | Material: {opt.material} "
        f"| Area: {opt.barrier_area_m2:g} m^2",
        f"  Limit:  <= {opt.limit_msv:g} mSv/yr at {conf_pct:.0f}% confidence (ANSI/ANS-6.4.3 GP buildup)",
        "",
        f"  Deterministic nominal (MicroShield/NCRP): {opt.deterministic_cm:6.2f} cm  "
        f"(USD {opt.deterministic_cost_usd:8,.0f})  -> TRUE P(exceed) = {det_p_pct:5.1f}%!",
        f"  Probabilistic GP-buildup ({conf_pct:.0f}% conf):     {opt.probabilistic_gp_cm:6.2f} cm  "
        f"(USD {opt.probabilistic_gp_cost_usd:8,.0f}, {opt.probabilistic_gp_mass_kg:,.0f} kg)  "
        f"-> P(exceed) = {opt.probabilistic_gp_exceedance_prob * 100:5.1f}%",
        f"  Epistemic envelope (worst model):         {opt.epistemic_envelope_cm:6.2f} cm  "
        f"(USD {opt.epistemic_envelope_cost_usd:8,.0f})",
        f"  Hidden uncertainty margin:               +{opt.uncertainty_margin_cm:6.2f} cm  "
        f"(+USD {opt.uncertainty_cost_premium_usd:,.0f})",
        "",
        "  Epistemic thickness spread by shielding model:",
    ]
    for m_name, cm in opt.thickness_by_model_cm.items():
        lines.append(f"    {m_name:<16} {cm:6.2f} cm")
    if voi:
        lines += [
            "",
            "  ALARA Value-of-Information (expected savings from measuring/fixing each uncertain input):",
            f"    {'Parameter':<18} {'cm saved':>9} {'USD saved':>11} {'Share':>8}",
        ]
        for row in voi:
            if row.thickness_saved_cm > 0 or row.cost_saved_usd > 0:
                lines.append(
                    f"    {row.parameter:<18} {row.thickness_saved_cm:9.2f} "
                    f"{row.cost_saved_usd:11,.0f} {row.share_of_uncertainty_premium * 100:7.1f}%"
                )
    return "\n".join(lines)
