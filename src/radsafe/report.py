"""Turn a :class:`SimulationResult` into a JSON-able report and a text table."""

from __future__ import annotations

from typing import Any

import numpy as np

from radsafe.montecarlo import UNSHIELDED, SimulationResult
from radsafe.risk import NOMINAL_DETRIMENT_PER_SV, exceedance, nominal_detriment, summarize
from radsafe.scenario import Limit
from radsafe.sensitivity import spearman_sensitivity

REFERENCE_MODEL = "narrow_beam"


def _exceedance_row(dose_usv: np.ndarray, limit: Limit) -> dict[str, Any]:
    e = exceedance(dose_usv, limit.value_msv)
    return {
        "limit": limit.name,
        "limit_msv": limit.value_msv,
        "probability": e.probability,
        "ci_low": e.ci_low,
        "ci_high": e.ci_high,
    }


def build_report(result: SimulationResult) -> dict[str, Any]:
    """Collect every headline number into plain Python types (doses in mSv)."""
    sc = result.scenario
    results: dict[str, Any] = {}
    reference = result.dose_usv.get(REFERENCE_MODEL)
    for name in result.series:
        dose = result.dose_usv[name]
        s = summarize(dose)
        entry: dict[str, Any] = {
            "nominal_msv": result.nominal_dose_usv[name] / 1000.0,
            "mean_msv": s.mean,
            "median_msv": s.median,
            "p5_msv": s.p5,
            "p95_msv": s.p95,
            "p99_msv": s.p99,
            "exceedance": [_exceedance_row(dose, limit) for limit in sc.limits],
            "sensitivity": [
                {"input": t.input, "rho": t.rho, "share": t.share} for t in spearman_sensitivity(result, name)
            ],
            "nominal_detriment": {pop: nominal_detriment(s.mean, pop) for pop in NOMINAL_DETRIMENT_PER_SV},
        }
        if reference is not None and name not in (UNSHIELDED, REFERENCE_MODEL):
            entry["median_ratio_to_narrow_beam"] = float(np.median(dose) / np.median(reference))
        results[name] = entry

    return {
        "scenario": {
            "name": sc.name,
            "description": sc.description,
            "nuclide": sc.source.nuclide,
            "gamma_constant_usv_m2_per_mbq_h": sc.source.gamma_constant,
            "activity_mbq_nominal": sc.activity_mbq.nominal,
            "distance_m_nominal": sc.distance_m.nominal,
            "duration_h_nominal": sc.duration_h.nominal,
            "layers": [
                {"material": layer.material, "thickness_cm_nominal": layer.thickness_cm.nominal}
                for layer in sc.layers
            ],
            "warnings": list(sc.warnings),
        },
        "simulation": {"n_samples": result.n_samples, "seed": result.seed},
        "limits": [{"name": lim.name, "msv": lim.value_msv} for lim in sc.limits],
        "results": results,
    }


# --------------------------------------------------------------------------
# Text rendering
# --------------------------------------------------------------------------


def _g(x: float) -> str:
    return f"{x:.3g}"


def _h(x: float) -> str:
    """Header-style number: thousands separators, no exponent below 1e6 ("3,700")."""
    return f"{x:,.4g}"


def _pct(p: float, ci_high: float) -> str:
    if p == 0:
        return f"0  (<{ci_high * 100:.3g}%)"
    return f"{p * 100:.3g}%"


def _pretty(key: str) -> str:
    """'shield1_lead_thickness_cm' -> 'lead thickness' ; 'gamma_constant' -> 'gamma constant'."""
    if key.startswith("shield"):
        _, material, *rest = key.split("_")
        field = {"thickness": "thickness", "mu": "attenuation (mu)", "hvl": "HVL"}[rest[0]]
        return f"{material} {field}"
    return {
        "activity_mbq": "activity",
        "distance_m": "distance",
        "duration_h": "duration",
        "gamma_constant": "gamma constant",
    }[key]


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    """Fixed-width text table: first column left-aligned, the rest right-aligned."""
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]

    def fmt(row: list[str]) -> str:
        cells = zip(row, widths, strict=True)
        return "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(cells))

    return [fmt(header), "  ".join("-" * w for w in widths), *[fmt(r) for r in rows]]


def format_text(report: dict[str, Any]) -> str:
    sc, sim, results = report["scenario"], report["simulation"], report["results"]
    lines = [
        f"RadSafe Scenario Simulator - {sc['name']}",
        "",
        f"Source    {sc['nuclide']}, {_h(sc['activity_mbq_nominal'])} MBq nominal "
        f"(gamma constant {_g(sc['gamma_constant_usv_m2_per_mbq_h'])} uSv m2/(MBq h))",
        f"Geometry  {_g(sc['distance_m_nominal'])} m nominal, {_g(sc['duration_h_nominal'])} h nominal",
    ]
    if sc["layers"]:
        stack = " + ".join(f"{_g(x['thickness_cm_nominal'])} cm {x['material']}" for x in sc["layers"])
        lines.append(f"Shield    {stack}")
    seed = "random" if sim["seed"] is None else sim["seed"]
    lines.append(f"Samples   {sim['n_samples']:,} (seed {seed})")
    for w in sc["warnings"]:
        lines.append(f"WARNING   {w}")

    lines += ["", "Dose over the exposure period (mSv)"]
    rows = [
        [name, *(_g(r[k]) for k in ("nominal_msv", "median_msv", "mean_msv", "p5_msv", "p95_msv", "p99_msv"))]
        for name, r in results.items()
    ]
    lines += _table(["model", "nominal", "median", "mean", "P5", "P95", "P99"], rows)

    if "narrow_beam" in results and len(results) > 2:
        ratios = ", ".join(
            f"{n} {r['median_ratio_to_narrow_beam']:.2f}x"
            for n, r in results.items()
            if "median_ratio_to_narrow_beam" in r
        )
        lines += ["", f"Median dose relative to narrow_beam: {ratios}"]

    if report["limits"]:
        lines += ["", "P(dose > limit)   [95% interval is Monte Carlo sampling error only]"]
        header = ["model", *(f"{lim['name']} ({_g(lim['msv'])} mSv)" for lim in report["limits"])]
        rows = [
            [name, *(_pct(e["probability"], e["ci_high"]) for e in r["exceedance"])]
            for name, r in results.items()
        ]
        lines += _table(header, rows)

    lines += ["", "Largest contributors to dose uncertainty (Spearman rho^2 share)"]
    for name, r in results.items():
        top = ", ".join(f"{_pretty(s['input'])} {s['share']:.0%}" for s in r["sensitivity"][:3])
        lines.append(f"  {name}: {top}")

    lines += [
        "",
        "Educational / research tool. Not a shielding design or regulatory compliance calculation.",
    ]
    return "\n".join(lines)
