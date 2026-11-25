"""Command-line interface: ``radsafe run | validate | list``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from radsafe import __version__
from radsafe.materials import list_materials
from radsafe.montecarlo import run_simulation
from radsafe.optimizer import (
    FACILITY_PRESETS,
    format_optimization_text,
    optimize_barrier,
    pareto_frontier,
    problem_from_scenario,
    value_of_information,
)
from radsafe.report import build_report, format_text
from radsafe.scenario import load_scenario
from radsafe.shielding import MODELS
from radsafe.sources import list_sources


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radsafe",
        description="Simulate controlled-source radiation exposure scenarios with uncertainty.",
    )
    parser.add_argument("--version", action="version", version=f"radsafe {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a scenario file")
    run.add_argument("scenario", type=Path)
    run.add_argument("-n", "--n-samples", type=int, help="override simulation.n_samples")
    run.add_argument("--seed", type=int, help="override simulation.seed")
    run.add_argument("--json", type=Path, metavar="PATH", help="also write the full report as JSON")
    run.add_argument(
        "--plot", type=Path, metavar="PNG", help="also write a summary figure (needs radsafe[plot])"
    )

    validate = sub.add_parser("validate", help="check scenario files without running them")
    validate.add_argument("scenarios", type=Path, nargs="+")

    sub.add_parser("list", help="list built-in sources, materials, shielding models, and facility presets")

    opt = sub.add_parser(
        "optimize",
        help="inverse probabilistic NCRP/TG-108 barrier optimizer (ANSI/ANS-6.4.3 GP buildup + ALARA VoI)",
    )
    opt.add_argument("scenario", type=Path, nargs="?", help="scenario YAML file (or use --preset)")
    opt.add_argument(
        "--preset", choices=sorted(FACILITY_PRESETS), help="built-in NCRP / TG-108 facility preset"
    )
    opt.add_argument("--material", default="concrete", help="barrier material (concrete, lead, iron, water)")
    opt.add_argument("--limit", type=float, default=None, help="annual dose limit [mSv/yr]")
    opt.add_argument(
        "--confidence", type=float, default=0.95, help="target compliance confidence (default 0.95)"
    )
    opt.add_argument("--area-m2", type=float, default=None, help="barrier area [m^2] for cost/mass")
    opt.add_argument("-n", "--n-samples", type=int, default=20_000)
    opt.add_argument("--seed", type=int, default=42)
    opt.add_argument("--voi", action="store_true", help="compute ALARA Value-of-Information dollar savings")
    opt.add_argument("--json", action="store_true", help="print result as JSON")

    par = sub.add_parser(
        "pareto",
        help="multi-material Cost (USD) vs. Exceedance Probability Pareto frontier",
    )
    par.add_argument("scenario", type=Path, nargs="?", help="scenario YAML file (or use --preset)")
    par.add_argument(
        "--preset", choices=sorted(FACILITY_PRESETS), help="built-in NCRP / TG-108 facility preset"
    )
    par.add_argument("--materials", default="concrete,lead,iron", help="comma-separated materials to compare")
    par.add_argument("--limit", type=float, default=None, help="annual dose limit [mSv/yr]")
    par.add_argument("--confidence", type=float, default=0.95)
    par.add_argument("--area-m2", type=float, default=None)
    par.add_argument("-n", "--n-samples", type=int, default=12_000)
    par.add_argument("--seed", type=int, default=42)
    par.add_argument("--json", action="store_true")
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    kwargs = {} if args.seed is None else {"seed": args.seed}
    result = run_simulation(scenario, n_samples=args.n_samples, **kwargs)
    report = build_report(result)
    print(format_text(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    if args.plot:
        try:
            from radsafe.plotting import plot_summary

            plot_summary(result, args.plot)
        except ImportError:
            print("error: plotting needs matplotlib: pip install 'radsafe[plot]'", file=sys.stderr)
            return 2
        print(f"wrote {args.plot}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    for path in args.scenarios:
        scenario = load_scenario(path)
        print(f"ok  {path}  ({scenario.name})")
        for warning in scenario.warnings:
            print(f"    warning: {warning}")
    return 0


def _cmd_list(_: argparse.Namespace) -> int:
    print("Sources  (H*(10) rate constant, uSv m2/(MBq h); effective energy)")
    for s in list_sources():
        unc = f"+/-{s.gamma_rel_unc:.0%}"
        print(f"  {s.nuclide:<8} {s.gamma_constant:<7g} {unc:<6} {s.effective_energy_mev} MeV")
    print("\nMaterials  (density g/cm3)")
    for m in list_materials():
        print(f"  {m.name:<9} {m.density_g_cm3:<7g} {m.energies_mev[0]}-{m.energies_mev[-1]} MeV")
    print("\nShielding models")
    for name, model in MODELS.items():
        print(f"  {name:<15} {model.summary}")
    print("\nFacility presets (for `radsafe optimize --preset ...` / `radsafe pareto --preset ...`)")
    for key, prob in FACILITY_PRESETS.items():
        print(f"  {key:<20} {prob.name}")
    return 0


def _resolve_problem(args: argparse.Namespace):
    import dataclasses

    if args.preset:
        prob = FACILITY_PRESETS[args.preset]
        if args.limit is not None:
            prob = dataclasses.replace(prob, limit_msv=args.limit)
        if args.area_m2 is not None:
            prob = dataclasses.replace(prob, barrier_area_m2=args.area_m2)
        return prob
    if args.scenario is None:
        raise ValueError("supply either a scenario YAML path or --preset")
    sc = load_scenario(args.scenario)
    return problem_from_scenario(sc, limit_msv=args.limit, barrier_area_m2=args.area_m2 or 12.0)


def _cmd_optimize(args: argparse.Namespace) -> int:
    prob = _resolve_problem(args)
    opt = optimize_barrier(
        prob,
        material=args.material,
        confidence=args.confidence,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    voi = (
        value_of_information(
            prob, material=args.material, confidence=args.confidence, n_samples=args.n_samples, seed=args.seed
        )
        if args.voi
        else None
    )
    if args.json:
        payload = opt.to_dict()
        if voi is not None:
            payload["value_of_information"] = [
                {
                    "parameter": r.parameter,
                    "thickness_if_known_cm": r.thickness_if_known_cm,
                    "thickness_saved_cm": r.thickness_saved_cm,
                    "cost_saved_usd": r.cost_saved_usd,
                    "share_of_uncertainty_premium": r.share_of_uncertainty_premium,
                }
                for r in voi
            ]
        print(json.dumps(payload, indent=2))
        return 0
    print(format_optimization_text(opt, voi))
    return 0


def _cmd_pareto(args: argparse.Namespace) -> int:
    prob = _resolve_problem(args)
    mats = [m.strip() for m in args.materials.split(",") if m.strip()]
    res = pareto_frontier(
        prob, materials=mats, confidence=args.confidence, n_samples=args.n_samples, seed=args.seed
    )
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    print(f"RadSafe Cost-vs-Exceedance Pareto Frontier — {res['problem_name']}")
    print(
        f"  Limit <= {res['limit_msv']:g} mSv/yr | Target confidence {res['confidence'] * 100:.0f}% "
        f"| Area {res['barrier_area_m2']:g} m^2"
    )
    cc = res["cheapest_compliant"]
    if cc:
        print(
            f"  Cheapest compliant option: {cc['material']} @ {cc['thickness_cm']:.2f} cm "
            f"(USD {cc['cost_usd']:,.0f}, {cc['mass_kg']:,.0f} kg, "
            f"P(exceed)={cc['exceedance_prob'] * 100:.1f}%)"
        )
    print(
        f"\n  {'Material':<10} {'Thick (cm)':>10} {'Cost (USD)':>12} "
        f"{'Mass (kg)':>10} {'P(exceed)':>10} {'OK':>4}"
    )
    for row in res["pareto_front"]:
        ok_lbl = "YES" if row["compliant"] else "no"
        print(
            f"  {row['material']:<10} {row['thickness_cm']:10.2f} {row['cost_usd']:12,.0f} "
            f"{row['mass_kg']:10,.0f} {row['exceedance_prob'] * 100:9.1f}% {ok_lbl:>4}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handler = {
        "run": _cmd_run,
        "validate": _cmd_validate,
        "list": _cmd_list,
        "optimize": _cmd_optimize,
        "pareto": _cmd_pareto,
    }[args.command]
    try:
        return handler(args)
    except (ValueError, OSError) as exc:  # ScenarioError, DataError, DistributionError, ... are ValueErrors
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
