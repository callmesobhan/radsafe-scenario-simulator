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

    sub.add_parser("list", help="list built-in sources, materials and shielding models")
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
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handler = {"run": _cmd_run, "validate": _cmd_validate, "list": _cmd_list}[args.command]
    try:
        return handler(args)
    except (ValueError, OSError) as exc:  # ScenarioError, DataError, DistributionError, ... are ValueErrors
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
