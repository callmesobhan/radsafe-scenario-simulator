"""Tests for ANSI/ANS-6.4.3 GP buildup, inverse probabilistic barrier optimization, Pareto, and ALARA VoI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from radsafe.buildup_gp import gp_buildup_factor
from radsafe.cli import main
from radsafe.materials import get_material
from radsafe.optimizer import (
    FACILITY_PRESETS,
    optimize_barrier,
    pareto_frontier,
    problem_from_scenario,
    value_of_information,
)
from radsafe.scenario import load_scenario
from radsafe.shielding import LayerSample, get_model


def test_gp_buildup_matches_ansi_ans_643_benchmarks():
    # Exact boundary: B(E, 0) = 1.0 for every material
    for mat in ("lead", "iron", "concrete", "water"):
        assert gp_buildup_factor(mat, 1.0, 0.0) == pytest.approx(1.0)
        assert gp_buildup_factor(mat, 1.0, 1.0) > 1.0

    # At 1 MeV, 10 mfp: lead B ~ 4.0-5.0, concrete B ~ 18-24, water B ~ 18-26 (ANSI/ANS-6.4.3)
    b_pb = float(gp_buildup_factor("lead", 1.0, 10.0))
    b_conc = float(gp_buildup_factor("concrete", 1.0, 10.0))
    b_h2o = float(gp_buildup_factor("water", 1.0, 10.0))
    assert 3.0 < b_pb < 6.5
    assert 15.0 < b_conc < 28.0
    assert 15.0 < b_h2o < 30.0
    # Low-Z concrete builds up far more scattered photons than high-Z lead at the same mfp
    assert b_conc > 3.0 * b_pb


def test_gp_buildup_transmission_strictly_above_narrow_beam_and_monotonic():
    gp = get_model("gp_buildup")
    nb = get_model("narrow_beam")
    mu = get_material("concrete").linear_attenuation(0.511)
    xs = np.array([0.0, 5.0, 15.0, 30.0, 50.0])
    t_gp = gp.transmission([LayerSample("concrete", xs, mu, energy_mev=0.511)])
    t_nb = nb.transmission([LayerSample("concrete", xs, mu, energy_mev=0.511)])
    assert t_gp[0] == pytest.approx(1.0)
    assert np.all(t_gp[1:] > t_nb[1:])
    assert np.all(np.diff(t_gp) < 0)


def test_inverse_optimizer_exposes_hidden_exceedance_of_deterministic_design():
    prob = FACILITY_PRESETS["pet_f18_uptake"]
    opt = optimize_barrier(prob, material="concrete", confidence=0.95, n_samples=8000, seed=7)

    # Deterministic nominal design has ~30-60% true exceedance probability under workload/occupancy unc!
    assert opt.deterministic_true_exceedance_prob > 0.25
    assert opt.probabilistic_gp_exceedance_prob <= 0.055
    assert opt.probabilistic_gp_cm > opt.deterministic_cm
    assert opt.uncertainty_margin_cm > 2.0
    assert opt.uncertainty_cost_premium_usd > 0.0
    # Epistemic ordering on low-Z concrete at 511 keV: GP buildup requires thicker concrete than narrow-beam
    assert opt.thickness_by_model_cm["gp_buildup"] > opt.thickness_by_model_cm["narrow_beam"]


def test_alara_value_of_information_ranks_dominant_uncertainties():
    prob = FACILITY_PRESETS["pet_f18_uptake"]
    voi = value_of_information(prob, material="concrete", confidence=0.95, n_samples=6000, seed=11)
    assert len(voi) == 7
    # Sorted descending by USD saved; top uncertain driver (occupancy/distance/workload/mu) saves real money
    assert voi[0].cost_saved_usd >= voi[-1].cost_saved_usd
    assert voi[0].cost_saved_usd > 100.0
    assert voi[0].thickness_saved_cm > 0.2


def test_multi_material_pareto_frontier_is_strictly_non_dominated():
    prob = FACILITY_PRESETS["hdr_ir192_suite"]
    res = pareto_frontier(
        prob, materials=("concrete", "lead", "iron"), confidence=0.95, n_samples=4000, seed=3
    )
    front = res["pareto_front"]
    assert len(front) >= 3
    # Cost monotonically increases while exceedance probability strictly decreases along the Pareto front
    costs = [p["cost_usd"] for p in front]
    probs = [p["exceedance_prob"] for p in front]
    assert all(c1 <= c2 for c1, c2 in zip(costs, costs[1:], strict=False))
    assert all(p1 > p2 for p1, p2 in zip(probs, probs[1:], strict=False))
    assert res["cheapest_compliant"] is not None
    assert res["cheapest_compliant"]["exceedance_prob"] <= 0.05


def test_scenario_yaml_and_cli_optimize_and_pareto(capsys):
    repo = Path(__file__).resolve().parents[1]
    sc = load_scenario(repo / "scenarios" / "cs137_bench_source.yaml")
    prob = problem_from_scenario(sc, limit_msv=1.0)
    opt = optimize_barrier(prob, material="lead", confidence=0.95, n_samples=4000, seed=1)
    assert opt.probabilistic_gp_cm > opt.deterministic_cm

    assert (
        main(
            [
                "optimize",
                "--preset",
                "pet_f18_uptake",
                "--material",
                "concrete",
                "--voi",
                "-n",
                "3000",
                "--json",
            ]
        )
        == 0
    )
    opt_json = json.loads(capsys.readouterr().out)
    assert "value_of_information" in opt_json
    assert (
        opt_json["probabilistic_gp_design"]["thickness_cm"] > opt_json["deterministic_design"]["thickness_cm"]
    )

    assert main(["pareto", "--preset", "cs137_calibrator", "-n", "2500", "--json"]) == 0
    par_json = json.loads(capsys.readouterr().out)
    assert par_json["cheapest_compliant"] is not None
