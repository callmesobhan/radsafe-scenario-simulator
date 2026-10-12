import math

import numpy as np
import pytest

from conftest import make_scenario
from radsafe.materials import get_material
from radsafe.montecarlo import UNSHIELDED, layer_key, nominal_run, run_simulation


def test_nominal_run_matches_hand_calculation():
    sc = make_scenario()
    mu = get_material("lead").linear_attenuation(0.662)
    nominal = nominal_run(sc)
    unshielded = 1000 * 0.100 / 1.0**2 * 1.0  # uSv
    assert nominal[UNSHIELDED] == pytest.approx(unshielded)
    assert nominal["narrow_beam"] == pytest.approx(unshielded * math.exp(-mu * 1.0))
    assert nominal["linear_buildup"] == pytest.approx(unshielded * (1 + mu) * math.exp(-mu))


def test_unshielded_scenario_has_only_reference_series():
    sc = make_scenario(shielding={})
    res = run_simulation(sc, n_samples=100)
    assert res.series == (UNSHIELDED,)
    assert res.transmission == {}


def test_analytic_log_variance_for_power_law():
    """Dose ~ A / d^2: Var[ln D] = (ln gsd_A)^2 + (2 ln gsd_d)^2 for independent log-normals."""
    gsd_a, gsd_d = 1.20, 1.10
    sc = make_scenario(
        source={
            "nuclide": "Test",
            "activity_mbq": {"dist": "lognormal", "median": 1000, "gsd": gsd_a},
            "gamma_constant_usv_m2_per_mbq_h": 0.1,
            "gamma_rel_unc": 0.0,
            "effective_energy_mev": 0.662,
        },
        geometry={"distance_m": {"dist": "lognormal", "median": 2.0, "gsd": gsd_d}},
        shielding={},
    )
    res = run_simulation(sc, n_samples=400_000, seed=1)
    expected = math.hypot(math.log(gsd_a), 2 * math.log(gsd_d))
    assert np.std(np.log(res.dose_usv[UNSHIELDED])) == pytest.approx(expected, rel=0.01)
    # and the median is the dose at the median inputs
    assert np.median(res.dose_usv[UNSHIELDED]) == pytest.approx(1000 * 0.1 / 4.0, rel=0.01)


def test_shield_attenuation_uncertainty_is_exponentially_amplified():
    """sigma[ln T] = x * sigma[mu] for narrow beam, so thicker shields => wider spread."""
    spread = []
    for thickness in (1.0, 4.0):
        sc = make_scenario(
            shielding={
                "models": ["narrow_beam"],
                "layers": [{"material": "lead", "thickness_cm": thickness, "mu_rel_unc": 0.05}],
            }
        )
        res = run_simulation(sc, n_samples=50_000, seed=3)
        spread.append(np.std(np.log(res.transmission["narrow_beam"])))
    assert spread[1] / spread[0] == pytest.approx(4.0, rel=0.05)


def test_same_seed_reproduces_and_different_seed_differs(scenario):
    a = run_simulation(scenario, seed=5)
    b = run_simulation(scenario, seed=5)
    c = run_simulation(scenario, seed=6)
    for key in a.inputs:
        np.testing.assert_array_equal(a.inputs[key], b.inputs[key])
    np.testing.assert_array_equal(a.dose_usv["narrow_beam"], b.dose_usv["narrow_beam"])
    assert not np.array_equal(a.dose_usv["narrow_beam"], c.dose_usv["narrow_beam"])


def test_models_share_samples_and_adding_a_model_does_not_reshuffle_inputs():
    two = make_scenario()
    one = make_scenario(
        shielding={"models": ["narrow_beam"], "layers": [{"material": "lead", "thickness_cm": 1.0}]}
    )
    r_two, r_one = run_simulation(two, seed=11), run_simulation(one, seed=11)
    for key in r_one.inputs:
        np.testing.assert_array_equal(r_one.inputs[key], r_two.inputs[key])
    np.testing.assert_array_equal(r_one.dose_usv["narrow_beam"], r_two.dose_usv["narrow_beam"])


def test_paired_ratio_is_the_buildup_factor(scenario):
    res = run_simulation(scenario, seed=2)
    mfp = res.inputs[layer_key(0, "lead", "mu_per_cm")] * res.inputs[layer_key(0, "lead", "thickness_cm")]
    ratio = res.dose_usv["linear_buildup"] / res.dose_usv["narrow_beam"]
    np.testing.assert_allclose(ratio, 1 + mfp)


def test_overrides(scenario):
    res = run_simulation(scenario, n_samples=321)
    assert res.n_samples == 321 and len(res.dose_usv["narrow_beam"]) == 321
    assert res.seed == scenario.seed
    assert run_simulation(scenario, n_samples=10, seed=None).seed is None
    with pytest.raises(ValueError):
        run_simulation(scenario, n_samples=1)


def test_practical_hvl_runs_when_hvl_given():
    sc = make_scenario(
        shielding={
            "models": ["narrow_beam", "practical_hvl"],
            "layers": [{"material": "lead", "thickness_cm": 1.3, "hvl_cm": 0.65}],
        }
    )
    assert nominal_run(sc)["practical_hvl"] == pytest.approx(100.0 * 0.25)
