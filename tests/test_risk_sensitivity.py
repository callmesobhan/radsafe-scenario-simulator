import numpy as np
import pytest

from conftest import make_scenario
from radsafe.montecarlo import run_simulation
from radsafe.risk import (
    NOMINAL_DETRIMENT_PER_SV,
    exceedance,
    nominal_detriment,
    summarize,
    wilson_interval,
)
from radsafe.sensitivity import spearman_sensitivity


def test_summarize_converts_to_msv():
    s = summarize(np.linspace(0, 1000, 10_001))  # uSv
    assert s.median == pytest.approx(0.5)
    assert s.mean == pytest.approx(0.5)
    assert s.p5 < s.median < s.p95 < s.p99


def test_exceedance_extremes_and_interval_contains_estimate():
    doses = np.full(1000, 500.0)  # 0.5 mSv
    above = exceedance(doses, 0.1)
    assert above.probability == 1.0 and above.ci_high == 1.0 and above.ci_low < 1.0
    none = exceedance(doses, 1.0)
    assert none.probability == 0.0 and none.ci_low == 0.0 and 0 < none.ci_high < 0.01
    half = exceedance(np.r_[np.full(500, 0.0), np.full(500, 2000.0)], 1.0)
    assert half.ci_low < half.probability == 0.5 < half.ci_high


def test_exceedance_is_strictly_greater_than():
    assert exceedance(np.array([1000.0]), 1.0).probability == 0.0


def test_wilson_symmetry_and_validation():
    lo, hi = wilson_interval(50, 100)
    assert lo + hi == pytest.approx(1.0)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_nominal_detriment_is_linear():
    per_msv = NOMINAL_DETRIMENT_PER_SV["population"] / 1000
    assert nominal_detriment(1.0, "population") == pytest.approx(per_msv)
    assert nominal_detriment(20.0) == pytest.approx(20 * nominal_detriment(1.0))


def test_sensitivity_identifies_dominant_input():
    sc = make_scenario(
        source={
            "nuclide": "Cs-137",
            "activity_mbq": {"dist": "lognormal", "median": 1000, "gsd": 1.5},  # dominant
        },
        geometry={"distance_m": {"dist": "lognormal", "median": 1.0, "gsd": 1.02}},
        exposure={"duration_h": 1.0},  # constant: must be omitted
    )
    res = run_simulation(sc, n_samples=20_000)
    sens = spearman_sensitivity(res, "narrow_beam")
    assert sens[0].input == "activity_mbq" and sens[0].share > 0.8
    assert sum(s.share for s in sens) == pytest.approx(1.0)
    assert "duration_h" not in {s.input for s in sens}
    assert sens[0].rho > 0
    # distance enters as 1/d^2, so its rank correlation with dose must be negative
    assert next(s for s in sens if s.input == "distance_m").rho < 0


def test_sensitivity_empty_when_dose_is_constant():
    res = run_simulation(make_scenario(), n_samples=100)  # everything constant except gamma / mu
    # gamma and mu are always uncertain, so make a fully deterministic one explicitly
    sc = make_scenario(
        source={
            "nuclide": "Test",
            "activity_mbq": 100,
            "gamma_constant_usv_m2_per_mbq_h": 0.1,
            "gamma_rel_unc": 0.0,
            "effective_energy_mev": 0.662,
        },
        shielding={},
    )
    assert spearman_sensitivity(run_simulation(sc, n_samples=50), "unshielded") == []
    assert spearman_sensitivity(res, "narrow_beam")  # non-empty
