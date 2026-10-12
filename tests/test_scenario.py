import copy

import pytest

from conftest import SCENARIO_DIR, make_scenario
from radsafe.scenario import ScenarioError, load_scenario, scenario_from_dict

EXAMPLES = sorted(SCENARIO_DIR.glob("*.yaml"))


def test_example_scenarios_exist():
    assert len(EXAMPLES) >= 2


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_scenarios_load(path):
    sc = load_scenario(path)
    assert sc.name and sc.layers and sc.limits


def test_defaults():
    sc = make_scenario(shielding={"layers": [{"material": "lead", "thickness_cm": 1}]})
    assert sc.models == ("narrow_beam", "linear_buildup")
    assert sc.layers[0].mu_rel_unc == 0.05


def test_custom_source():
    sc = make_scenario(
        source={
            "nuclide": "Na-22",
            "activity_mbq": 10,
            "gamma_constant_usv_m2_per_mbq_h": 0.3,
            "gamma_rel_unc": 0.05,
            "effective_energy_mev": 0.8,
        }
    )
    assert sc.source.nuclide == "Na-22" and sc.source.gamma_constant == 0.3


def test_builtin_source_override():
    sc = make_scenario(
        source={"nuclide": "Cs-137", "activity_mbq": 10, "gamma_constant_usv_m2_per_mbq_h": 0.2}
    )
    assert sc.source.gamma_constant == 0.2 and sc.source.effective_energy_mev == 0.662


def test_tiny_distance_warns():
    assert make_scenario(geometry={"distance_m": 0.05}).warnings
    assert not make_scenario().warnings


def _bad(mutator):
    data = {
        "name": "x",
        "source": {"nuclide": "Cs-137", "activity_mbq": 1000},
        "geometry": {"distance_m": 1.0},
        "exposure": {"duration_h": 1.0},
        "shielding": {"layers": [{"material": "lead", "thickness_cm": 1.0}]},
    }
    data = copy.deepcopy(data)
    mutator(data)
    return data


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda d: d.update(bogus=1), "unknown key"),
        (lambda d: d.pop("source"), "missing required key 'source'"),
        (lambda d: d["source"].update(nuclide="Xx-1"), "not built in"),
        (lambda d: d["source"].update(activity_mbq=0), "must be > 0"),
        (lambda d: d["geometry"].update(distance_m=0), "must be > 0"),
        (lambda d: d["exposure"].update(duration_h=-1), "must be >= 0"),
        (lambda d: d["shielding"]["layers"][0].update(material="unobtainium"), "built-in materials"),
        (lambda d: d["shielding"]["layers"][0].update(thickness_cm=-2), "must be >= 0"),
        (lambda d: d["shielding"]["layers"][0].update(mu_rel_unc=-0.1), "mu_rel_unc"),
        (lambda d: d["shielding"].update(models=["magic"]), "unknown model"),
        (lambda d: d["shielding"].update(models=["narrow_beam", "narrow_beam"]), "duplicate"),
        (lambda d: d["shielding"].update(models=["practical_hvl"]), "hvl_cm"),
        (lambda d: d["shielding"].update(layers=[], models=["narrow_beam"]), "no shielding.layers"),
        (lambda d: d.update(limits=[{"name": "l", "msv": 0}]), "must be > 0"),
        (lambda d: d.update(limits=[{"name": "l"}]), "missing required key 'msv'"),
        (lambda d: d.update(simulation={"n_samples": 1}), "n_samples"),
        (lambda d: d.update(simulation={"nsamples": 100}), "unknown key"),
        (lambda d: d["geometry"].update(distance_m={"dist": "normal", "mean": 1}), "sd"),
    ],
)
def test_validation_errors(mutator, message):
    with pytest.raises(ScenarioError, match=message):
        scenario_from_dict(_bad(mutator))


def test_attenuation_data_range_checked_against_source_energy():
    data = _bad(
        lambda d: d["source"].update(
            nuclide="HiE",
            gamma_constant_usv_m2_per_mbq_h=0.1,
            gamma_rel_unc=0.05,
            effective_energy_mev=10.0,
        )
    )
    with pytest.raises(ScenarioError, match="no attenuation data"):
        scenario_from_dict(data)


def test_load_errors_name_the_file(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nbogus: 1\n")
    with pytest.raises(ScenarioError, match="bad.yaml"):
        load_scenario(bad)
    broken = tmp_path / "broken.yaml"
    broken.write_text("name: [unclosed\n")
    with pytest.raises(ScenarioError, match="invalid YAML"):
        load_scenario(broken)
