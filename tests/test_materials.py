import math

import pytest

from radsafe.materials import get_material, list_materials
from radsafe.sources import DataError, get_source, list_sources


def test_table_values_round_trip():
    lead = get_material("lead")
    # NIST XCOM, lead at 1.25 MeV: 5.876e-2 cm2/g
    assert lead.mass_attenuation(1.25) == pytest.approx(5.876e-2)
    assert lead.linear_attenuation(1.25) == pytest.approx(5.876e-2 * 11.35)


def test_interpolation_is_monotonic_and_bracketed():
    lead = get_material("lead")
    lo, mid, hi = (lead.mass_attenuation(e) for e in (0.6, 0.662, 0.8))
    assert hi < mid < lo


def test_lead_cs137_narrow_beam_hvl_is_physically_sensible():
    mu = get_material("lead").linear_attenuation(get_source("Cs-137").effective_energy_mev)
    hvl_cm = math.log(2) / mu
    assert 0.4 < hvl_cm < 0.7  # ~0.55 cm narrow-beam; practical broad-beam values are larger


def test_denser_materials_attenuate_more_per_cm():
    e = 0.662
    mus = {m.name: m.linear_attenuation(e) for m in list_materials()}
    assert mus["lead"] > mus["iron"] > mus["concrete"] > mus["water"]


def test_refuses_to_extrapolate():
    with pytest.raises(DataError, match="no attenuation data"):
        get_material("lead").mass_attenuation(0.1)


def test_unknown_names_list_alternatives():
    with pytest.raises(DataError, match="built-in materials"):
        get_material("unobtainium")
    with pytest.raises(DataError, match="built-in sources"):
        get_source("Xx-999")


@pytest.mark.parametrize("alias", ["Cs-137", "cs137", "CS 137", "cs-137"])
def test_source_name_normalisation(alias):
    assert get_source(alias).nuclide == "Cs-137"


def test_all_sources_have_attenuation_data_for_all_materials():
    for source in list_sources():
        for material in list_materials():
            assert material.linear_attenuation(source.effective_energy_mev) > 0
