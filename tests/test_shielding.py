import numpy as np
import pytest

from radsafe.compare import transmission_curve
from radsafe.shielding import (
    LayerSample,
    LinearBuildup,
    NarrowBeam,
    PracticalHVL,
    ShieldingError,
    get_model,
    optical_depth,
)


def layer(x, mu=0.5, hvl=None, material="lead"):
    return LayerSample(material, x, mu, hvl)


def test_zero_thickness_is_fully_transparent_for_every_model():
    for model in (NarrowBeam(), LinearBuildup(), PracticalHVL()):
        assert model.transmission([layer(0.0, hvl=1.0)]) == pytest.approx(1.0)


def test_narrow_beam_is_beer_lambert():
    assert NarrowBeam().transmission([layer(2.0, mu=0.5)]) == pytest.approx(np.exp(-1.0))


def test_layers_multiply():
    both = NarrowBeam().transmission([layer(1.0, mu=0.5), layer(2.0, mu=0.25, material="iron")])
    single = NarrowBeam().transmission([layer(1.0, mu=0.5)]) * NarrowBeam().transmission(
        [layer(2.0, mu=0.25)]
    )
    assert both == pytest.approx(single)
    assert optical_depth([layer(1.0, mu=0.5), layer(2.0, mu=0.25)]) == pytest.approx(1.0)


def test_buildup_never_below_narrow_beam():
    x = np.linspace(0, 20, 50)
    narrow = NarrowBeam().transmission([layer(x)])
    build = LinearBuildup().transmission([layer(x)])
    assert np.all(build >= narrow)
    assert np.all(build <= 1.0 + 1e-12)


def test_linear_buildup_formula():
    mfp = 3.0
    assert LinearBuildup().transmission([layer(mfp / 0.5, mu=0.5)]) == pytest.approx((1 + mfp) * np.exp(-mfp))


def test_practical_hvl_halves_per_hvl():
    t = PracticalHVL().transmission([layer(3.0, hvl=1.0)])
    assert t == pytest.approx(0.125)


def test_practical_hvl_requires_hvl():
    with pytest.raises(ShieldingError, match="hvl_cm"):
        PracticalHVL().transmission([layer(1.0)])


def test_monotonic_in_thickness():
    x = np.linspace(0, 10, 100)
    for model in (NarrowBeam(), PracticalHVL()):
        t = model.transmission([layer(x, hvl=1.0)])
        assert np.all(np.diff(t) <= 0)


def test_negative_inputs_rejected():
    with pytest.raises(ShieldingError):
        LayerSample("lead", -1.0, 0.5)
    with pytest.raises(ShieldingError):
        LayerSample("lead", 1.0, -0.5)
    with pytest.raises(ShieldingError):
        LayerSample("lead", 1.0, 0.5, hvl_cm=0.0)


def test_unknown_model():
    with pytest.raises(ShieldingError, match="unknown shielding model"):
        get_model("magic")


def test_transmission_curve_helper():
    curve = transmission_curve("Cs-137", "lead", [0, 1, 2], models=("narrow_beam", "linear_buildup"))
    assert set(curve) == {"narrow_beam", "linear_buildup"}
    assert curve["narrow_beam"][0] == pytest.approx(1.0)
    assert np.all(curve["linear_buildup"] >= curve["narrow_beam"])
