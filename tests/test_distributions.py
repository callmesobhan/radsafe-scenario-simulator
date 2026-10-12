import math

import numpy as np
import pytest

from radsafe.distributions import (
    Constant,
    DistributionError,
    LogNormal,
    Normal,
    Triangular,
    Uniform,
    from_spec,
)

N = 200_000


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_constant(rng):
    d = from_spec(3.5)
    assert isinstance(d, Constant) and d.nominal == 3.5
    assert np.all(d.sample(10, rng) == 3.5)


def test_uniform_moments(rng):
    x = Uniform(2, 6).sample(N, rng)
    assert x.mean() == pytest.approx(4.0, rel=0.01)
    assert x.min() >= 2 and x.max() <= 6


def test_triangular(rng):
    d = Triangular(1, 2, 6)
    x = d.sample(N, rng)
    assert x.mean() == pytest.approx((1 + 2 + 6) / 3, rel=0.01)
    assert d.nominal == 2


def test_normal_truncation_respected(rng):
    x = Normal(1.0, 1.0, lower=0.0).sample(N, rng)
    assert x.min() >= 0.0
    assert x.mean() > 1.0  # truncating the lower tail pushes the mean up


def test_normal_impossible_truncation_errors(rng):
    with pytest.raises(DistributionError, match="exclude almost all"):
        Normal(0.0, 1.0, lower=50.0).sample(10, rng)


def test_lognormal_median_and_gsd(rng):
    x = LogNormal(100.0, 1.2).sample(N, rng)
    assert np.median(x) == pytest.approx(100.0, rel=0.01)
    assert np.exp(np.std(np.log(x))) == pytest.approx(1.2, rel=0.01)


def test_lognormal_from_rel_sd_matches_cv(rng):
    x = LogNormal.from_rel_sd(10.0, 0.3).sample(N, rng)
    assert x.std() / x.mean() == pytest.approx(0.3, rel=0.03)
    assert isinstance(LogNormal.from_rel_sd(10.0, 0.0), Constant)


def test_from_spec_nonnegative_default_bound():
    d = from_spec({"dist": "normal", "mean": 1.0, "sd": 1.0}, nonnegative=True)
    assert d.lower == 0.0
    assert from_spec({"dist": "normal", "mean": 1.0, "sd": 1.0}).lower is None


def test_numeric_strings_are_accepted():
    # PyYAML (YAML 1.1) parses 3.7e6 as the string '3.7e6'
    import yaml

    assert isinstance(yaml.safe_load("3.7e6"), str)
    assert from_spec("3.7e6").nominal == 3.7e6
    assert from_spec({"dist": "lognormal", "median": "3.7e6", "gsd": 1.1}).median == 3.7e6


def test_from_spec_rel_sd():
    d = from_spec({"dist": "normal", "mean": 200, "rel_sd": 0.05})
    assert d.sd == pytest.approx(10.0)
    g = from_spec({"dist": "lognormal", "median": 5, "rel_sd": 0.1})
    assert g.gsd == pytest.approx(math.exp(math.sqrt(math.log1p(0.01))))


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"dist": "weibull"}, "'dist' must be one of"),
        ({"dist": "normal", "mean": 1}, "exactly one of 'sd' or 'rel_sd'"),
        ({"dist": "normal", "mean": 1, "sd": 1, "rel_sd": 0.1}, "exactly one"),
        ({"dist": "normal", "mean": 1, "sd": 1, "sigma": 2}, "unknown key"),
        ({"dist": "lognormal", "median": -1, "gsd": 1.1}, "median must be > 0"),
        ({"dist": "lognormal", "median": 1, "gsd": 0.9}, "gsd must be >= 1"),
        ({"dist": "triangular", "low": 3, "mode": 2, "high": 4}, "low <= mode <= high"),
        ({"dist": "uniform", "low": 2, "high": 1}, "low"),
        ({"dist": "uniform", "low": "a", "high": 1}, "must be a number"),
        ("three", "expected a number or mapping"),
        (True, "expected a number or mapping"),
        (float("nan"), "finite"),
    ],
)
def test_from_spec_errors(spec, message):
    with pytest.raises(DistributionError, match=message):
        from_spec(spec)
