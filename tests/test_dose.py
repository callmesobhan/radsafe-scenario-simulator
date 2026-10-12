import numpy as np
import pytest

from radsafe.dose import dose_from_rate, unshielded_dose_rate


def test_known_value():
    # 1 GBq Cs-137 at 1 m with gamma constant 0.1 uSv m2/(MBq h) -> 100 uSv/h
    assert unshielded_dose_rate(1000, 1.0, 0.1) == pytest.approx(100.0)


@pytest.mark.parametrize("factor", [2.0, 3.0, 10.0])
def test_inverse_square(factor):
    near = unshielded_dose_rate(500, 1.5, 0.1)
    far = unshielded_dose_rate(500, 1.5 * factor, 0.1)
    assert near / far == pytest.approx(factor**2)


def test_linear_in_activity_and_time():
    r1 = unshielded_dose_rate(100, 2.0, 0.3)
    r2 = unshielded_dose_rate(300, 2.0, 0.3)
    assert r2 / r1 == pytest.approx(3.0)
    assert dose_from_rate(r1, 8.0) == pytest.approx(8.0 * r1)


def test_vectorised():
    out = unshielded_dose_rate([100, 200], [1.0, 2.0], 0.1)
    assert out == pytest.approx(np.array([10.0, 5.0]))


@pytest.mark.parametrize("args", [(100, 0.0, 0.1), (100, -1.0, 0.1), (-1, 1.0, 0.1), (100, 1.0, 0.0)])
def test_rejects_nonphysical_inputs(args):
    with pytest.raises(ValueError):
        unshielded_dose_rate(*args)


def test_rejects_negative_duration():
    with pytest.raises(ValueError):
        dose_from_rate(1.0, -1.0)
