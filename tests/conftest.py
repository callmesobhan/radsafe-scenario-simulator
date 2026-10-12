from pathlib import Path

import pytest

from radsafe.scenario import scenario_from_dict

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = REPO_ROOT / "scenarios"


def make_scenario(**overrides):
    """A small Cs-137 scenario; top-level sections can be replaced via keyword."""
    data = {
        "name": "test scenario",
        "source": {"nuclide": "Cs-137", "activity_mbq": 1000},
        "geometry": {"distance_m": 1.0},
        "exposure": {"duration_h": 1.0},
        "shielding": {
            "models": ["narrow_beam", "linear_buildup"],
            "layers": [{"material": "lead", "thickness_cm": 1.0}],
        },
        "limits": [{"name": "limit", "msv": 0.05}],
        "simulation": {"n_samples": 5000, "seed": 7},
    }
    data.update(overrides)
    return scenario_from_dict(data)


@pytest.fixture
def scenario():
    return make_scenario()
