"""How much shield do you need? Compare the shielding models across thickness.

Prints, for Cs-137, the thickness of lead and of concrete at which each model first
reaches a target transmission. The gap between models is the cost of getting the
shielding assumption wrong.

    python examples/thickness_sweep.py
"""

import numpy as np

from radsafe.compare import transmission_curve

NUCLIDE = "Cs-137"
TARGETS = (1e-2, 1e-3)
MODELS = ("narrow_beam", "linear_buildup", "practical_hvl")
#: Illustrative practical (broad-beam) half-value layers for Cs-137; verify before relying on them.
PRACTICAL_HVL_CM = {"lead": 0.65, "concrete": 4.8}
GRID_CM = np.linspace(0.0, 80.0, 8001)


def thickness_for(transmission: np.ndarray, target: float) -> float:
    """Smallest grid thickness whose transmission is <= target."""
    return float(GRID_CM[np.argmax(transmission <= target)])


def main() -> None:
    for material, hvl in PRACTICAL_HVL_CM.items():
        curves = transmission_curve(NUCLIDE, material, GRID_CM, MODELS, hvl_cm=hvl)
        print(f"\n{NUCLIDE} in {material}: thickness (cm) needed for a given dose reduction")
        print(f"  {'model':<15}" + "".join(f"{'T <= ' + format(t, 'g'):>12}" for t in TARGETS))
        for name in MODELS:
            row = "".join(f"{thickness_for(curves[name], t):>12.2f}" for t in TARGETS)
            print(f"  {name:<15}{row}")


if __name__ == "__main__":
    main()
