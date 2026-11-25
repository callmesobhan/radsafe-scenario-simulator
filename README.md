<div align="center">

# RadSafe Scenario Simulator 🛡️☢️

**Radiation-Protection Exposure Scenario & Shielding Uncertainty Simulator**  
*Compare simplified gamma shielding models side-by-side and propagate uncertainty in distance, time, activity, and attenuation via Monte Carlo*

[![CI](https://github.com/callmesobhan/radsafe-scenario-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/callmesobhan/radsafe-scenario-simulator/actions/workflows/ci.yml)
[![License: Non-Commercial (Research & Personal)](https://img.shields.io/badge/License-Non--Commercial%20(Research%20%26%20Personal)-f59e0b.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Physics](https://img.shields.io/badge/Models-Inverse--Square%20%7C%20Buildup%20%7C%20HVL%20%7C%20Monte%20Carlo-10b981)](#what-it-answers)

</div>

> [!WARNING]
> **Educational and research tool.** It uses deliberately simplified models and
> illustrative reference data. It is **not** a shielding design tool and must not be
> used for regulatory compliance, dose assignment, or safety decisions. Those need
> a qualified radiation protection expert and validated codes. See
> [Limitations](#limitations).

## Simulation Architecture

```mermaid
flowchart LR
    A["📄 YAML Scenario (Source, Distance, Time, Shielding)"] --> B["Monte Carlo Parameter Sampler"]
    B --> C1["Unshielded Inverse-Square"]
    B --> C2["Narrow-Beam Exponential"]
    B --> C3["Linear Buildup Factor"]
    B --> C4["Practical HVL / TVL"]
    C1 & C2 & C3 & C4 --> D["Dose Distributions + Exceedance Probability + Spearman Sensitivity"]
```

## What it answers

| Question | How |
|---|---|
| How much dose accumulates at distance *d* over time *t*, behind a shield? | Inverse-square point source × shielding transmission × time |
| How much does the *shielding assumption* matter? | Narrow-beam, linear-buildup and practical-HVL models run on identical samples |
| How uncertain is the answer? | Monte Carlo propagation of activity, distance, duration, shield thickness and attenuation (μ) |
| How likely is it to exceed a limit or constraint? | Exceedance probability with a Monte Carlo confidence interval |
| What should I measure or control more carefully? | Spearman rank sensitivity of dose to each input |

## Quickstart

```bash
pip install -e ".[dev]"          # numpy + pyyaml (+ matplotlib, pytest, ruff for dev)

radsafe run scenarios/cs137_bench_source.yaml --plot summary.png
radsafe run scenarios/co60_irradiator_corridor.yaml --json report.json
radsafe optimize --preset pet_f18_uptake --material concrete --confidence 0.95 --voi
radsafe pareto --preset hdr_ir192_suite --materials concrete,lead,iron --confidence 0.95
radsafe validate scenarios/*.yaml
radsafe list                     # built-in sources, materials, shielding models & NCRP/TG-108 presets
```

## Inverse Probabilistic Barrier Optimizer, ANSI/ANS-6.4.3 GP Buildup & ALARA Value-of-Information

Deterministic shielding tools (**MicroShield**, NCRP spreadsheets) compute one nominal thickness—hiding the true probability that uncertain patient workload growth, occupancy factor $T$, or concrete density tolerance will push annual dose over regulatory limits.

RadSafe now ships:
* **ANSI/ANS-6.4.3 Geometric Progression (GP) buildup (`gp_buildup`)** over `0.1–3.0 MeV` for lead, iron, ordinary concrete, and water (covering `Tc-99m`, `I-131`, `Ir-192`, `F-18` PET, `Cs-137`, and `Co-60`).
* **Inverse Probabilistic Barrier Optimizer (`radsafe optimize`):** solves for the exact minimum thickness achieving $P(\text{dose} > \text{limit}) \le 1 - \text{confidence}$ under uncertain NCRP 147/151/155 & AAPM TG-108 parameters, and exposes the **hidden exceedance risk** of a deterministic nominal design.
* **Cost-vs-Exceedance Pareto Frontier (`radsafe pareto`):** compares concrete, lead, and iron on a non-dominated USD vs. exceedance-risk curve to select the cheapest compliant barrier.
* **ALARA Value-of-Information (`--voi`):** translates parameter uncertainty into **exact construction dollars saved** if the RSO measures or pins occupancy $T$, concrete density, workload, or distance instead of pouring thicker walls.

Example (`scenarios/cs137_bench_source.yaml`: 3.7 GBq Cs-137, worker at ~2 m, 3 cm lead, ~150 h/year):

```
Dose over the exposure period (mSv)
model           nominal  median  mean     P5    P95    P99
--------------  -------  ------  ----  -----  -----  -----
unshielded         13.9    13.8  14.6   8.68   23.2   29.4
narrow_beam       0.316   0.315  0.34  0.172  0.592  0.778
linear_buildup     1.51    1.51  1.61  0.863   2.72   3.52
practical_hvl     0.566   0.565   0.6  0.343  0.973   1.25

P(dose > limit)
model           Facility constraint (illustrative) (0.3 mSv)  ICRP 103 public limit (1 mSv)
narrow_beam                                            55.3%                         0.179%
linear_buildup                                          100%                          88.5%
practical_hvl                                          98.3%                          4.29%
```

Same source, same shield, same inputs: the answer to "does this exceed 1 mSv?" is
*0.2%*, *4%* or *88%* depending on the shielding assumption. The spread between
models (median dose differs by 1.8-4.8x here, 9-14x in the Co-60 example) is comparable to
or larger than the spread from all input uncertainties combined, which is why the models
are compared rather than picked.

![Exceedance curves and sensitivity for the Cs-137 example](docs/img/example_cs137.png)

## Describing a scenario

A scenario is a YAML file. Any numeric value can be a constant or a distribution.

```yaml
name: Cs-137 bench source behind lead
source:
  nuclide: Cs-137
  activity_mbq: { dist: lognormal, median: 3700, gsd: 1.05 }
geometry:
  distance_m: { dist: normal, mean: 2.0, sd: 0.25, lower: 0.5 }
exposure:
  duration_h: { dist: triangular, low: 100, mode: 150, high: 200 }
shielding:
  models: [narrow_beam, linear_buildup, practical_hvl]
  layers:
    - material: lead
      thickness_cm: { dist: normal, mean: 3.0, sd: 0.1 }
      mu_rel_unc: 0.05
      hvl_cm: 0.65
limits:
  - { name: ICRP 103 public limit, msv: 1.0 }
```

Full reference: [docs/scenario-schema.md](docs/scenario-schema.md).

## Python API

```python
from radsafe import load_scenario, run_simulation, build_report, format_text

scenario = load_scenario("scenarios/cs137_bench_source.yaml")
result = run_simulation(scenario, n_samples=50_000, seed=1)

result.dose_usv["narrow_beam"]  # numpy array of per-sample doses, uSv
report = build_report(result)  # plain dict (what --json writes)
print(format_text(report))
```

## Repository layout

```
src/radsafe/
  distributions.py   input uncertainty: constant / uniform / normal / lognormal / triangular
  sources.py         nuclide data (gamma constant, effective energy)       -> data/sources.yaml
  materials.py       NIST XCOM attenuation, log-log interpolation          -> data/materials.yaml
  dose.py            inverse-square dose rate, dose = rate x time
  shielding.py       narrow_beam / linear_buildup / practical_hvl models
  scenario.py        validated scenario dataclasses + strict YAML loader
  montecarlo.py      sampling + propagation (common random numbers across models)
  sensitivity.py     Spearman rank sensitivity
  risk.py            summaries, exceedance + Wilson CI, ICRP limits and nominal detriment
  report.py          JSON-able report and text tables
  plotting.py        optional matplotlib figure
  cli.py             radsafe run | validate | list
scenarios/           illustrative example scenarios
tests/               physics checks, analytic Monte Carlo checks, validation, CLI
docs/                physics, uncertainty, schema, architecture, roadmap
examples/            standalone scripts
```

## Physics in one paragraph

Dose rate from an unshielded point source is `Γ·A / d²`, with `Γ` the ambient dose
equivalent rate constant H\*(10) in µSv·m²/(MBq·h). Shields multiply that by a
transmission `T`: `exp(-Σμx)` for narrow beam (optimistic, unscattered photons only),
`(1+Σμx)·exp(-Σμx)` for a crude broad-beam buildup correction, or `2^-(Σx/HVL)`
using practical half-value layers you supply. Attenuation coefficients μ come from NIST
XCOM tables. Dose is rate × duration. Details and validity limits:
[docs/physics.md](docs/physics.md).

## Data provenance

* **Attenuation (μ/ρ)** for lead, iron, concrete and water: NIST X-Ray Mass
  Attenuation Coefficients, 0.5–2 MeV, transcribed in
  [`materials.yaml`](src/radsafe/data/materials.yaml).
* **Gamma constants** for Cs-137 and Co-60 are mid-range values of published
  tabulations that differ by a few percent; a 5% uncertainty is carried through. They
  are marked *illustrative: verify before use* in
  [`sources.yaml`](src/radsafe/data/sources.yaml).
* **Practical HVLs** in the example scenarios are common rule-of-thumb values, marked
  illustrative in the scenario files.
* **Dose limits / risk coefficients**: ICRP Publication 103 (2007).

Replace any of these with your own validated data; they are plain YAML.

## Testing

```bash
pytest          # 100+ tests, < 1 s
```

Beyond unit tests, the suite checks the Monte Carlo engine against closed-form results
(e.g. for `D ∝ A/d²` with log-normal inputs, `Var[ln D]` is known exactly), verifies that
adding a shielding model does not reshuffle the random streams of other inputs, and
checks that thicker shields amplify attenuation uncertainty exponentially, as theory says.

## Limitations

Point isotropic source; slab shields at normal incidence; no room scatter or skyshine;
one effective photon energy (so no broad spectra such as Ir-192 or X-ray tubes); no
source decay; no neutron, beta or internal exposure; inputs sampled independently.
See [docs/physics.md](docs/physics.md#what-is-not-modelled) for the full list and
[docs/roadmap.md](docs/roadmap.md) for what is planned.

## Development

```bash
make install   # editable install with dev extras
make check     # ruff + pytest
```

CI runs lint and tests on Python 3.10–3.13.

## License

**Non-Commercial Personal & Academic Research License (NC-PARL 1.0)** — see [LICENSE](LICENSE). Free for personal use and non-commercial academic/scientific research with citation (`Sobhan Shahbazi (2026). radsafe-scenario-simulator. https://github.com/callmesobhan/radsafe-scenario-simulator`). Commercial use requires a written license.
