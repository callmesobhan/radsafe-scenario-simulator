# Architecture

## Data flow

```
scenario.yaml ──► scenario.py ──► Scenario (frozen, validated)
                      │                │
          distributions.py             │         sources.yaml   materials.yaml
          sources.py, materials.py ◄───┤              (packaged reference data)
                                       ▼
                               montecarlo.py
              ┌──────────────────────────────────────────────┐
              │ _draw():   one independent RNG stream/input  │
              │ _evaluate(): dose.py  ──►  shielding.py      │  same code path for the
              │              rate = Γ A / d²   T per model   │  nominal run (scalars)
              │              dose = rate · T · duration      │  and the MC run (arrays)
              └──────────────────────────┬───────────────────┘
                                         ▼
                                 SimulationResult
                      ┌──────────────────┼──────────────────┐
                      ▼                  ▼                  ▼
               sensitivity.py        risk.py           plotting.py
                      └──────────┬───────┘
                                 ▼
                             report.py ──► cli.py (text / JSON / PNG)
```

## Layers and their rules

| Layer | Modules | Rule |
|---|---|---|
| Reference data | `data/*.yaml`, `sources.py`, `materials.py` | Data is YAML, never hard-coded. Lookups fail loudly (unknown name, out of range). |
| Physics | `dose.py`, `shielding.py` | Pure, vectorised functions over numpy arrays. No I/O, no randomness, no scenario knowledge. |
| Uncertainty | `distributions.py`, `montecarlo.py` | The only place randomness lives. |
| Scenario I/O | `scenario.py` | The only place YAML is parsed and validated. Everything downstream may assume a valid `Scenario`. |
| Analysis | `sensitivity.py`, `risk.py` | Pure functions of a `SimulationResult` / dose array. |
| Presentation | `report.py`, `plotting.py`, `cli.py` | No physics. `plotting` is the only module needing matplotlib and is imported lazily. |

Dependencies point downward only. The core install needs just `numpy` and `pyyaml`.

## Key design decisions

1. **One evaluation path for nominal and Monte Carlo.** `_evaluate` takes scalars or
   arrays, so the deterministic result and the Monte Carlo result cannot drift apart.
2. **Common random numbers across models.** Models are compared on identical samples;
   see [uncertainty.md](uncertainty.md#sampling-mechanics).
3. **Stable RNG streams.** One `SeedSequence.spawn` stream per input in fixed order;
   reproducible and insensitive to adding models.
4. **Model-form uncertainty is exposed, not averaged.** Multiple shielding models
   side by side; no blended "best" answer.
5. **Strict scenario validation.** A typo in a safety-adjacent input file should be an
   error, never silently ignored (`dosage:` vs `duration_h:`).
6. **Honest data.** Reference values live in YAML with provenance notes and are
   flagged illustrative where they are not primary-source values.
7. **Units in names** (`distance_m`, `thickness_cm`, `duration_h`, `activity_mbq`,
   `dose_usv`) so unit mistakes are visible at the call site.

## Extension points

| To add… | Do this |
|---|---|
| a nuclide | add an entry to `data/sources.yaml` (needs a gamma constant, uncertainty, effective energy) |
| a material or energies | add rows to `data/materials.yaml` (NIST XCOM) |
| a shielding model | subclass `ShieldingModel` in `shielding.py`, implement `transmission(layers)`, add it to `MODELS`; it is then selectable in scenario files and appears in every report |
| a distribution | add a frozen dataclass with `sample`/`nominal` in `distributions.py` and a branch in `from_spec` |
| an output format | consume `build_report(result)`, a plain dict |

## Testing strategy

* **Physics invariants**: inverse-square scaling, Beer–Lambert, `T(0)=1`, monotonicity,
  buildup ≥ narrow beam, HVL halving.
* **Data**: NIST round-trip values, interpolation bracketing, no extrapolation.
* **Statistics against closed forms**: `Var[ln D]` for log-normal power-law inputs; μ-uncertainty
  amplification `σ[ln T] ∝ x`.
* **Engine properties**: seed reproducibility, stream stability, paired-sample ratios.
* **Validation**: every error path in the scenario loader, and every bundled scenario loads.
* **CLI**: end-to-end runs, JSON output, clean error exit codes.
