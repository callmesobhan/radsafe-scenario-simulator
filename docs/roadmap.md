# Roadmap

Roughly in order of value for the effort. None of this is implemented yet.

## Physics fidelity

- [ ] **Tabulated buildup factors** (ANS-6.4.3 / Taylor or Berger fits) as a fourth shielding
      model, so the simplified models can be compared against a physically grounded one.
- [ ] **Polyenergetic sources** (Ir-192, Se-75, Am-241, X-ray spectra): multi-line
      emission with per-line attenuation; add Ir-192 once this exists.
- [ ] **Source decay** over the exposure period (decay-corrected activity).
- [ ] **Oblique incidence** and finite-size slab geometry.
- [ ] **Two-component TVL model** (`TVL₁`, `TVLₑ`, NCRP 151 style) in place of the single practical HVL.

## Uncertainty analysis

- [ ] **Sobol / variance-based sensitivity** (first-order and total), replacing the Spearman
      ranking where input interactions matter.
- [ ] **Latin hypercube / quasi-Monte Carlo** sampling for faster convergence of rare-event
      probabilities.
- [ ] **Correlated inputs** (copulas or explicit joint distributions).
- [ ] **Model-form weights**: optional user-chosen weights over shielding models to produce a
      mixture distribution, kept explicit and opt-in.
- [ ] **Bayesian updating** of activity/geometry from measured dose rates.

## Scenarios and risk

- [ ] **Multi-source and multi-receptor** scenarios; occupancy/workload schedules.
- [ ] **ALARA what-if**: sweep distance / thickness / time and report the cheapest change that
      brings P(exceed) under a target.
- [ ] Optional **effective-dose conversion** coefficients (ICRP 74/116) instead of reporting H\*(10).

## Validation

- [ ] Benchmarks against published point-kernel results and Monte Carlo transport (MCNP/Geant4)
      for a few canonical geometries; record agreement and known disagreements in `docs/`.
- [ ] Independent review of the reference data by a health physicist.

## Packaging / UX

- [ ] Jupyter notebook walkthrough (`notebooks/`).
- [ ] `radsafe sweep` subcommand wrapping `examples/thickness_sweep.py`.
- [ ] Publish to PyPI; add a changelog and release workflow.
