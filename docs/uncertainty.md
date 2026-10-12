# Uncertainty propagation

## Two kinds of uncertainty, treated differently

| Kind | Example | How RadSafe handles it |
|---|---|---|
| **Parameter uncertainty** | where the worker stands, how long they stay, source activity, shield thickness, μ | **Sampled** (Monte Carlo) |
| **Model-form uncertainty** | narrow-beam vs buildup vs empirical HVL | **Not sampled**: every model is run on the *same* samples and shown side by side |

Mixing the two into one distribution would hide the fact that model choice is a
judgement, not a random draw. Keeping them separate lets you say "under assumption X
the exceedance probability is p₁, under Y it is p₂".

## Input distributions

| Input | Typical choice | Why |
|---|---|---|
| Activity | log-normal (`gsd` ≈ certificate tolerance) | positive, multiplicative error |
| Distance | normal, truncated at a lower bound | measured/positional spread; truncation avoids ≈0 m |
| Duration | triangular (low / mode / high) | expert judgement with limited data |
| Shield thickness | normal, truncated at 0 | manufacturing/installation tolerance |
| μ (per layer) | log-normal, `mu_rel_unc` (default 5%) | data, density and composition spread |
| Γ (gamma constant) | log-normal, `gamma_rel_unc` from `sources.yaml` | spread between tabulations |

Available kinds: `constant`, `uniform`, `normal` (with `lower`/`upper`), `lognormal`
(`gsd` or `rel_sd`), `triangular`. `normal` accepts `sd` or `rel_sd`.

> Normal distributions on distance have unbounded probability near zero unless you set
> `lower`; `1/d²` makes that tail dominate the mean. The scenario loader applies
> `lower: 0` by default and you should set a physically meaningful bound.

## Why shielding dominates thick-shield uncertainty

For the narrow-beam model `ln T = -μx`, so

```
σ[ln T] ≈ x · σ[μ]    (when μ is the uncertain quantity)
```

A 5% error in μ at 13 mean free paths is a ±0.65 error in ln T, a factor of ~1.9 in dose,
far more than the same 5% in activity. Thicker shield ⇒ wider dose distribution. The test
suite checks this scaling. It is also why the Co-60/concrete example attributes 60%+ of
the variance to μ while the 3 cm lead example is dominated by distance.

## Sampling mechanics

* One `numpy.random.SeedSequence(seed)` is spawned into one independent stream per
  input, in a **fixed order**, whether or not that input is constant. Consequences:
  results are reproducible for a seed; adding a shielding model, or changing which inputs
  are constant, does not change the random numbers any other input receives.
* **Common random numbers**: all shielding models consume the same input samples, so
  per-sample ratios between models are exact (e.g. `linear_buildup / narrow_beam = 1+τ`
  sample-by-sample) and differences in exceedance probability are not sampling noise.
* Inputs are sampled **independently**. If thickness and distance are physically coupled
  (a thicker door pushes the worker further away) you need a joint distribution; see the
  roadmap.

## Reading the outputs

* **Percentiles (P5/P95/P99)** and mean/median describe the dose distribution due to
  input uncertainty. For heavy-tailed (log-normal-ish) doses the mean exceeds the median.
* **Exceedance probability** `P(dose > limit)` is a count of samples above the limit. Its
  95% interval (Wilson) quantifies **Monte Carlo sampling error only**. It says nothing
  about model-form uncertainty or whether the input distributions are right. Roughly,
  the standard error is `√(p(1-p)/n)`: use ≥10⁵ samples for probabilities near 10⁻³, and
  treat `0 (<x%)` as "not observed in n samples", not "impossible".
* **Sensitivity** is the Spearman rank correlation of each input with dose, reported as
  the normalised share of ρ². It is a cheap *ranking* of which inputs matter, valid for
  monotone input→dose relationships (true here). It is not an exact variance
  decomposition; Sobol indices are on the roadmap. A model that ignores an input shows
  ≈0 for it (e.g. `practical_hvl` ignores μ).
* **Nominal** column: dose with every input at its nominal value (mean/median/mode/
  midpoint by distribution). Nominal ≠ median in general: compare them to see how
  skewed the input uncertainty makes the answer.
