# Scenario file reference

Scenarios are YAML. The loader is **strict**: unknown keys, missing keys, non-physical
values and unavailable data are errors with the offending path in the message
(`shielding.layers[0].thickness_cm: ...`). `radsafe validate FILE...` checks files
without running them.

```yaml
name: <string>                         # required
description: <string>                  # optional

source:                                # required
  nuclide: Cs-137                      # built-in: see `radsafe list`
  activity_mbq: <value>                # required, > 0
  # Optional overrides / custom source. A nuclide that is not built in needs ALL THREE:
  gamma_constant_usv_m2_per_mbq_h: 0.100
  gamma_rel_unc: 0.05
  effective_energy_mev: 0.662

geometry:                              # required
  distance_m: <value>                  # > 0

exposure:                              # required
  duration_h: <value>                  # >= 0, total hours over the period the limits refer to

shielding:                             # optional; omit for an unshielded scenario
  models: [narrow_beam, linear_buildup, practical_hvl]   # default: narrow_beam, linear_buildup
  layers:                              # ordered source -> receptor (order is not used by the models)
    - material: lead                   # lead | iron | concrete | water
      thickness_cm: <value>            # >= 0
      mu_rel_unc: 0.05                 # optional, 1-sigma relative uncertainty on mu (default 0.05)
      hvl_cm: <value>                  # required for every layer iff practical_hvl is used

limits:                                # optional
  - { name: <string>, msv: <number> }  # compared against the dose over `duration_h`

simulation:                            # optional
  n_samples: 100000                    # default 100000, minimum 2
  seed: 12345                          # default 12345; null for a non-reproducible run
```

## `<value>`

A number (constant) or a mapping with `dist`:

| `dist` | keys |
|---|---|
| `constant` | `value` |
| `uniform` | `low`, `high` |
| `normal` | `mean`, one of `sd` / `rel_sd`; optional `lower`, `upper` (truncation) |
| `lognormal` | `median`, one of `gsd` / `rel_sd` (`rel_sd` is the coefficient of variation) |
| `triangular` | `low`, `mode`, `high` |

For `normal`, quantities that cannot be negative (activity, distance, duration, thickness,
HVL) get `lower: 0` unless you set one. Numeric strings such as `"3.7e6"` are accepted:
PyYAML reads `3.7e6` (no sign on the exponent) as a string, so this is handled for you.

## Built-in data

`radsafe list` prints it. Sources: Cs-137, Co-60. Materials: lead, iron, concrete
(ordinary, 2.30 g/cm³), water, tabulated 0.5–2 MeV. Add rows to
`src/radsafe/data/*.yaml` to extend; lookups refuse to extrapolate.

## Output

`radsafe run` prints a text report; `--json PATH` writes the same content as JSON
(`scenario`, `simulation`, `limits`, `results.<model>.{nominal,mean,median,p5,p95,p99}_msv`,
`exceedance`, `sensitivity`, `median_ratio_to_narrow_beam`, `nominal_detriment`).
`results` always includes an `unshielded` reference.
