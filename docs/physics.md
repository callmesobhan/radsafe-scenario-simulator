# Physics, assumptions and validity

RadSafe implements the classical *time, distance, shielding* picture of external
photon exposure from a sealed source. This page states every equation, every
assumption behind it, and what is outside the model.

## Quantity and units

The dose quantity is **ambient dose equivalent, H\*(10)**, in µSv (reported in mSv).
For external photon fields H\*(10) is a standard operational quantity that generally
*overestimates* effective dose, so it is a sensible protection-side surrogate.
Internally: activity in MBq, distance in m, thickness in cm, duration in h, dose rate
in µSv/h, dose in µSv.

## Equations

**1. Unshielded dose rate (point source, inverse-square law)**

```
Ḣ₀ = Γ · A / d²                       [µSv/h]
```

`Γ` is the H\*(10) rate constant in µSv·m²/(MBq·h) (`sources.yaml`), `A` the
activity and `d` the distance.

**2. Shielded dose rate**

```
Ḣ = Ḣ₀ · T          D = Ḣ · Δt          [µSv]
```

`T` is the dimensionless transmission, `0 < T ≤ 1`, from one of three models.
Optical depth for a stack of layers is `τ = Σᵢ μᵢ xᵢ` (number of mean free paths).

| Model | `T` | Character |
|---|---|---|
| `narrow_beam` | `exp(-τ)` | Counts only photons that cross the shield without interacting. **Optimistic**: ignores scattered photons that still reach the receptor. A lower bound on `T`. |
| `linear_buildup` | `(1+τ)·exp(-τ)` | Crude broad-beam correction with buildup factor `B = 1+τ`. Typically *overstates* buildup in high-Z shields (lead) and can *understate* it in low-Z shields (water, concrete) at many mean free paths. Neither a bound nor an estimate; a middle scenario. |
| `practical_hvl` | `2^-(Σᵢ xᵢ/HVLᵢ)` | Uses user-supplied *practical* (broad-beam, empirical) half-value layers, e.g. from NCRP 151 or vendor data. Only as good as the numbers you give it. |

`μᵢ = (μ/ρ)ᵢ · ρᵢ`, with total mass attenuation coefficient `μ/ρ` from NIST XCOM
(coherent scattering included), log-log interpolated in energy, evaluated at the
source's **effective energy** (Cs-137: 0.662 MeV; Co-60: 1.25 MeV, the mean of its two
gammas). Out-of-range energies raise an error rather than extrapolating.

Why three models and no "best" one: real buildup depends on material, energy, geometry
and depth, and a rigorous answer needs ANSI/ANS-6.4.3 buildup data or full transport
(MCNP, Geant4, MicroShield-class codes). The point of the simplified models is to show
how large the spread between reasonable assumptions is. In the bundled examples the
median dose differs between models by a factor of about 2–5 (3 cm of lead, Cs-137) and
9–14 (1 m of concrete, Co-60); the gap widens with shield thickness.

**3. Multi-layer shields** multiply per-layer transmissions (sum optical depths; sum
halvings for HVL). This ignores layer-order effects (scatter from one layer into the next,
spectrum hardening, and k-edge effects in high-Z/low-Z sandwiches).

## Validity conditions

The unshielded formula holds when the source is small compared to `d` (the code warns if
the nominal distance is below 0.1 m), air attenuation is negligible (true for MeV photons
over metres), and the receptor is not so close that scatter from nearby structures
dominates. Slab shields are assumed laterally infinite and the beam normal to them. For
an oblique path through a slab of thickness `x` at angle θ, use `x/cos θ`.

## What is not modelled

* **Source decay.** Activity is constant over the exposure period. If the activity you
  enter is the start-of-year value, a full year of exposure is overstated by about 6% for
  Co-60 (half-life 5.27 y; the source has lost ~12% by year end) and about 1% for Cs-137
  (30.1 y; ~2% lost). Negligible for hours or days.
* **Scatter and skyshine**: room walls, floor, ceiling, and over-the-roof paths.
* **Polyenergetic sources**: Ir-192, Se-75, Am-241, X-ray tubes. A single effective
  energy mis-states attenuation in thick shields because the spectrum hardens with depth.
  (Ir-192 is deliberately not in `sources.yaml` for this reason.)
* **Finite source size, self-shielding, source encapsulation, collimation.**
* **Neutrons, beta/bremsstrahlung, internal contamination.**
* **Dose to tissue/organs**: no phantom, no conversion to effective dose.
* **Occupancy and workload models**: duration is simply the total exposure time over
  the period the limit applies to. Compare against an annual limit by giving annual hours.
* **Correlated inputs**: all inputs are sampled independently (see
  [uncertainty.md](uncertainty.md)).

## Reference values used

* ICRP Publication 103 (2007): public limit 1 mSv/y; occupational 20 mSv/y averaged
  over 5 years with no more than 50 mSv in any single year; detriment-adjusted nominal
  risk coefficients 5.7×10⁻² Sv⁻¹ (whole population) and 4.2×10⁻² Sv⁻¹ (adult workers).
  The risk numbers are linear no-threshold, population-level figures for low doses and
  dose rates: **not a prediction for any individual**.
* NIST X-Ray Mass Attenuation Coefficients (XCOM) for lead, iron, concrete (ordinary),
  water.
* Gamma constants: see the notes and tolerance in `sources.yaml`.
