# Figure contract

## Method-chain schematic

- **Core conclusion:** a predictive disagreement scale becomes useful only
  after deployment-audit calibration defines an ambiguity set and the same set
  is queried in decision-relevant directions.
- **Evidence chain:** clean/perturbed world models -> smoothed disagreement ->
  max/ellipsoid conformal set -> adjoint support or adversarial rollout ->
  independent closed-loop evaluation.
- **Archetype:** schematic-led single figure (not a panel composite).
- **Backend:** Python/matplotlib exclusively.
- **Export:** 183 mm by 67 mm; editable SVG/PDF text, 300 DPI PNG preview, and
  optional 600 DPI TIFF master.
- **Review risks:** the audit guarantee is marginal for one random transition;
  arrows must not imply trajectory safety or automatic control improvement.

Quantitative plots cite their own CSV/JSON source files in the figure manifest.

## Final quantitative figures

All plots use Python/matplotlib exclusively, a white background, editable
SVG/PDF text, and a 600 DPI TIFF master. Each file contains one conclusion;
multi-panel composites are not used.

### Function-space coverage--tightness frontier

- **Core conclusion:** deployment calibration restores coverage, while set
  geometry determines how much decision-relevant support is paid for it.
- **Archetype:** single quantitative comparison.
- **Evidence:** independent combined-shift coverage versus mean adjoint
  support for source L2, audit L2, anisotropic ellipsoid, and simultaneous box.
- **Reviewer risk:** an undersized set can look tight; the nominal 90% target
  and observed coverage must appear on the same plot.

### Closed-loop tail-risk comparison

- **Core conclusion:** uncertainty is useful for control only if matched-case
  p90 cost improves relative to nominal MPC without a material mean-cost loss.
- **Archetype:** single quantitative comparison with paired raw cases retained
  in source data.
- **Evidence:** p90 cost for nominal, scalar-tube, adjoint, box-adjoint, and
  adversarial controllers; the number of actuator-gain cases is stated.
- **Reviewer risk:** a p90 change from a small deterministic gain sweep is
  mechanism evidence, not a population confidence interval.

### Trajectory max-score audit

- **Core conclusion:** a separately calibrated max-over-time-and-coordinate
  score can cover an entire held-out behavior-policy rollout near its nominal
  target.
- **Archetype:** single validation plot.
- **Evidence:** held-out trajectory coverage, target line, horizon, audit and
  test trajectory counts.
- **Reviewer risk:** the plot must state that it does not certify
  counterfactual MPC trajectories.

### Value-bound coverage--conservatism frontier

- **Core conclusion:** at comparable independent-test coverage, useful bounds
  are smaller and more highly utilized than a global maximum recursion.
- **Archetype:** single quantitative frontier.
- **Evidence:** coverage and mean bound for global, local, adjoint, and
  adjoint-plus-curvature constructions, all value-calibrated on a disjoint
  split.
- **Reviewer risk:** utilization is interpreted only together with coverage.

### Scaling witness

- **Core conclusion:** the fixed-policy value gap is linear in epsilon and
  quadratic in effective horizon when the dynamics are non-expansive.
- **Archetype:** two separate log--log validation files, one exponent per file.
- **Evidence:** fitted slopes and analytic reference laws.
- **Reviewer risk:** this verifies theorem sharpness, not learned-FNO quality.

### NS2D simultaneous band

- **Core conclusion:** the calibrated max-score band achieves simultaneous
  function-field coverage on held-out official 128-by-128 pairs.
- **Archetype:** separate image plates and quantitative validation files.
- **Evidence:** target/prediction/error/half-width fields, reliability curve,
  coverage--width curve, and disagreement--error localization.
- **Reviewer risk:** the public benchmark has no action channel, so no 2D
  control claim is made.
