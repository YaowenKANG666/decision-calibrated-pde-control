# Preliminary paired-factorial results

These results are mechanism evidence, not a final statistical study.

For each of three training/evaluation seeds, the same initial field, viscosity,
and boundary conditions were evaluated at actuator gains
`[-0.5, 0.0, 0.5, 1.0, 1.5]`. The gain is hidden from the controller.

| Seed | ID L2 coverage | Audit ellipsoid coverage | Nominal mean | Adversarial mean | Mean change | Nominal p90 | Adversarial p90 | p90 change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.283 | 0.922 | 0.4244 | 0.4261 | +0.41% | 0.5693 | 0.5483 | -3.68% |
| 1 | 0.206 | 0.950 | 0.1799 | 0.1791 | -0.41% | 0.1832 | 0.1803 | -1.59% |
| 2 | 0.222 | 0.878 | 2.5504 | 2.5543 | +0.15% | 2.8257 | 2.7814 | -1.57% |

Across seeds, the mean of the paired relative p90 changes is `-2.28%`, while
the mean of the relative mean-cost changes is `+0.05%`. The audit-L2 tube
improves p90 by `1.74%` on the same calculation.

Interpretation: the nonlinear ellipsoidal adversary reduced upper-tail cost in
all three initial states with negligible average change in mean cost. The
sample is far too small for a population claim. Next experiments must add more
initial states, confidence intervals, audit-size sweeps, and stronger inner
maximization checks.
