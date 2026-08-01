# Final result summary

Generated directly from stored JSON files. Smaller closed-loop cost is better.

## Controlled Burgers

### Function-space ambiguity sets under combined shift

| Set | Coverage | Mean radius | Decision coverage | Mean decision support |
|---|---|---|---|---|
| id_l2 | 0.741 | 0.0114 | 0.820 | 0.0065 |
| audit_l2 | 0.884 | 0.0188 | 0.934 | 0.0107 |
| audit_ellipsoid | 0.906 | 0.0324 | 0.986 | 0.0201 |
| audit_simultaneous_box | 0.906 | 0.2273 | 1.000 | 0.0442 |

### Closed-loop control

| Method | Mean cost | p90 cost | Mean change (%) | p90 change (%) |
|---|---|---|---|---|
| uncontrolled | 0.6948 | 0.6948 | +19.85 | -35.38 |
| nominal_mpc | 0.5797 | 1.0752 | +0.00 | +0.00 |
| id_l2_robust_mpc | 0.5724 | 1.0250 | -1.27 | -4.66 |
| audit_l2_robust_mpc | 0.5686 | 0.9914 | -1.92 | -7.79 |
| adjoint_robust_mpc | 0.5747 | 1.0201 | -0.88 | -5.12 |
| box_adjoint_robust_mpc | 0.5688 | 0.9586 | -1.89 | -10.84 |
| adversarial_robust_mpc | 0.5764 | 1.0532 | -0.58 | -2.04 |

Trajectory max-score coverage: 0.880 at horizon 10.

## Independent value-bound comparison

| Method | Coverage | Mean bound | Median utilization | p90 utilization | Max utilization |
|---|---|---|---|---|---|
| Global max | 0.938 | 0.09871 | 0.210 | 0.688 | 3.919 |
| Local recursion | 0.944 | 0.08704 | 0.322 | 0.788 | 2.190 |
| Adjoint support | 0.925 | 0.20345 | 0.265 | 0.854 | 1.766 |
| Adjoint + curvature | 0.925 | 0.20345 | 0.265 | 0.854 | 1.766 |

## Official NS2D function-valued benchmark

- Simultaneous test coverage: 0.883
- Mean field RMSE: 0.163472
- Mean full band width: 4.340887
- Pointwise scale-error Pearson r: 0.169
- Proper train/audit/test sizes: 800/200/300
- Scope: function-valued uncertainty only; the public pairs have no action channel.
