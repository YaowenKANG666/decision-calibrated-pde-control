# Release data audit

Overall status: **PASS**.

This audit recomputes reported summaries from the saved JSON and CSV source files.
It checks internal numerical consistency; it is not a cryptographic provenance proof.

## Numerical checks

| Check | Status | Detail |
|---|---|---|
| Burgers backbone is FNO | PASS | fno |
| FNO architecture recorded | PASS | {"layers": 4, "modes": 14, "width": 32} |
| FNO pair parameter count is positive | PASS | parameters=240454 |
| uncontrolled: control shape | PASS | costs=(24,), actions=(24, 20) |
| uncontrolled: finite values | PASS | costs and actions are finite |
| uncontrolled: mean | PASS | observed=0.694795177512, stored=0.694795177512, abs_diff=0 |
| uncontrolled: standard deviation | PASS | observed=0, stored=0, abs_diff=0 |
| uncontrolled: p90 | PASS | observed=0.694795177512, stored=0.694795177512, abs_diff=0 |
| uncontrolled: mean absolute action | PASS | observed=0, stored=0, abs_diff=0 |
| nominal_mpc: control shape | PASS | costs=(24,), actions=(24, 20) |
| nominal_mpc: finite values | PASS | costs and actions are finite |
| nominal_mpc: mean | PASS | observed=0.579732690873, stored=0.579732690873, abs_diff=0 |
| nominal_mpc: standard deviation | PASS | observed=0.319327319755, stored=0.319327319755, abs_diff=0 |
| nominal_mpc: p90 | PASS | observed=1.07517036468, stored=1.07517036468, abs_diff=0 |
| nominal_mpc: mean absolute action | PASS | observed=1.27825014907, stored=1.27825014907, abs_diff=0 |
| id_l2_robust_mpc: control shape | PASS | costs=(24,), actions=(24, 20) |
| id_l2_robust_mpc: finite values | PASS | costs and actions are finite |
| id_l2_robust_mpc: mean | PASS | observed=0.5723875722, stored=0.5723875722, abs_diff=0 |
| id_l2_robust_mpc: standard deviation | PASS | observed=0.294196003046, stored=0.294196003046, abs_diff=0 |
| id_l2_robust_mpc: p90 | PASS | observed=1.02501602405, stored=1.02501602405, abs_diff=0 |
| id_l2_robust_mpc: mean absolute action | PASS | observed=1.18544519758, stored=1.18544519758, abs_diff=0 |
| audit_l2_robust_mpc: control shape | PASS | costs=(24,), actions=(24, 20) |
| audit_l2_robust_mpc: finite values | PASS | costs and actions are finite |
| audit_l2_robust_mpc: mean | PASS | observed=0.568603518224, stored=0.568603518224, abs_diff=0 |
| audit_l2_robust_mpc: standard deviation | PASS | observed=0.275854376336, stored=0.275854376336, abs_diff=0 |
| audit_l2_robust_mpc: p90 | PASS | observed=0.99143293983, stored=0.99143293983, abs_diff=0 |
| audit_l2_robust_mpc: mean absolute action | PASS | observed=1.11521384622, stored=1.11521384622, abs_diff=0 |
| adjoint_robust_mpc: control shape | PASS | costs=(24,), actions=(24, 20) |
| adjoint_robust_mpc: finite values | PASS | costs and actions are finite |
| adjoint_robust_mpc: mean | PASS | observed=0.574650360689, stored=0.574650360689, abs_diff=0 |
| adjoint_robust_mpc: standard deviation | PASS | observed=0.293606204447, stored=0.293606204447, abs_diff=0 |
| adjoint_robust_mpc: p90 | PASS | observed=1.02009089233, stored=1.02009089233, abs_diff=0 |
| adjoint_robust_mpc: mean absolute action | PASS | observed=1.2298369652, stored=1.2298369652, abs_diff=0 |
| box_adjoint_robust_mpc: control shape | PASS | costs=(24,), actions=(24, 20) |
| box_adjoint_robust_mpc: finite values | PASS | costs and actions are finite |
| box_adjoint_robust_mpc: mean | PASS | observed=0.568792174581, stored=0.568792174581, abs_diff=0 |
| box_adjoint_robust_mpc: standard deviation | PASS | observed=0.258394595494, stored=0.258394595494, abs_diff=0 |
| box_adjoint_robust_mpc: p90 | PASS | observed=0.958605546137, stored=0.958605546137, abs_diff=0 |
| box_adjoint_robust_mpc: mean absolute action | PASS | observed=1.10068324311, stored=1.10068324311, abs_diff=0 |
| adversarial_robust_mpc: control shape | PASS | costs=(24,), actions=(24, 20) |
| adversarial_robust_mpc: finite values | PASS | costs and actions are finite |
| adversarial_robust_mpc: mean | PASS | observed=0.576382804805, stored=0.576382804805, abs_diff=0 |
| adversarial_robust_mpc: standard deviation | PASS | observed=0.310360559833, stored=0.310360559833, abs_diff=0 |
| adversarial_robust_mpc: p90 | PASS | observed=1.05323955469, stored=1.05323955469, abs_diff=0 |
| adversarial_robust_mpc: mean absolute action | PASS | observed=1.26234031308, stored=1.26234031308, abs_diff=0 |
| Matched control cases share viscosity | PASS | unique_values=1 |
| Matched control cases share left_boundary | PASS | unique_values=1 |
| Matched control cases share right_boundary | PASS | unique_values=1 |
| Actuator-gain sweep is strictly ordered | PASS | n=24, range=[-0.5, 1.5] |
| Burgers combined shift id_l2: integer denominator | PASS | 593/800=0.74125 |
| Burgers combined shift audit_l2: integer denominator | PASS | 707/800=0.88375 |
| Burgers combined shift audit_ellipsoid: integer denominator | PASS | 725/800=0.90625 |
| Burgers combined shift audit_simultaneous_box: integer denominator | PASS | 725/800=0.90625 |
| Burgers behavior trajectory: integer denominator | PASS | 352/400=0.88 |
| Value-bound test row count | PASS | rows=160, stored=160 |
| Value bound global: coverage | PASS | observed=0.9375, stored=0.9375, abs_diff=0 |
| Value bound global: mean_bound | PASS | observed=0.0987066784424, stored=0.0987066784424, abs_diff=0 |
| Value bound global: median_utilization | PASS | observed=0.209975606593, stored=0.209975606593, abs_diff=0 |
| Value bound global: p90_utilization | PASS | observed=0.687853085156, stored=0.687853085156, abs_diff=0 |
| Value bound global: max_utilization | PASS | observed=3.9191907918, stored=3.9191907918, abs_diff=0 |
| Value bound global: mean_value_gap | PASS | observed=0.0343359909109, stored=0.0343359909109, abs_diff=0 |
| Value bound global: integer denominator | PASS | 150/160=0.9375 |
| Value bound local: coverage | PASS | observed=0.94375, stored=0.94375, abs_diff=0 |
| Value bound local: mean_bound | PASS | observed=0.0870352640208, stored=0.0870352640208, abs_diff=0 |
| Value bound local: median_utilization | PASS | observed=0.321788145177, stored=0.321788145177, abs_diff=0 |
| Value bound local: p90_utilization | PASS | observed=0.787550700584, stored=0.787550700584, abs_diff=0 |
| Value bound local: max_utilization | PASS | observed=2.18986193517, stored=2.18986193517, abs_diff=0 |
| Value bound local: mean_value_gap | PASS | observed=0.0343359909109, stored=0.0343359909109, abs_diff=0 |
| Value bound local: integer denominator | PASS | 151/160=0.94375 |
| Value bound adjoint: coverage | PASS | observed=0.925, stored=0.925, abs_diff=0 |
| Value bound adjoint: mean_bound | PASS | observed=0.203451630005, stored=0.203451630005, abs_diff=0 |
| Value bound adjoint: median_utilization | PASS | observed=0.264926920255, stored=0.264926920255, abs_diff=0 |
| Value bound adjoint: p90_utilization | PASS | observed=0.85400200327, stored=0.85400200327, abs_diff=0 |
| Value bound adjoint: max_utilization | PASS | observed=1.76645750397, stored=1.76645750397, abs_diff=0 |
| Value bound adjoint: mean_value_gap | PASS | observed=0.0343359909109, stored=0.0343359909109, abs_diff=0 |
| Value bound adjoint: integer denominator | PASS | 148/160=0.925 |
| Value bound adjoint_curvature: coverage | PASS | observed=0.925, stored=0.925, abs_diff=0 |
| Value bound adjoint_curvature: mean_bound | PASS | observed=0.203451630005, stored=0.203451630005, abs_diff=0 |
| Value bound adjoint_curvature: median_utilization | PASS | observed=0.264926920255, stored=0.264926920255, abs_diff=0 |
| Value bound adjoint_curvature: p90_utilization | PASS | observed=0.85400200327, stored=0.85400200327, abs_diff=0 |
| Value bound adjoint_curvature: max_utilization | PASS | observed=1.76645750397, stored=1.76645750397, abs_diff=0 |
| Value bound adjoint_curvature: mean_value_gap | PASS | observed=0.0343359909109, stored=0.0343359909109, abs_diff=0 |
| Value bound adjoint_curvature: integer denominator | PASS | 148/160=0.925 |
| Value-gap epsilon slope | PASS | observed=1, stored=1, abs_diff=0 |
| Value-gap effective-horizon slope | PASS | observed=2, stored=2, abs_diff=8.88e-16 |
| Value-gap maximum numerical error | PASS | observed=2.55351295664e-15, stored=2.55351295664e-15, abs_diff=0 |
| NS2D metric is named RMSE | PASS | sqrt(mean((prediction-target)^2)) over all output coordinates, averaged over test samples |
| NS2D split sizes are positive | PASS | train/validation/audit/test=800/50/200/300 |
| NS2D training history length | PASS | epochs=20 |
| NS2D training history is finite and nonnegative | PASS | all stored train/validation MSE values checked |
| NS2D 90% curve coverage | PASS | observed=0.883333333333, stored=0.883333333333, abs_diff=0 |
| NS2D 90% quantile | PASS | observed=30.6790180206, stored=30.6790180206, abs_diff=0 |
| NS2D 90% full width | PASS | observed=4.34088710002, stored=4.3408870697, abs_diff=3.03e-08 |
| NS2D simultaneous field: integer denominator | PASS | 265/300=0.883333333333 |

## Coverage denominators and sampling uncertainty

The intervals below are descriptive 95% Wilson intervals for the realized test
proportions. They are not replacements for the split-conformal theorem.

| Event | Count | Observed | Target | Wilson 95% | Target inside |
|---|---:|---:|---:|---:|---|
| Burgers combined shift id_l2 | 593/800 | 0.7412 | 0.90 | [0.7098, 0.7704] | no |
| Burgers combined shift audit_l2 | 707/800 | 0.8838 | 0.90 | [0.8597, 0.9041] | yes |
| Burgers combined shift audit_ellipsoid | 725/800 | 0.9062 | 0.90 | [0.8841, 0.9246] | yes |
| Burgers combined shift audit_simultaneous_box | 725/800 | 0.9062 | 0.90 | [0.8841, 0.9246] | yes |
| Burgers behavior trajectory | 352/400 | 0.8800 | 0.90 | [0.8445, 0.9083] | yes |
| Value bound global | 150/160 | 0.9375 | 0.90 | [0.8888, 0.9657] | yes |
| Value bound local | 151/160 | 0.9437 | 0.90 | [0.8966, 0.9701] | yes |
| Value bound adjoint | 148/160 | 0.9250 | 0.90 | [0.8735, 0.9566] | yes |
| Value bound adjoint_curvature | 148/160 | 0.9250 | 0.90 | [0.8735, 0.9566] | yes |
| NS2D simultaneous field | 265/300 | 0.8833 | 0.90 | [0.8421, 0.9149] | yes |

## Metric correction

The NS2D value 0.163472 is a mean field RMSE, not a relative L2 error.
The implementation, manuscript, README, and stored metric key now use the RMSE label.
No numerical value was changed by this semantic correction.

## Scope

The Burgers controller table contains 24 matched actuator-gain cases from one
trained FNO pair. The NS2D result also conditions on one trained FNO pair. These
checks support internal consistency and mechanism-level interpretation, not
population-level uncertainty across training seeds.
