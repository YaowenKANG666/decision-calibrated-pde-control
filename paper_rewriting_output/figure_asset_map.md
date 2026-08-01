# Figure asset map

| Figure | Intended message | Source |
|---|---|---|
| Fig. 1 | Prediction uncertainty becomes an adjoint-supported ambiguity set and robust MPC objective | to be drawn from method equations |
| Fig. 2 | ID calibration fails under joint shift; audit calibration restores coverage | metrics JSON |
| Fig. 3 | Closed-loop cost and tail-risk comparison | multi-seed metrics, pending |
| Fig. 4 | Same prediction error magnitude, different decision impact | controlled synthetic counterexample, pending |
| Fig. 5 | Radius, support, and error correlation along a trajectory | rollout logging, pending |

Fig. 3 cannot be finalized from a three-case smoke test.

## Standalone exports added

| Stem | Intended message | Status |
|---|---|---|
| `value_01_epsilon_scaling` | Exact linear epsilon dependence | available |
| `value_02_discount_scaling` | Sharp squared effective-horizon dependence | available |
| `value_03_burgers_gap_bound` | Controlled-Burgers gap versus recursion | available |
| `value_04_rollout_error` | Multi-step observed error versus envelope | available |
| `value_05_fno_gap_discount` | Learned-FNO gap across gamma | available |
| `value_06_fno_gap_envelope` | Observed gap versus finite-visited-set envelope | available |
| `bound_01_coverage`--`bound_07_utilization_distribution` | Coverage and tightness of four value bounds | available independent-test run |
| `ns2d_01_input`--`ns2d_11_error_scale` | Two-dimensional function-valued UQ evidence | available Colab run |

No composite panel is used in the release figure suite.
