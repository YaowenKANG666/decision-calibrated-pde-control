# Evidence bank

| Claim ID | Candidate claim | Current evidence | Status |
|---|---|---|---|
| E1 | ID calibration can fail under joint shift | seed-0 smoke: 0.611 L2 coverage on combined shift | preliminary |
| E2 | A small shifted audit improves one-step coverage | seed-0 smoke: 0.822 L2 and 0.856 weighted coverage | preliminary |
| E3 | Coverage improvement alone need not improve control | seed-0 smoke: all first-generation robust controllers failed to beat uncontrolled | supported negative result |
| E4 | Adjoint ellipsoid support has a closed form | algebra plus numerical equality unit test | supported |
| E5 | One-step uniform error yields geometric rollout error | proof plus recurrence unit test | supported under assumptions |
| E6 | Value gap is order epsilon/(1-gamma)^2 | deterministic proof sketch | conditional; policy assumptions need tightening |
| E7 | Nonlinear ellipsoid MPC improves paired upper-tail cost | three seeds, five paired gain levels per seed; p90 improves in all seeds by 1.57–3.68% | preliminary mechanism evidence |
| E8 | Tail improvement does not require a material mean-cost sacrifice | mean relative change averaged across seeds is +0.05% | preliminary mechanism evidence |
| E9 | Sharp value-gap witness is linear in epsilon | six epsilon levels; fitted log-log slope 1.0000000000000004 | exact numerical verification |
| E10 | Sharp value-gap witness scales quadratically with effective horizon after dividing by gamma times epsilon | six gamma levels; fitted log-log slope 1.9999999999999987 | exact numerical verification |
| E11 | Local recursion can be less conservative than the global maximum | 60 calibration/160 independent test trajectories: mean bounds 0.0870 versus 0.0987 | independent-test evidence |
| E12 | Adjoint value bound reaches approximately matched test coverage | independent-test coverage 0.925; mean bound 0.2035 | independent-test evidence; not the tightest bound |
| E13 | Curvature feature improves the adjoint bound | fitted coefficient was zero; result identical to adjoint | unsupported/negative result |
| E14 | NS2D evaluates controlled dynamics | official data contain vorticity pairs but no action | forbidden claim |
| E15 | NS2D max-score band reaches nominal 90% simultaneous coverage | 300 independent tests give 0.883 coverage and mean full width 4.3409 | not supported at the realized sample proportion; report exactly |
| E16 | Perturbed-model disagreement accurately localizes NS2D error | pointwise scale-error Pearson r is 0.169 | weak support; motivates a better uncertainty mechanism |

No headline positive claim may use E7 until repeated seeds support it.
