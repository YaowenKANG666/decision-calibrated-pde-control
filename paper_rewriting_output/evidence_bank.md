# Evidence bank

| Claim ID | Candidate claim | Current evidence | Status |
|---|---|---|---|
| E1 | ID calibration can fail under joint shift | seed-0 smoke: 0.611 L2 coverage on combined shift | preliminary |
| E2 | A small shifted audit improves one-step coverage | seed-0 smoke: 0.822 L2 and 0.856 weighted coverage | preliminary |
| E3 | Coverage improvement alone need not improve control | seed-0 smoke: all first-generation robust controllers failed to beat uncontrolled | supported negative result |
| E4 | Adjoint ellipsoid support has a closed form | algebra plus numerical equality unit test | supported |
| E5 | One-step uniform error yields geometric rollout error | proof plus recurrence unit test | supported under assumptions |
| E6 | Value gap is order epsilon/(1-gamma)^2 | deterministic proof sketch | conditional; policy assumptions need tightening |
| E7 | Calibrated geometry changes closed-loop tail cost | final seed-27 run, 24 matched actuator gains: box-adjoint p90 is 10.84% below nominal | supported mechanism evidence; one controlled sweep |
| E8 | The tail improvement does not require a mean-cost sacrifice in the same sweep | box-adjoint mean is 1.89% below nominal; audit-L2 mean is 1.92% below nominal | supported within the tested sweep |
| E9 | Sharp value-gap witness is linear in epsilon | six epsilon levels; fitted log-log slope 1.0000000000000004 | exact numerical verification |
| E10 | Sharp value-gap witness scales quadratically with effective horizon after dividing by gamma times epsilon | six gamma levels; fitted log-log slope 1.9999999999999987 | exact numerical verification |
| E11 | Local recursion can be less conservative than the global maximum | 60 calibration/160 independent test trajectories: mean bounds 0.0870 versus 0.0987 | independent-test evidence |
| E12 | Adjoint value bound reaches approximately matched test coverage | independent-test coverage 0.925; mean bound 0.2035 | independent-test evidence; not the tightest bound |
| E13 | Curvature feature improves the adjoint bound | fitted coefficient was zero; result identical to adjoint | unsupported/negative result |
| E14 | NS2D evaluates controlled dynamics | official data contain vorticity pairs but no action | forbidden claim |
| E15 | NS2D max-score band reaches nominal 90% simultaneous coverage | 300 independent tests give 0.883 coverage and mean full width 4.3409 | not supported at the realized sample proportion; report exactly |
| E16 | Perturbed-model disagreement accurately localizes NS2D error | pointwise scale-error Pearson r is 0.169 | weak support; motivates a better uncertainty mechanism |

| E17 | Deployment audit restores combined-shift field coverage | 800 independent transitions: source L2 0.741; audit ellipsoid and box 0.906 | independent-test evidence |
| E18 | Box and ellipsoid with equal realized field coverage have different decision support | both cover 0.906; mean support 0.04422 versus 0.02012 | independent-test geometry evidence |

E7 may support only a scoped mechanism claim until repeated training and
control seeds establish uncertainty across runs.
