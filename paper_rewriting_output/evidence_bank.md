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

No headline positive claim may use E7 until repeated seeds support it.
