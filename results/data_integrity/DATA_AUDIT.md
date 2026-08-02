# Public result audit

**Status: PASS**

| Check | Status | Detail |
|---|---|---|
| Independent test-case count | PASS | rows=100, stored=100 |
| Unique case identifiers | PASS | unique=100 |
| All released values are finite | PASS | costs and efforts |
| uncontrolled: mean cost | PASS | observed=0.589950965608, stored=0.589950965608, abs_diff=0 |
| uncontrolled: median cost | PASS | observed=0.524921133919, stored=0.524921133919, abs_diff=0 |
| uncontrolled: p90 cost | PASS | observed=1.05992130827, stored=1.05992130827, abs_diff=0 |
| uncontrolled: mean control effort | PASS | observed=0, stored=0, abs_diff=0 |
| uncontrolled: failure rate | PASS | observed=0, stored=0, abs_diff=0 |
| pde_oracle_mpc: mean cost | PASS | observed=0.161033476306, stored=0.161033476306, abs_diff=0 |
| pde_oracle_mpc: median cost | PASS | observed=0.137744308803, stored=0.137744308803, abs_diff=0 |
| pde_oracle_mpc: p90 cost | PASS | observed=0.2874916385, stored=0.2874916385, abs_diff=0 |
| pde_oracle_mpc: mean control effort | PASS | observed=1.13882184946, stored=1.13882184946, abs_diff=0 |
| pde_oracle_mpc: failure rate | PASS | observed=0, stored=0, abs_diff=0 |
| Paired mean difference | PASS | observed=-0.428917489303, stored=-0.428917489303, abs_diff=0 |
| Paired p90 difference | PASS | observed=-0.77242966977, stored=-0.77242966977, abs_diff=0 |
| Fraction oracle better | PASS | observed=1, stored=1, abs_diff=0 |
| Standard-budget first-20 mean | PASS | observed=0.166242967407, stored=0.166242967407, abs_diff=2.78e-17 |
| Stored high-budget audit improves the standard budget | PASS | high-budget case-level outputs are not redistributed; this check validates the stored aggregate only |

## Scope

The audit recomputes the headline costs from 100 released case-level rows.
It checks the paired design and the stored task-validity summary.
It does not certify experiment provenance, FNO performance, conformal
coverage, or global optimality of the finite-budget CEM planner.
