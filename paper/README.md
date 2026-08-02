# Working preprint

The current manuscript is
[decision_calibrated_robust_control.pdf](decision_calibrated_robust_control.pdf).

The paper now follows one argument: a frozen neural PDE world model produces a
spatial uncertainty scale, a labelled target-domain audit split calibrates a
field-valued ambiguity set, and exact support functions connect its geometry
to finite-horizon robust control decisions.

The main text contains only the persistently forced controlled-Burgers
benchmark. The current quantitative result is the task-validity gate comparing
uncontrolled dynamics with finite-budget PDE-oracle MPC on 100 independently
sampled plants. The manuscript explicitly defers learned-controller claims
until the locked five-seed evaluation is complete.

The previous NS2D stress test, projected-gradient adversary, curvature fit,
and deterministic value-bound study have been removed from the manuscript
main line. They are not used to support the present contribution.
