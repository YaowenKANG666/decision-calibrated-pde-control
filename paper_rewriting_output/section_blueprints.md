# Section blueprints

## Abstract

Open with the predictive-versus-decision calibration mismatch. Define the
adjoint score and robust support in one sentence. State only verified theory.
Leave quantitative superiority as a placeholder until multi-seed results pass.

## 1 Introduction

Paragraph 1: learned dynamics make PDE control computationally feasible but
errors become consequential under shift. Paragraph 2: predictive coverage
weights irrelevant and relevant function-space directions alike. Paragraph 3:
neighboring conformal robust control and decision-focused learning. Paragraph
4: contributions and explicit scope.

## 2 Problem

Controlled function-valued dynamics, finite-horizon MPC, deployment shift, and
small audit set. Define what “effective for control decisions” means: lower
closed-loop mean/tail cost or fewer constraint violations at matched coverage.

## 3 Method

FNO mean/scale model; audit split; L2 baseline; adjoint nonconformity score;
ellipsoidal ambiguity set; exact support function; adjoint-robust MPC.

## 4 Theory

Finite-sample one-step conformal proposition; exact support proposition;
first-order robust-counterpart remainder; deterministic rollout and discounted
value bounds. Keep stochastic and deterministic statements separate.

## 5 Experiments

Controlled Burgers, shift axes, fair calibration budgets, baselines, metrics,
paired seeds, failure cases. Architecture ablations last.

## 6 Limitations

Exchangeability of audit data, first-order approximation, weak learned scales,
small PDE benchmark, empirical local Lipschitz constant.

## 7 Conclusion

Return to the question: uncertainty becomes useful only after its geometry is
matched to the downstream value sensitivity and verified by closed-loop tests.
