# Section blueprints

## Abstract

Define the prediction-to-decision mismatch, summarize the perturbation scale,
disjoint audit, function-space geometries, adjoint/PGD robust MPC, separated
theory, Burgers control result, NS2D negative result, and exact scope.

## 1 Introduction

1. Learned operators make repeated PDE planning feasible.
2. Average field error and control impact use different geometries.
3. Research question: convert predictive uncertainty into a decision-effective
   dynamics set.
4. Related work in operator learning/UQ, conformal robust control, and
   decision-aware model error.
5. Prior-work table clarifies novelty and guarantee scope.
6. Four contributions, followed by the original closed-loop schematic.

## 2 Preliminaries and problem setting

1. Continuous field dynamics and fixed-grid discretization.
2. Proper train, validation, deployment audit, independent test, and separate
   trajectory splits.
3. Split-conformal quantile with the atom-at-infinity convention.
4. Definition of predictive validity, efficiency, and decision effectiveness.

## 3 Methodology

1. Residual FNO inputs, residual blocks, boundary projection, and explicit
   statement that FNO is replaceable.
2. Clean/perturbed operators, smoothing, and training-derived floor.
3. L2, normalized ellipsoid, and max-type scores with their exact sets.
4. Five-step practical implementation protocol.

## 4 Decision-effective robust MPC

1. Rectangular min-max rollout as a planning model.
2. Exact ellipsoid and box support functions in the full finite-horizon
   adjoint direction.
3. Nonlinear projected-gradient adversarial rollout.
4. Common CEM outer budget and receding-horizon plant feedback.

## 5 Theoretical analysis

1. One-step split-conformal field coverage.
2. Separate behavior-policy trajectory score.
3. Stacked Taylor remainder and conditional curvature bound.
4. Uniform-error rollout recurrence.
5. Fixed-policy discounted value bound.
6. Policy-transfer bound with explicit optimization error.
7. Guarantee ledger explaining non-implications.

## 6 Experimental design

Five research questions; controlled Burgers equation and cost; official NS2D
field-UQ benchmark; exact split table; matched baselines, metrics, and training
details.

## 7 Results

1. World-model fit and combined-shift coverage table/shift figure.
2. Closed-loop cost table and p90 figure.
3. NS2D table, absolute-error field, and scale-error association.
4. Deterministic epsilon/effective-horizon scaling.
5. Independent value-bound table and coverage-width frontier.

## 8 Discussion and limitations

Separate supported mechanism evidence from unproved safety/generalization;
retain adversarial, curvature, bound-tightness, and NS2D negative results;
specify the next controlled 2D and multi-seed experiments.

## 9 Conclusion

Return to the controlling motivation. The contribution is the calibrated
uncertainty-to-decision interface and its falsifiable evaluation ledger, not a
new neural-operator architecture.
