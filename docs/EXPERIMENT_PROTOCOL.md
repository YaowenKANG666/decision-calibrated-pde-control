# Locked experimental protocol

This document separates the validated task gate from the experiments required
for the paper's final learned-controller claim.

## 1. Task-validity gate: complete

The primary plant is

$$
u_t+u u_x=\nu u_{xx}+f_{\mathrm{ext}}(x,t)
+g_{\mathrm{act}}\sum_{k=1}^{2}a_{t,k}b_k(x).
$$

The reference is zero. Persistent forcing prevents passive viscous decay from
solving the task. The state cost is the spatial mean squared error, the action
weight is 0.002, and the terminal tracking weight is four.

One hundred independently sampled plants use the following test population:

| Quantity | Distribution |
|---|---:|
| viscosity | Uniform(0.008, 0.018) |
| each boundary value | Uniform(-0.03, 0.03) |
| latent actuator gain | Uniform(0.65, 1.35) |
| forcing amplitude | Uniform(0.40, 0.80) |
| forcing frequency | Uniform(0.50, 1.50) |
| forcing phase | Uniform(0, 2 pi) |
| initial-field amplitude | Uniform(0.03, 0.12) |

Both controllers receive the same draw within each pair. PDE-oracle MPC uses
the numerical transition model with CEM horizon six, 64 candidates, eight
elites, and three iterations. “Oracle” means model access, not global
optimization. Five thousand bootstrap resamples form the reported intervals.

## 2. Learned-model gate

Before robust-control comparisons, nominal FNO MPC must achieve a meaningful
cost between PDE-oracle MPC and uncontrolled dynamics on the forced task. If it
does not, model fitting or task observability must be corrected before any
ambiguity-set result is interpreted.

## 3. Primary controller comparison

Use five independently trained FNO seeds. For each seed, evaluate 100–200
independent trajectories drawn from the same locked joint test population.
Every controller uses the same trajectories and common CEM random numbers.

Controllers:

1. uncontrolled;
2. PDE-oracle MPC;
3. nominal FNO MPC;
4. target-audit isotropic L2 MPC;
5. target-audit ellipsoid-adjoint MPC;
6. target-audit box-adjoint MPC.

Report mean cost with a 95% interval, median, p90 with a bootstrap interval,
average control effort, and paired differences from nominal MPC. A failure
metric may be included only if its threshold is fixed before examining the
controller comparison and produces a scientifically meaningful event.

## 4. Calibration-source comparison

Hold the FNO and scale fixed. Compare source calibration with labelled
target-domain audit calibration on identical target test transitions. Report
field-level coverage, conformal multiplier, and set size together. The target
audit may recalibrate quantiles but may not update FNO weights.

## 5. Audit-size experiment

Use

$$
n_{\mathrm{audit}}\in\{20,50,100,200,300\}.
$$

For each size, draw 10–20 audit subsets, recompute the conformal quantile, and
leave the FNO frozen. Report coverage, quantile, set width or support, mean
cost, and p90 cost. This experiment supports the phrase “few-shot
calibration,” not “few-shot model adaptation.”

## 6. Scale ablation

Compare:

- constant scale;
- twin-FNO perturbation disagreement;
- deep-ensemble variance;
- residual-quantile scale head.

Record the label and compute budgets for every method. Evaluate Pearson and
Spearman association, top-10% error recall or AUPRC, conformal multiplier,
matched-coverage set width, adjoint support, and closed-loop cost. The twin-FNO
scale is a plug-in estimator borrowed from prior operator-UQ work, not the
paper's novelty.

## 7. Geometry test

The primary comparison is ellipsoid versus box at matched field-level sample
coverage. The box uses a max score and has simultaneous coordinate coverage
for one random field. The ellipsoid uses a normalized L2 score and has joint
field-membership coverage. Therefore the common comparison is called
“matched field-level coverage,” not “matched simultaneous coverage.”

Report:

- observed field-level coverage;
- mean set width or radius;
- adjoint support;
- applied actions;
- mean and p90 closed-loop cost.

The intended claim is supported only if geometry changes decision-relevant
support and the paired closed-loop comparison is stable across training seeds.

## 8. Excluded from the primary claim

Uncontrolled NS2D prediction, finite-step PGD inner maximization, fitted
curvature corrections, and deterministic value-gap scaling are not part of
the minimum submission experiment. They may be revisited only if they answer
a distinct reviewer question without diluting the main result.
