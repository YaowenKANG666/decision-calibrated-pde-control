# Experimental and ablation protocol

This document records what is fixed and what is varied in every reported
comparison. It is intended to prevent a change in data, model, calibration,
and controller from being interpreted as a single-factor ablation.

## FNO world model

The controlled-Burgers experiments use a residual Fourier Neural Operator.
The clean and perturbed operators have identical architectures:

| Setting | Value |
|---|---:|
| spatial grid | 64 |
| hidden width | 32 |
| retained Fourier modes | 14 |
| residual Fourier blocks | 4 |
| training epochs | 60 |
| batch size | 64 |
| optimizer | AdamW |
| learning rate | 0.002 |
| weight decay | 0.00001 |
| label-noise multiplier | 0.05 times target standard deviation |
| disagreement smoothing | width-5 moving average |
| scale floor | 10% of median proper-training disagreement |
| total parameters in the FNO pair | 240,454 |

Both FNOs use all 4,000 proper-training inputs. The base FNO uses clean
targets. The replica uses one fixed Gaussian-perturbed copy of those targets.
The 600 validation transitions select checkpoints. Neither model uses source
calibration, deployment-audit, or test labels during optimization.

## Data roles

| Benchmark | Proper train | Validation | Source calibration | Deployment audit | Test | Additional split |
|---|---:|---:|---:|---:|---:|---:|
| Burgers one-step | 4,000 | 600 | 800 | 300 | 800 per regime | 200/400 trajectory audit/test |
| NS2D field UQ | 800 | 50 | - | 200 | 300 | - |
| Value bounds | fixed trained FNO | - | - | 60 trajectories | 160 trajectories | horizon 20 |

All roles are disjoint. Burgers one-step tests contain four separately
generated regimes: in-distribution, parameter shift, boundary shift, and
compound shift. The trajectory audit and trajectory test are disjoint from all
one-step samples.

## Ablation A: calibration source and ambiguity geometry

This group uses the same trained FNO pair, perturbation scale, target coverage
of 0.90, and 800 compound-shift test transitions.

1. Source $L^2$ versus audit $L^2$ changes only the calibration population.
2. Audit $L^2$, audit ellipsoid, and audit simultaneous box use the same 300
   deployment-audit transitions and change only the nonconformity score and
   induced set geometry.
3. Every variant is evaluated on the identical ordered test transitions.

Reported outcomes are field coverage, mean radius, decision-direction
coverage, and mean decision support. Coverage and width are reported together.

## Ablation B: robust-control interface

This group uses the same FNO pair and the same 24 plants. The plants form a
deterministic actuator-gain sweep from -0.5 to 1.5. Viscosity, boundary values,
and the initial field are common to all cases. Every controller is evaluated
for 20 receding-horizon steps. The CEM random seed is shared across controller
variants for each case and time step.

| Variant | Calibration/set | Inner robust query |
|---|---|---|
| nominal MPC | none | none |
| source-$L^2$ robust MPC | source $L^2$ tube | isotropic robust penalty |
| audit-$L^2$ robust MPC | deployment-audit $L^2$ tube | isotropic robust penalty |
| ellipsoid-adjoint MPC | audit ellipsoid | exact first-order support |
| box-adjoint MPC | audit coordinate box | exact first-order support |
| adversarial ellipsoid MPC | audit ellipsoid | projected-gradient rollout |

The source-$L^2$ versus audit-$L^2$ pair isolates calibration access. The two
adjoint rows isolate geometry at a common first-order interface. The
ellipsoid-adjoint versus adversarial-ellipsoid pair isolates the inner query
while holding the calibrated set fixed. Mean cost, empirical p90 cost, and mean
absolute action are computed from the same 24 matched cases.

## Ablation C: value-bound construction

Global maximum, local recursion, adjoint support, and adjoint plus curvature
use the same 60 calibration trajectories and the same 160 independent test
trajectories. The action sequence, horizon 20, discount factor 0.95, observed
value gap, and trained FNO are fixed within each trajectory.

Each raw construction is separately scaled on the common calibration split to
a 0.90 value-level target. Test coverage is therefore interpreted jointly with
mean bound and utilization. The curvature coefficient selected on calibration
was zero, so the last two variants coincide in the current run.

## NS2D protocol

The official NeuralOperator Reynolds-500 archive supplies $128\times128$
vorticity input-output fields. A base FNO and a perturbed-label FNO use the same
800 proper-training indices, which are disjoint from 50 validation, 200 audit,
and 300 test indices. The model uses 16 Fourier modes, 32 hidden channels, four
layers, 20 epochs, batch size 8, a 0.05 label-noise multiplier, and a
$15\times15$ smoothing window.

The reported prediction metric is field RMSE, defined as the square root of
the mean squared coordinate error for each sample and then averaged over 300
test samples. It is not a relative $L^2$ error. Simultaneous coverage uses the
maximum normalized coordinate residual as one score per test field.

## Replication boundary

The primary Burgers and NS2D tables condition on one trained model pair with
seed 27. Repeated calibration/test partitions or matched plant cases quantify
conditional variability, not training-seed variability. The current ablations
support mechanism isolation but not population-level superiority.
