# Full experiment record

This file records the current full seed-27 mechanism experiment and the
independent value-bound and NS2D audits. The raw JSON/CSV files, rather than
this prose summary, are the source of truth.

## Theorem-scaling checks

The deterministic sharpness witness uses

\[
G(x)=L_Gx,\qquad \widehat G(x)=L_Gx+\epsilon,\qquad
r(x)=-L_r|x|.
\]

It gives the exact fixed-policy gap

\[
|V_G-V_{\widehat G}|=
\frac{\gamma L_r\epsilon}
{(1-\gamma)(1-\gamma L_G)}.
\]

Across six epsilon values and six discount factors, the numerical checks give:

| Check | Result |
|---|---:|
| log-log slope versus epsilon | 1.000000 |
| log-log slope of gap divided by gamma times epsilon versus effective horizon | 2.000000 |
| maximum simulation-to-closed-form relative error | below 5e-15 |
| maximum controlled-Burgers gap / finite-horizon recursion bound | 0.3586 |

The first two rows test a deterministic uniform-error theorem. They are not
conformal-coverage results.

## Controlled Burgers: predictive sets

The residual FNO and label-perturbed FNO were trained for 60 epochs. The final
perturbed-model validation MSE was `9.87e-5`. The table uses independent
combined-shift transitions.

| Set | Function coverage | Decision-direction coverage | Mean support |
|---|---:|---:|---:|
| source-calibrated L2 | 0.741 | 0.820 | 0.00650 |
| deployment-audit L2 | 0.884 | 0.934 | 0.01072 |
| audit ellipsoid | 0.906 | 0.986 | 0.02012 |
| audit simultaneous box | 0.906 | 1.000 | 0.04422 |

Coverage is not tightness: the simultaneous box has the strongest projected
coverage and the largest mean support.

## Controlled Burgers: closed-loop cost

All controllers use the same learned world model and 24 matched combined-shift
cases of horizon 20. Lower cost is better; changes are relative to nominal MPC.

| Controller | Mean cost | Mean change | empirical p90 | p90 change |
|---|---:|---:|---:|---:|
| nominal MPC | 0.5797 | -- | 1.0752 | -- |
| source-L2 robust MPC | 0.5724 | -1.27% | 1.0250 | -4.66% |
| audit-L2 robust MPC | 0.5686 | -1.92% | 0.9914 | -7.79% |
| ellipsoid-adjoint MPC | 0.5747 | -0.88% | 1.0201 | -5.12% |
| box-adjoint MPC | 0.5688 | -1.89% | 0.9586 | -10.84% |
| adversarial MPC | 0.5764 | -0.58% | 1.0532 | -2.04% |

This run supports a geometry-dependent control benefit. It does not show that
one robust optimizer universally dominates the others.

The separately calibrated max-over-time-and-coordinate band obtained 0.88
coverage on held-out length-10 behavior-policy trajectories at a nominal 0.90
level. This is reported as a finite test proportion and applies only to the
audited rollout distribution, not to counterfactual MPC action sequences.

## Four independently calibrated value bounds

Each raw bound was scaled using 60 calibration trajectories and evaluated on
160 disjoint test trajectories at horizon 20 and gamma 0.95.

| Method | Coverage | Mean bound | Median utilization | p90 utilization | Max utilization |
|---|---:|---:|---:|---:|---:|
| global maximum | 0.938 | 0.0987 | 0.210 | 0.688 | 3.919 |
| local recursion | 0.944 | 0.0870 | 0.322 | 0.788 | 2.190 |
| adjoint support | 0.925 | 0.2035 | 0.265 | 0.854 | 1.766 |
| adjoint plus curvature | 0.925 | 0.2035 | 0.265 | 0.854 | 1.766 |

The local recursion is the tightest by mean bound in this experiment. The
adjoint construction is not automatically a tighter scalar value certificate.
The audit-calibrated curvature coefficient was zero, so no curvature benefit is
claimed.

## Official NeuralOperator NS2D audit

The experiment uses 800 proper-training, 200 conformal-audit, and 300 disjoint
test pairs at 128 by 128 resolution.

| Mean relative L2 error | max-score q90 | Simultaneous test coverage | Mean full band width | Scale-error Pearson r |
|---:|---:|---:|---:|---:|
| 0.1635 | 30.679 | 0.883 | 4.3409 | 0.169 |

The realized 0.883 proportion is not rounded into a 0.90 guarantee. The public
dataset has no action channel, so these results test high-dimensional
function-space calibration only, not two-dimensional closed-loop control.
The weak scale-error correlation also shows that the current perturbed-model
disagreement is a poor spatial error localizer on this benchmark; a stronger
uncertainty mechanism is required before claiming tight high-dimensional sets.

## Guarantee boundary

Split conformal prediction gives a marginal statement for a new exchangeable
transition, or for a separately calibrated exchangeable trajectory when the
entire trajectory is the conformal example. The rollout and value theorem
instead assumes a deterministic uniform error on a specified forward-invariant
region. Neither statement implies the other.
