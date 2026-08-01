# Research dossier

## Closest research lines

1. **Neural operators.** FNO, DeepONet, and the general neural-operator
   framework establish fast maps between discretized functions. They motivate
   the world model but do not supply calibrated uncertainty or robust control.
2. **Uncertainty for operator learning.** Ensembles, probabilistic neural
   operators, Bayesian/linearized neural operators, risk-controlling quantile
   operators, and conformalized DeepONets produce structured field
   uncertainty. The closest perturbation method trains clean-label and
   perturbed-label FNOs, then uses a max score for simultaneous Navier--Stokes
   coverage in a data-scarce regime.
3. **Conformal robust control.** Chee et al. use weighted CP, constraint
   tightening, and predictive reference generation. SODA-MPC and
   conformalized system-level synthesis address OOD finite-dimensional
   control. These works establish that CP can enter controllers, so novelty
   cannot be claimed at that generic level.
4. **Decision-aware error.** Value-aware model learning and robust MBRL show
   that errors matter through downstream value, while simulation lemmas
   propagate uniform one-step error to long-horizon value error.

## Defensible gap

The gap is the interface between calibrated field uncertainty and the
finite-horizon objective. Operator-UQ papers stop at field coverage and width;
conformal-control papers largely use finite-dimensional state tubes or
constraints. The present work asks whether a calibrated anisotropic field set
can be queried through its support in an MPC sensitivity direction and whether
that conversion improves independent closed-loop mean or tail cost.

## Implemented construction

The implemented score is predictive, not adjoint-calibrated. For residual
`r`, perturbation scale `sigma`, and deployment audit multiplier `q`, the two
primary sets are

\[
\mathcal U_2=\{\widehat G+\Delta:
\|\Delta/\sigma\|_{2,n}\le q_2\},\qquad
\mathcal U_\infty=\{\widehat G+\Delta:
|\Delta_j|\le q_\infty\sigma_j\}.
\]

After calibration, the controller queries the sets in the finite-horizon
adjoint direction `lambda`:

\[
h_{\mathcal U_2}(\lambda)=q_2\|\lambda\odot\sigma\|_{2,n},
\qquad
h_{\mathcal U_\infty}(\lambda)=
\frac{q_\infty}{n}\sum_j|\lambda_j|\sigma_j.
\]

This ordering is central: calibration constructs a predictive set; decision
weighting is a downstream support query. The paper does not claim conformal
validity for an action-dependent adjoint score.

## Theory ledger

- Split CP: finite-sample marginal one-step field coverage on a fixed grid,
  conditional on the trained models and exchangeability.
- Trajectory CP: simultaneous time/coordinate coverage only for a predeclared
  exchangeable behavior-policy trajectory distribution.
- Support function: exact algebra for ellipsoid and box.
- Curvature: conditional Taylor remainder for the stacked product set.
- Rollout/value bounds: deterministic uniform one-step error and Lipschitz
  assumptions, with explicit CEM optimization error in policy transfer.
- No conformal result is used to instantiate the uniform epsilon assumption.

## Current evidence and falsification

- Combined-shift source L2 coverage is 0.741; audit ellipsoid and box coverage
  are 0.906 on 800 test transitions.
- In 24 matched actuator-gain cases, box-adjoint MPC changes mean/p90 cost by
  -1.89%/-10.84% versus nominal MPC. This is one controlled sweep, not a
  multi-seed population claim.
- The nonlinear adversarial controller is not best; the adjoint value bound is
  wider than local recursion; the fitted curvature coefficient is zero.
- NS2D simultaneous coverage is 0.883, full width 4.3409, and scale-error
  correlation 0.169. A reviewer should interpret this as a high-dimensional
  failure mode, not a success claim.
- The central empirical claim fails if multi-seed controlled-PDE experiments
  erase the audit-box tail advantage or if it loses to an audit-matched
  isotropic baseline under equal compute.
