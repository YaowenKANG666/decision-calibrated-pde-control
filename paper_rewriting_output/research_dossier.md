# Research dossier

## Established neighboring results

1. Conformal residual regions have been incorporated into model-based control
   through constraint tightening and can obtain finite-sample marginal
   guarantees under exchangeability (Chee et al., 2024).
2. Weighted conformal covariance models have recently been combined with
   system-level synthesis for nonlinear OOD MPC (Srinivasan et al., 2026).
3. Conformal uncertainty sets have a direct robust-optimization
   interpretation (Johnstone and Cox, 2021).
4. Decision-aware or end-to-end conformal optimization calibrates uncertainty
   with respect to downstream loss rather than prediction alone.
5. Value-aware model learning predates this project: not all model errors are
   equally relevant to a policy (Farahmand et al., 2017).
6. Neural-operator uncertainty methods can fail under moderate OOD shift;
   ensemble disagreement is a strong baseline (Mouli et al., 2024).

## Defensible gap

The gap is not “conformal control has never been done.” It is narrower:
existing learned-control uncertainty sets are commonly judged by state-space
coverage or safety constraints, while function-valued dynamics can have large
errors in control-irrelevant directions and small errors in value-sensitive
directions. The project asks whether an audit-calibrated support function in
the finite-horizon adjoint direction improves closed-loop decisions at a fixed
or smaller robustness budget.

## Proposed object

For learned dynamics mean \(\mu_\theta(x,a)\), scale
\(\sigma_\theta(x,a)\), and finite-horizon cost sensitivity
\(\lambda=\nabla_{x^+}\widehat J_H\), define

\[
s_i=\frac{|\langle\lambda_i,y_i-\mu_i\rangle|}
{\|\lambda_i\odot\sigma_i\|_2+\delta},\qquad
q=\operatorname{Quantile}_{1-\alpha}^{\rm conf}\{s_i\}.
\]

The induced ellipsoid has support

\[
\sup_{\Delta\in\mathcal U(x,a)}
\langle\lambda,\Delta\rangle
=q\|\lambda\odot\sigma_\theta(x,a)\|_2.
\]

This quantity enters the robust MPC objective directly. L2 tubes remain the
main baseline.

## Falsification criteria

- If adjoint calibration improves coverage but not mean, tail, or worst-case
  closed-loop cost across seeds, the central empirical claim fails.
- If gains disappear against an audit-calibrated L2 set, the result is audit
  adaptation rather than decision calibration.
- If the FNO scale has weak error correlation, replace it with a deep ensemble
  or residual quantile model before interpreting robust-control results.
- If first-order support is inaccurate for large sets, restrict the radius,
  add second-order curvature, or solve an inner adversarial rollout.
