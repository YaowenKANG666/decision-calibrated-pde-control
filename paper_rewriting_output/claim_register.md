# Claim register

1. **Method claim:** a conformal score based on cost-to-go sensitivity defines
   an ambiguity support value directly usable in robust MPC.
2. **Theory claim:** the support function is exact for the stated diagonal
   ellipsoid; the conformal guarantee is marginal under exchangeability.
3. **Propagation claim:** uniform one-step state error implies a geometric
   rollout bound and a conditional discounted value bound.
4. **Empirical claim:** pending multi-seed evidence; must compare nominal,
   ID-L2, audit-L2, and audit-adjoint controllers.
5. **Scope claim:** FNO is an implementation choice, not a new architecture.

Forbidden current claims: certified trajectory safety, arbitrary-shift
coverage, global Lipschitz certification, or superiority across PDEs.

## 2026-08-01 implementation update

6. **Uncertainty-mechanism claim:** clean/perturbed model disagreement is an
   error-localization scale only; conformal calibration, not disagreement,
   supplies coverage.
7. **Sharpness claim:** the scalar witness attains slope 1.000 in epsilon; after
   normalizing by \(\gamma\epsilon\), it attains slope 2.000 against the
   effective discount horizon, within numerical precision.
8. **Bound tightness claim:** on 60 calibration and 160 independent test
   trajectories, local recursion has a smaller mean bound than global maximum;
   this is an independent-test experiment, not a distribution-wide theorem.
9. **NS2D scope claim:** the official public dataset supports function-valued
   UQ experiments but not closed-loop control because it has no action channel.

The curvature-augmented method showed no improvement in the current run;
the fitted curvature coefficient was zero. No positive curvature claim is
permitted until a larger experiment or a certified Hessian bound supports it.
