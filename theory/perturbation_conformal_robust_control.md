# Calibrated field geometry and decision support

This note records the mathematical core used by the preprint. It deliberately
excludes the earlier PGD, curvature, and value-gap branches.

## 1. One-step dynamics model

After spatial discretization, the controlled PDE induces

$$
x_{t+1}=G_\star(x_t,a_t;\xi),\qquad x_t\in\mathbb R^n.
$$

The residual FNO approximates this map by

$$
\widehat G_\theta(x,a;\xi)=x+\delta_\theta(x,a;\xi).
$$

A second operator uses the same inputs and perturbed labels. Its disagreement
with the base operator gives a positive plug-in scale

$$
\sigma_\theta(z)=\operatorname{Smooth}
\left(|\widehat G_\theta(z)-\widetilde G_{\widetilde\theta}(z)|\right)+\tau_0.
$$

The scale is not a calibrated standard deviation. It only shapes the residual
geometry before conformal calibration.

## 2. Split-conformal field sets

For a disjoint target-domain audit pair \((z_i,y_i)\), define
\(r_i=y_i-\widehat G_\theta(z_i)\). The ellipsoid and max scores are

$$
S_i^{(2)}=\left\|r_i\odot\sigma_i^{-1}\right\|_{2,n},
\qquad
S_i^{(\infty)}=\left\|r_i\odot\sigma_i^{-1}\right\|_\infty.
$$

At target miscoverage \(\alpha\),

$$
q_g=S^{(g)}_{(\lceil(n_{\mathrm{audit}}+1)(1-\alpha)\rceil)}.
$$

The corresponding ambiguity set is

$$
\mathcal U_g(x,a)=\left\{
\widehat G_\theta(x,a)+\Delta:
\left\|\Delta\odot\sigma_\theta(x,a)^{-1}\right\|_{g,n}\le q_g
\right\}.
$$

Under exchangeability between the audit pairs and one new target pair, the
new complete field belongs to its set with marginal probability at least
\(1-\alpha\). The box additionally expresses simultaneous coordinate coverage
for that random field. The ellipsoid expresses joint field membership. Their
common comparison is matched field-level coverage.

## 3. Exact local support functions

Use the normalized inner product

$$
\langle\lambda,\Delta\rangle_n=\frac1n\lambda^\top\Delta.
$$

For the ellipsoid,

$$
\sup_{\|\Delta\odot\sigma^{-1}\|_{2,n}\le q_2}
\langle\lambda,\Delta\rangle_n
=q_2\|\lambda\odot\sigma\|_{2,n}.
$$

For the coordinate box,

$$
\sup_{\|\Delta\odot\sigma^{-1}\|_\infty\le q_\infty}
\langle\lambda,\Delta\rangle_n
=q_\infty\|\lambda\odot\sigma\|_{1,n}.
$$

Both formulas are exact consequences of norm duality. They show why two sets
with the same field-level coverage can affect a controller differently.

## 4. Adjoint query

For a nominal rollout
\(\widehat x_{h+1}=\widehat G_\theta(\widehat x_h,a_h)\), the discrete adjoint
satisfies

$$
\lambda_H=\nabla_x\ell_T(\widehat x_H),
$$

$$
\lambda_h=\nabla_x\ell(\widehat x_h,a_h)
+D_x\widehat G_\theta(\widehat x_h,a_h)^\top\lambda_{h+1}.
$$

The first-order robust objective adds the stagewise supports

$$
J_{\mathrm{rob}}(a_{0:H-1})
=J_{\mathrm{nom}}(a_{0:H-1})
+\sum_{h=0}^{H-1}h_{\mathcal U_g}(\lambda_{h+1}).
$$

The support is exact for the linearized cost perturbation. It is not a global
certificate for the nonlinear, state-dependent FNO rollout.

## 5. Guarantee boundary

The following statements are distinct:

1. split conformal provides marginal one-step field membership under
   exchangeability;
2. norm duality provides exact support for a local linear functional;
3. receding-horizon deployment produces empirical closed-loop performance.

None implies the next without additional assumptions. In particular, offline
one-step coverage does not certify counterfactual MPC trajectories.
