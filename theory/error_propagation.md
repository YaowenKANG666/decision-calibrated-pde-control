# Calibration, robust support, error propagation, and control performance

This note states the current theoretical closure. It separates exact
finite-sample and convex-analytic statements, conditional deterministic
bounds, and planning approximations.

## Proposition 1: finite-sample dynamics-set coverage

Let a trained dynamics model output a mean and a positive pointwise scale,
$\widehat G_\theta(x,a)$ and $\sigma_\theta(x,a)$. For an independent
deployment-audit sample $Z_i=(x_i,a_i,y_i)$, define

$$
S_i=
\left\lVert
\frac{y_i-\widehat G_\theta(x_i,a_i)}
{\sigma_\theta(x_i,a_i)}
\right\rVert_{2,n},
\qquad
\lVert v\rVert_{2,n}^2=\frac1n\sum_{j=1}^n v_j^2.
$$

For $m$ audit points, take the order statistic with rank
$k=\lceil(m+1)(1-\alpha)\rceil$, using the usual atom at $+\infty$ if
$k>m$, and call it $q$.
If the audit points and a new deployment point are exchangeable conditional
on the trained model, the standard split-conformal rank argument gives

$$
\Pr\left\lbrace
G(x,a)\in\mathcal U_\theta(x,a)
\right\rbrace\ge 1-\alpha,
$$

where

$$
\mathcal U_\theta(x,a)=
\left\lbrace
\widehat G_\theta(x,a)+\sigma_\theta(x,a)\odot z:
\lVert z\rVert_{2,n}\le q
\right\rbrace.
$$

This is marginal one-step coverage on the audited deployment distribution. It
does not imply coverage under an arbitrary further shift.

For a fixed horizon and predeclared rollout procedure, a separate
max-over-time-and-coordinate score can instead be conformalized on independent
trajectories. The same rank argument then gives marginal coverage of the whole
random rollout. This statement is specific to the audited trajectory
distribution; behavior-policy trajectories do not certify counterfactual MPC
action sequences.

## Proposition 2: exact decision support of the ambiguity set

Use the normalized inner product
$\langle v,w\rangle_n=n^{-1}\sum_jv_jw_j$. For any cost-to-go sensitivity
$\lambda$,

$$
\sup_{\Delta:\lVert\Delta/\sigma\rVert_{2,n}\le q}
\langle\lambda,\Delta\rangle_n
=q\lVert\lambda\odot\sigma\rVert_{2,n}.
$$

To prove this, substitute $\Delta=\sigma\odot z$ and apply Cauchy--Schwarz.
Equality is attained by

$$
z^\star
=q\frac{\lambda\odot\sigma}
{\lVert\lambda\odot\sigma\rVert_{2,n}}.
$$

Thus a predictive, anisotropic dynamics ellipsoid becomes decision-effective
by querying its support in the finite-horizon adjoint direction. No
direction-specific recalibration is required, so the conformal set and the
MPC support use the same $q$.

The nonlinear controller instead uses the rectangular product of these sets:

$$
\min_{a_{0:H-1}}\max_{\lVert z_t\rVert_{2,n}\le q}
\sum_{t=1}^{H}\ell(x_t)+\sum_{t=0}^{H-1}r(a_t),
\quad
x_{t+1}=\widehat G_\theta(x_t,a_t)
+\sigma_\theta(x_t,a_t)\odot z_t.
$$

Projected gradient ascent gives a computable lower bound on the inner maximum,
not a certificate that the global maximum was found. Also, marginal one-step
conformal coverage does not imply that all $H$ rectangular sets contain the
true transitions simultaneously.

## Proposition 3: first-order robust objective and its remainder

Let $J_H(x^+)$ denote the remaining nominal horizon cost after a predicted
next state, and let $\lambda=\nabla J_H(\widehat x^+)$. The linearized
worst-case cost over the calibrated set is exactly

$$
J_H(\widehat x^+)
+q\lVert\lambda\odot\sigma_\theta(x,a)\rVert_{2,n}.
$$

If $J_H$ is twice differentiable and its Hessian has operator norm at most
$M_H$ in the normalized Euclidean geometry on the ambiguity set, Taylor's
theorem yields the conditional upper bound

$$
\sup_{\Delta\in\mathcal U_\theta}
J_H(\widehat x^++\Delta)
\le
J_H(\widehat x^+)
+q\lVert\lambda\odot\sigma\rVert_{2,n}
+\frac{M_H}{2}q^2\lVert\sigma\rVert_\infty^2.
$$

The current controller implements the first two terms. It is therefore an
exact robust counterpart for the linearized cost, and a first-order
approximation for the nonlinear cost. A certified nonlinear claim requires
estimating or bounding $M_H$ and adding the third term.

## Setting

Let the controlled dynamics and learned world model be

$$
u_{t+1}=G(u_t,a_t), \qquad
\widehat u_{t+1}=\widehat G(\widehat u_t,a_t).
$$

Assume on a forward-invariant set:

1. the true transition is $L_G$-Lipschitz in state;
2. the deterministic one-step operator error is uniformly bounded:
   $\lVert G(u,a)-\widehat G(u,a)\rVert\leq\epsilon$;
3. the same open-loop action sequence is applied to both systems.

This uniform assumption is not a consequence of Proposition 1. Conformal
coverage controls a probability for a random audited transition; the
$\epsilon$ assumption controls every state--action pair in a specified
region. The code tests their consequences in separate experiments.

## Proposition 4: multi-step rollout error

Writing $e_t=\lVert u_t-\widehat u_t\rVert$, the triangle inequality gives

$$
e_{t+1}
\leq
\lVert G(u_t,a_t)-G(\widehat u_t,a_t)\rVert
+
\lVert G(\widehat u_t,a_t)-\widehat G(\widehat u_t,a_t)\rVert
\leq L_Ge_t+\epsilon.
$$

Therefore

$$
e_h
\leq
\epsilon\sum_{j=0}^{h-1}L_G^j
=
\begin{cases}
\epsilon(1-L_G^h)/(1-L_G),&L_G\neq1,\\
h\epsilon,&L_G=1.
\end{cases}
$$

This recursion motivates the robust MPC uncertainty tube

$$
R_{t+1}=L R_t+\rho(\widehat u_t,a_t).
$$

## Proposition 5: discounted value error

Suppose the state-dependent stage reward is $L_r$-Lipschitz and
$\gamma L_G<1$. For any fixed open-loop action sequence,

$$
|V_G-V_{\widehat G}|
\leq
L_r\sum_{t\geq0}\gamma^t e_t
\leq
\frac{\gamma L_r\epsilon}
{(1-\gamma)(1-\gamma L_G)}.
$$

For a feedback-policy class, assume every true and learned closed-loop map
$x\mapsto G(x,\pi(x))$ and
$x\mapsto\widehat G(x,\pi(x))$ is $L_G$-Lipschitz and the one-step model
error bound holds uniformly over the visited state-action set. Also assume the
induced closed-loop reward $r_\pi(x)$, including any action penalty through
$\pi(x)$, is uniformly $L_r$-Lipschitz. Then the fixed-policy bound is
uniform over that class. The standard two-model optimality decomposition gives

$$
V_G^{\pi^\star}-V_G^{\widehat\pi}
\leq
\frac{2\gamma L_r\epsilon}
{(1-\gamma)(1-\gamma L_G)}.
$$

In the non-expansive case $L_G\leq1$,

$$
V_G^{\pi^\star}-V_G^{\widehat\pi}
\leq
\frac{2\gamma L_r}{(1-\gamma)^2}\epsilon,
$$

which has the requested $O(\epsilon/(1-\gamma)^2)$ form.

### Sharpness witness and reward experiment

The squared effective-horizon dependence is attainable, rather than only an
artifact of two loose inequalities. Consider the scalar true and learned
dynamics

$$
G(x)=L_Gx,
\qquad
\widehat G(x)=L_Gx+\epsilon,
\qquad x_0=0,
$$

with the $L_r$-Lipschitz reward $r(x)=-L_r|x|$. The true trajectory stays
at zero, while

$$
\widehat x_t
=\epsilon\sum_{k=0}^{t-1}L_G^k.
$$

Consequently,

$$
|V_G-V_{\widehat G}|
=L_r\sum_{t\ge0}\gamma^t|\widehat x_t|
=\frac{\gamma L_r\epsilon}
{(1-\gamma)(1-\gamma L_G)}.
$$

For $L_G=1$, equality becomes

$$
|V_G-V_{\widehat G}|
=\frac{\gamma L_r\epsilon}{(1-\gamma)^2}.
$$

Thus the fixed-policy rate is sharp. The coefficient two for the regret of a
policy optimized in the learned model comes from comparing the two optimal
values through the uniform fixed-policy bound; this witness does not assert
that the factor two is always attained. The reproducible analytic and Burgers
experiments are in `experiments/reward_value_gap/`.

## Important limitations

- Split conformal calibration provides marginal, not uniform, one-step
  coverage. The implemented trajectory max-score gives a distinct marginal
  whole-rollout guarantee only for exchangeable behavior-policy trajectories;
  arbitrary adaptive MPC trajectories require sequential or policy-aware
  methods.
- The empirical 95th-percentile local Lipschitz estimate used by the code is a
  planning heuristic, not a certified global Lipschitz bound.
- The adjoint robust term is exact for the linearized cost. Nonlinear
  certification requires the curvature term in Proposition 3.
- Feedback policies require the stated common closed-loop Lipschitz condition;
  otherwise an additional policy-Lipschitz term appears.
- The policy-transfer decomposition assumes an exact optimizer in the learned
  model. Finite CEM search introduces a separate optimization-error term.
- The theorem must specify the norm and invariant state-action set.

These limitations are part of the research agenda rather than hidden
assumptions.
