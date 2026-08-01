# Perturbation-Conformal Neural World Models for Robust PDE Control

## 1. Synthesis and research question

This construction combines two ingredients without treating their guarantees
as interchangeable:

1. a base FNO and a label-perturbed FNO provide a spatially adaptive scale for
   function-valued prediction error;
2. conformal error regions are converted into exact constraint-tightening
   margins or support-function penalties for model-based control.

The research question is not whether an FNO can replace a finite-dimensional
neural dynamics model. It is:

> Which calibrated function-space geometry gives the best trade-off between
> simultaneous predictive coverage, conservatism, and closed-loop control
> performance under dynamics shift?

## 2. Controlled function-valued dynamics

Let a controlled PDE, after spatial discretization on `n` coordinates and one
control interval of numerical integration, induce

\[
x_{t+1}=G_\star(x_t,a_t;\xi),\qquad x_t\in\mathbb R^n.
\]

The deployment variable `xi` may contain viscosity, boundary values, forcing,
or an unobserved actuator gain. The base world model is

\[
\widehat G_0(x,a)=\mu(x,a).
\]

In the prototype, `G_hat` is a one-step residual FNO. The mathematical
construction does not require the FNO architecture.

## 3. A data-efficient perturbation scale

Train `G_hat_0` on a proper training set `(x_i,a_i,y_i)`. Form perturbed labels

\[
\widetilde y_i=y_i+\varepsilon_i,\qquad
\varepsilon_i\sim\mathcal N(0,c^2s_y^2I),
\]

and train a second operator `G_hat_eps` on the same inputs and perturbed
labels. Define the raw coordinate disagreement

\[
d_j(x,a)=|\widehat G_{0,j}(x,a)-\widehat G_{\varepsilon,j}(x,a)|.
\]

The implemented one-dimensional Burgers version uses an odd moving-average
window `K` and a proper-training-set floor:

\[
\bar d_j=\frac1{|N_K(j)|}\sum_{k\in N_K(j)}d_k,
\qquad
\sigma_j(x,a)=\max\{\bar d_j(x,a),\tau_0\},
\]

\[
\tau_0=0.1\,\operatorname{median}_{i,j}\bar d_j(x_i,a_i).
\]

The scale is only a ranking and localization device. Its numerical value is
not a calibrated probability or standard deviation. Validity comes from the
next conformal step.

## 4. Spatially simultaneous conformal ambiguity box

For an independent deployment-audit set of size `m`, define the max-type score

\[
S_i=\max_{1\le j\le n}
\frac{|y_{i,j}-\mu_j(x_i,a_i)|}{\sigma_j(x_i,a_i)}.
\]

For target coverage `1-alpha`, take

\[
q=S_{(\lceil(m+1)(1-\alpha)\rceil)}.
\]

The resulting ambiguity box is

\[
\mathcal U_\infty(x,a)=
\left\{
\mu(x,a)+\Delta:
|\Delta_j|\le q\sigma_j(x,a),\;j=1,\ldots,n
\right\}.
\]

Conditional on the two trained operators and their perturbation noise, if the
audit examples and a new deployment example are exchangeable, then

\[
\Pr\left\{
G_\star(x,a)_j\in
[\mu_j-q\sigma_j,\mu_j+q\sigma_j]
\text{ for every }j
\right\}\ge 1-\alpha.
\]

This is simultaneous-in-coordinate coverage for one random function-valued
transition. It is not uniform coverage over all possible inputs, and it is not
yet simultaneous coverage along an adaptively controlled trajectory.

## 5. Two geometries and their exact decision supports

The repository compares the max box above with the normalized ellipsoid

\[
\mathcal U_2(x,a)=
\left\{
\mu+\sigma\odot z:\|z\|_{2,n}\le q_2
\right\},\qquad
\|z\|_{2,n}^2=n^{-1}\sum_jz_j^2.
\]

For the normalized inner product
`<lambda,Delta>_n = n^{-1} sum_j lambda_j Delta_j`, the support functions are

\[
h_{\mathcal U_\infty}(\lambda)
=q_\infty\frac1n\sum_{j=1}^n|\lambda_j|\sigma_j,
\]

and

\[
h_{\mathcal U_2}(\lambda)
=q_2\left(\frac1n\sum_{j=1}^n\lambda_j^2\sigma_j^2\right)^{1/2}.
\]

These formulas turn a predictive set into a decision-aware quantity without
recalibrating a different score. In MPC, `lambda` is the finite-horizon
adjoint of the nominal cost. Thus the same statistically calibrated set is
queried only in directions that affect the decision.

## 6. Exact Chee-style constraint tightening

Let the next-state constraint be

\[
Ax_{t+1}\le b,
\]

and let row `i` of `A` be `a_i^T`. Requiring every state in the conformal box
to satisfy the constraint is equivalent to

\[
a_i^T\mu(x_t,a_t)
+\sup_{|\Delta_j|\le q\sigma_j}a_i^T\Delta
\le b_i.
\]

The inner supremum is exact:

\[
\sup_{|\Delta_j|\le q\sigma_j}a_i^T\Delta
=q\sum_{j=1}^n|a_{ij}|\sigma_j.
\]

Therefore the tightened constraint is

\[
A\mu(x_t,a_t)\le b-\beta(x_t,a_t),
\qquad
\beta_i=q\sum_j|a_{ij}|\sigma_j.
\]

For the normalized ellipsoid, the exact margin is

\[
\beta_i=q_2\sqrt n\|a_i\odot\sigma\|_2.
\]

If the true next state belongs to the calibrated set, satisfaction of the
tightened nominal constraint implies satisfaction of the original true-state
constraint. Hence the one-step constraint statement inherits the same
`1-alpha` probability under the exchangeability assumptions.

## 7. Robust planning objective

The nonlinear robust controller considers

\[
\min_{a_{0:H-1}}
\max_{\Delta_t\in\mathcal U(x_t,a_t)}
\sum_{t=1}^{H}\ell(x_t)+\sum_{t=0}^{H-1}r(a_t),
\]

\[
x_{t+1}=\mu(x_t,a_t)+\Delta_t.
\]

The implementation uses CEM for the outer action sequence and projected
gradient ascent for the inner adversary. Projection is coordinate clipping for
the max box and normalized-L2 projection for the ellipsoid. Finite-step PGD
is an approximate inner maximum, not a global certificate.

The fast controller linearizes the finite-horizon objective and adds

\[
\sum_{t=0}^{H-1}h_{\mathcal U_t}(\lambda_{t+1}).
\]

This term is exact for the linearized objective. A nonlinear upper certificate
requires a Hessian/curvature remainder.

## 8. Multi-step error: three different guarantees

These three statements must not be conflated.

### 8.1 Union-bound trajectory coverage

If each visited transition satisfies a valid per-step miscoverage bound
`alpha_t`, regardless of dependence between steps, then

\[
\Pr\{G_\star(x_t,a_t)\in\mathcal U_t\;\forall t<H\}
\ge1-\sum_{t=0}^{H-1}\alpha_t.
\]

Choosing `alpha_t = delta/H` gives trajectory probability at least `1-delta`.
The difficult condition is validity at controller-selected state-action pairs.
Offline split conformal on behavior-policy data does not automatically supply
it. Online weighted conformal or a correctly sampled deployment audit is
needed, and its shift correction must be reported.

### 8.2 Direct trajectory conformalization

A stronger experimental extension calibrates entire rollouts with

\[
S_i^{\rm traj}=\max_{0\le t<H}\max_j
\frac{|x_{i,t+1,j}-\widehat x_{i,t+1,j}|}
{\sigma_{i,t,j}}.
\]

Split conformal then covers all calibrated horizon-time-coordinate entries at
once for a random trajectory drawn from the same trajectory distribution. It
avoids a union-bound inflation but remains marginal over trajectories and does
not cover arbitrary counterfactual MPC action sequences.

### 8.3 Deterministic propagation

If the true transition is `L_G`-Lipschitz and the one-step model error is
uniformly at most `epsilon`, then

\[
e_{t+1}\le L_Ge_t+\epsilon,
\qquad
e_h\le\epsilon\sum_{k=0}^{h-1}L_G^k.
\]

This is deterministic but requires a uniform error assumption that marginal
conformal coverage does not establish.

## 9. What is inherited and what is new

Inherited from perturbation-conformal operator learning:

- two same-data operators with clean and perturbed labels;
- smoothed/floored disagreement scale;
- max-type simultaneous coordinate score;
- finite-sample marginal validity under exchangeability.

Inherited from conformal controller robustification:

- online/receding-horizon uncertainty updating as an optional extension;
- exact dynamic constraint tightening;
- high-probability open-loop constraint reasoning.

The proposed research contribution must be tested rather than assumed:

- comparison of function-space box and ellipsoid geometries at matched audit
  data and coverage;
- exact adjoint support for decision-aware planning;
- closed-loop control benefit under PDE parameter, boundary, and actuator
  shifts;
- trajectory-level calibration or a transparent union-bound alternative;
- demonstration that stronger coverage may become too conservative for
  control.

## 10. Implementation map

- `src/unoc/models.py`
  - `PerturbationScaleWorldModel`: base/perturbed disagreement, smoothing,
    stabilized floor.
- `src/unoc/data.py`
  - `TransitionArrays.with_perturbed_targets`: same inputs, noisy labels,
    exact boundary preservation.
- `src/unoc/calibration.py`
  - `estimate_perturbation_floor`: proper-training median floor;
  - `norm_kind="max"`: max-type score and simultaneous box;
  - `polyhedral_tightening_margin`: exact box/ellipsoid margins.
- `src/unoc/mpc.py`
  - exact adjoint support for box and ellipsoid;
  - coordinate-clipped or normalized-L2 projected adversary.
- `src/unoc/experiment.py`
  - `--uncertainty perturbation` trains both FNOs;
  - reports L2, ellipsoid, and simultaneous-box coverage;
  - compares nominal, tube, ellipsoid-adjoint, box-adjoint, and nonlinear
    adversarial MPC.

## 11. Required experiments

1. Matched label budget: perturbation scale vs learned uncertainty head,
   unscaled conformal, MC dropout, and an ensemble.
2. Geometry: max box vs normalized ellipsoid at the same audit data and target
   coverage.
3. Calibration sizes: 30, 60, 90, 180, 300.
4. Shift severity: viscosity, boundary, actuator gain, and combined shift.
5. Control: paired initial conditions and latent actuator gains; report mean,
   median, p90, worst case, constraint violation, and action magnitude.
6. Horizon: one-step spatial coverage, direct trajectory coverage, and
   empirical propagated-tube coverage.
7. Ablations: label-noise multiplier, smoothing window, floor, adjoint term,
   and adversary iterations.

The current smoke run only validates executability. It must not be used as a
scientific result.
