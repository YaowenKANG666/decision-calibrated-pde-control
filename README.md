# Decision-Calibrated Robust Control

**From Predictive Uncertainty to Decision-Effective Dynamics Ambiguity Sets
for Robust Control under Deployment Shift**

This project is about uncertainty-aware robust control, not about proposing a
new neural-operator architecture. It studies a decision-focused question:
how can predictive model uncertainty be converted into a dynamics ambiguity
set that is effective for downstream control decisions?

The one-step model outputs

\[
\widehat u_{t+1},\qquad \sigma_\theta(u_t,a_t),
\]

and a small held-out deployment audit set conformalizes the anisotropic set

\[
\mathcal U_\theta(u,a)=
\left\{\widehat G_\theta(u,a)+\sigma_\theta(u,a)\odot z:
\|z\|_{2,n}\leq q\right\}.
\]

The primary controller approximately solves a nonlinear robust inner problem
over this same set by projected gradient ascent. A faster ablation queries the
set in its finite-horizon adjoint direction:

\[
\sup_{\Delta\in\mathcal U_\theta}
\langle\lambda,\Delta\rangle_n
=q\|\lambda\odot\sigma_\theta\|_{2,n}.
\]

This converts predictive uncertainty into either a nonlinear adversarial
rollout or a first-order robust cost without recalibrating a different score.
Isotropic L2 tubes are the main baseline. Residual FNO is the initial world
model; TNO, DSC-DNO, and MoE are deferred architecture-independence checks.

## Current scope

- controlled 1D viscous Burgers equation;
- non-homogeneous Dirichlet boundary inputs;
- viscosity, boundary, and combined deployment shifts;
- architecture-agnostic learned dynamics, instantiated by FNO, doubled-grid
  finite-section TNO, DSC-DNO, and a mixture of operator experts;
- heteroscedastic uncertainty head plus split-conformal calibration;
- small deployment-audit conformalization of a heteroscedastic ellipsoid;
- nominal, in-distribution-L2, audit-L2, adjoint-support, and adversarial
  robust CEM-MPC;
- explicit multi-step error and value-gap theorem targets.

## Quick start

The quick experiment is CPU-compatible:

```bash
python -m pip install -e ".[dev]"
dcurc-experiment --model fno --quick --output-dir results
dcurc-experiment --model tno --quick --output-dir results
dcurc-experiment --model dscdno --quick --output-dir results
```

For the intended experiment:

```bash
dcurc-experiment --model fno --output-dir results
dcurc-experiment --model tno --output-dir results
dcurc-experiment --model moe --output-dir results
```

Each run trains a probabilistic one-step world model, calibrates L2 and
heteroscedastic ellipsoidal sets, evaluates coverage under four regimes, and compares
uncontrolled, nominal MPC, in-distribution-L2 robust MPC, deployment-audit-L2
robust MPC, adjoint-support MPC, and nonlinear adversarial MPC under joint shift.

[`RESULTS_PRELIMINARY.md`](RESULTS_PRELIMINARY.md) reports the initial
three-seed paired actuator-gain sweep. It is explicitly labeled mechanism
evidence rather than a final statistical claim.

## Architecture ablations, not the main contribution

`fno`
: Standard periodic Fourier spectral convolution plus local residual maps.

`tno`
: Zero-embedded doubled-grid finite-section spectral convolution, intended to
  reduce the wrap-around bias caused by non-periodic boundaries.

All three backbones use ResNet-style skip connections inside every operator
block. ResNet is not treated as a separate world-model family.

`dscdno`
: Dynamic displacement-structured convolutional DNO. Input-conditioned
  generator pairs define low-displacement-rank spatial maps, followed by an
  internal residual update.

`moe`
: Input-conditioned mixture of FNO, TNO, and DSC-DNO experts. Total predictive
  variance includes both within-expert scale and between-expert disagreement.

MoE is treated as a hypothesis, not assumed to win.

## Theory

[`theory/error_propagation.md`](theory/error_propagation.md) derives:

\[
\|u_h-\widehat u_h\|
\leq \epsilon\sum_{j=0}^{h-1}L_G^j
\]

and, under a common closed-loop Lipschitz condition,

\[
V_G^{\pi^\star}-V_G^{\widehat\pi}
\leq
\frac{2\gamma L_r\epsilon}
{(1-\gamma)(1-\gamma L_G)}.
\]

For \(L_G\leq1\), this reduces to the target
\(O(\epsilon/(1-\gamma)^2)\) dependence.

## Scientific status

This is a research prototype. Split conformal prediction provides marginal
one-step coverage; the empirical Lipschitz tube is not yet a certified
trajectory-level guarantee. The repository states these limitations explicitly
and will test whether calibrated uncertainty improves decisions rather than
reporting calibration alone.

## License

MIT.
