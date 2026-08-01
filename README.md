# Decision-Calibrated Robust Control

**From Predictive Uncertainty to Decision-Effective Dynamics Ambiguity Sets
for Robust Control under Deployment Shift**

This project is about uncertainty-aware robust control, not about proposing a
new neural-operator architecture. It studies a decision-focused question:
how can predictive model uncertainty be converted into a dynamics ambiguity
set that is effective for downstream control decisions?

The default implementation trains a base FNO and a second FNO on the same
inputs with slightly perturbed labels. Their smoothed disagreement supplies a
local scale. A learned heteroscedastic head remains available as a baseline.
The one-step wrapper outputs

\[
\widehat u_{t+1},\qquad \sigma_\theta(u_t,a_t),
\]

and a small held-out deployment audit set conformalizes either the anisotropic set

\[
\mathcal U_\theta(u,a)=
\left\{\widehat G_\theta(u,a)+\sigma_\theta(u,a)\odot z:
\|z\|_{2,n}\leq q\right\}.
\]

or the simultaneous coordinate box

\[
\mathcal U_\infty(u,a)=
\left\{\widehat G_\theta(u,a)+\Delta:
|\Delta_j|\leq q_\infty\sigma_{\theta,j}(u,a),\;\forall j\right\}.
\]

The max-type score gives simultaneous spatial coverage for one random
function-valued transition. It does not by itself give trajectory coverage.
For a fixed rollout horizon we therefore calibrate a separate
max-over-time-and-coordinate score on independent behavior-policy
trajectories. That split-conformal band covers an entire random rollout under
the audited trajectory distribution; it is not advertised as a certificate
for counterfactual MPC action sequences.

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
- a residual FNO world model for the current paper experiments;
- perturbation-disagreement scale plus a learned-head baseline;
- max-type simultaneous spatial conformal boxes;
- a separate max-over-time-and-coordinate trajectory band;
- small deployment-audit conformalization of a heteroscedastic ellipsoid;
- nominal, in-distribution-L2, audit-L2, adjoint-support, and adversarial
  robust CEM-MPC;
- explicit multi-step error and value-gap theorem targets.

## Quick start

The quick experiment is CPU-compatible:

```bash
python -m pip install -e ".[dev]"
dcurc-experiment --model fno --quick --output-dir results
```

Use `--uncertainty head` for the learned-scale baseline; perturbation
disagreement is the default.

## Notebook-first Colab workflow

[`notebooks/Decision_Calibrated_PDE_Control_Colab.ipynb`](notebooks/Decision_Calibrated_PDE_Control_Colab.ipynb)
is the canonical reproducible workflow. It installs the project, downloads the
official 128-by-128 NeuralOperator Navier--Stokes archive from Zenodo, trains
the clean/perturbed two-FNO uncertainty model, runs controlled Burgers and the
four-bound comparison, exports one conclusion per figure, and packages results
for download.

The public NS2D pairs contain no action channel. They validate simultaneous
function-valued uncertainty coverage, not closed-loop control. Raw data are
downloaded at runtime and are excluded from Git and release archives; see
[`docs/DATASET.md`](docs/DATASET.md).

For the intended experiment:

```bash
dcurc-experiment --model fno --output-dir results
```

Each run trains a one-step world model and its chosen uncertainty mechanism,
then calibrates scalar L2 tubes, normalized ellipsoids, and simultaneous
coordinate boxes. It evaluates coverage under four regimes and compares
uncontrolled, nominal MPC, in-distribution-L2 robust MPC, deployment-audit-L2
robust MPC, ellipsoid/box adjoint-support MPC, and nonlinear adversarial MPC
under joint shift. It also audits direct max-over-time-and-coordinate coverage
on held-out behavior-policy trajectories; that audit is not a certificate for
counterfactual MPC trajectories.

[`RESULTS_PRELIMINARY.md`](RESULTS_PRELIMINARY.md) reports the initial
three-seed paired actuator-gain sweep. It is explicitly labeled mechanism
evidence rather than a final statistical claim.

[`RESULTS_FULL.md`](RESULTS_FULL.md) records the full seed-27 controlled
Burgers run, deterministic theorem-scaling checks, the 60/160 independent
value-bound comparison, and the official 128-by-128 NS2D audit. It also states
the guarantee boundary for every table.

## Deferred architecture ablations

The current manuscript reports FNO only. The following backbones remain in the
code as future architecture-independence checks and are not part of the present
evidence:

`fno`
: Standard periodic Fourier spectral convolution plus local residual maps.

`tno`
: Zero-embedded doubled-grid finite-section spectral convolution, intended to
  reduce the wrap-around bias caused by non-periodic boundaries.

`dscdno`
: Dynamic displacement-structured convolutional DNO. Input-conditioned
  generator pairs define low-displacement-rank spatial maps, followed by an
  internal residual update.

`moe`
: Input-conditioned mixture of FNO, TNO, and DSC-DNO experts. Total predictive
  variance includes both within-expert scale and between-expert disagreement.

MoE is treated as a hypothesis, not assumed to win.
All backbones use ResNet-style skip connections inside their operator blocks.
ResNet is not treated as a separate world-model family.

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

[`theory/perturbation_conformal_robust_control.md`](theory/perturbation_conformal_robust_control.md)
gives the combined construction, exact polyhedral tightening margins, support
functions, guarantee boundaries, and implementation map.

[`experiments/reward_value_gap_colab_final/`](experiments/reward_value_gap_colab_final/) contains a
reproducible reward/value experiment that attains the
`epsilon/(1-gamma)^2` fixed-policy rate analytically, tests finite-horizon
propagation in controlled Burgers dynamics, and audits the trained FNO on
visited joint-shift trajectories. Source CSV files and PNG/SVG/PDF figure
exports are included.

[`experiments/bound_comparison_colab_final/`](experiments/bound_comparison_colab_final/) contains the
full independent-test results comparing
global, local-recursion, adjoint-support, and adjoint-plus-curvature value
bounds using disjoint calibration and test trajectories. Coverage and bound
utilization are reported together so that an undersized invalid bound cannot
appear artificially tight.

After a full run, create traceable manuscript tables and the three standalone
decision-effectiveness figures with:

```bash
python scripts/build_paper_results.py \
  --burgers-metrics results/fno_burgers_seed27/fno_metrics.json \
  --bound-summary experiments/bound_comparison_colab_final/results/summary.json \
  --ns2d-metrics experiments/ns2d_colab_v2/results/metrics.json \
  --output-dir results/final_summary

python scripts/build_decision_figures.py \
  --metrics results/fno_burgers_seed27/fno_metrics.json \
  --output-dir results/final_summary/figures
```

The first command recomputes all relative changes from JSON instead of copying
numbers by hand. The second exports PNG, editable SVG/PDF, and 600 DPI TIFF
files, one conclusion per figure.

The current paper-like report is
[`paper_rewriting_output/final_paper/main.pdf`](paper_rewriting_output/final_paper/main.pdf).

## Scientific status

This is a research prototype. One-step split conformal prediction gives a
marginal transition statement. A separately calibrated max-over-time-and-space
score gives a simultaneous statement only for exchangeable behavior-policy
rollouts; neither result certifies counterfactual MPC actions. The deterministic
rollout/value theorem instead assumes a uniform one-step error on a
forward-invariant region. Marginal conformal coverage does not imply that
assumption. The repository keeps these claims separate and tests whether
calibrated uncertainty improves decisions rather than reporting calibration
alone.

## License

MIT.
