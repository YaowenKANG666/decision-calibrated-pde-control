# Decision-Effective Conformal Ambiguity Sets for Neural PDE Control

This repository studies one question:

> How can predictive uncertainty from a neural PDE world model be converted
> into a dynamics ambiguity set that changes control decisions in a useful and
> testable way?

The contribution is not a new FNO architecture. A residual Fourier Neural
Operator supplies a differentiable one-step world model. A disjoint labelled
target-domain audit split calibrates field-valued ambiguity sets, and their
support functions connect uncertainty geometry to robust model predictive
control (MPC).

![Offline-calibrated, online-applied method overview](figures/method_01_chain_schematic.png)

The upper half of the diagram is offline. The FNO pair, uncertainty scale, and
conformal multiplier are frozen before deployment. The lower half is the
online receding-horizon feedback loop. The method is therefore
**offline-calibrated and online-applied**, not online conformal calibration.

## Scientific status

The revised persistent-forcing Burgers task has passed its task-validity gate.
The learned multi-seed robust-control comparison is not yet complete. The
current release does not claim that robust FNO MPC outperforms nominal MPC.

Across 100 independently sampled plants:

| Controller | Mean cost (95% CI) | Median | p90 cost (95% CI) |
|---|---:|---:|---:|
| Uncontrolled | 0.5900 (0.5176, 0.6661) | 0.5249 | 1.0599 (0.9632, 1.3942) |
| PDE-oracle MPC | 0.1610 (0.1410, 0.1812) | 0.1377 | 0.2875 (0.2322, 0.3401) |

PDE-oracle MPC reduced cumulative cost in all 100 matched cases. “Oracle”
means direct access to the numerical PDE transition model; CEM still uses a
finite optimization budget and is not globally exact.

## Method in one page

For a field state $u$, action $a$, and observed parameters $\xi$, the
residual FNO predicts

```math
\widehat G_\theta(u,a;\xi)=u+\delta_\theta(u,a;\xi).
```

A perturbed-label replica gives a plug-in spatial scale

```math
\sigma_\theta(z)=\operatorname{Smooth}
\left(|\widehat G_\theta(z)-\widetilde G_{\widetilde\theta}(z)|\right)+\tau_0.
```

Split conformal calibration on a disjoint target-domain audit set constructs

```math
\mathcal U_g(u,a)=\left\{
\widehat G_\theta(u,a)+\Delta:
\left\|\Delta\odot\sigma_\theta(u,a)^{-1}\right\|_{g,n}\le q_g
\right\},
\qquad g\in\{2,\infty\}.
```

For the normalized field inner product, the exact local support functions are

```math
h_{\mathcal U_2}(\lambda)=q_2\|\lambda\odot\sigma\|_{2,n},
\qquad
h_{\mathcal U_\infty}(\lambda)=q_\infty\|\lambda\odot\sigma\|_{1,n}.
```

These terms weight uncertainty by finite-horizon adjoint sensitivity. The
ellipsoid and box may have matched field-level coverage while producing
different support, actions, and closed-loop tail cost.

## Primary benchmark

The controlled plant is

```math
u_t+u u_x=\nu u_{xx}+f_{\mathrm{ext}}(x,t)
+g_{\mathrm{act}}\sum_{k=1}^{2}a_{t,k}b_k(x).
```

Persistent forcing prevents zero control from winning through passive viscous
decay. Two localized actuators make the action spatially structured. The
actuator gain is latent to the learned model and supplies a controlled dynamics
shift.

## Reproduce the validated result

Python 3.10 or newer is required.

    git clone https://github.com/YaowenKANG666/decision-calibrated-pde-control.git
    cd decision-calibrated-pde-control
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"

Run the task-validity experiment:

    dcurc-forced-oracle \
      --cases 100 \
      --rollout-horizon 20 \
      --cem-horizon 6 \
      --cem-candidates 64 \
      --cem-elites 8 \
      --cem-iterations 3 \
      --seed 27 \
      --output-dir results/forced_oracle_validation

Audit the saved statistics:

    python scripts/audit_release_results.py

A portable Jupyter workflow is available at
[notebooks/Decision_Calibrated_PDE_Control.ipynb](notebooks/Decision_Calibrated_PDE_Control.ipynb).
It uses ordinary Python paths and contains no platform-specific notebook
commands.

## Locked minimum submission experiment

The final comparison will contain:

- five independent FNO training seeds;
- 100–200 independent joint-shift trajectories per seed;
- uncontrolled, PDE-oracle, nominal, audit-$L^2$, ellipsoid-adjoint, and
  box-adjoint controllers;
- source calibration versus target audit calibration;
- audit sizes $20,50,100,200,300$, with repeated audit subsampling;
- constant, twin-FNO, ensemble, and residual-quantile scale ablations;
- paired mean, median, p90, control-effort, coverage, width, and adjoint-support
  statistics.

The main empirical claim will be tested only after these runs:

```math
\text{matched field-level coverage}
\quad\Longrightarrow\quad
\text{geometry changes support, actions, and upper-tail cost}.
```

## Repository layout

    src/unoc/                 Burgers solvers, neural models, calibration, and MPC
    experiments/forced_oracle_validation/
                              Reproducible task-validity entry point
    results/forced_oracle_validation/
                              Case-level data, summaries, and standalone figures
    notebooks/                Portable Jupyter workflow
    docs/                     Locked experimental protocol and figure contract
    theory/                   Ambiguity-set and support-function derivation
    paper/                    Current working preprint PDF
    tests/                    Numerical and structural smoke tests

## Scope and future work

One-step split-conformal coverage is marginal over a new audited transition.
It does not certify counterfactual MPC rollouts or closed-loop safety. Immediate
work is the locked five-seed evaluation. Longer-term extensions include
feedback-aware sequential calibration, trajectory-coupled ambiguity sets,
certified nonlinear robust inner problems, controlled two-dimensional flows,
and architecture-portability tests with FNO, TNO, or related operators.

## License

MIT.
