# Figure 1 contract

**Core conclusion.** A perturbation-derived predictive scale becomes useful for
control only after disjoint split-conformal calibration constructs a
function-space ambiguity set and the set is queried by the downstream MPC
objective.

**Archetype.** Schematic-led composite.

**Target/output.** Two-column preprint figure; Python/Matplotlib only; editable
SVG and PDF plus 300 dpi PNG and 600 dpi TIFF.

**Panel map.**

- **a, proper training:** clean and label-perturbed FNOs produce a mean and a
  spatial disagreement scale.
- **b, deployment audit:** a disjoint calibration split produces a finite-sample
  conformal multiplier.
- **c, ambiguity geometry:** ellipsoidal and simultaneous-box dynamics sets
  represent different support functions.
- **d, robust MPC:** adjoint support or nonlinear adversarial rollout evaluates
  candidate actions, then closes the loop with the deployed PDE.
- **e, guarantee ledger:** separates marginal conformal validity, deterministic
  uniform-error propagation, and empirical closed-loop evidence.

**Evidence hierarchy.** The hero evidence is the uncertainty-to-decision loop;
the guarantee ledger is the conceptual validation; numerical results appear in
later figures rather than being duplicated here.

**Reviewer risks.** The graphic must not imply that marginal one-step conformal
coverage certifies counterfactual MPC rollouts, that the deterministic
uniform-error theorem follows from conformal calibration, or that the FNO
architecture is the contribution.
