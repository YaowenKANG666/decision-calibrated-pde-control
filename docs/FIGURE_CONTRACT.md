# Figure contract

Every manuscript figure must carry one conclusion and be exported as PDF, SVG,
and high-resolution PNG. Composite grids are avoided unless a direct paired
comparison cannot be understood in separate figures.

## Figure 1: method overview

- **Role:** whole-paper scheme, not FNO architecture.
- **Conclusion:** offline target auditing turns field uncertainty into a
  geometry that robust MPC can query inside an online feedback loop.
- **Required distinction:** calibration is offline; receding-horizon control is
  online.
- **Risk:** do not imply that one-step conformal coverage certifies the closed
  loop.

## Figure 2: paired task-validity cost

- **Conclusion:** persistent forcing makes active control valuable.
- **Evidence:** 100 paired uncontrolled and PDE-oracle costs with identity line.
- **Risk:** “oracle” means numerical-model access under finite-budget CEM, not
  globally optimal control.

## Final submission figures

The remaining figures are generated only after the locked five-seed run:

1. source versus target-audit coverage and set size;
2. audit-size coverage and support;
3. scale ablation at matched field-level coverage;
4. ellipsoid-versus-box adjoint support;
5. paired mean and p90 closed-loop cost;
6. representative actions and state trajectories.

Coverage must always be shown beside width or support. Control-cost claims must
use independent joint-shift trajectories and paired uncertainty intervals.
