# Figure contract: persistent-forcing task validation

## Core conclusion

On independently sampled, persistently forced Burgers plants, model-predictive
control with direct access to the numerical PDE lowers matched closed-loop cost
relative to zero control. This establishes that the benchmark rewards active
control before any learned world model is introduced.

## Evidence chain

- `oracle_01_paired_cost`: every point is one shared physical case; the diagonal
  is the no-improvement boundary.
- `oracle_02_paired_difference`: shows effect heterogeneity and the direction of
  the paired effect without hiding individual cases behind a bar chart.
- `oracle_03_tracking_trajectory`: verifies the mechanism on a preselected
  representative case (case index zero).
- `oracle_04_control_actions`: documents the control effort that produces the
  representative tracking trajectory.

## Archetype and export

Each artifact is a standalone single-column quantitative figure. Python and
Matplotlib are the exclusive rendering backend. Editable SVG and PDF, 600-dpi
PNG, and 600-dpi TIFF are exported from the same source data and script.

## Review risks

- “Oracle” means access to the true numerical transition model, not global
  optimization; finite-budget CEM remains approximate.
- Test cases are independent joint-distribution draws and both controllers use
  identical cases.
- Main uncertainty is summarized with paired bootstrap intervals; the original
  actuator-gain sweep is not used as population evidence.
- The failure threshold is fixed in the task configuration and reported with
  the output rather than selected after inspecting controller outcomes.
