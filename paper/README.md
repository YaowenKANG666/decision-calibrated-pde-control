# Preprint

The current manuscript is
[`decision_calibrated_robust_control.pdf`](decision_calibrated_robust_control.pdf).

The preprint, *From Predictive Coverage to Robust PDE Control: Calibrated
Dynamics Ambiguity Sets for Neural Operators*, reports a residual FNO world
model, perturbation-scaled conformal
ambiguity sets, robust MPC, deterministic error propagation, and the controlled
Burgers and NS2D experiments. Its preliminaries distinguish the continuous PDE
solution operators from the 64-dimensional Burgers and 16,384-dimensional
Navier--Stokes maps learned on fixed grids. The code and source result files in
the repository are the reproducibility record. A linked three-level contents
page provides direct navigation to the PDE definitions, calibration method,
robust-control construction, theoretical guarantees, and ablation studies.
The framing and derivations are supported by 37 source-matched references
across operator learning, uncertainty quantification, conformal prediction,
and robust decisions and control.
Its Data Availability statement distinguishes study-generated Burgers samples
from the reused NeuralOperator Reynolds-500 archive and cites the exact Zenodo
record and `nsforcing_128.tgz` file.
The manuscript now separates preliminaries, data preparation, methodology, and
experimental evaluation. The Data preparation section records the Burgers
sampling regimes, the public NS2D source, discretized learned maps, and split
protocol before the method and result comparisons.
