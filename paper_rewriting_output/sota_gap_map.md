# SOTA gap map

| Neighbor | What it already solves | Remaining question used here |
|---|---|---|
| Conformal robust MPC | Calibrated dynamics regions and constraint tightening | Which error directions matter for cost in function-valued systems? |
| OOD conformal SLS | Shift-aware covariance sets and reachable tubes | Can a small audit calibrate a decision support function for PDE control? |
| Decision-aware conformal optimization | Downstream-loss-aware sets | How should this be instantiated for multi-step learned dynamics? |
| Value-aware model learning | Value-sensitive model loss | Can value sensitivity define a calibrated dynamics ambiguity set? |
| Neural-operator UQ | Predictive uncertainty for PDE surrogates | Does that uncertainty improve closed-loop control under shift? |

Candidate contribution: a finite-horizon adjoint-calibrated dynamics
ambiguity set plus an evaluation protocol that distinguishes predictive
coverage, audit adaptation, and actual control improvement.
