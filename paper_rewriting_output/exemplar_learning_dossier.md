# Exemplar learning dossier

## Structural lessons from the control exemplar

The L4DC conformal-control exemplar places a system/controller loop beside the
robustification pipeline, states the formal problem before the method, recalls
CP notation explicitly, and connects every uncertainty object to a concrete
controller modification. We transfer that architecture, not its text or
graphic: our schematic uses a PDE/FNO loop, disjoint data roles, ambiguity
geometry, support queries, and a guarantee ledger.

## Structural lessons from the operator-UQ exemplar

The data-scarce Navier--Stokes exemplar separates problem formulation, split-CP
preliminaries, perturbation-based nonconformity, scale construction,
calibration, practical implementation, experimental protocol, quantitative
tables, ablations, and qualitative spatial diagnostics. We reuse this readable
sequence while adding the missing decision layer: ambiguity geometry,
decision-effective robust MPC, multi-step propagation, and closed-loop tests.

## Resulting paper spine

1. state the predictive-versus-decision mismatch;
2. position against operator UQ, conformal control, and value-aware learning;
3. define discretization and strict train/validation/audit/test roles;
4. construct perturbation scale and split-conformal field sets;
5. query exact support or solve a nonlinear adversarial rollout in MPC;
6. keep conformal, deterministic, and empirical claims separate;
7. answer five research questions with tables and standalone figures;
8. preserve negative results and end with the value of the interface, not FNO
   architecture novelty.
