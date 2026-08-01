# Figure asset map

All paper figures are standalone, Python-generated, and traceable to committed
source data or method equations.

| Paper figure | Core conclusion | Source | Export status |
|---|---|---|---|
| Fig. 1 method schematic | Predictive scale becomes a calibrated field set, then a robust MPC query; claim types remain separate | `scripts/make_method_schematic.py`, method equations, figure contract | PNG/SVG/PDF/TIFF available |
| Fig. 2 coverage by shift | Source calibration degrades under compound shift; deployment audit restores coverage at different widths | `results/fno_burgers_seed27/fno_metrics.json` | PNG/SVG/PDF available |
| Fig. 3 control p90 | Robust-control benefit is geometry-dependent; box-adjoint has the lowest tested upper tail | same metrics JSON | PNG/SVG/PDF available |
| Fig. 4 NS2D absolute error | High-dimensional FNO residuals are spatially structured | NS2D test prediction | PNG/SVG/PDF available |
| Fig. 5 NS2D association | Perturbation scale weakly localizes absolute error (r=0.169) | NS2D independent test outputs | PNG/SVG/PDF available |
| Fig. 6 value scaling | Deterministic witness has exact squared effective-horizon scaling after normalization | `analytic_sharpness.csv` | PNG/SVG/PDF available |
| Fig. 7 bound frontier | Conservatism must be compared at matched coverage; local recursion dominates adjoint in mean width here | `coverage_bound_curve.csv` | PNG/SVG/PDF available |

The method schematic uses one neutral/blue/gold/teal/violet palette, editable
text, a deployed-PDE feedback loop, and a lower guarantee ledger. The NS2D
target/prediction images use a shared color scale in the release suite. No
figure converts undercoverage into a positive tightness claim.
