# Final artifact manifest

## Updated preprint package (2026-08-01)

- `final_paper/main.tex`: updated nine-page technical preprint.
- `final_paper/main.pdf`: compiled and visually verified PDF.
- `../../experiments/reward_value_gap_colab_final/`: exact, controlled-Burgers, and
  learned-FNO value-gap evidence with individual figure exports.
- `../../experiments/bound_comparison_colab_final/`: independent calibration/test run
  and seven individual figure exports.
- `../../experiments/ns2d_colab_v2/`: official 128-by-128 NS2D audit metrics and
  eleven standalone figure exports.
- `../../notebooks/Decision_Calibrated_PDE_Control_Colab.ipynb`: notebook-first
  Colab workflow with full Burgers, value-gap, bound, and NS2D commands.

## Primary deliverables

- `final_paper/main.tex`: editable English preprint draft.
- `final_paper/references.bib`: verified initial bibliography.
- `final_paper/main.pdf`: compiled and visually checked nine-page draft.
- `../RESULTS_PRELIMINARY.md`: human-readable three-seed result summary.
- `../results/three_seed_summary.json`: machine-readable aggregate.

## Reproducibility

- `../src/unoc/`: simulator, world model, uncertainty calibration, and MPC.
- `../tests/test_smoke.py`: seven current numerical/structural tests.
- `../theory/error_propagation.md`: extended theorem and limitation note.
- `../README.md`: setup and experiment commands.

## Integrity

- `integrity_audit.md`: LaTeX gate READY.
- `source_map.md`, `evidence_bank.md`, and `claim_register.md`: provenance and
  claim boundaries.

## Deliberately excluded claims

The current results do not establish population-level superiority,
simultaneous trajectory coverage, certified global inner maximization, or
architecture independence.
