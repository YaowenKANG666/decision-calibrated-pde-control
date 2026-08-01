# Final artifact manifest

## Updated preprint package (2026-08-01)

- `final_paper/main.tex`: editable 15-page English preprint source.
- `final_paper/main.pdf`: compiled and page-by-page verified manuscript.
- `final_paper/references.bib`: 27-entry cited bibliography.
- `../figures/method_01_chain_schematic.{png,svg,pdf}`: original pipeline and
  guarantee-ledger schematic in raster and editable vector formats.
- `figure_contract.md`: conclusion, evidence logic, export, and review-risk
  contract used for the schematic.

## Evidence included in the manuscript

- `../../experiments/reward_value_gap_colab_final/`: exact,
  controlled-Burgers, and learned-FNO value-gap evidence.
- `../../experiments/bound_comparison_colab_final/`: independent
  calibration/test bound comparison and standalone figures.
- `../../experiments/ns2d_colab_v2/`: official 128-by-128 NS2D audit metrics
  and standalone figures.
- `../../notebooks/Decision_Calibrated_PDE_Control_Colab.ipynb`: notebook-first
  Colab workflow covering Burgers, value-gap, bound, and NS2D commands.

## Reproducibility

- `../../src/unoc/`: simulator, FNO world model, uncertainty calibration, and
  robust MPC components.
- `../../tests/`: numerical and structural regression tests.
- `../../theory/error_propagation.md`: extended derivations and limitation
  notes.
- `../../README.md`: setup, data, training, and experiment commands.

## PaperSpine integrity artifacts

- `integrity_audit.md`: artifact, reasoning, evidence, and integrity gates;
  release state READY.
- `research_dossier.md`, `exemplar_learning_dossier.md`, and
  `section_blueprints.md`: paper architecture and source-learning record.
- `source_map.md`, `evidence_bank.md`, `claim_register.md`, and
  `citation_support_bank.md`: provenance and claim boundaries.
- `writing_rationale_matrix.md`: section-level rhetorical and evidential
  rationale.

## Deliberately excluded claims

The manuscript does not claim that marginal conformal coverage implies a
uniform one-step error bound, counterfactual MPC-rollout coverage, or closed-loop
safety. It does not claim population-level controller superiority from the
single-seed Burgers sweep, continuum function-space coverage from a fixed grid,
certified global adversarial optimization, or NS2D control performance without
an action channel.
