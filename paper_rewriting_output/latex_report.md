# LaTeX report

## 2026-08-01 full manuscript rebuild

- Reorganized the draft into a complete 15-page paper: introduction, related
  work, preliminaries and problem setting, methodology, decision-effective
  robust MPC, theory, experiments, results, limitations, and conclusion.
- Added an original closed-loop schematic that distinguishes the training,
  calibration, ambiguity-set, robust-planning, and deployment interfaces.
- Added perturbation-based uncertainty scales; normalized L2, ellipsoidal, and
  simultaneous max-type conformal scores; and a practical split-calibration
  procedure.
- Added exact ellipsoidal and box support functions, an adjoint robust
  counterpart, nonlinear PGD rollouts, deterministic multi-step error
  propagation, and a policy-transfer bound with an explicit optimization-error
  term.
- Separated marginal split-conformal coverage, behavior-policy trajectory
  coverage, deterministic uniform-error propagation, and matched closed-loop
  evidence. None is presented as implying the others.
- Added the controlled Burgers results, high-dimensional 2D Navier--Stokes
  stress test, value-gap scaling audit, and independent bound comparison.
- Expanded the bibliography to 27 cited primary articles, conference papers,
  preprints, and datasets.

## Build and visual QA

- Engine: pdfLaTeX via latexmk (TeX Live 2025).
- Build status: PASS.
- Output: `final_paper/main.pdf`, 15 US-letter pages, 641,033 bytes.
- Bibliography: BibTeX; all citations and cross-references resolved.
- Log audit: no overfull boxes, undefined citations/references, oversized
  floats, or LaTeX warnings matched by the release check.
- Visual QA: all 15 pages rendered with Poppler at 110 DPI and inspected in
  contact sheets after the final compile.
- Checked: title hierarchy, contribution ordering, schematic placement,
  equations, theorem statements, tables, standalone plots, references, page
  numbers, clipping, overlap, and glyph rendering.
- Final visual status: PASS.
- Repository QA: Ruff passed and all 14 pytest regression tests passed.
