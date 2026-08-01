# LaTeX report

## 2026-08-01 rebuild

- Replaced the obsolete learned-scale-head description with the implemented
  clean/perturbed two-operator disagreement scale.
- Added the simultaneous max-score box and its guarantee boundary.
- Added the exact value-gap sharpness experiment and one standalone figure.
- Added an independent 60/160 value-bound table and reported the zero
  curvature result honestly.
- Added actual NS2D metrics while explicitly excluding a control claim.
- `latexmk -pdf` completed successfully; all nine rendered pages were visually
  inspected with no clipping, overlap, missing glyphs, or unreadable tables.

- Engine: pdfLaTeX via latexmk (TeX Live 2025).
- Build status: PASS.
- Output: `final_paper/main.pdf`, 9 US-letter pages.
- Bibliography: BibTeX, 10 cited primary sources and datasets.
- Cross-references: resolved.
- Visual QA: every page rendered with Poppler at 110 DPI.
- Checked: title hierarchy, equations, theorem blocks, preliminary-results
  table, page numbers, references, clipping, overlap, and glyph rendering.
- Final visual status: PASS; no clipped or overlapping content.
- Known warning: LaTeX changed table placement from `h` to `ht`; the table is
  legible at the top of page 5 and precedes the limitations section.
