# LaTeX report

- Engine: pdfLaTeX via latexmk (TeX Live 2025).
- Build status: PASS.
- Output: `final_paper/main.pdf`, 5 US-letter pages.
- Bibliography: BibTeX, 9 cited primary sources.
- Cross-references: resolved.
- Visual QA: every page rendered with Poppler at 120 DPI.
- Checked: title hierarchy, equations, theorem blocks, preliminary-results
  table, page numbers, references, clipping, overlap, and glyph rendering.
- Final visual status: PASS; no clipped or overlapping content.
- Known warning: LaTeX changed table placement from `h` to `ht`; the table is
  legible at the top of page 5 and precedes the limitations section.
