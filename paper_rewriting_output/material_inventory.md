# Material inventory

Generated through the PaperSpine `material_inventory.py` workflow. The full
55-file machine inventory is stored in `source_inventory.md` and
`source_inventory.json`.

## Materials used to build the draft

- `src/unoc/`: executable simulator, world model, calibration, and MPC method;
- `tests/test_smoke.py`: numerical checks of set support and rollout recursion;
- `theory/error_propagation.md`: theorem statements and limitations;
- `results/*/fno_metrics.json`: preliminary results, used only when identified
  as smoke diagnostics;
- `README.md`: implementation scope and reproducibility commands;
- verified primary literature indexed in `citation_support_bank.md`.

## Excluded from positive claims

- interrupted or obsolete experimental runs;
- model checkpoints without a matching metrics file;
- planned TNO, DSC-DNO, and MoE architecture ablations;
- unverified items in the citation screening pool.
