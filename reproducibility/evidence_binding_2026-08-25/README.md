# Evidence-binding reproducibility snapshot — 2026-08-25

This directory publishes the paper-bound, public-safe evidence-binding
reproduction materials prepared from the validated reviewer package.

## Browsable subset

- `src/tinylm_slicer/`: byte-exact minimum import closure used by the evaluator
- `configs/`: path-free stress and held-out configurations
- `paper/swegca/scripts/`: portable evaluator
- `provenance/original_evaluator.py`: byte-exact original evaluator
- `runs/*/report.path-sanitized.json`: result reports without workstation paths
- `validation/`: exact-schema ledger validation receipts
- `reviewer_tools/validate_case_ledger.py`: ledger validator

The generated 102,400-case development ledger and consumed 10,240-case
held-out ledger are omitted from the browsable tree to avoid duplicating 36 MB
of line-oriented data. They remain byte-exact inside
`SWEGCA_evidence_binding_reproducibility_bundle.zip`, together with the full
package manifest and README.

## Archive identity

```text
59a7959e3ef12d27ac90720f6a38c56fa1cd5198a66ca1ea34481e17a01c5690  SWEGCA_evidence_binding_reproducibility_bundle.zip
```

The archive was originally labeled a confidential reviewer supplement. The
author authorized public release on 2026-09-04 after its allowlist, UTF-8,
absolute-path, identifier, and package-hash checks passed. Public release does
not turn the consumed held-out replay into a new independent confirmation.

Install the declared NumPy and PyTorch dependencies before reproducing the
commands documented inside the archive. Compare outcome counts and integrity
fields, not environment-dependent latency or Git metadata.
