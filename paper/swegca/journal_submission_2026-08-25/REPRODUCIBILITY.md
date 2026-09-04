# Reproducibility and artifact lineage

## Frozen lineage

- implementation commit: `7e4ae52010c3e3a18fa73dbb34fa6b1c7214d95c`
- held-out freeze commit: `fa4e9eb18ff8f98ded83e2b8e9d8637bd055ec89`
- statistical configuration SHA-256: `2f54cead86a2aad187fe927be6db5a1ef2f87e1f0cc4970e81c235cd011200d1`
- held-out freeze manifest SHA-256: `17092235477dc7212fb2ef4eadf29269c1f0e358207abe68a71a0c19924ca1e4`

Complete source hashes and result/ledger hashes are recorded in `RESULT_LEDGER.md` and the freeze manifest.

## Focused regression

From the repository root with the package installed or `src` on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q tests/test_mosaic_bounded_world_write.py tests/test_mosaic_evidence_revision.py tests/test_mosaic_world_memory_transaction.py tests/test_rozephine_p2_n128_world_linked_memory_transaction.py tests/test_evaluate_rozephine_p2_n129_natural_closed_loop.py tests/test_evaluate_rozephine_p2_n164_current_evidence_world_write_gate.py tests/test_evaluate_rozephine_p2_n200_hash_bound_camera_proposal.py tests/test_evaluate_rozephine_p2_n201_reversible_camera_world_write.py tests/test_evaluate_rozephine_p2_n202_frozen_autonomous_camera_loop.py tests/test_evaluate_rozephine_p3_g1_fresh_core_safety.py tests/test_evaluate_rozephine_p3_g5_physical_closed_loop.py
```

Expected narrow rerun: 60 passed.

## Ordered component contract checks

The following development tests cover adaptive parallel fan-out and worker
isolation, fixed-order memory activation, and the local external-effect state
machine described in the component ledger:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q tests/test_mosaic_synapse_arbiter.py tests/test_mosaic_memory_activation.py tests/test_mosaic_autonomous_cognition.py
```

Expected result on the audited snapshot: 39 passed. These tests do not turn
parallel workers into independent evidence and do not replace Run 009/010.

## Statistical evaluator

The original held-out was consumed once and must not be treated as fresh again. Re-executing it can reproduce deterministic outcome counts but is not another independent confirmation. Latency fields, absolute paths, Git metadata, and report hashes may differ by environment.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python paper/swegca/scripts/evaluate_evidence_binding_statistics.py --config configs/swegca_evidence_binding_statistical_evaluation_v1.json --split heldout --frozen-manifest configs/swegca_evidence_binding_heldout_freeze_v1.json --output reviewer_outputs/heldout_reproduction/report.json
```

The committed evaluator refuses held-out execution when the configuration or any of the five bound source hashes differs from the freeze manifest. Use a new preregistered seed and identity namespace for any new scientific confirmation.

## Distribution boundary

The private working repository, private Rozephine data, unrelated experiments, and absolute local paths are excluded from journal/public packages. A reviewer supplement may contain reviewed source files, frozen configurations, sanitized reports, and case ledgers. Public release remains a separate author-controlled action.

The built confidential reviewer supplement contains byte-exact Run 009/010
ledgers, path-sanitized reports, a path-free portable held-out manifest, the
byte-exact original evaluator, a portable evaluator differing only in Git
metadata fallback, and the allowlisted import closure. Its internal 35-entry
SHA-256 manifest, ZIP CRC, source-to-package ledger equality, ledger validators,
portable freeze binding, UTF-8 no-BOM check, and prohibited identifier/path scan
all pass. The package does not convert a held-out reproduction into a second
independent confirmation.

After extracting the reviewer supplement, install its complete declared runtime
dependencies before executing the packaged commands:

```powershell
python -m pip install -r requirements.txt
```

The requirements file declares both PyTorch and NumPy because the bounded-state
hash path calls `Tensor.numpy()`.
