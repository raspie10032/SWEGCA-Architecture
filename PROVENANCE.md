# Extraction provenance

## Source

- source repository: `raspie10032/tinylm-slicer` (private working source at extraction time)
- initial paper-final source commit: `67915dd8010c2a19dcd9e8ad6ca474c36a97dc80`
- pre-submission manuscript revision commit: `73bfb874ac9b2d6f7cbf1c0071131daaf1acf63f`
- architecture implementation evaluated at clean commit: `168e9b760b8f6b3e8ff767e1ebb80b310b1a6a28`
- extraction date: 2026-08-20
- author: PARK DONGJUN

## Included implementation

Thirteen architecture and direct-support modules were copied from `src/tinylm_slicer/` and placed under `src/swegca/`. Imports and hard-coded evaluator source paths were mechanically changed from the original package namespace to `swegca`.

Two large product/benchmark dependencies were not copied wholesale:

- `mosaic_omni.py`: only the exact `SLOT_ROLES`, `SurfaceResidualRef`, and `WorldState` compatibility contracts required by the architecture were extracted;
- `mosaic_sqlite_memory.py`: only the exact `_fts_query` helper required by external memory was extracted.

The product-specific database build tool used by one test was replaced with a minimal schema-equivalent fixture in `tests/_database.py`.

## Excluded

No model, model runtime, model weight, dataset, image/audio/video asset, product API, chat memory, browser/game integration, frozen report, Google Drive artifact, compiler binary, secret, or user-specific absolute path is included.

An exact file-hash manifest is generated after standalone validation.

## Standalone validation anchor

- initial standalone source commit: `99000b4023838c86f06616592b1709b50fde4184`
- private repository: `raspie10032/SWEGCA-Architecture`
- architecture tests: 73 passed
- standalone fixed contract matrix: passed
- standalone evidence-address binding contract: failed as expected and retained
  as an explicit negative result

## Journal snapshot and public release

- journal manuscript snapshot: 2026-08-25
- anonymous manuscript SHA-256:
  `a5110bd43b43d1ab7807ca4824f13783f593e9f1eb8348cc05c7f5fc7b514e71`
- paper-bound reproduction archive SHA-256:
  `59a7959e3ef12d27ac90720f6a38c56fa1cd5198a66ca1ea34481e17a01c5690`
- public release authorization: PARK DONGJUN, 2026-09-04

The journal snapshot does not replace the initial extraction. It adds the
later repaired guarded-path manuscript and a public-safe reproduction package
whose source closure, configurations, generated case ledgers, reports, and
validation receipts retain their original hashes. The private source
repository, its Git history, unrelated Rozephine experience, models,
checkpoints, contact files, cover letter, and title page remain excluded.
