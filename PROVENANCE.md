# Extraction provenance

## Source

- source repository: `raspie10032/tinylm-slicer` (private working source at extraction time)
- paper-final source commit: `67915dd8010c2a19dcd9e8ad6ca474c36a97dc80`
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
