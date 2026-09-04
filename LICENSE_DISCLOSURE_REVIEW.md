# License and disclosure review

## Assets reviewed

- standalone Python source and tests extracted from the author's private source
  repository;
- SWEGCA manuscript, architecture specification, terminology, evaluation
  scripts, and result JSON;
- packaging and repository documentation; and
- the declared PyTorch runtime dependency.

No model weights, datasets, third-party code snapshots, images, audio, video,
fonts, or compiled binaries are included.

## License requirements

PARK DONGJUN selected the Mozilla Public License 2.0 (`MPL-2.0`) for the SWEGCA
source repository on 2026-08-20. The complete unmodified license text is in
`LICENSE`, and the source-form notice is in `NOTICE`. MPL-2.0 applies file-level
copyleft: distributed modifications to covered source files remain under
MPL-2.0, while separate files in a Larger Work may use other terms subject to
the license. PyTorch is a runtime dependency and is not redistributed in this
repository; users obtain it under PyTorch's own license.

## Required attribution

The initial copyright holder is identified as PARK DONGJUN in `NOTICE`. License,
copyright, patent, warranty, and liability notices in covered source must not be
removed or materially altered except to correct factual inaccuracies. The
repository's AI-use disclosure must remain with research submissions derived
from these materials.

## Redistribution notes

The author authorized public release on 2026-09-04. Visibility is independent
of the MPL-2.0 grant. If covered executable form is distributed, the covered
source must be made available as required by MPL-2.0 Section 3.2. Before the
GitHub visibility change, the exact-tree secret, provenance, dependency,
generated-artifact, anonymous-PDF, and reviewer-package integrity checks were
rerun.

## Risks and unknowns

- MPL-2.0 licenses the SWEGCA source code. The arXiv manuscript distribution
  license is separate and has not been selected; that choice belongs to the
  author during submission.
- The manuscript text and PDF remain a pre-peer-review preprint. No separate
  manuscript distribution license has been selected, so they remain all rights
  reserved unless the author later states otherwise.
- Legacy `mosaic_*` and Rozephine-named compatibility identifiers remain for
  reproducibility and do not imply inclusion of the original product.

## Release action

Retain `LICENSE`, `NOTICE`, the MPL-2.0 package metadata, provenance, and AI-use
disclosure. Publish the paper-bound snapshot while preserving the initial
standalone negative result as a separate versioned finding.
