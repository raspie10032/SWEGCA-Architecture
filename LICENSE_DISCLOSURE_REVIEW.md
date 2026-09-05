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

PARK DONGJUN changed the license of the current SWEGCA source repository to the
MIT License in September 2026. The complete license text is in `LICENSE`, and a
short copyright/license notice is in `NOTICE`. The MIT License permits use,
copying, modification, merging, publication, distribution, sublicensing, and
sale of copies, provided that the copyright and permission notice are retained
in copies or substantial portions of the Software.

Earlier published repository revisions that were distributed under MPL-2.0
remain subject to the license terms that accompanied those revisions. This
license change applies to the current repository version and later revisions
unless the author states otherwise.

PyTorch is a runtime dependency and is not redistributed in this repository;
users obtain it under PyTorch's own license.

## Required attribution

The copyright holder is identified as PARK DONGJUN in `LICENSE` and `NOTICE`.
Redistributions under the MIT License must retain the copyright notice and the
MIT permission notice in copies or substantial portions of the Software. The
repository's AI-use disclosure must remain with research submissions derived
from these materials where applicable.

## Redistribution notes

The author authorized public release on 2026-09-04. Repository visibility is
independent of the license grant. Before the GitHub visibility change, the
exact-tree secret, provenance, dependency, generated-artifact, anonymous-PDF,
and reviewer-package integrity checks were rerun.

## Risks and unknowns

- The MIT License applies to the SWEGCA source code in the current repository
  version. The manuscript distribution license is separate.
- The manuscript text and PDF remain a pre-peer-review preprint. No separate
  manuscript distribution license has been selected, so they remain all rights
  reserved unless the author later states otherwise.
- Legacy `mosaic_*` and Rozephine-named compatibility identifiers remain for
  reproducibility and do not imply inclusion of the original product.

## Release action

Retain `LICENSE`, `NOTICE`, provenance, and AI-use disclosure. Publish the
paper-bound snapshot while preserving the initial standalone negative result as
a separate versioned finding.
