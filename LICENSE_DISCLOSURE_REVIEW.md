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

The repository is marked `All Rights Reserved` by PARK DONGJUN. No permission
for public copying, modification, or redistribution is granted. PyTorch is a
runtime dependency and is not redistributed in this repository; users obtain it
under PyTorch's own license.

## Required attribution

The author is identified as PARK DONGJUN. The repository's AI-use disclosure
must remain with any research submission derived from these materials.

## Redistribution notes

The repository is intended to remain private. A separate public-release review
is required before changing visibility or granting a redistribution license.

## Risks and unknowns

- The arXiv submission license has not been selected; that choice belongs to the
  author during submission.
- Public release would require a fresh dependency and provenance audit against
  the exact release tree.
- Legacy `mosaic_*` and Rozephine-named compatibility identifiers remain for
  reproducibility and do not imply inclusion of the original product.

## Recommended action

Keep the GitHub repository private and retain the current all-rights-reserved
notice. Do not make it public until PARK DONGJUN explicitly chooses a source
license and the exact release tree passes a new license and disclosure review.
