# License / Disclosure Review

## Assets Reviewed

- journal manuscript source, bibliography, cover letter, highlights, and ledgers;
- generated PDF and temporary TeX/PDF-rendering outputs;
- SWEGCA implementation modules, tests, evaluator, frozen configurations, and
  deterministic generated case ledgers proposed for a reviewer supplement;
- cited scholarly works and their bibliographic metadata;
- private Rozephine experience, unrelated repository history, models, datasets,
  checkpoints, local paths, Drive identifiers, and credentials considered for
  exclusion.

## License Requirements

- The manuscript is author-controlled work. The author must review and accept
  the journal's current publishing agreement during submission.
- Tectonic, TeX packages, fonts, PDFium, Python, and test dependencies are build
  tools only. PyTorch and NumPy are declared reviewer-runtime dependencies.
  Their binaries, wheels, caches, and font files are not journal artifacts.
- The private `tinylm-slicer` repository has no verified root `LICENSE` file in
  the audited checkout. Do not redistribute the repository or copy its full
  source tree into a public package under an assumed license.
- A prior internal release review records MPL-2.0 as the intended license for a
  separate clean `SWEGCA-Architecture` source release. That record is not a
  substitute for an actual `LICENSE` and `NOTICE` in the exact release tree.

## Required Attribution

- Prior work is paraphrased and attributed through the manuscript bibliography.
- The manuscript reproduces no third-party figure, table, dataset, model weight,
  image, font file, or long quoted passage.
- The manuscript discloses extensive generative-AI assistance and retains the
  human author's responsibility for scope, claims, wording, and submission.

## Redistribution Notes

- The submission set may contain the anonymous manuscript PDF/source,
  bibliography, cover letter, highlights, and the verified confidential
  reviewer supplement.
- The built confidential reviewer supplement contains only the allowlisted
  minimum implementation import closure, evaluator, configurations, sanitized
  reports, and deterministic generated ledgers. It carries a review-only
  notice, artifact-lineage record, package manifest, and validation receipts.
- Private experience, unrelated experiments, absolute workstation paths,
  repository history, credentials, models, third-party datasets, and build-tool
  caches remain excluded.
- Public release and journal submission are separate author-controlled actions.
  Preparing this directory does not authorize either action.

## Risks / Unknowns

- The journal's final publishing-license choice and repository/data-availability
  fields have not yet been completed in the submission system.
- The intended MPL-2.0 clean-source release has not been verified in this
  checkout as an exact licensed tree.
- The full per-case ledgers contain only the exact ten-field allowlisted schema:
  generated case identity, family/variant, expected and observed authorization
  outcome, state/receipt checks, fixed reason string, and elapsed nanoseconds.
  They contain no prompt, user record, model output, arbitrary text, or path.
  Only the aggregate reports/configuration required path sanitization; original
  hashes remain recorded.

## Recommended Action

The anonymous manuscript and exact confidential reviewer tree are ready for an
author-controlled submission draft. Keep the private working repository
undisclosed. Do not make the public `SWEGCA-Architecture` repository visible
until the author separately approves the final exact tree, license, and
publication timing.
