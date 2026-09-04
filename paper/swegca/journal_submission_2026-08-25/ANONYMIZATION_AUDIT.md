# Double-anonymization audit

## Anonymous upload set

- `main.tex`
- `references.bib`
- generated anonymous manuscript PDF
- `HIGHLIGHTS.txt`
- reviewer supplement ZIP

## Prohibited identifiers

The anonymous upload set must contain none of:

- PARK DONGJUN or Korean-script name variants;
- Independent Researcher affiliation;
- postal address, email, phone, GitHub handle, Drive identifier, or local user
  profile path;
- author-valued PDF metadata;
- acknowledgements or declaration text that identifies the author.

## Permitted project identifiers

`SWEGCA`, `Rozephine`, and `MOSAIC` name the architecture and audited
implementation, not the author. Their use is necessary for a precise methods
description. Repository URLs and public handles remain excluded during double
anonymized review.

## Artifact-path boundary

Opaque source revision hashes may be retained for integrity because the private
repository is not accessible to reviewers and the supplement contains no Git
history or remote metadata. Raw local paths must be replaced by package-relative
paths while their source-artifact hashes remain recorded.

## Current verified result

- Anonymous PDF: 11 pages; `/Author` absent; extracted text contains no author
  name, affiliation, local user/handle, repository URL, email, or Windows path.
- Reviewer ZIP: 36 members; CRC passed; all 35 inner manifest entries matched.
- Primary and held-out ledgers: 102,400 and 10,240 rows respectively; packaged
  bytes match the source ledgers exactly.
- Ledger schema: exactly ten allowlisted fields and ten allowlisted contract
  families; no prompts, user data, raw observations, model outputs, or arbitrary
  text fields.
- Sanitized report path fields: zero remaining after package extraction.
- Full package scan: zero prohibited identifier/path hits and zero UTF-8 BOM
  files.
