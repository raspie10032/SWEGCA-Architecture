# Journal v2 artifact index

| Path | Type | Role | Source | Status | Verification / next action |
|---|---|---|---|---|---|
| `main.tex` | LaTeX source | Double-anonymized journal research paper | Copied byte-exact from frozen v1, then evidence-binding and ordered-component claims synchronized | Submission draft | 220-word abstract; approximately 5,450 words through the conclusion; terminology, citation, build, identity scan, and visual QA passed |
| `references.bib` | BibTeX | Bibliography | Frozen v1 bibliography | Copied; unchanged | BibTeX build resolved without undefined citations |
| `TERMINOLOGY.md` | Markdown | Controlled vocabulary and claim boundaries | Frozen v1 terminology with guarded-binding and ordered-component updates | Verified draft | Terminology checker passed over 26 manuscript files, including the reviewer-defense documents |
| `COMPONENT_LEDGER.md` | Markdown | Ordered runtime component and invariant cross-check | Canonical SWEGCA definition, registered implementation, and tests | Verified draft | Runtime steps 0--11 and invariants I01--I12 match the manuscript; targeted component tests passed 39/39 |
| `CLAIM_EVIDENCE_MATRIX.md` | Markdown | Journal v2 claim audit | Final implementation, Run 009, Run 010, regression | Draft | Cross-check against final manuscript |
| `RESULT_LEDGER.md` | Markdown | Exact counts, intervals, and hashes | Frozen reports and case ledgers | Draft | Cross-check every manuscript number |
| `EXPERIMENT_GAPS.md` | Markdown | Remaining limits | Completion report and v1 gaps | Draft | Keep excluded claims explicit |
| `REPRODUCIBILITY.md` | Markdown | Frozen lineage and commands | Committed evaluator/config/tests | Verified draft | Corrected reviewer supplement built and cross-machine macOS arm64 unpacked/verified; not external replication |
| `LICENSE_DISCLOSURE.md` | Markdown | License, attribution, AI-use, and redistribution boundary | Exact checkout and prior release review | Verified for confidential review package | Public license and journal publishing agreement remain author-controlled |
| `HIGHLIGHTS.txt` | Text | Journal submission highlights | Journal v2 contributions | Verified draft | Five highlights; lengths 72, 70, 69, 67, and 73 characters |
| `COVER_LETTER.md` | Markdown | Research-paper cover letter | Journal scope and manuscript evidence | Draft | Author must perform submission-time declarations |
| `REVIEWER_DEFENSE.md` | Markdown | Internal anticipated-concern and evidence-bound response map | Journal v2 claims, results, gaps, reproducibility artifacts, and official journal guidance | Internal working document | P0 statistical-unit repair is flagged; no hypothetical comment may be presented as an actual review |
| `REVIEW_SCOPE_BOUNDARY.md` | Markdown | Exact-claim versus expanded-interpretation boundary and response protocol | Claim/evidence matrix, experiment gaps, and recurring scope-shift axes | Internal working document | Requires neutral boundary clarification and forbids using scope control to evade a real defect |
| `REVIEW_EVIDENCE_CARDS.md` | Markdown | Reviewer-lens routing, claim zones, advanced technical attacks, and revision experiment cards | User-supplied idea source checked against journal v2 manuscript, implementation, artifacts, and primary literature | Internal working document | Current facts are separated from unexecuted P0/P1 cards; no prepared experiment is a result |
| `REVIEW_LITERATURE_MATRIX.md` | Markdown | Closest-work defense matrix | Checked primary sources for CK, MemTX, LedgerMind, Kumiho, and CoALA plus journal v2 ledgers | Internal working document | Source-qualified comparison only; no ranking, superiority, first, or exhaustive-review claim |
| `REVIEW_RESPONSE_LEDGER.md` | Markdown | Point-by-point reviewer-response and revision-verification template | Elsevier response guidance and journal v2 artifact identity | Ready template | Empty until an exact decision letter and reviewer comments are received |
| `scripts/check_terminology.py` | Python | Local terminology gate | Frozen v1 checker | Copied; unchanged | Passed over 26 files in the current journal directory |
| `output/pdf/main.pdf` | Generated PDF | Superseded author-identified review copy | Pre-anonymization `main.tex` + `references.bib` | Internal only | Never upload for double-anonymized review |
| `output/pdf/SWEGCA_anonymized_manuscript.pdf` | Generated PDF | Anonymous manuscript | Current `main.tex` + `references.bib` | Verified submission PDF | 11 pages; no author metadata or prohibited identifier; zero layout/citation warnings; every page visually checked |
| `scripts/validate_case_ledger.py` | Python | Exact ledger/report gate | Frozen case schema and outcomes | Verified | Full Run 009/010 validation passed; rejects extra text fields and reordered histories |
| `scripts/build_reviewer_supplement.py` | Python | Allowlisted supplement builder | Validated runs plus minimal source closure | Verified | Declares NumPy and PyTorch runtime dependencies; tests passed; deterministic ZIP; full-tree identifier/path scan |
| `submission/output/SWEGCA_reviewer_supplement.zip` | ZIP | Confidential reviewer supplement | Byte-exact ledgers, path-sanitized reports, portable reproduction files | Verified | 36 members; CRC and 35-entry inner manifest passed; 0 prohibited identifier/path hits; clean macOS arm64 install and rerun passed |
| `cleanroom/macos_arm64_20260825/` | Markdown + JSON | Internal cross-machine reproduction report and receipts | Corrected reviewer ZIP executed on user-owned M2 Mac | Verified internal evidence | Original missing-NumPy failure retained; corrected-package primary and held-out outcome fields matched; not external replication or a fresh held-out confirmation |

## Current SHA-256 manifest

| Path | SHA-256 |
|---|---|
| `main.tex` | `d9792073dcc09b31460a2d712c7739c136f326c164c132cd1291ac6d94498ad2` |
| `references.bib` | `23d8108f854470777cb37b3b16d4e597ae8a43eebf02bcb00dc3ad41ee2c1e55` |
| `output/pdf/SWEGCA_anonymized_manuscript.pdf` | `a5110bd43b43d1ab7807ca4824f13783f593e9f1eb8348cc05c7f5fc7b514e71` |
| `COVER_LETTER.md` | `00ab8de10ddef416a1de2ca5b78ebcf57ab58fe6dffd2919c0ae84955a974764` |
| `HIGHLIGHTS.txt` | `4343db6a67d5f639df61ff90379207a5ca02c0da6b92070de1d6e91afc4f4ec7` |
| `COMPONENT_LEDGER.md` | `8943978c87238da2df18ea31d02273183be698b44fdd449338e8ba67d7b76ebb` |
| `REPRODUCIBILITY.md` | `16f65dcabe37af4e4629536af12749fff836227014befd88a49def458a1ab51d` |
| `LICENSE_DISCLOSURE.md` | `941bf016db7a595cae1c3a9d2f6200743fdf0e6feb677b4800186ee2b97cafe2` |
| `REVIEWER_DEFENSE.md` | `05ed3f3be34191ac490778247b4cb29d6711e1ca6e6f0abc5a69ec2478ab7ce1` |
| `REVIEW_SCOPE_BOUNDARY.md` | `c66f143144a4e1336dc0d05bba72dd07ee0833ea9a9da5a3c29b700533f2b5ef` |
| `REVIEW_EVIDENCE_CARDS.md` | `03121dc0e9bf37c165403b0df9dae88948b687e81e9fc9b3f686c833ee5feb14` |
| `REVIEW_LITERATURE_MATRIX.md` | `cc2e3a8ddd9ebb84108774bba011161d14e78b0fd7c2f1cc87ff2254a9743d20` |
| `REVIEW_RESPONSE_LEDGER.md` | `09d2f71328a5b5cc918d292b3cda509e183f854c17a13d300828ec14c13d9362` |
| `scripts/validate_case_ledger.py` | `84be51480534ae5161276f95f7bfb6caa7dd612f84b0ab0b849510c9263ce661` |
| `scripts/build_reviewer_supplement.py` | `117d99ee39b9044eaf6ef2590c7586c663087aec0e75d374821b9ecc7dccbcb9` |
| `submission/output/SWEGCA_reviewer_supplement.zip` | `59a7959e3ef12d27ac90720f6a38c56fa1cd5198a66ca1ea34481e17a01c5690` |
| `submission/output/SWEGCA_reviewer_supplement.zip.sha256` | `f24fd2d298cc38981eb04e5b1b38b26cd326e3b210bf801685004171da31bcb9` |
| `cleanroom/macos_arm64_20260825/CLEANROOM_REPORT.md` | `7cff8e6724992e7f3c2b278673261f3407e39c29f5212998975db5c3776081fc` |
| `cleanroom/macos_arm64_20260825/primary_report.json` | `7d04030a4f3797b75e071d241030c95f245b313c229670246e229510c10f81e9` |
| `cleanroom/macos_arm64_20260825/heldout_report.json` | `112c34626a0312a917ab5fe6146017085950e53432a9bae12ec2cff72ff740ec` |
| `cleanroom/macos_arm64_20260825/primary_validation.json` | `313e83a877b36c073961e79589e905fc5c4e1607e3aa096b30f0652db52309a1` |
| `cleanroom/macos_arm64_20260825/heldout_validation.json` | `819c67f02d5d34a4ceabea5009527a138ebca97e910d661805952727f578f64d` |

Generated LaTeX auxiliaries and PDFs are not source files. The immutable v1 remains under `paper/swegca/` and is not superseded or modified by this directory.
