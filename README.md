# SWEGCA Architecture

Standalone implementation of **SWEGCA — Single-World Evidence-Gated Cognitive Agency** by PARK DONGJUN.

**Pronunciation:** `SWEGCA` is pronounced **“스웨카”** in Korean. The Korean reading **스웨카** is the reference pronunciation used by this project.

> SWEGCA does not admit uncurated observations directly into state; it constructs a persistent, revisable internal state through evidence validation and conflict resolution.

## Scope

This repository contains only the architecture boundary evaluated in the SWEGCA paper:

- one main-owned persistent `CognitiveState`;
- authority-limited transient proposals;
- status-unfiltered experience access with zero read authority;
- evidence accumulation with `accept`, `reject`, and `abstain`;
- coded same-slot directional-conflict no-commit behavior;
- a bounded verification-slot writer with receipts and rollback; and
- local journaled World/semantic-memory promotion, retraction, and recovery.

SWEGCA는 비정제 관측을 곧바로 상태로 받아들이지 않고, 증거 검증과 충돌 해결을 거쳐 지속 가능한 내부 상태를 구성한다.

Legacy `mosaic_*` filenames and a small number of Rozephine-named receipt fields are retained to preserve the audited implementation contract and paper reproducibility. The original product runtime, models, model serving, chat system, datasets, frozen outputs, private reports, and unrelated tools are not included.

## Versioned evidence boundary

The initial 2026-08-20 standalone extraction preserves an important negative
result: its guarded writer accepts matching, empty, and decision-mismatched
proposal evidence addresses when its other gates pass. The 2026-08-25 journal
snapshot separately records the repaired guarded path, its exact-field binding
boundary, and deterministic development and held-out contract evidence. Neither
version claims semantic truth, natural-world reliability, or general agent
safety.

The paper-bound source closure, portable evaluator, path-sanitized reports, and
full generated case ledgers are published under
`reproducibility/evidence_binding_2026-08-25/`. Keeping both snapshots visible
prevents the repaired result from rewriting the earlier negative finding.

## Test

Python 3.11+ and PyTorch are required.

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
```

The extraction acceptance target is the 73 architecture tests carried from the audited repository snapshot.

## Paper and evaluation

- architecture specification: `paper/swegca/ARCHITECTURE_SPEC.md`
- normative terminology: `paper/swegca/TERMINOLOGY.md`
- initial manuscript source: `paper/swegca/main.tex`
- adversarial review: `paper/swegca/ADVERSARIAL_REVIEW.md`
- fixed evaluation scripts and hash-bound result JSON: `paper/swegca/scripts/` and `paper/swegca/results/`
- submitted journal preprint snapshot, including the verified anonymous PDF:
  `paper/swegca/journal_submission_2026-08-25/`
- paper-bound evidence-binding reproduction package:
  `reproducibility/evidence_binding_2026-08-25/`

The standalone rerun preserves both outcomes: the fixed synthetic contract
matrix passes, while the evidence-address binding falsification remains a
negative result. The `*_standalone.json` files bind those results to the initial
standalone source commit.

## Publication status and distribution

This repository was released publicly by the author on 2026-09-04. The source
code is licensed under the [MIT License](LICENSE), effective for the current
repository version from the license-change commit onward. Earlier published
revisions remain available under the license terms that accompanied those
revisions at the time of distribution.

The manuscript text and PDF are a pre-peer-review preprint and remain copyright
PARK DONGJUN with all rights reserved unless a separate manuscript license is
stated.

The 2026-08-25 manuscript snapshot was submitted to *Cognitive Systems
Research*. It has not been peer reviewed or accepted, and this repository is not
an Elsevier publication. Public preprint availability does not convert the
submission into a published journal article.

Extensive generative-AI use in the associated research is disclosed in the
manuscript.

The full disclosure is preserved in `AI_USE_DISCLOSURE.md`.
