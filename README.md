# SWEGCA Architecture

Standalone implementation of **SWEGCA — Single-World Evidence-Gated Cognitive Agency** by PARK DONGJUN.

> SWEGCA does not admit uncurated observations directly into state; it constructs a persistent, revisable internal state through evidence validation and conflict resolution.

## Scope

This private repository contains only the architecture boundary evaluated in the SWEGCA paper:

- one main-owned persistent `CognitiveState`;
- authority-limited transient proposals;
- status-unfiltered experience access with zero read authority;
- evidence accumulation with `accept`, `reject`, and `abstain`;
- coded same-slot directional-conflict no-commit behavior;
- a bounded verification-slot writer with receipts and rollback; and
- local journaled World/semantic-memory promotion, retraction, and recovery.

SWEGCA는 비정제 관측을 곧바로 상태로 받아들이지 않고, 증거 검증과 충돌 해결을 거쳐 지속 가능한 내부 상태를 구성한다.

Legacy `mosaic_*` filenames and a small number of Rozephine-named receipt fields are retained to preserve the audited implementation contract and paper reproducibility. The original product runtime, models, model serving, chat system, datasets, frozen outputs, private reports, and unrelated tools are not included.

## Important negative result

The current guarded writer accepts matching, empty, and decision-mismatched proposal evidence addresses when its other gates pass. Complete decision-to-proposal and tensor-semantic evidence binding is a target invariant, not a proven current property. See `docs/KNOWN_LIMITATIONS.md` and the paper artifacts.

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
- manuscript source: `paper/swegca/main.tex`
- adversarial review: `paper/swegca/ADVERSARIAL_REVIEW.md`
- fixed evaluation scripts and hash-bound result JSON: `paper/swegca/scripts/` and `paper/swegca/results/`

The standalone rerun preserves both outcomes: the fixed synthetic contract
matrix passes, while the evidence-address binding falsification remains a
negative result. The `*_standalone.json` files bind those results to the initial
standalone source commit.

## Distribution

This repository is private and all rights are reserved. No public redistribution license has been granted. Extensive generative-AI use in the associated research is disclosed in the manuscript.

The full disclosure is preserved in `AI_USE_DISCLOSURE.md`.
