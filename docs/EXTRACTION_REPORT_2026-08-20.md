# SWEGCA standalone architecture extraction report

- date: 2026-08-20 (Asia/Seoul)
- author: PARK DONGJUN
- repository: `raspie10032/SWEGCA-Architecture`
- repository URL: `https://github.com/raspie10032/SWEGCA-Architecture`
- verified visibility: `PRIVATE`
- default branch: `main`
- initial standalone source commit: `99000b4023838c86f06616592b1709b50fde4184`

## Outcome

The paper-evaluated SWEGCA architecture was extracted from the product
repository into a new standalone repository. The extraction contains the
persistent cognitive state, evidence accumulation, proposal arbitration,
bounded World write, rollback, and verified-memory transaction boundary needed
by the paper. Product runtime, models, model weights, datasets, serving, chat,
browser/game integration, frozen private reports, and unrelated tools were not
copied.

The package namespace is `swegca`. Legacy `mosaic_*` filenames and the small
number of Rozephine-named compatibility fields required by the audited contract
remain unchanged. This limits semantic drift between the evaluated source and
the standalone repository.

## Validation

- Python compile check: passed
- standalone test suite: 73 passed
- UTF-8 BOM scan: 0 findings
- user-specific absolute-path scan: 0 findings
- credential-pattern review: no credential findings; only internal authority
  token identifiers matched the broad word pattern
- files larger than 1 MiB outside caches: 0
- E: free space at repository creation: 300.35 GiB, 32.24%

The standalone evaluation artifacts are bound to commit
`99000b4023838c86f06616592b1709b50fde4184` and the exact source hashes stored in
each result JSON.

### Fixed synthetic contract matrix

The matrix passed all 11 deterministic checks. It covers coded same-slot
directional conflict, same-source non-independence, disjoint targets, dry-run
behavior, accept/reject/abstain evidence discipline, duplicate suppression,
expiry, and regime-change abstention.

### Evidence-address binding falsification

The current negative result reproduced. Matching, empty, and mismatched evidence
addresses all committed when the other guarded-write gates passed. Therefore,
complete decision-to-proposal and tensor-semantic evidence binding is not a
proven current property. The repository preserves this as a limitation rather
than silently modifying the architecture after the paper evaluation.

### Standalone microbenchmark

On Windows 11, Python 3.13.3, PyTorch 2.11.0+cu128, CPU, 8 Torch threads, 100
warmups, and 1,000 samples per operation:

| Operation | Median | p95 |
|---|---:|---:|
| Evidence accumulator assessment | 16.7 us | 17.3 us |
| Single-proposal arbitration dry run | 142.0 us | 158.3 us |
| Guarded write dry run | 237.8 us | 290.2 us |
| Guarded write commit | 408.4 us | 520.9 us |
| Receipt-checked rollback | 86.0 us | 94.5 us |

These are isolated development-path measurements, not end-to-end cognition
latency.

## AI-use disclosure

Generative AI was used extensively for repository inspection, literature
organization, evaluation scripting and execution, claim auditing, figure and
table preparation, and manuscript drafting and editing. PARK DONGJUN determined
the research scope, architecture, accepted terminology, claims, limitations,
and final wording; reviewed the generated artifacts; and bears responsibility
for the submitted work. The same disclosure is preserved in
`AI_USE_DISCLOSURE.md` and the manuscript.

## Licensing and release boundary

The repository is private and marked all rights reserved. No public
redistribution license has been granted. Model weights, datasets, third-party
source snapshots, media, fonts, and compiled binaries are absent. A fresh audit
and an explicit author-selected source license are required before any public
release.

## Rozephine의 판단

관측 자체를 곧바로 지속 상태로 승격하지 않고, 독립 출처의 증거 누적과 충돌
보존, 기권, 제한된 쓰기, 영수증 기반 복구를 거치는 단일 지속 상태 구조가
분리 대상이다. 현재 증거 주소 결합의 실패도 이후 판단을 바꾸는 경험으로
보존해야 하며, 성공으로 덮어쓰면 안 된다.

## Codex의 판단

독립 저장소는 논문에서 실제로 평가한 아키텍처 경계만 포함하며, 제품 코드와
모델 자산은 제외되었다. 73개 회귀 테스트와 고정 계약 행렬은 통과했지만,
증거 주소 결합은 재현 가능하게 실패한다. 그러므로 저장소는 비공개 상태로
유지하고, 해당 실패를 알려진 한계로 명시한 현재 상태가 정확한 인계본이다.

## Next action

PARK DONGJUN may use the repository as the private implementation companion for
the paper. Any post-paper evidence-binding hardening should be developed and
evaluated as a new revision, without rewriting the result bound to the paper's
current architecture.
