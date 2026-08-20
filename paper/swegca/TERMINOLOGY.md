# SWEGCA Terminology Ledger

This ledger is normative for the paper. Code identifiers remain unchanged and appear in backticks. A manuscript term may differ from an implementation identifier only where the mapping below is explicit.

## Identity and title

| Item | Canonical form | Rule |
|---|---|---|
| Acronym | **SWEGCA** | Never expand differently. |
| Exact expansion | **Single-World Evidence-Gated Cognitive Agency** | `Agency`, not `Architecture`, is the final acronym word. |
| Working title | **SWEGCA: An Architecture for Single-World Evidence-Gated Cognitive Agency in Persistent AI Agents** | The title may describe SWEGCA as an architecture while preserving the exact acronym expansion. |
| Author | **PARK DONGJUN** | Use this order and capitalization in the English manuscript. |
| Canonical Korean sentence | **SWEGCA는 비정제 관측을 곧바로 상태로 받아들이지 않고, 증거 검증과 충돌 해결을 거쳐 지속 가능한 내부 상태를 구성한다.** | Preserve verbatim. |
| Canonical English rendering | **SWEGCA does not admit uncurated observations directly into state; it constructs a persistent, revisable internal state through evidence validation and conflict resolution.** | Use in the abstract/introduction unless later evidence requires narrowing. |

## Epistemic objects

| Canonical paper term | Korean | Definition | Do not conflate with |
|---|---|---|---|
| **observation** | 관측 | A time-indexed input presented to cognition, such as a sensor result, tool result, user statement, or retrieved datum. | evidence, experience, state |
| **uncurated observation** | 비정제 관측 | An observation not preselected for correctness, success, agreement, or semantic authority. It must still be lawful and provenance-preserving. | provenance-free data, unauthorized data, verified evidence |
| **experience record** | 경험 기록 | A retained, addressable record of an observation, outcome, failed attempt, contradiction, or derived episode that may inform later cognition. | authoritative belief, semantic-memory fact |
| **status-unfiltered experience access** | 상태 비필터 경험 접근 | Paper term for the registered implementation path that does not exclude records by success/failure, verification status, or file type. Code/docs may call this `unrestricted experience`. | unrestricted acquisition rights, write authority, truth |
| **evidence** | 증거 | Addressable support or refutation evaluated under a declared admission policy for a specific hypothesis or mutation. Experience becomes evidence only in this claim-relative, policy-evaluated sense. | any retrieved experience, confidence |
| **evidence address** | 증거 주소 | A stable reference identifying the supporting or refuting artifact/record. | provenance metadata as a whole |
| **provenance** | 출처·계보 | Metadata describing source, revision, derivation, time, and relevant verification/outcome state. | proof of correctness |
| **proposal** | 제안 | A transient `SynapseProposal`: source, bounded-delta candidate, target mask, confidence, contradiction, uncertainty, and optional evidence addresses. | second state, accepted belief, commit |
| **authoritative evidence decision** | 권위 있는 증거 결정 | An `AccumulatorDecision` created by the registered accumulator capability. It may be `accept`, `reject`, or `abstain`. | producer confidence, proposal weight, factual truth |

## State and authority

| Canonical paper term | Definition | Usage rule |
|---|---|---|
| **persistent Cognitive State** | The sole main-owned `CognitiveState`, including semantic, executive, and scratch tensors plus structured and metadata components. | Capitalize `Cognitive State` when referring to the architecture object; use code form `CognitiveState` for the class. |
| **world-facing state** | The parts of persistent cognition and verified memory that encode claims about the external world. | Prefer this when the implementation spans `CognitiveState` and semantic memory. |
| **legacy/OMNI `WorldState`** | The separate compatibility representation with its own 32-by-256 default. | Never call it the universal current SWEGCA state. |
| **internal state** | Reader-friendly umbrella term for the persistent Cognitive State and its authorized world-facing content. | Do not use bare `state` where the target object could be ambiguous. |
| **Single-World** | Exactly one authoritative persistent Cognitive State owner for the registered cognitive entity. | It does not mean one metaphysically correct world, one hypothesis, or deletion of contradictions. |
| **authority** | The capability to cause a persistent mutation, semantic-memory promotion, external action, training/model update, distribution, or P3 promotion. | Always name the authority domain when ambiguity matters. |
| **producer** | A model, specialist, tool, worker, manager, or retrieval component that emits observations or transient proposals. | Producers are not separate persistent agents and do not own cognitive state. |
| **main process** | The sole owner of identity, persistent Cognitive State, and accumulated experience. | Do not call request-local managers/workers independent agents in the architecture section. |

## Decision and mutation operations

| Canonical term | Definition | Distinction |
|---|---|---|
| **evidence validation** | Structural, freshness, diversity, policy, and support checks that determine whether evidence is admissible for a decision. | Validation checks admissibility; it does not prove truth. |
| **evidence gate** | The policy/capability boundary that converts accumulated evidence into permission or no permission for a scoped write. | Not the proposal-merging arbiter. |
| **Single-World Arbiter** | The component that weights, bounds, combines, and suppresses conflicting transient proposals. | The low-level `SingleWorldArbiter(commit=True)` is authority-sensitive but is not by itself the fully guarded commit path. |
| **conflict resolution** | Falsification-first handling of accepted distinct-source proposals that target the same slot in opposing directions; the current implementation zeros the unresolved slot. | Do not imply that every semantic contradiction is solved. `abstention` is a valid resolution outcome. |
| **abstain / no-commit** | Insufficient or unresolved evidence results in no persistent mutation. | Use `hold` only when quoting historical reports; the accumulator's canonical status is `abstain`. |
| **reject** | Evidence is sufficient to support a negative decision under the accumulator policy. | Rejection is not the same as uncertainty/abstention. |
| **bounded write** | An authorized delta restricted to the registered verification slot and configured norm limits. | Not arbitrary World mutation. |
| **commit** | Application of an accepted delta to the persistent Cognitive State. | Proposal generation, retrieval, and dry-run authorization are not commits. |
| **semantic-memory promotion** | Admission of an episodic/candidate record into verified semantic memory under a linked accepted decision and World receipt. | Not equivalent to a Cognitive State tensor commit. |
| **rollback** | Receipt-checked restoration to the exact pre-write state after a known write. | Different from later retraction and startup recovery. |
| **retraction** | Removal of a still-active fact/update while preserving unrelated later cognition where supported. | Not a destructive reset of all later state. |
| **recovery** | Repair of an incomplete journaled World/memory operation, including restart-time handling. | Do not claim distributed atomicity. |

## Quantities and evidence strength

| Term | Meaning | Prohibited interpretation |
|---|---|---|
| **confidence** | Producer-provided scalar used in proposal weighting. | Calibrated probability of truth or mutation authority |
| **contradiction score** | Producer scalar that discounts its proposal weight. | The arbiter's slot-level `unresolved_contradiction` mask |
| **uncertainty** | Producer scalar that discounts proposal weight. | A universal epistemic uncertainty measure |
| **source diversity** | Count/proxy used by the accumulator policy. | Statistical or causal independence by itself |
| **distinct source** | Current arbiter comparison `left.source != right.source`. | Proof of independent evidence |
| **causal lower bound** | Minimum Wilson lower bound across configured evidence axes in the current accumulator. | General causal identification or intervention proof |
| **development evidence** | Results produced during architecture development, threshold choice, correction, or repeated-set evaluation. | Held-out scientific confirmation |
| **held-out confirmation** | Evaluation data not used for code, threshold, weight, curriculum, or stopping choices. | Reused or outcome-exposed development cases |

## Required claim qualifiers

- Use **“on the audited guarded verification-slot path”** for current positive enforcement claims.
- Use **“registered unrestricted-experience path”** only when naming the implementation; otherwise prefer **“status-unfiltered experience access.”**
- Use **“persistent, revisable internal state”** for the English rendering of `지속 가능한 내부 상태`.
- Use **“development evidence”** for G1, N253-N272, and same-50K Language VRS results unless a fresh held-out protocol says otherwise.
- State that proposal evidence-address and decision-to-delta binding remain open whenever describing the current guarded path as evidence-gated.

## Prohibited drift

- `SWEGCA = ... Architecture`
- `raw data` as a synonym for uncurated observations; reserve `raw` for undecoded/source sensor bytes or pixels
- `experience = evidence = belief`
- `confidence = authority` or `confidence = probability of truth`
- `arbiter = evidence gate`
- `rejected = uncertain`
- `rollback = retraction = recovery`
- `WorldState = CognitiveState`
- `status-unfiltered = provenance-free`
- `single-world = single hypothesis`
- `zero test failures = general safety`
