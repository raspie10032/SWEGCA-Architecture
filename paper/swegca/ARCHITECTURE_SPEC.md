# SWEGCA Architecture Specification

Status: Phase D draft. Terminology is governed by `TERMINOLOGY.md`. This document describes both the intended SWEGCA contract and the narrower contract enforced by the audited Rozephine/MOSAIC implementation.

## 1. Scope and positioning

> SWEGCA는 비정제 관측을 곧바로 상태로 받아들이지 않고, 증거 검증과 충돌 해결을 거쳐 지속 가능한 내부 상태를 구성한다.

> SWEGCA does not admit uncurated observations directly into state; it constructs a persistent, revisable internal state through evidence validation and conflict resolution.

SWEGCA addresses a mutation-authority problem, not a data-cleaning problem. An observation or experience record may remain available even if it is failed, rejected, contradictory, or unverified. Availability does not grant authority to mutate the persistent Cognitive State, promote semantic memory, execute an external action, update a model, or claim P3 completion.

The current implementation claim is limited to the registered status-unfiltered experience path and the guarded verification-slot write path. The low-level arbiter remains callable with `commit=True`, and proposal evidence addresses plus exact decision-to-delta semantics are not universally bound.

## 2. System boundary

The **main process** owns identity, accumulated experience, and one persistent `CognitiveState`. Models, tools, specialists, request-local managers, and workers are **producers**. They may read a detached state snapshot and status-unfiltered experience, then return observations or transient proposals. They do not own a second persistent cognitive state and do not directly authorize external effects.

The audited core path is:

```text
uncurated observation / experience record
        -> addressable candidate with provenance
        -> authority-limited producer proposal
        -> evidence accumulation and decision
        -> evidence gate
        -> Single-World Arbiter and conflict abstention
        -> bounded verification-slot write
        -> receipt-bound semantic-memory promotion
        -> rollback, retraction, or journal recovery when required
```

## 3. Problem formulation

### 3.1 Objects

At logical time `t`:

- `S_t` is the main-owned persistent Cognitive State.
- `o_t` is an observation. It may be uncurated.
- `U_t = {r_j}` is the status-unfiltered experience universe. Each retained record has an address, source/revision metadata, and whatever verification or outcome state is available.
- `C_t` is the set of experience candidates retrieved for the current query.
- `rho_t` is the replayable experience-selection receipt.
- `p_{i,t}` is producer `i`'s transient proposal.
- `D_t` is an authoritative evidence decision with status `accept`, `reject`, or `abstain`.
- `Delta_t` is the bounded aggregate delta proposed for the persistent Cognitive State.
- `R_t` is the bounded-write receipt, if a commit occurs.

Status-unfiltered experience selection is modeled as

```text
Select(q_t, U_t) -> (C_t, J_t, rho_t)
Authority(rho_t) = empty
```

where `J_t` records selected and rejected candidate judgments. `Authority(rho_t) = empty` means that selection alone grants no World, memory, external-action, training, distribution, or P3 authority.

A producer operates on an observation, a detached state view, and retrieved candidates:

```text
Producer_i(o_t, detach(S_t), C_t) -> p_{i,t}
```

The implemented proposal has the form

```text
p_i = (source_i, delta_i, mask_i, confidence_i,
       contradiction_i, uncertainty_i, addresses_i)
```

It is a candidate delta, never a second Cognitive State.

### 3.2 Failure model

The architecture considers at least these failures:

1. an uncurated observation is mistaken for authoritative state;
2. a failed or unverified experience is retrieved and treated as permission;
3. a high-confidence producer bypasses evidence admission;
4. stale, duplicate, or source-concentrated evidence is accepted;
5. opposing proposals are averaged into a misleading residual;
6. an accepted decision for hypothesis A authorizes a delta for hypothesis B;
7. a proposal changes slots outside its declared target;
8. a World-facing write and semantic-memory promotion become torn;
9. a worker mutates shared state or retains identity across requests; and
10. rollback/retraction is attempted after unrelated state has changed.

### 3.3 Authority domains

Authority is scoped. The paper distinguishes:

- Cognitive State commit authority;
- verified semantic-memory promotion authority;
- external-action authority;
- training/model-update authority;
- distribution authority; and
- P3 promotion authority.

No read, retrieval, confidence value, producer identity, or model name implicitly grants any of these domains.

## 4. Architecture

### 4.1 Single-World persistent Cognitive State

`CognitiveState` holds semantic, executive, and scratch tensors plus a structured world graph, evidence references, goal/value/self metadata, and one `owner_id`. Its `persistent_state_count` is one. Request-local worker execution receives detached snapshots and returns proposals.

“Single-World” is an ownership rule: only one persistent Cognitive State is authoritative for the registered cognitive entity. It does not require one hypothesis, erase conflicting experience, or assert that the represented world is correct.

Implementation boundary: this property is strong on registered core paths, but Phase A did not prove process-wide uniqueness for every external integration.

### 4.2 Status-unfiltered experience access

The registered `mosaic_unrestricted_experience.py` path discovers regular files under declared experience roots without filtering by verification state, outcome status, or file type. It creates stable addresses and a metadata-snapshot digest, builds a cold-path address/semantic-key index, and serves hot lookups without per-judgment file, JSON, SQLite, network, or hash work.

Runtime cognition judges every retrieved candidate and records selection, rejection, relevance, contradiction, verification state, revision, rationale, and rejection evidence. The receipt has all consequential authority flags set to false.

This component preserves access to negative experience but does not validate its content. An experience record becomes evidence only when it is addressable, claim-relevant, and admitted by the evidence policy.

### 4.3 Authority-limited producers

Each producer returns a `SynapseProposal` with a finite delta matching the state shape, a boolean target mask, bounded scalar scores, a nonempty source identifier, and optional batch evidence-address tuples. Proposal construction and validation do not mutate `S_t`.

The current low-level type permits `addresses_i` to be empty. Therefore, “every proposal is evidence-carrying” is a target contract, not a universal implementation fact.

### 4.4 Evidence accumulation and authoritative decision

The accumulator maintains configured evidence axes, including counterfactual and intervention axes in the current default. It discounts stale/duplicate contributions, measures effective samples and source/context diversity, computes Wilson intervals, checks recent regime change, and emits:

```text
D_t.status in {accept, reject, abstain}
```

The current decision rule abstains on insufficient per-axis samples, insufficient source diversity, insufficient context diversity, or suspected regime change. It accepts when the minimum axis lower bound exceeds the configured threshold, rejects when the overall upper bound remains below it, and otherwise abstains.

The implementation marks registered accumulator state and decisions with process-local authority capabilities. A forged data object with matching visible fields is not equivalent to an authoritative decision.

### 4.5 Evidence gate and the binding gap

For the guarded verification-slot path, the implemented authorization predicate checks:

- accepted evidence status;
- current evidence and accumulator revision;
- safe runtime context;
- minimum causal lower bound;
- minimum source and context diversity;
- complete definitions;
- counterfactual and intervention support flags;
- absence of suspected regime change;
- slot, device, and capacity gates; and
- an authentic gate capability and digest at commit time.

It also requires a batch-one persistent Cognitive State and a proposal targeting only the verification slot.

The intended SWEGCA predicate additionally requires semantic binding:

```text
Authorize_target(p_i, D_t) = Authorize_impl(p_i, D_t)
                              and Nonempty(addresses_i)
                              and Bind(D_t, addresses_i, delta_i, mask_i)
```

`Bind` means that the decision was produced from the named evidence for the same hypothesis and semantic delta. Phase E reproduced that the current guarded writer commits both an empty-address proposal and a proposal citing an unrelated address when other gates pass. A prototype hardening was rolled back after it broke current frozen evaluator contracts. The paper therefore reports E001/E002 as negative results and keeps this gap explicit.

### 4.6 Single-World arbitration and conflict abstention

For proposal `i`, the current arbiter computes batch weight

```text
w_i = confidence_i * (1 - contradiction_i) * (1 - uncertainty_i)
```

and accepts the proposal for aggregation when `w_i` reaches the configured minimum. The proposal delta is first masked and clipped to the maximum per-slot norm. For two accepted proposals with different `source` strings and a shared target slot, the arbiter computes their per-slot directional inner product. A negative product marks an unresolved contradiction for that slot. The aggregate delta on such a slot is set to zero before the whole-world norm bound is applied.

This is falsification-first no-commit behavior for the coded directional conflict case. Different source strings are a proxy, not proof of statistically or causally independent evidence. The arbiter does not resolve every semantic contradiction.

### 4.7 Bounded persistent commit and receipt

The guarded writer invokes the arbiter as a dry run, rejects failed gates or insufficient proposal weight, and commits only after validating the authority capability. The implemented write changes only the verification slot. It produces a receipt binding:

- before/after state hashes;
- before/after slot hashes;
- applied-delta hash;
- target role and revision;
- evidence references copied from the proposal; and
- prior bounded-write metadata.

The transition is

```text
S_{t+1} = S_t                         if not authorized or conflict abstains
S_{t+1} = Commit_verification(S_t, Delta_t, R_t)  otherwise
```

Because proposal addresses may be empty, the presence of a receipt does not by itself prove complete evidence binding.

### 4.8 Semantic-memory promotion and recovery

A semantic-memory candidate may be promoted only when:

- the current Cognitive State hash matches the linked World-write receipt;
- the accumulator decision is accepted and meets causal/diversity thresholds;
- candidate evidence references include the World receipt's evidence references and `world-write:<receipt_id>`; and
- the versioned-memory mutation matches the prepared journal.

The journal advances through `prepared`, `memory_committed`, `state_committed`, and `completed` stages. Rollback and retraction bind the memory receipt to the World receipt and restore or remove only under current-hash/head checks. Startup recovery rolls back incomplete verified-memory changes under the tested stages.

This is a journaled, recoverable local SQLite/versioned-memory protocol. It is not a claim of distributed atomicity across arbitrary services or crashes.

### 4.9 Abstention, rejection, rollback, retraction, and recovery

- `abstain`: evidence is insufficient or unresolved; no authority is granted.
- `reject`: sufficient evidence supports a negative decision under the configured policy.
- conflict no-commit: the arbiter zeros a coded opposing shared-slot update.
- rollback: a receipt restores the bit-exact pre-write state when the current state still matches the receipt.
- retraction: an active linked fact/update is removed while unrelated later cognition is preserved where the head checks allow it.
- recovery: an incomplete journal is repaired, including restart-time cleanup of linked memory changes.

These terms are not interchangeable.

## 5. Architectural invariants

| Invariant | Failure prevented | Enforcement point | Current evidence | Remaining limitation |
|---|---|---|---|---|
| I01: one authoritative persistent Cognitive State | Split identity/state ownership | `CognitiveState`, registered main/worker boundary | C001; R001 | Process-wide external integration inventory pending |
| I02: producer output is not belief | Direct generator-to-state promotion | `SynapseProposal`, explicit commit | C002; R001 | Raw low-level commit remains callable |
| I03: state-changing evidence is traceable | Unreceipted mutation | `CognitiveEvent`, guarded gate, write receipt | C003/C004; R005 negative | Proposal addresses may be empty and are not bound to the decision; current implementation claim is narrowed |
| I04: undefined remains insufficient | Prior-filled fabricated state | accumulator abstention | C007; R001 | Representation-wide unknown semantics unproven |
| I05: confidence is not authority | Confident unsupported commit | accumulator capability and guarded writer | C005 | Universal form unsupported outside guarded path |
| I06: missing critical evidence fails closed | Mutation under incomplete definitions/evidence | accumulator and write-gate checks | C004/C007 | Registered route enumeration pending |
| I07: persistent mutation is provenance-bound and reversible | Torn or unrecoverable World/memory state | bounded receipt and transaction journal | C004/C010; R001 | Local implemented/tested stages only |
| I08: producer identity is not truth rank | Producer privilege | diversity checks and no fixed truth rank | C006/C007 | Distinct source is not independence proof |
| I10: capability is separate from automatic authority | Capability-to-permission collapse | authority tokens/digests | C004/C005 | Broader action/training/distribution gates outside this draft |
| Derived experience-authority separation | Failed/unverified experience becomes permission | zero-authority selection receipts | C012; R003 | Empirical usefulness and all loader coverage unproven |

Canonical SWEGCA-I09, I11, and I12 remain candidate/limitation items in `INVARIANT_LEDGER.md`; this draft does not promote them beyond current evidence.

## 6. Implementation mapping

| Architecture role | Rozephine/MOSAIC implementation | Paper claim strength |
|---|---|---|
| Persistent Cognitive State | `mosaic_cognitive_kernel.py:176-275` | Proven on registered core paths |
| Provenance-bearing event | `mosaic_cognitive_kernel.py:68-104` | Nonempty refs enforced for `CognitiveEvent` |
| Transient proposal | `mosaic_synapse_arbiter.py:54-104` | Implemented; addresses optional |
| Proposal arbitration/conflict | `mosaic_synapse_arbiter.py:219-355` | Implemented for coded weight, bound, and directional conflict |
| Evidence accumulation | `mosaic_evidence_accumulator.py:49-93,97-257,285-428` | Implemented; natural-world calibration unproven |
| Guard capability | `mosaic_bounded_world_write.py:58-173` | Implemented on registered path |
| Bounded verification write | `mosaic_bounded_world_write.py:286-442` | Bounds/receipt/rollback implemented; proposal-decision binding gap remains |
| World/semantic-memory protocol | `mosaic_world_memory_transaction.py:27-180,212-361` | Journaled local protocol; tested failure stages only |
| Status-unfiltered experience | `mosaic_unrestricted_experience.py:55-81,159-293,342-511` | Registered path verified by R003 |

## 7. Architecture-critical open questions

1. Can a future compatibility-safe binding layer reject empty and decision-mismatched proposal evidence without changing frozen evaluator semantics? E001/E002 currently answer no for the audited writer.
2. Can a valid evidence decision authorize a semantically unrelated tensor delta? The current address attack demonstrates the structural gap but not full semantic interpretation.
3. Does retaining uncurated/failed experience improve fresh held-out judgment without increasing unauthorized commits or excessive abstention? E009 must compare policies.
4. Does the coded source-string distinction correspond to genuinely independent evidence in evaluated scenarios? The paper must report the proxy and its limits.

Until these questions are resolved, SWEGCA is an audited guarded-path architecture with explicit counterexamples and limits, not a universal safety theorem.
