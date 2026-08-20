# SWEGCA adversarial manuscript review

Status: Phase G review complete; final package verification pending.

The review treats every positive sentence as a potential overclaim and checks it against `CLAIM_EVIDENCE_MATRIX.md`, `RESULT_LEDGER.md`, `ARCHITECTURE_SPEC.md`, the Phase E result artifacts, the fifteen-work prior-art audit, and the current manuscript.

## Findings and dispositions

| ID | Attack | Finding | Disposition |
|---|---|---|---|
| G001 | Read the conclusion as a current universal enforcement claim | “Only scoped and validated evidence may authorize” could be read as an implemented fact despite the three-case counterexample. | Fixed: the conclusion now calls this the **target boundary** and says evidence **should** authorize mutation. |
| G002 | Challenge process-local capability language | “Prove” overstated what a process-local capability and receipt establish. | Fixed: the discussion now says “establish, within the audited process” and scopes reversal to receipt checks. |
| G003 | Reproduce the implementation snapshot | The manuscript named artifact hashes but did not name the evaluated architecture commit in its reproducibility section. | Fixed: added clean Phase E implementation commit `168e9b760b8f6b3e8ff767e1ebb80b310b1a6a28`. |
| G004 | Hide the failed binding invariant behind passing counts | Abstract and evaluation could have foregrounded only 73 passing tests and 11/11 matrix checks. | No defect: matching, empty, and mismatched proposals committing is stated in the abstract, a dedicated table, discussion, limitations, and conclusion. |
| G005 | Conflate distinct source with independent evidence | Directional conflict handling could be misread as semantic contradiction resolution. | No defect: manuscript repeatedly states that source-string inequality is a proxy, same-source opposition is missed, and not every semantic conflict is solved. |
| G006 | Conflate experience availability with evidence or belief | Status-unfiltered access could be read as authority for failed/unverified material. | No defect: selection authority is empty; experience becomes evidence only under claim-relative admission; usefulness remains unevaluated. |
| G007 | Present a regression count as scientific confirmation | The 73-test result could be treated as held-out safety or task performance. | No defect: it is labeled architecture regression health; all matrices are development evaluations; natural-world and held-out claims are prohibited. |
| G008 | Treat local recovery as distributed atomicity | Journal language could imply cross-service atomic transactions. | No defect: the paper names a local SQLite/versioned-memory protocol and explicitly denies distributed atomicity. |
| G009 | Claim novelty for established components | Evidence ledgers, provenance, TMS, arbitration, memory transactions, and confidence-authority separation are prior art. | No defect: the contribution is an audited implementation-level composition; “first architecture” language is absent. |
| G010 | Minimize AI involvement | “Automated tools assisted” understated the actual workflow. | Fixed at the author's request: the manuscript now says generative AI was used **extensively** and lists repository inspection, literature organization, evaluation scripting/execution, auditing, figures/tables, drafting, and editing. |
| G011 | Infer public code availability | A future standalone repository will initially be private. | No defect: the manuscript does not claim that a public code repository currently exists. |

## Numerical cross-check

| Manuscript value | Bound source | Review result |
|---|---|---|
| 73 architecture tests | Phase E final SWEGCA-only test run; Phase F rerun `73 passed in 2.67s` | Consistent; paper retains the recorded Phase E time of 2.71 seconds only in the Phase E description |
| 11/11 fixed checks | `results/contract_matrix_phase_e.json` | Consistent |
| residual norm 1.4142 | same-source opposition case | Consistent rounded display |
| 16.8/17.5 microseconds | accumulator median/p95 | Consistent |
| 142.4/215.9 microseconds | arbitration median/p95 | Consistent |
| 236.5/302.7 microseconds | guarded dry-run median/p95 | Consistent |
| 408.0/560.0 microseconds | commit median/p95 | Consistent |
| 85.5/122.2 microseconds | rollback median/p95 | Consistent |
| 2,472 files, 246 records | static mutation-path inventory | Consistent and labeled non-reachability evidence |
| 15 cited works | `references.bib` plus full-text literature matrix | All 15 keys cited; no missing or unused key |

## Remaining limitations, intentionally unresolved

- dynamic coverage of every persistent owner and mutation entry point;
- nonempty and decision-matched proposal evidence addresses;
- exact hypothesis/tensor-semantic binding;
- proof that source labels represent independent evidence;
- a complete fault-stage matrix beyond implemented tests;
- a fresh held-out status-unfiltered versus validated-only versus no-experience evaluation; and
- end-to-end cognition latency and memory/throughput measurements.

These items remain future work. None is converted into a positive submission claim.
