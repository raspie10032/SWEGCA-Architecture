# SWEGCA ordered component ledger

This ledger fixes the journal description to one runtime order. It is not a list
of interchangeable modules. A later stage cannot grant authority retroactively
to an earlier producer, and a failure at a required stage yields no persistent
mutation.

## Runtime sequence

| Step | Component | Input -> output | Authority boundary | Implementation and evidence | Journal status |
|---:|---|---|---|---|---|
| 0 | Main process and persistent `CognitiveState` | Current identity, state, and request context -> detached request snapshot | Sole owner of persistent cognition and accumulated experience | `mosaic_cognitive_kernel.py`; cognitive-kernel and worker-isolation tests | Supported on registered paths; process-wide external inventory remains open |
| 1 | Observation and active evidence | Sensor/tool/user result -> typed observation or bounded evidence-action result | Observation is not evidence, belief, memory authority, or action authority | `mosaic_cognitive_kernel.py`; `mosaic_autonomous_cognition.py`; autonomy tests | Development-tested state machine; natural perception is not evaluated here |
| 2 | Status-unfiltered experience universe and hot lookup | Declared lawful roots -> addressed candidates and selection receipt | Lookup and selection grant no write, memory, action, training, distribution, or P3 authority | Registered unrestricted-experience module and receipt tests | Supported on the registered path; external loaders are not exhaustively audited |
| 3 | Déjà vu -> Recall -> Replay -> Re-evidence | Query, current cues, immutable hot snapshot, and current evidence judge -> activation receipt | Familiarity is anonymous; replay is not truth; all consequential authority flags remain false | `mosaic_memory_activation.py`; `test_mosaic_memory_activation.py` and VRS bridge tests | Fixed order and snapshot binding unit-tested; downstream utility not measured by Run 009/010 |
| 4 | Request-local manager and adaptive parallel workers | Operation type, route, detached snapshot, and request -> selected producer results | Manager and workers are transient; they cannot own state or commit | `run_dynamic_cognition` in `mosaic_synapse_arbiter.py`; primary-only, barrier-confirmed parallel fan-out, cleanup, source-drift, and isolation tests | Implemented and development-tested; end-to-end accuracy, throughput, and latency remain open |
| 5 | Synapse proposal | Producer result -> source, hypothesis, delta, target mask, scalars, and evidence addresses | Proposal is not belief or permission; producer source is not a trust credential | `SynapseProposal` and adapters in `mosaic_synapse_arbiter.py` and `mosaic_cognition/` | Implemented; generic compatibility type remains permissive outside the guarded path |
| 6 | Evidence accumulator | Claim-relative observations grouped by source family/context and evidence axis -> authoritative `accept`, `reject`, or `abstain` decision | Repetition within one correlated group does not create arbitrary independent sample size; confidence is not authority | `mosaic_evidence_accumulator.py`; seven accumulator checks in the 11-check contract matrix | Implemented for configured axes and thresholds; natural-world calibration is unmeasured |
| 7 | Evidence gate and exact proposal binding | Authentic accepted decision plus one proposal -> process-local scoped capability or no capability | Requires same hypothesis, nonempty accepted-address subset, unchanged proposal digest, and policy gates | `world_write_gates_from_decision` and digest checks in `mosaic_bounded_world_write.py`; Run 009 and Run 010 | 100K development plus one-time held-out confirmation on the guarded verification-slot path |
| 8 | Single-World Arbiter | Bound transient proposal set -> accepted masks, bounded aggregate delta, and unresolved-conflict mask | Arbitration is preview-only on the registered route; worker multiplicity is not evidence independence | `SingleWorldArbiter`; four arbitration checks and focused tests | Implemented for weight/mask/norm and distinct-source directional conflict; not general semantic arbitration |
| 9 | Main bounded commit and receipt | Authorized preview, unchanged proposal, and current state -> updated verification slot plus receipt, or no commit | Only the main guarded writer commits; exact target and capability are revalidated | `bounded_verification_write`; binding runs and focused regressions | Proven for the audited guarded path; low-level and other mutation routes remain outside this result |
| 10 | Memory promotion and World/memory transaction | Current World receipt, accepted evidence, and memory candidate -> episodic/semantic update and journal receipt | Cognitive write and semantic promotion are distinct authorities; rollback, retraction, and recovery have distinct preconditions | `mosaic_memory_promotion.py`; `mosaic_world_memory_transaction.py`; transaction and fault-stage tests | Supported for selected local stages; no distributed atomicity claim |
| 11 | External effects and outcome observation | Verified provenance, allowlisted reversible proposal, confidence/policy checks -> action permission and observed result, or abstention | State or memory verification does not itself authorize action, model update, distribution, or P3 promotion | `mosaic_autonomous_cognition.py`; autonomy transition tests | Local reversible action loop development-tested; broader effect domains remain separately gated and outside Run 009/010 |

## Cross-cutting invariants

These rules apply across the sequence and are not additional runtime stages.

| ID | Invariant | Consequence in the journal claim |
|---|---|---|
| I01 | Single authoritative World | One persistent owner on registered paths; multiple proposals and contradictory records may coexist |
| I02 | Proposal is not belief | No producer or worker commits directly |
| I03 | Evidence addressability | A guarded commit cites a nonempty subset of the accepted decision's addresses |
| I04 | Unknown is first-class | Insufficiency and unresolved conflict may remain `abstain` without forced closure |
| I05 | Confidence is not authority | Producer scalars influence weight only; an authentic evidence capability is still required |
| I06 | Fail closed | Missing, stale, forged, mismatched, or unresolved requirements produce no persistent mutation |
| I07 | Reversible mutation | Registered writes are bounded and receipt-linked; rollback, retraction, and recovery are distinguished |
| I08 | No producer privilege | Model, tool, memory, manager, worker, and specialist identity does not establish truth rank |
| I09 | Causal claims need causal evidence | Configured counterfactual/intervention labels do not by themselves prove causal identification |
| I10 | Capability and authority are separate | Reading, proposing, verifying, remembering, acting, training, distributing, and P3 promotion are separate domains |
| I11 | Cognition and rendering are separate | A language model may propose or render but is not the persistent cognitive authority |
| I12 | Identity requires continuity evidence | Naming or repeated appearance alone does not establish cross-time identity |

## Independence rule for parallel verification

Parallel execution reduces wall-clock dependence on a single producer; it does
not create epistemic independence. Results sharing a source family, seed,
template, context, model lineage, or derived evidence must be grouped or marked
as correlated. The accumulator and gate evaluate addressed evidence for the
named hypothesis, the arbiter handles only its coded proposal-conflict class,
and the main process alone decides whether a guarded mutation is authorized.
