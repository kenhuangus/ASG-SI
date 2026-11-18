# Research protocol and paper alignment

Base repository commit: `c1da9a1d15883d04518f4a7213ccba612b0e6b28`.
Implementation branch: `evidence-gated-library`.

| Reviewer concern | Implemented response | Evidence to inspect |
| --- | --- | --- |
| Scripted behavior described as learning | Explicit mechanisms and zero parameter updates; frozen model adapter | `run.json`, policy metadata |
| Same policy seed across runs | Derived task/policy/promotion streams, recorded per seed | `seeds.json`, regression tests |
| Duplicate verifications counted as skills | Boolean insertion result and distinct submission/acceptance/insertion counts | `decisions.jsonl`, `checkpoints.jsonl` |
| Verifier never sees faulty candidates | Seven disclosed mutations of a frozen candidate pool | Per-fault acceptance counts |
| Tiny passing samples treated as sufficient | Minimum unique cases and named boundary coverage | Three-way decisions and insufficiency tests |
| Evaluation contaminates acquisition | Separate splits and frozen checkpoint evaluation | Split hashes, immutable-state tests |
| Schema-only compatibility | Declared units and checked transitions; nested skill calls | Composition report and traces |
| Reward changes confound improvement | Fixed success metric; post-hoc reuse sensitivity labeled separately | Metric hashes and raw components |
| Audit volume mistaken for auditability | Reconstruct complete decisions, outputs, constraints, and rewards | Full audit command and corruption tests |
| Unsupported significance | Independent-run descriptive CIs and paired differences | Raw denominators and report JSON |

## Scope of the controlled benchmark

Three valid workflows are compiled from successful trajectories with explicit
bindings; seven faulty controls exercise memorization, incorrect binding,
unit mismatch, prohibited simulated writes, changed artifacts, stale tool
versions, and skipped validation. The narrow discovery examples deliberately
permit every candidate to demonstrate one successful outcome. In particular,
the unit mismatch succeeds on an empty example: its zero output masks the
unit error. This is a controlled failure case, not a natural candidate
prevalence estimate.

The same candidate pool, candidate order within seed, promotion suite, and
maximum number of tests are used across admission conditions. In-process
checking and independent replay use the same criteria; this isolates execution
placement and is expected to yield equal verdicts for these pure functions.
Adding different data coverage or criteria to only one condition would
confound a claimed effect of independence.

The three valid variants implement the same task in slightly different ways.
Bootstrap intervals over five runs measure task/order/policy variation,
conditional on those 10 candidates. They do not measure uncertainty over
future candidate populations. Zero faulty admissions on these controls does
not establish a zero real-world failure rate. Provenance-invalid submissions
can share an executable with a valid submission; acceptance decisions and
unique insertions therefore have separate denominators.

Final task selection is uniform over available skills of the correct kind.
If no skill exists, the policy abstains and task success is zero. No direct-tool
fallback is used in this benchmark. This choice makes faulty library reuse
visible but limits external validity; future evaluations should compare
retrieval policies, abstention, and fallback costs.

The online condition is staged admission of a frozen pool. It shows library
accumulation and utilization over checkpoints, while leaving adaptive candidate
synthesis and continuing acquisition for later experiments.

## Composition experiment

Primitive preparation and rendering skills pass independent fixture tests
before pairing. Preparation in milliliters is individually correct, but its
output violates the renderer's liter precondition. Final-output tests alone
can miss this incompatibility on zero-total data. Runtime unit checks block
all such pairings in the controlled test, including accidental zero matches.
The graph does not infer arbitrary logical contracts, synthesize plans, or
prove compatibility for all inputs.

## Evidence and isolation

Bundles include canonical executable content, full SHA-256 identities, test
inputs and expected labels, observed/intermediate outputs, source hashes,
tool contracts and versions, configuration, reasons, and per-test reward
components. The exact Python version is recorded. Replay requires that
recorded environment; it explicitly refuses mismatched source or runtime.

A separate Python worker separates execution and data flow from the policy
callback. It shares the OS identity and filesystem permissions. This is not
an adversarial policy/verifier boundary. Unsigned manifests also do not provide
nonrepudiation; full-run replacement requires an externally retained digest
or a future protected signing service to detect.

## Remaining milestones

1. Connect a real frozen LLM provider to the common policy interface and run
   paired model experiments without injected policy errors. Record actual
   model/backend identifiers, prompts, sampling settings, provider failures,
   retries if added, tokens, latency, and cost.
2. Collect naturally generated successful and faulty candidates on a broader
   benchmark. Keep the frozen-pool experiment for admission comparisons and
   add true online acquisition as a separately labeled condition.
3. Implement OS-level policy/verifier/evidence separation with protected
   tests and an externally anchored or signed manifest; evaluate deliberate
   tampering across that boundary.
4. Extend composition to more tasks, units, depths, and held-out combinations;
   test transitions and routing separately.
5. If policy optimization is retained as a paper contribution, implement and
   evaluate the actual optimizer. A frozen model with an evolving library is
   sufficient for a narrower admission-gating question.
6. For stochastic outcomes, predefine an equivalence or noninferiority margin
   and choose the procedure for the outcome type. Failure to reject a
   difference test does not establish equivalence.
7. Rewrite the abstract and results around the completed experiments, then
   verify related-work distinctions against primary papers. Do not claim all
   prior skill libraries omit self-verification.
