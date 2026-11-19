# ASG-SI implementation progress

Completed and revalidated on **2026-11-18**, using Python **3.12.14**.
Base repository commit: `c1da9a1d15883d04518f4a7213ccba612b0e6b28`.
Local implementation branch: `evidence-gated-library`.

| Step | Implemented result | Validation |
| --- | --- | --- |
| 1. Experimental correctness | Separate seeds, public observations, honest policy metadata, unique insertion counts | Regression tests pass |
| 2. Admission gate | ACCEPT / REJECT / INSUFFICIENT_EVIDENCE, minimum test count, boundary coverage, artifact and dependency checks | All seven faulty controls rejected by held-out replay |
| 3. Multi-step execution | Explicit bindings, bounded tool/skill calls, semantic unit checks | Compatible and incompatible composition tests pass |
| 4. Library lifecycle | Versioned dependencies, explicit transitions, transitive revocation | Mutation, deduplication, and revocation tests pass |
| 5. Evidence and replay | Full SHA-256 manifests, retrievable artifacts, source/runtime snapshots, decision and reward reconstruction | 400 bundles and 7,000 downstream tasks reproduced |
| 6. Experiment/report pipeline | Four admission methods, five independent run seeds, static and staged online conditions, raw JSONL and descriptive CIs | Five-seed run and full audit complete |
| 7. Handoff | Source, tests, configuration, protocol, patch, complete raw evidence package | Downloadable package prepared with this report |

**37 regression tests passed** after the implementation was restored following
workspace maintenance. The results below come from the fresh completed run,
not from the interrupted earlier run.

## Controlled admission results

Mean final-checkpoint results across seeds 7, 42, 123, 456, and 789. Static and
staged online conditions finish with the same candidate submissions and final
library; online checkpoint trajectories are recorded separately.

| Admission method | Faulty candidates accepted | Valid candidates rejected | Downstream success | Constraint violations | Verification tool calls per run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Admit after first success | 100% | 0% | 45.3% | 17.3% | 48 |
| Additional checking in process | 0% | 0% | 100% | 0% | 570 |
| Independent replay worker | 0% | 0% | 100% | 0% | 570 |
| Replay plus composition checks | 0% | 0% | 100% | 0% | 495 |

Each method receives 10 submissions per run: 3 valid and 7 faulty. For each
method and phase, there are 35 faulty and 15 valid candidate decisions across
five runs. The 150 final evaluation task executions per method and phase
comprise 30 tasks per run; stress results use a separate 20 tasks per run.
Detailed confidence intervals, paired differences, costs, unique promotions,
and fault-specific counts are generated in `runs/review_run/report.json` and
`per_run.csv`.

The identical in-process and independent replay outcomes are an expected
result for pure deterministic tools with identical checks. They do not show a
performance benefit from process isolation. Composition checks save 75 tool
calls per run here by rejecting an incompatible chain before tool execution.

## Controlled composition results

All primitive skills first pass their individual tests. Pairings are withheld
until the final composition test.

| Pairing | Flat verified library | Checked graph |
| --- | --- | --- |
| Compatible liter workflow, 30 tasks | 30 completed correctly | 30 completed correctly |
| Incompatible milliliter/liter workflow, 30 tasks | 24 incorrect completions; 6 accidental zero-total matches | All 30 blocked; no incorrect completions |

The checked incompatible condition uses 120 tool calls versus 150 for the flat
condition. These are two-skill controlled pairings, not a broad composition
benchmark or universal proof of compatibility.

## Audit result

The independent reconstruction returned `REPRODUCED` for:

- 400 candidate admission bundles.
- 3,700 candidate test executions and their per-test rewards.
- 7,000 downstream task executions, policy selections, constraints, and rewards.

Deletion, artifact modification, manifest modification, changed environments,
and forged reward totals have explicit regression tests. Evidence manifests
are unsigned; whole-run replacement requires an externally retained digest or
future protected signing service to detect.

## What the paper can claim now

This implementation supports a small deterministic experiment showing that
held-out checks reject the seven controlled faulty candidates, and that
semantic transition checks detect a unit incompatibility that can be hidden
by zero-output examples. It supports reproducible admission and task-level
evidence under the recorded environment.

Real frozen-LLM results, policy optimization, natural candidate populations,
true adaptive online synthesis, longer-term retention, and OS-enforced
adversarial isolation remain future milestones. The in-process condition is
a checking surrogate, not an LLM self-verification result. The five-run CIs
are conditional on this small frozen candidate pool.

The connected GitHub account has read access to `kenhuangus/ASG-SI`; this
implementation is delivered as a local branch and downloadable patch/package.
No upstream branch or pull request has been published.
