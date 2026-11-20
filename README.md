# ASG-SI: evidence-gated skill libraries

This revision supplies a controlled research harness for the question:
**Does verification before admission reduce faulty-skill accumulation and reuse, at what cost?**
It retains `asg_si_demo.py` as a compatibility entry point and adds a modular
package, explicit admission decisions, executable composition, and reproducible evidence.
The implemented evaluation is a deterministic simulation. No LLM weights or routing
parameters are optimized.

## Setup and run

Use Python 3.10 or newer. The harness and tests need only the standard library.
Run commands from this directory; no API key or model server is needed.

```bash
python3 -m unittest discover -s tests -v
python3 -m experiments.run --config configs/smoke.json --output runs/my_run
python3 -m experiments.report runs/my_run
python3 -m experiments.audit runs/my_run
python3 -m experiments.composition --seed 7 --output runs/my_composition.json
```

Output directories must be empty. A run records the exact Python version and
source hashes. Auditing a supplied evidence bundle requires those recorded
versions. On another Python version, create and audit a fresh run using the
commands above. The bundled reviewed run was produced with Python 3.12.14.

The default configuration runs seeds 7, 42, 123, 456, and 789. Each seed uses the
same frozen pool of 10 candidate submissions (3 valid trajectory-compiled
programs and 7 controlled faulty variants), with separate promotion,
evaluation, stress, candidate-order, and policy random streams.

## Admission conditions

| Condition | Checks | Execution |
| --- | --- | --- |
| `first_success` | One successful discovery outcome | In process |
| `self_verification` | Held-out outcome, constraints, artifact digest, dependency pins, coverage | In process |
| `independent_replay` | Identical checks and test inputs to self-verification | Separate Python worker |
| `independent_composition` | Replay plus declared transition and runtime unit checks | Separate Python worker |

`self_verification` is a deterministic in-process checking surrogate, not an
LLM self-critique experiment. Process separation alone is expected to leave
its decisions identical to independent replay. The comparison does not establish
that isolation improves accuracy or provides adversarial security.

Each method has the same allocated maximum test budget; actual executed tests,
tool calls, latency including process startup, and evidence bytes are reported.
First-success checking uses less actual work. Budget equality does not mean
identical realized cost.

The gate returns `ACCEPT`, `REJECT`, or `INSUFFICIENT_EVIDENCE`. Passing one or two
examples cannot satisfy the default five-case minimum or required boundary
coverage. Canonical task content detects overlap even when task IDs differ.
Private expected answers are absent from the observation given to a policy.
The worker receives promotion cases; it receives neither the policy object nor
final evaluation cases.

## Skill programs and graph

Programs use ordered tool or skill calls, explicit references to task inputs and
previous outputs, literal constants, version pins, and a maximum step budget.
The controlled workflow reads records, validates fields, converts units,
aggregates, and simulates writing a report. Simulated writes record permission
violations; the benchmark performs no external writes.

The graph stores unique executable identities, declared transition bindings,
checked status, and dependencies. Revocation propagates to dependents. Tool
version changes invalidate affected skills until they are re-verified and
explicitly reintroduced through an appropriate lifecycle extension; automatic
re-admission of the same revoked identity is not implemented.

Compatibility uses the intended logical reading `Post_A => Pre_B`. A known
unit disagreement can refute compatibility; passing replay examples provide
empirical evidence and never prove universal compatibility.

`experiments.composition` individually tests three primitive skills, then holds
out their two-skill pairings. It compares a compatible liter workflow with an
incompatible milliliter-to-liter workflow, using both a flat library and checked
execution. This is a small controlled demonstration, not broad compositional
generalization or automatic graph planning.

## Results and audit artifacts

Each run contains:

- `run.json`: configuration, fixed metric, source and runtime versions.
- `decisions.jsonl`: one record per candidate decision, including actual costs.
- `tasks.jsonl`: one record per downstream task, with outputs and reward evidence.
- `checkpoints.jsonl`: actual submission, unique-insertion, and library counts.
- Per-seed split manifests, hidden-suite adjudication, and frozen libraries.
- Per-decision candidate, input cases, observations, intermediate outputs,
  verdicts, reward components, and full SHA-256 manifests.
- `report.md`, `report.json`, and `per_run.csv` generated from raw events.
- `audit_report.json` after complete replay.

Candidate validity is measured independently on evaluation/stress suites using
a Decimal reference implementation, constraint records, and provenance/version
requirements. Unknown coverage gives unknown validity. The promotion verifier's
own verdict is never the candidate's validity label.

Evaluation copies or reconstructs a frozen library and uses a fresh policy RNG;
it does not compile candidates, change policy parameters, or update memory.
The primary fixed reward is constraint-respecting task success, with reuse and
composition weights zero. Post-hoc reward sensitivity with a reuse bonus is
reported separately; it is not a reward-training ablation.

Confidence intervals bootstrap independent run seeds and are conditional on
the small frozen candidate pool. There are no p-value or stochastic-equivalence
claims. The `online` condition schedules submissions from that frozen pool into
successive checkpoints; it is not adaptive open-ended skill discovery.

A single decision can also be checked with:

```bash
python3 -m asg_si.audit_replay PATH_TO_BUNDLE --manifest-sha256 RECORDED_DIGEST
```

The complete audit regenerates decisions, policy selections, task outputs,
constraint outcomes, and rewards without reading the report tables. Deleted
artifacts raise an incomplete-evidence error. Changed artifacts raise an
integrity error. A hash does not regenerate missing bytes. The manifests are
unsigned: retain a digest outside the run directory to detect replacement of
the entire run. The worker is not an OS-enforced adversarial sandbox.

## Review-response experiments

These experiments address the ICML 2026 reviews; `docs/RESPONSE_TO_REVIEWS.md` maps each
comment to a change. Results are stored in `results/review/`, and `paper/assemble.py` builds
every table in the revised paper from those files.

| Module | Question | Output |
|---|---|---|
| `experiments.llm_candidates` | Do frozen LLMs write faulty skills that pass one example, and which admission method keeps them out? Compares first success, LLM self-verification, SmartSnap-style evidence verification, independent replay, and replay with composition checks | `llm_<model>_*.json` |
| `experiments.stochastic` | Exact replay versus a sequential Clopper–Pearson gate when tools are stochastic and partially observable | `stochastic.json` |
| `experiments.composition_depth` | Interface checks on chains of 2 to 5 skills over unit and validation tags, versus a flat library and end-to-end replay | `composition_depth.json` |
| `experiments.selection` | Four selection rules in libraries of 10, 50, and 200 skills, with and without the gate | `selection.json` |
| `experiments.tamper` | Which defence (hashes, external anchor, HMAC signature, audit re-execution) detects which evidence attack; a latent poisoned skill | `tamper.json` |
| `experiments.retention` | A silent tool upgrade with no pins, with pins, and with revocation plus re-verification | `retention.json` |
| `experiments.alfworld_skills` | Model-written `solve(env, goal)` skills for the six ALFWorld task types, repaired from discovery-episode feedback, admitted by first success, LLM self-verification, SmartSnap-style evidence verification, or replay on ten held-out games, and adjudicated on all 274 validation games | `alfworld/summary_<model>.json` |

The language-model experiment calls OpenAI-compatible endpoints configured by `CLOVER_BASE_URL`
and `DEEPSEEK_BASE_URL`. Every request and response is cached under `runs/llm/llm_cache/` by the
SHA-256 of the request; the cache is published in `results/review/llm_cache.zip`, and `--offline`
re-runs the experiment from it without model access. Bundles can be signed with an HMAC key held
outside the run directory (`write_bundle(..., signing_key=...)`, `read_bundle(..., signing_key=...)`).

The ALFWorld experiment runs episodes under Linux (`alfworld` 0.4.2 with the ALFWorld data in
`$ALFWORLD_DATA`); on Windows it runs them in WSL through stdin and stdout. Its stages are
`prepare`, `generate`, `execute`, `judge`, and `analyze`; `generate` and `judge` accept `--offline`
to replay the response cache in `results/review/alfworld/cache/`, and `analyze` needs only the
recorded files in `results/review/alfworld/`.

## Compatibility demo

```bash
python3 asg_si_demo.py --seed 7 --iters 2 --num-tasks 17
python3 run_multi_seed.py --seeds 7 42 --iters 2 --num-tasks 17 --evaluation-tasks 13 --output runs/legacy_check
python3 analyze_results.py --input runs/legacy_check/multi_seed_results.json --output runs/legacy_analysis.json
```

The scripted policy's error injection is disabled by default and can be enabled
explicitly with `--fault-error-rate`. Its reuse probability is a disclosed
simulation parameter, not learned behavior. `--model` uses a frozen Ollama model
configured through `OLLAMA_BASE_URL` and `OLLAMA_MODEL`, requires a running
server, and records provider/parsing/execution failures without scripted
fallback. Model integration has mocked regression coverage; real model
performance has not been evaluated in this revision.

Historical JSON results and audit logs are retained as historical artifacts.
Their old significance, learning, and security claims must not be inferred
from this revision's results. See [research protocol](docs/RESEARCH_PROTOCOL.md)
and [implementation progress](docs/IMPLEMENTATION_PROGRESS.md).

## Recorded validation results

See [results/README.md](results/README.md) for the five-seed results, per-run
metrics, confidence intervals, PDF progress report, full evidence archive, and
commands for replaying the supplied artifacts.
