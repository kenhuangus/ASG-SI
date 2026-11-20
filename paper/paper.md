% title Audited Skill-Graph Self-Improvement: Evidence-Gated Admission of Agent Skills
% authors Jerry Huang, Yingjun Chen, Pin-Yu Chen, Ken Huang
% affiliations Kleiner Perkins · Georgia Tech · IBM · DistributedApps.ai

ABSTRACT: Agents that add reusable skills to a library while deployed can also add faulty skills, and later tasks then reuse those faults. Audited Skill-Graph Self-Improvement (ASG-SI) treats a skill library as a graph whose nodes enter only through an admission gate: a candidate program compiled from a successful trajectory is replayed on held-out tests with required boundary coverage, checked against declared interface contracts, and admitted, rejected, or held for insufficient evidence. Every decision is written to a hashed, optionally signed evidence bundle from which an independent audit re-executes the decision. We evaluate the gate in seven experiments. On candidates written by two frozen local language models (a 35B mixture-of-experts model sampled in two independent runs, and a 748B DeepSeek model), first-success admission accepted all 30 faulty candidates (29 used a shortcut tool that ignores units and validity flags; one skipped validation), and evaluation success with the resulting libraries fell to 40–63 %. Replay with coverage admitted none of them and rejected none of the 32 valid candidates. Same-model self-verification also excluded the faults but rejected 19 of the 32 valid candidates, and a SmartSnap-style verifier over proposer-chosen evidence rejected 9. With stochastic, partially observable tools, exact replay accepted an adequate skill in only 67.4 % of trials, while a sequential Clopper–Pearson gate with a reliability threshold fixed in advance accepted every adequate skill and no inadequate one across 500 trials per candidate. Contract checks blocked every incompatible chain of two to five skills without tool calls, at the cost of also blocking 6 chains whose unit errors cancelled. A silent tool upgrade reduced unpinned success to 60 %, and version pins with revocation and re-verification kept it at 100 %. The audit detected each evidence attack only when the matching defence was present. The policy is frozen throughout; the paper makes no claim about reward-driven learning, and all environments are small and synthetic.

## 1 Introduction

Language-model agents increasingly keep what works. Voyager stores successful programs as callable skills (Wang et al., 2023); reinforcement-learning formulations add skill generation and reuse to the training objective (Wang et al., 2025); and agentic reinforcement learning more broadly treats tool use, memory, and self-improvement as parts of one interactive policy (Zhang et al., 2025). A library that grows during deployment is useful because later tasks reuse earlier work. The same property makes it a liability: a skill that passed once on a narrow example and is wrong in general is reused on every later task that selects it.

This paper studies one control point in such systems: the decision to admit a skill into the library. The question is whether verification before admission keeps faulty skills out, under which conditions it fails, and what it costs. We call the overall design Audited Skill-Graph Self-Improvement (ASG-SI). The unit of improvement is an admitted skill with an explicit program, pinned tool versions, a declared interface, and an evidence bundle. An admission gate separate from the policy replays each candidate on held-out tests before it becomes callable, and an independent audit can re-execute every admission decision from the stored evidence.

Two uses of verification are combined. Verifiable-reward work uses deterministic evaluators as training signals (Wen et al., 2025; Wu et al., 2025). ASG-SI uses verification to gate which capabilities exist at all, and it records the evidence so that a third party can reproduce the decision. Prior skill libraries also verify candidates, usually with the same model that wrote them (Wang et al., 2023; Shinn et al., 2023). Section 6.2 compares the two forms of verification on naturally generated candidates.

The contributions are the following.

- An admission gate with three outcomes (accept, reject, insufficient evidence), coverage requirements, version pins, interface checks on composed chains, and a sequential statistical variant for stochastic tools (Section 4).
- An evidence format with full SHA-256 manifests, optional HMAC signatures, and an independent audit that re-executes decisions, together with a measured matrix of which defence detects which attack (Sections 4.6 and 6.6).
- Seven experiments: a controlled fault benchmark, candidates from two frozen language models (three runs) compared with LLM self-verification and SmartSnap-style evidence verification, stochastic tools, composition chains up to length five, selection in libraries of up to 200 skills, evidence tampering and poisoning, and retention under a tool upgrade (Section 6).
- A public implementation, raw evidence, and an audit command that regenerates every reported decision (Appendix A).

Section 2 reviews related work, Section 3 states the setting and threat model, Section 4 defines the method, Section 5 describes the experimental setup, Section 6 reports results, and Section 7 discusses limitations.

## 2 Related Work

Related work falls into five groups: skill libraries, self-verification, verifiable rewards, tool-use learning, and software supply-chain integrity. Table 1 compares the systems closest to ASG-SI on the properties that matter for admission.

**Skill libraries and self-improving agents.** Voyager builds a code skill library in Minecraft and adds a program after an LLM critic judges the task complete (Wang et al., 2023). Wang et al. integrate skill generation and use into reinforcement-learning training, adding skills after successful use (Wang et al., 2025). Skill-it orders training data by skills rather than storing callable programs (Chen et al., 2023). SuperIntelliAgent pairs a trainable learner with a frozen LLM verifier that labels the learner's outputs for continual DPO training (Lin et al., 2025). ASG-SI keeps the library idea but moves the admission decision to held-out replay by a separate process and records the evidence.

**Self-verification and evidence-seeking agents.** Reflexion and CRITIC use model feedback, optionally with tools, to revise outputs (Shinn et al., 2023; Gou et al., 2023). Huang et al. report that models often fail to correct their own reasoning without external feedback (Huang et al., 2023). SmartSnap trains a GUI agent to collect snapshot evidence of task completion for an LLM verifier (Cai et al., 2025). SmartSnap and SuperIntelliAgent have public code, but they target Android GUI control and diffusion-model training respectively; we did not run them on our benchmark. Instead, Section 6.2 re-implements their admission rule (an LLM verifier judging evidence, with the evidence curated by the proposer in the SmartSnap variant) inside the same harness.

**Verifiable rewards and measurement.** Outcome verifiers and process verifiers improve reasoning training (Cobbe et al., 2021; Lightman et al., 2023), and reinforcement learning with verifiable rewards improves reasoning in base models (Wen et al., 2025). Wu et al. document measurement gaps, including contamination and missing parity controls, that can overstate such gains (Wu et al., 2025). ASG-SI applies these concerns to admission: separate splits, a fixed metric, and evidence from which every number is regenerated.

**Tool use, memory, and agent infrastructure.** Toolformer and ToolRL study how models learn tool calls and how reward structure affects tool learning (Schick et al., 2023; Qian et al., 2025). Progressive reward shaping stages structural and outcome rewards (Su et al., 2025). MEM1 and Memory-R1 train memory operations with reinforcement learning (Zhou et al., 2025; Yan et al., 2025). AgentRL and Agent Lightning decouple rollout and training (Zhang et al., 2025; Luo et al., 2025); the ASG-SI compiler and gate can run as separate services in such pipelines. Experience synthesis generates training interactions from a learned environment model (Chen et al., 2025). Continual and constraint benchmarks measure forgetting and outcome-driven violations (Joshi et al., 2025; Li et al., 2025).

**Admission control and supply-chain integrity.** Proof-carrying code requires a program to ship with a proof that the host checks before execution (Necula, 1997). The simplex architecture keeps a verified controller as a fallback for an unverified one (Sha, 2001). in-toto records signed metadata for each step of a software supply chain so that a consumer can verify the whole chain (Torres-Arias et al., 2019). ASG-SI applies the same pattern, check before admission and keep verifiable provenance, to skills produced by an agent, with empirical replay instead of proofs.

Table 1 contrasts the closest systems. Each entry describes the published method; the table does not rank overall quality.

| System | What enters the library or model | Verifier | Held-out tests | Evidence for third-party replay | Revocation on change |
|---|---|---|---|---|---|
| Voyager | code skill | same LLM (critic) | no | no | no |
| Wang et al. (2025) | skill | task success | no | no | no |
| SmartSnap | trajectory reward | LLM judging agent-chosen evidence | no | snapshots | no |
| SuperIntelliAgent | learner update | frozen LLM | no | no | no |
| ASG-SI | compiled skill program | separate replay process | yes, with coverage | hashed, signed bundles | yes, transitive |
Widths: 1.2,1.25,1.3,0.85,1.0,0.8
Table: Table 1: Admission properties of systems that add skills or updates after verification

## 3 Setting, Threat Model, and Scope

An agent solves a stream of tasks with tools. A task has a public observation (its inputs) and private evaluator data (the expected output), and the policy never receives the private part. Tools have declared contracts and versions. A successful trajectory can be compiled into a program and submitted as a candidate skill. Admitted skills become callable for later tasks.

**Determinism assumption.** Exact replay requires deterministic tools, or tools whose randomness is seeded and recorded. Sections 4.3 and 6.3 relax this with a statistical gate for tools that are stochastic and only partly observable.

**Adversary.** We consider three tiers. An external attacker controls task inputs and tool outputs. A supply-chain attacker can modify a skill artifact, the verifier, or stored evidence. The self-improving agent itself can propose skills that pass tests while violating requirements outside the tested distribution. The security goals are auditability (each decision can be re-executed from evidence), integrity of admission (only candidates that pass held-out checks become callable), controlled composition (chains are admitted only when declared interfaces agree), and measurement reliability (reported numbers are regenerated from raw records).

> **Scope of claims.** The experiments support claims about admission: which candidates a gate admits, what that does to downstream success, what it costs, and which attacks an audit detects. No policy is trained; every model is frozen. All environments are small and synthetic. The guarded-fallback edge type and learned reward schedule described in earlier drafts are design proposals and are not evaluated.

## 4 Method

ASG-SI has five components: a frozen policy runtime, a trajectory log, a skill compiler, an admission gate, and a skill graph with an evidence store and an independent audit. Figure 1 shows how they connect. Candidates move from the compiler to the gate; accepted skills enter the graph and become available to the policy; every decision produces an evidence bundle that the audit can re-execute.

![Figure 1: ASG-SI components and data flow](figures/figure1_architecture.png)

\pagebreak
### 4.1 Definitions

Table 2 lists the notation used in the rest of the paper.

| Symbol | Meaning |
|---|---|
| x, y* | public task inputs and private expected output |
| T, v(T) | tool and its version string |
| C(T) = (args, out, units) | declared tool contract: argument types, output type, input and output units |
| P = (s_1 … s_k, o, pins) | program: ordered steps with bindings to inputs, earlier outputs, or literals; output step o; exact version pin for every tool |
| id(P) | SHA-256 of the canonical JSON of P |
| I(P) = (Pre, Post) | interface: partial assignments over declared tags (unit, validated); Post values may inherit |
| G = (V, E) | skill graph: admitted programs and checked composition edges |
| D_promo, D_eval, D_stress | disjoint promotion, evaluation, and stress splits; disjointness is checked on task content |
| B = (k, b) | coverage requirement: at least k unique cases spanning boundary classes b |
| tau, alpha | minimum acceptable success rate and one-sided error for the statistical gate |
| E | evidence bundle: candidate, tests, configuration, decision, manifest, optional signature |
Widths: 1.6,4.8
Table: Table 2: Notation

A **skill** is a program P with its identity id(P) and interface I(P). An **edge** (A, B) is composition-ok when every tag required by Pre(B) holds in the state produced by propagating Post(A); Section 4.4 gives the procedure. An **admission decision** is one of ACCEPT, REJECT, or INSUFFICIENT_EVIDENCE, with reason codes.

### 4.2 Skill compilation

The compiler records each tool call of a successful trajectory with the binding of every argument and converts the trajectory into a program. Algorithm 1 gives the procedure. Only trajectories that succeed on their discovery example are compiled, so every candidate in this paper has at least one success.

```algorithm Algorithm 1: Compile a successful trajectory into a program
input: trajectory events e_1..e_k (tool, bindings, tool version), task inputs x, success flag
if not success or any event lacks a tool or bindings: return none
steps <- [(id_i, tool_i, bindings_i) for each event e_i]   # input key, earlier step, or literal
pins  <- {tool_i: version_i}                               # exact versions observed
P     <- (steps, output = id_k, pins, input_keys = keys(x))
return P, id(P) = SHA-256(canonical_json(P))
```

The compiler does not segment long trajectories into reusable sub-skills and does not infer interfaces from data. Interfaces come from the declared tool contracts.

### 4.3 Admission gate

The gate receives a candidate, its claimed digest, and the promotion split. Algorithm 2 shows the exact-replay gate. The pre-checks reject artifacts whose digest does not match, programs with missing or stale pins, and duplicated test content; insufficient coverage returns INSUFFICIENT_EVIDENCE rather than ACCEPT. Each test passes only if the output matches the reference and no constraint was violated.

```algorithm Algorithm 2: Exact-replay admission gate
input: candidate (P, claimed digest), promotion cases D, coverage B = (k, b), tool registry R
if claimed digest != id(P): return REJECT("artifact_digest_mismatch")
if tools(P) != keys(pins(P)) or any R.version(t) != pins(P)[t]: return REJECT("pin")
cases <- relevant cases of D, at most max_cases
if |unique(cases)| < k or b not subset of boundaries(cases): return INSUFFICIENT_EVIDENCE
for c in cases:                                       # in a separate worker process
    r <- execute(P, c.inputs, R, enforce_pins=True)
    record r (events, outputs, violations) and reward components
    if r.error or r.output != c.expected or r.violations: return REJECT("replay_check_failed")
return ACCEPT
```

For tools whose outputs vary from run to run, exact replay is the wrong test: an adequate skill can fail one execution and an inadequate skill can pass all of them by chance. The statistical gate in Algorithm 3 fixes a minimum acceptable success rate tau before any data are seen and uses one-sided Clopper–Pearson bounds (Clopper & Pearson, 1934). Every execution is seeded, and the seeds are recorded, so the audit can replay the stochastic executions exactly.

```algorithm Algorithm 3: Sequential statistical gate for stochastic tools
input: program P, promotion cases D, tau = 0.90, alpha = 0.05, budgets (60, 240)
successes, n <- 0, 0
for budget in (60, 240):
    while n < budget: run P on D[n mod |D|] with a fresh recorded seed; update successes, n
    if lower_CP(successes, n, alpha) >= tau: return ACCEPT
    if upper_CP(successes, n, alpha) <  tau: return REJECT
return REJECT("inconclusive at maximum budget")
```

### 4.4 Composition checking

A composed skill calls admitted skills in sequence. The checker propagates interface tags left to right: the state starts empty, each skill's Pre must hold in the current state, and the state is then replaced by the skill's Post, with inherited tags copied from the incoming state. The first failing edge is reported with the tag, the required value, and the value found. Figure 2 gives an example in which a milliliter output feeds a renderer that requires liters, and the same chain with a rescaling skill inserted.

![Figure 2: Interface propagation refutes one chain and accepts the other](figures/figure3_composition.png)

The check refutes only declared incompatibilities. It does not prove correctness for properties that no contract declares, and it blocks chains whose declared tags disagree even when their numeric errors cancel. Section 6.4 measures both effects.

### 4.5 Library lifecycle and selection

The graph stores admitted identities, their dependencies, and checked edges. Revoking a skill revokes every skill that depends on it. When the tool registry changes a version, every skill pinned to the old version is revoked; candidates are then re-pinned and re-verified against the current registry, and only those that pass return. Figure 3 shows the lifecycle from candidate to admission, rejection, revocation, and re-verification.

![Figure 3: Admission and revocation lifecycle](figures/figure2_lifecycle.png)

When several admitted skills apply to a task, the policy in Sections 6.1 and 6.2 selects uniformly at random among them, with a recorded seed, and abstains when none applies. Section 6.5 compares this rule with cost-greedy selection, ranking by admission evidence, and an upper-confidence-bound rule that learns from runtime-observable outcomes.

### 4.6 Evidence and audit

Each decision writes a bundle with four artifacts (candidate, test inputs and labels, gate configuration, decision with every execution trace and reward component) and a manifest. The manifest records the SHA-256 and byte length of each artifact, the Python version, the tool registry with versions and contracts, and the SHA-256 of every source file. Three optional controls strengthen the bundle: an external anchor (the manifest digest stored where the agent cannot write), an HMAC-SHA256 signature with a key held outside the run directory, and the independent audit, which checks the recorded environment against the auditor's code and re-executes the decision. The audit reports REPRODUCED only if the re-executed decision, traces, and rewards are byte-identical after canonical serialization.

### 4.7 Rewards

The decomposed reward of earlier drafts (tool validity, outcome, reuse, composition, memory) is computed and stored for every execution so that alternative weightings can be evaluated after the fact. All results in this paper use a fixed metric: success means a correct output with no constraint violation, with every other weight set to zero. Appendix C gives the proposed phase schedule as a design for future policy training; no reported result depends on it.

## 5 Experimental Setup

**Task.** A volume-report workflow: read records, validate fields (drop records whose valid flag is false), convert milliliters and liters to liters, sum, and write the report to the allowed destination. Ground truth comes from an independent Decimal reference implementation. Each split cycles through five boundary classes: ordinary, mixed units, invalid records, negative values, and empty input. Splits are disjoint on task content.

**Seeds and statistics.** Five run seeds (7, 42, 123, 456, 789) derive separate random streams for splits, candidate order, and selection. We report means with 95 % percentile bootstrap intervals over seeds where intervals have nonzero width; the statistical unit is the run seed. We make no significance claims, and intervals are conditional on the candidate pools used.

**Models.** Two frozen local models served through OpenAI-compatible endpoints: clover-35b-a3b, a 35B mixture-of-experts model in 4-bit MLX format on an Apple M4 Pro Mac mini with 64 GB memory, and a DeepSeek model with 748 billion parameters in Q8_0 GGUF format on an NVIDIA DGX Spark through llama.cpp. Clover was sampled in two independent runs. Run 2 requested qwen3.5-4b, but the serving software answered every request with its loaded model, and the model name returned with each response, which the cache records, identifies all 120 run-2 responses as clover-35b-a3b. An earlier version of this paper reported run 2 as qwen3.5-4b; the client now rejects any response whose model differs from the request. Every request and response is cached by the SHA-256 of the request, so the language-model experiments can be re-run offline from the recorded responses.

**Admission methods.** First success admits any candidate whose discovery example succeeded. LLM self-verification shows the same model the program, the specification, and one execution and asks for ACCEPT or REJECT. Evidence verification follows SmartSnap: the proposing model selects three test inputs, the harness executes the program on them, and the model judges the outputs. In-process checking and independent replay run Algorithm 2 in the same process or in a separate worker. Replay with composition checks adds the interface check of Section 4.4.

## 6 Results

### 6.1 Controlled fault benchmark

The first experiment uses a frozen pool of 10 candidates per run: three valid programs compiled from trajectories and seven controlled faults (memorized constant, incorrect binding, unit mismatch, prohibited write destination, modified artifact, stale tool pin, skipped validation). Every candidate succeeds on its discovery example; the unit mismatch succeeds only because its example is empty. Table 3 reports admission outcomes and downstream success.

| Admission method | Faulty admitted % | Valid rejected % | Eval. success % | Stress success % | Violations % | Verif. tool calls |
|---|---|---|---|---|---|---|
| Admit after first success | 100.0 | 0.0 | 45.3 (40.0–50.7) | 46.0 (37.0–54.0) | 17.3 (12.0–21.3) | 48 |
| In-process checking | 0.0 | 0.0 | 100.0 | 100.0 | 0.0 | 570 |
| Independent replay | 0.0 | 0.0 | 100.0 | 100.0 | 0.0 | 570 |
| Replay + composition checks | 0.0 | 0.0 | 100.0 | 100.0 | 0.0 | 495 |
Widths: 1.6,0.75,0.75,0.9,0.9,0.85,0.65
Table: Table 3: Controlled fault benchmark, mean over five run seeds with 95 % run-bootstrap intervals where nonzero. Per method: 35 faulty and 15 valid decisions, 150 evaluation and 100 stress tasks

Gating removed all 35 faulty admissions per method across the five runs and raised evaluation success from 45.3 % to 100 % on this benchmark, at 570 verification tool calls per run against 48. Composition checks reduced verification to 495 calls by refuting the unit-mismatch candidate before execution. In-process checking and independent replay returned identical decisions, as expected for pure functions under identical checks; the separate worker added about 890 ms of verification time per run on our Windows host and showed no accuracy benefit.

### 6.2 Candidates written by frozen language models

The second experiment replaces constructed faults with candidates written by models. Each model received the tool documentation, a task specification, and one example, under six conditions: documented or signature-only tools, crossed with a full specification, a terse request, or an example-only request. The registry includes one plausible shortcut tool, sum_values, which adds raw values without unit conversion or validity filtering and is documented truthfully. Table 4 summarizes generation.

| Model | Samples | Over token budget | Other errors | Failed example | Submitted | Unique | Valid | Faulty | Fresh-sample success |
|---|---|---|---|---|---|---|---|---|---|
| Clover, run 1 | 60 | 11 | 1 | 7 | 41 | 31 | 12 | 19 | 23.3 % |
| Clover, run 2 | 60 | 16 | 0 | 9 | 35 | 22 | 11 | 11 | 23.3 % |
| DeepSeek (748B) | 24 | 10 | 0 | 0 | 14 | 9 | 9 | 0 | 58.3 % |
Widths: 1.1,0.6,0.62,0.55,0.6,0.72,0.55,0.45,0.5,0.71
Table: Table 4: Candidate generation by frozen models over six prompt conditions. Fresh-sample success is the share of all samples whose program is valid on the hidden suite, which equals expected success when every task receives a new sample and no library

The two Clover runs wrote faulty programs for 19 of 31 (run 1) and 11 of 22 (run 2) unique compiled candidates. Of these 30 faults, 29 called sum_values and therefore ignored units, validity flags, or both; the remaining one skipped validation. Each passed its discovery example because that example contained only valid records in liters. All 30 came from the terse and example-only requests; with the full specification and documented tools, every compiled candidate of every model was valid. DeepSeek wrote no faulty candidate. With signature-only documentation, many responses reached the 4,096-token output budget before producing a program (11 in Clover run 1, 16 in run 2, 10 for DeepSeek). Without a library, a task that receives a fresh sample would succeed 23.3 % of the time in either Clover run and 58.3 % of the time with DeepSeek.

Table 5 reports admission outcomes on the unique compiled candidates of each run. Validity labels come from hidden adjudication on the union of evaluation and stress cases for all five seeds, never from an admission method.

| Model | Admission method | Faulty admitted | Valid rejected | Eval. success (range over seeds) | Verif. tool calls | Verif. LLM tokens |
|---|---|---|---|---|---|---|
| Clover, run 1 | Admit after first success | 19 / 19 | 0 / 12 | 54.0 % (46.7–63.3) | 89 | 0 |
| Clover, run 1 | LLM self-verification | 0 / 19 | 8 / 12 | 100.0 % | 89 | 30,673 |
| Clover, run 1 | LLM evidence verification (SmartSnap-style) | 0 / 19 | 6 / 12 | 100.0 % | 174 | 80,402 |
| Clover, run 1 | Independent replay | 0 / 19 | 0 / 12 | 100.0 % | 1335 | 0 |
| Clover, run 1 | Replay + composition checks | 0 / 19 | 0 / 12 | 100.0 % | 1335 | 0 |
| Clover, run 2 | Admit after first success | 11 / 11 | 0 / 11 | 51.3 % (40.0–63.3) | 73 | 0 |
| Clover, run 2 | LLM self-verification | 0 / 11 | 9 / 11 | 100.0 % | 73 | 24,292 |
| Clover, run 2 | LLM evidence verification (SmartSnap-style) | 0 / 11 | 3 / 11 | 100.0 % | 159 | 68,015 |
| Clover, run 2 | Independent replay | 0 / 11 | 0 / 11 | 100.0 % | 1095 | 0 |
| Clover, run 2 | Replay + composition checks | 0 / 11 | 0 / 11 | 100.0 % | 1095 | 0 |
| DeepSeek (748B) | Admit after first success | 0 / 0 | 0 / 9 | 100.0 % | 41 | 0 |
| DeepSeek (748B) | LLM self-verification | 0 / 0 | 2 / 9 | 100.0 % | 41 | 14,170 |
| DeepSeek (748B) | LLM evidence verification (SmartSnap-style) | 0 / 0 | 0 / 9 | 100.0 % | 123 | 16,029 |
| DeepSeek (748B) | Independent replay | 0 / 0 | 0 / 9 | 100.0 % | 615 | 0 |
| DeepSeek (748B) | Replay + composition checks | 0 / 0 | 0 / 9 | 100.0 % | 615 | 0 |
Widths: 1.0,1.65,0.7,0.7,1.0,0.6,0.75
Table: Table 5: Admission of model-written candidates, mean over five seeds (LLM verdicts do not depend on the seed). Counts are unique compiled candidates; success uses each admitted library with uniform selection on 30 evaluation tasks per seed

First-success admission accepted every faulty candidate, and evaluation success fell to 46.7–63.3 % in Clover run 1 and 40.0–63.3 % in run 2 across seeds. Replay, with or without composition checks, admitted no faulty candidate and rejected no valid one in any run. Both LLM verifiers also admitted no faulty candidate, so the models detect this fault class when asked to check for it, but both rejected valid candidates: self-verification rejected 8 of 12 and 9 of 11 valid candidates in the two Clover runs and 2 of 9 for DeepSeek, and evidence verification rejected 6 of 12, 3 of 11, and 0 of 9, counting unparseable verdicts as rejections. Over-rejection reduced library size but not success in this benchmark, because at least one valid skill remained in every library; in a task family with a single valid program per task kind, the same rejection rates would leave tasks without a skill. Replay used 615 to 1,335 tool calls per run and no model tokens; the LLM verifiers used 14,170 to 80,402 completion tokens per run across all candidates, in addition to their tool calls.

### 6.3 Stochastic and partially observable tools

The third experiment replaces the reader with read_records_noisy, which returns each record independently with probability 0.9, so a single call observes part of the input. Candidates reconcile one, two, or three independent reads; a fourth candidate reconciles three reads but ignores units. True success rates are estimated from 20,000 seeded executions on 200 evaluation tasks. Each gate runs 500 times per candidate with fresh promotion cases and seeds. Table 6 reports acceptance rates against the requirement tau = 0.90.

| Candidate | True success | Meets tau = 0.90 | Exact, 1 replay | Exact, all of 4 | Statistical gate | Statistical executions |
|---|---|---|---|---|---|---|
| one read | 0.755 | no | 0.8 % | 0.0 % | 0.0 % | 76 |
| two reads | 0.971 | yes | 67.4 % | 16.0 % | 100.0 % | 150 |
| three reads | 0.997 | yes | 94.4 % | 82.4 % | 100.0 % | 63 |
| three reads, ignores units | 0.474 | no | 0.0 % | 0.0 % | 0.0 % | 60 |
Widths: 1.55,0.7,0.75,0.8,0.8,0.8,0.9
Table: Table 6: Acceptance rates over 500 gate trials per candidate with stochastic reads (p_drop = 0.1)

Exact single replay accepted the two-read skill, which meets the requirement, in 67.4 % of trials, and requiring all of four replays reduced that to 16.0 %; even the three-read skill was rejected in 17.6 % of trials under the four-replay rule. The statistical gate accepted both adequate skills in every trial and rejected both inadequate skills in every trial, using 150 executions on average for the two-read skill and 63 for the three-read skill.

### 6.4 Composition at depth

The fourth experiment composes chains of two to five admitted primitives over two declared tags: unit (liters, milliliters, centiliters) and validated. Each chain is a preparation skill, zero to three rescaling skills, and a renderer that requires validated liters. All primitives pass their own fixture tests, and no chain is tested before the final evaluation. Compatible and incompatible chains are sampled in equal numbers where enough exist. Table 7 reports the flat library, end-to-end replay on five tasks, and the interface check.

| Length | Chains | Incompat. | Flat: wrong tasks | Incompat., always correct | Replay-5: incompat. admitted | Replay-5: calls per chain | Check: incompat. admitted | Check: compat. rejected |
|---|---|---|---|---|---|---|---|---|
| 2 | 6 | 5 | 56.7 % | 0 | 0 | 10 | 0 | 0 |
| 3 | 36 | 34 | 72.2 % | 0 | 0 | 15 | 0 | 0 |
| 4 | 64 | 60 | 68.8 % | 2 | 2 | 20 | 0 | 0 |
| 5 | 68 | 60 | 62.4 % | 4 | 4 | 25 | 0 | 0 |
Widths: 0.6,0.6,0.7,0.7,0.8,0.8,0.75,0.75,0.7
Table: Table 7: Composition at depth, 20 evaluation tasks per chain. The interface check uses no tool calls

Every incompatible chain produced some correct outputs, so per-task output checks on a few examples can pass an incompatible chain. End-to-end replay on five boundary-covering tasks admitted 6 incompatible chains, all of them functionally correct because their unit errors cancelled, and caught every chain that produced a wrong total, at 10 to 25 tool calls per chain. The interface check admitted no incompatible chain and rejected no compatible chain without any tool calls; it also blocked the 6 chains whose errors cancelled, which replay would have admitted.

### 6.5 Selection in larger libraries

The fifth experiment builds libraries of 10, 50, and 200 skills, 40 % valid variants with zero to three redundant steps and 60 % faulty variants from six fault templates, and evaluates 200 tasks per seed. Table 8 reports success and cost for four selection rules with and without the gate.

| Library size | Admission | Selection rule | Admitted (faulty) | Success (range over seeds) | Violations | Tool calls per task |
|---|---|---|---|---|---|---|
| 10 | ungated | uniform | 10 (6) | 51.3 % (49.0–54.5) | 10.7 % | 6.91 |
| 10 | ungated | fewest steps | 10 (6) | 0.0 % | 0.0 % | 1.00 |
| 10 | ungated | evidence-ranked | 10 (6) | 100.0 % | 0.0 % | 5.60 |
| 10 | ungated | runtime feedback (UCB) | 10 (6) | 56.7 % (53.0–58.5) | 2.0 % | 6.85 |
| 10 | gated | uniform | 4 (0) | 100.0 % | 0.0 % | 7.56 |
| 10 | gated | fewest steps | 4 (0) | 100.0 % | 0.0 % | 5.60 |
| 10 | gated | evidence-ranked | 4 (0) | 100.0 % | 0.0 % | 5.60 |
| 10 | gated | runtime feedback (UCB) | 4 (0) | 100.0 % | 0.0 % | 7.57 |
| 50 | ungated | uniform | 50 (30) | 56.2 % (52.5–62.5) | 9.9 % | 7.29 |
| 50 | ungated | fewest steps | 50 (30) | 0.0 % | 0.0 % | 1.00 |
| 50 | ungated | evidence-ranked | 50 (30) | 100.0 % | 0.0 % | 5.80 |
| 50 | ungated | runtime feedback (UCB) | 50 (30) | 47.9 % (42.0–53.0) | 4.0 % | 6.75 |
| 50 | gated | uniform | 20 (0) | 100.0 % | 0.0 % | 8.38 |
| 50 | gated | fewest steps | 20 (0) | 100.0 % | 0.0 % | 5.80 |
| 50 | gated | evidence-ranked | 20 (0) | 100.0 % | 0.0 % | 5.80 |
| 50 | gated | runtime feedback (UCB) | 20 (0) | 100.0 % | 0.0 % | 8.17 |
| 200 | ungated | uniform | 200 (120) | 52.8 % (46.5–59.5) | 10.6 % | 6.86 |
| 200 | ungated | fewest steps | 200 (120) | 0.0 % | 0.0 % | 1.00 |
| 200 | ungated | evidence-ranked | 200 (120) | 100.0 % | 0.0 % | 5.00 |
| 200 | ungated | runtime feedback (UCB) | 200 (120) | 32.2 % (31.0–34.0) | 5.6 % | 5.62 |
| 200 | gated | uniform | 80 (0) | 100.0 % | 0.0 % | 7.87 |
| 200 | gated | fewest steps | 80 (0) | 100.0 % | 0.0 % | 5.00 |
| 200 | gated | evidence-ranked | 80 (0) | 100.0 % | 0.0 % | 5.00 |
| 200 | gated | runtime feedback (UCB) | 80 (0) | 100.0 % | 0.0 % | 7.95 |
Widths: 0.65,0.75,1.35,0.9,1.3,0.7,0.75
Table: Table 8: Selection rules in libraries of 10 to 200 skills, mean over five seeds with 200 tasks each

Without the gate, uniform selection reached 51 % to 56 % success, and cost-greedy selection reached 0 % because the cheapest program is the memorized constant. The runtime-feedback rule, which observes only errors and constraint violations, fell to 32 % at 200 skills because wrong totals produce no runtime signal. Ranking by admission-test results reached 100 % even without the gate, because it uses the same verification evidence. With the gate, every rule reached 100 %, and cost-greedy selection also minimized tool calls.

### 6.6 Evidence attacks and poisoning

The sixth experiment applies six attacks to the bundle of a faulty candidate that the honest gate rejected and runs each defence layer on its own. Table 9 reports which layer detected each attack.

| Attack | Artifact hashes | External anchor | HMAC signature | Audit re-execution |
|---|---|---|---|---|
| no attack | not detected | not detected | not detected | not detected |
| edit decision only | detected | detected | detected | detected |
| edit decision and recompute manifest hashes | not detected | detected | detected | detected |
| delete test inputs | detected | detected | detected | detected |
| compromised verifier writes forged decision | not detected | not detected | not detected | detected |
| compromised verifier records modified source | not detected | not detected | not detected | detected |
Widths: 2.4,1.0,1.0,1.0,1.0
Table: Table 9: Detection of evidence attacks by each defence layer used alone

Per-artifact hashes detect edits but not an attacker who also recomputes the manifest. An external anchor or signature detects that attacker. A compromised verifier that writes a forged decision at decision time produces a correctly anchored and signed bundle; only the audit's re-execution of the decision detects it, and a verifier that records its own modified source is detected by the environment check.

The poisoning case is a skill that writes to the literal destination report. It passes every promotion case because all of them use that destination. Across five seeds the standard gate admitted it, and deployment on 200 tasks with 20 % alternate destinations produced 37 to 46 violations without monitoring. Adding an alternate-destination boundary class to the coverage requirement caused the gate to reject it in every seed. A runtime monitor that revokes a skill after its first violation limited deployment to one violation per seed.

### 6.7 Retention under a tool upgrade

The seventh experiment runs ten checkpoints of 30 tasks. Before checkpoint 4 the provider upgrades validate_fields to version 2.0.0, which keeps records whose valid flag is false, and publishes a filter_valid tool. Table 10 reports success before and after the change.

| Condition | Success before change | Success after change | Observable errors after change |
|---|---|---|---|
| unpinned | 100.0 % | 60.0 % | 0 |
| pinned, no lifecycle | 100.0 % | 0.0 % | 180 |
| pinned with revocation and re-verification | 100.0 % | 100.0 % | 0 |
Widths: 2.6,1.2,1.2,1.4
Table: Table 10: Retention across ten checkpoints of 30 tasks; the tool change takes effect at checkpoint 4

Unpinned skills continued to run and produced wrong totals on every task that contained an invalid record, and no error was raised. Version pins turned the change into errors on every task. With the lifecycle, the version change revoked the stale skill, re-verification rejected the stale program re-pinned to version 2.0.0, and admitted an adapted program that calls filter_valid, so success stayed at 100 %.

## 7 Discussion and Limitations

**What the results support.** Replay with coverage requirements excluded every controlled fault class and every faulty model-written candidate in our pools, at roughly ten times the verification cost of first-success admission. Same-model verification was a weaker filter on the same candidates (Section 6.2). Exact replay is the wrong test for stochastic tools, and a statistical gate with a threshold fixed in advance separated adequate from inadequate skills in our setting. Interface checks are a cheap pre-filter that is sound for declared tags and conservative otherwise. Evidence integrity requires layers: hashes, an external anchor or signature, and re-execution each detect a different attack.

**Limitations.** The environments are small and synthetic, and the model-written faults are dominated by one shortcut that the registry made available. Model-written candidates come from only two models, one of them sampled twice. Natural fault rates depend on prompts and documentation and do not transfer to other tasks. Coverage requirements only test what the boundary classes describe; the poisoning case passed the standard gate. The separate verifier worker shares the operating-system identity of the policy and is not an adversarial sandbox; signatures assume the key is held outside the agent's reach. No policy is trained, so the paper makes no claim about reward-driven learning, utilization drift under learning, or the reward schedule in Appendix C. SmartSnap and SuperIntelliAgent were not run end to end; we re-implemented their admission rule. Evidence bundles can contain sensitive data and need access control and retention rules in deployment.

**Next steps.** Evaluate the gate on an established interactive benchmark such as ALFWorld (Shridhar et al., 2021) with model-generated candidates, train a policy against the stored reward components, and enforce the policy, verifier, and evidence boundary at the operating-system level.

## 8 Conclusion

ASG-SI makes the admission of a skill an explicit, recorded decision. Across seven experiments, held-out replay with coverage and interface checks kept faulty skills out of libraries in which first-success admission and same-model verification admitted them or rejected valid skills, a statistical gate handled stochastic tools, and layered evidence controls detected each simulated attack. The results are limited to small synthetic environments and frozen models, and they are reproducible from the published evidence.

## References

REF Cai, S., Qin, Y., Lin, H., Xu, Z., Li, G., Shi, Y., Li, Z., Mao, Y., Cai, S., Tan, X., Liang, Y., Li, K., & Sun, X. (2025). SmartSnap: Proactive Evidence Seeking for Self-Verifying Agents. *arXiv preprint arXiv:2512.22322*. https://arxiv.org/abs/2512.22322
REF Chen, M. F., Roberts, N., Bhatia, K., Wang, J., Zhang, C., Sala, F., & Ré, C. (2023). Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models. *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2307.14430
REF Chen, Z., Zhao, Z., Zhang, K., Liu, B., Qi, Q., Wu, Y., Kalluri, T., Cao, S., Xiong, Y., Tong, H., Yao, H., Li, H., Zhu, J., Li, X., Song, D., Li, B., Weston, J., & Huynh, D. (2025). Scaling Agent Learning via Experience Synthesis. *arXiv preprint arXiv:2511.03773*. https://arxiv.org/abs/2511.03773
REF Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika, 26*(4), 404–413. https://doi.org/10.1093/biomet/26.4.404
REF Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). Training Verifiers to Solve Math Word Problems. *arXiv preprint arXiv:2110.14168*. https://arxiv.org/abs/2110.14168
REF Gou, Z., Shao, Z., Gong, Y., Shen, Y., Yang, Y., Duan, N., & Chen, W. (2023). CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing. *arXiv preprint arXiv:2305.11738*. https://arxiv.org/abs/2305.11738
REF Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., & Zhou, D. (2023). Large Language Models Cannot Self-Correct Reasoning Yet. *arXiv preprint arXiv:2310.01798*. https://arxiv.org/abs/2310.01798
REF Joshi, T., Chowdhury, S., & Uysal, F. (2025). SWE-Bench-CL: Continual Learning for Coding Agents. *arXiv preprint arXiv:2507.00014*. https://arxiv.org/abs/2507.00014
REF Li, M. Q., Fung, B. C. M., Weiss, M., Xiong, P., Al-Hussaeni, K., & Fachkha, C. (2025). A Benchmark for Evaluating Outcome-Driven Constraint Violations in Autonomous AI Agents. *arXiv preprint arXiv:2512.20798*. https://arxiv.org/abs/2512.20798
REF Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K. (2023). Let's Verify Step by Step. *arXiv preprint arXiv:2305.20050*. https://arxiv.org/abs/2305.20050
REF Lin, J., Pan, Z., Zhu, Y., Song, R., & Yang, J. (2025). Towards Continuous Intelligence Growth: Self-Training, Continual Learning, and Dual-Scale Memory in SuperIntelliAgent. *arXiv preprint arXiv:2511.23436*. https://arxiv.org/abs/2511.23436
REF Luo, X., Zhang, Y., He, Z., Wang, Z., Zhao, S., Li, D., Qiu, L. K., & Yang, Y. (2025). Agent Lightning: Train ANY AI Agents with Reinforcement Learning. *arXiv preprint arXiv:2508.03680*. https://arxiv.org/abs/2508.03680
REF Necula, G. C. (1997). Proof-carrying code. In *Proceedings of the 24th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages* (pp. 106–119). https://doi.org/10.1145/263699.263712
REF Qian, C., Acikgoz, E. C., He, Q., Wang, H., Chen, X., Hakkani-Tür, D., Tur, G., & Ji, H. (2025). ToolRL: Reward is All Tool Learning Needs. *arXiv preprint arXiv:2504.13958*. https://arxiv.org/abs/2504.13958
REF Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. *arXiv preprint arXiv:2302.04761*. https://arxiv.org/abs/2302.04761
REF Sha, L. (2001). Using simplicity to control complexity. *IEEE Software, 18*(4), 20–28. https://doi.org/10.1109/MS.2001.936213
REF Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *arXiv preprint arXiv:2303.11366*. https://arxiv.org/abs/2303.11366
REF Shridhar, M., Yuan, X., Côté, M.-A., Bisk, Y., Trischler, A., & Hausknecht, M. (2021). ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. *International Conference on Learning Representations*. https://arxiv.org/abs/2010.03768
REF Su, J., Zeng, X., Liu, L., Luo, C., Chen, Y., & Zhuang, Z. (2025). Enhancing Agentic RL with Progressive Reward Shaping and Value-based Sampling Policy Optimization. *arXiv preprint arXiv:2512.07478*. https://arxiv.org/abs/2512.07478
REF Torres-Arias, S., Afzali, H., Kuppusamy, T. K., Curtmola, R., & Cappos, J. (2019). in-toto: Providing farm-to-table guarantees for bits and bytes. In *Proceedings of the 28th USENIX Security Symposium* (pp. 1393–1410). https://www.usenix.org/conference/usenixsecurity19/presentation/torres-arias
REF Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., & Anandkumar, A. (2023). Voyager: An Open-Ended Embodied Agent with Large Language Models. *arXiv preprint arXiv:2305.16291*. https://arxiv.org/abs/2305.16291
REF Wang, J., Yan, Q., Wang, Y., Tian, Y., Mishra, S. S., Xu, Z., Gandhi, M., Xu, P., & Cheong, L. L. (2025). Reinforcement Learning for Self-Improving Agent with Skill Library. *arXiv preprint arXiv:2512.17102*. https://arxiv.org/abs/2512.17102
REF Wen, X., Liu, Z., Zheng, S., Ye, S., Wu, Z., Wang, Y., Xu, Z., Liang, X., Li, J., Miao, Z., Bian, J., & Yang, M. (2025). Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs. *arXiv preprint arXiv:2506.14245*. https://arxiv.org/abs/2506.14245
REF Wu, F., Tu, A., Xuan, W., Qi, H., Huang, X., Zeng, Q., Talaei, S., Xiao, Y., Xia, P., Tang, X., Zhuang, Y., Li, Y., Hu, B., Cao, H., Shi, W., Yang, R., Liu, N., Yao, H., Liu, G., … Choi, Y. (2025). Position: The Hidden Costs and Measurement Gaps of Reinforcement Learning with Verifiable Rewards. *arXiv preprint arXiv:2509.21882*. https://arxiv.org/abs/2509.21882
REF Yan, S., Yang, X., Huang, Z., Nie, E., Ding, Z., Li, Z., Ma, X., Bi, J., Kersting, K., Pan, J. Z., Schütze, H., Tresp, V., & Ma, Y. (2025). Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning. *arXiv preprint arXiv:2508.19828*. https://arxiv.org/abs/2508.19828
REF Zhang, G., Geng, H., Yu, X., Yin, Z., Zhang, Z., Tan, Z., Zhou, H., Li, Z., Xue, X., Li, Y., Zhou, Y., Chen, Y., Zhang, C., Fan, Y., Wang, Z., Huang, S., Piedrahita-Velez, F., Liao, Y., Wang, H., … Bai, L. (2025). The Landscape of Agentic Reinforcement Learning for LLMs: A Survey. *arXiv preprint arXiv:2509.02547*. https://arxiv.org/abs/2509.02547
REF Zhang, H., Liu, X., Lv, B., Sun, X., Jing, B., Iong, I. L., Hou, Z., Qi, Z., Lai, H., Xu, Y., Lu, R., Wang, H., Tang, J., & Dong, Y. (2025). AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework. *arXiv preprint arXiv:2510.04206*. https://arxiv.org/abs/2510.04206
REF Zhou, Z., Qu, A., Wu, Z., Kim, S., Prakash, A., Rus, D., Zhao, J., Low, B. K. H., & Liang, P. P. (2025). MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents. *arXiv preprint arXiv:2506.15841*. https://arxiv.org/abs/2506.15841

## Appendix A: Reproducibility

The implementation, tests, and evidence are at https://github.com/kenhuangus/ASG-SI on branch review-response; every reported result was produced by commit 71b7548e62c37b43db4f886dc06946c4740c6e85 with Python 3.13.14 on Windows 11, and results/review/ holds the result files, the response cache, the raw evidence of the main benchmark, and SHA-256 checksums. The audit compares the recorded source hashes and Python version with the auditor's checkout, so it must run with the same Python version and with line endings preserved as committed. The commands below regenerate every table from a clean checkout; the language-model experiments replay from the recorded response cache without access to the models.

```
python -m unittest discover -s tests
python -m experiments.run --config configs/smoke.json --output runs/main
python -m experiments.report runs/main
python -m experiments.audit runs/main
python -m experiments.llm_candidates --model clover --output runs/llm --n-per-spec 10 --offline
python -m experiments.stochastic --output runs/review/stochastic.json --trials 500
python -m experiments.composition_depth --output runs/review/composition_depth.json
python -m experiments.selection --output runs/review/selection.json
python -m experiments.tamper --output runs/review/tamper.json
python -m experiments.retention --output runs/review/retention.json
```

## Appendix B: Prompts for the language-model experiments

Each generation request contains a system message ("You write small tool-calling programs. Reply with a single JSON object and nothing else."), the tool documentation (full descriptions, or signatures only), the program format with binding rules, one of the three task statements below, and the discovery example with its expected output. Requests use temperature 0.8 and at most 4,096 output tokens; verification requests use temperature 0. The full prompt text is in experiments/llm_candidates.py, and every request and response is stored in the response cache.

- Full: "Write a reusable program for this task. Output the total volume, in liters, of all records whose valid field is true. Records can be in liters (L) or milliliters (mL). Write the report to the task's destination; writing anywhere other than allowed_destination is prohibited. The program must work for any input of this task, not only the example."
- Terse: "Write a program that totals the record volumes and writes the report."
- Example only: "Write a program that turns the example input into the example output."
- Self-verification: the full task statement, the candidate program, its execution on the discovery example, and the question "Will this program produce the correct report, without prohibited writes, for every valid input of this task?"
- Evidence verification: the proposing model first selects three task inputs that "would best reveal whether this program is correct"; the harness executes the program on them; the model then receives the program, the inputs, and the actual outputs and returns ACCEPT or REJECT.

\pagebreak
## Appendix C: Proposed reward schedule (design only)

The weights below are a proposal for future policy training and are not used by any result in this paper. Table 11 lists them by phase.

| Phase | Iterations | tool | outcome | reuse | compose | memory |
|---|---|---|---|---|---|---|
| structural | 1–2 | 0.5 | 0.3 | 0.1 | 0.0 | 0.1 |
| correctness | 3–4 | 0.2 | 0.4 | 0.2 | 0.1 | 0.1 |
| efficiency | 5–6 | 0.1 | 0.3 | 0.2 | 0.3 | 0.1 |
Table: Table 11: Proposed phase-dependent reward weights (not evaluated)
