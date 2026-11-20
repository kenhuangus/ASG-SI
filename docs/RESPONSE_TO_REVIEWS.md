# Response to ICML 2026 reviews (submission 3760)

Each comment from the area chair (AC), the three reviewers (iMLt, Df4u, EXwR), and the automated
PAT feedback is listed with the change made and where it appears. Section and table numbers refer to
the revised paper (`paper/paper.md`). Experiment code and result files are named so each claim can be checked.

## Area chair

| ID | Comment | Response | Where |
|---|---|---|---|
| AC-1 | Method unclear to the point of irreproducibility | Definitions, notation table, three algorithms; every table regenerated from published raw evidence; audit command re-executes decisions | §4, Table 2, Alg. 1–3, App. A |
| AC-2 | Insufficient positioning in the literature | Related work rewritten into five groups with 28 verified references and a comparison table | §2, Table 1 |
| AC-3 | Limited results: domain difficulty and baselines | Eight experiments, including candidates from two frozen LLMs (three runs), stochastic tools, composition depth, selection at scale, attacks, retention, and model-written skills on ALFWorld; baselines include LLM self-verification and a SmartSnap-style evidence verifier | §6 |

## Reviewer iMLt

| ID | Comment | Response | Where |
|---|---|---|---|
| iMLt-1 | Policy-improvement algorithm never specified | The paper no longer claims policy optimization. All models are frozen; the contribution is the admission gate. Scope stated in a box | §3 scope box, §4.5, §7 |
| iMLt-2 | Base model not described | Two frozen models (one sampled in two independent runs) with sizes, quantization, hardware, and serving stack; all responses cached by request hash | §5 Models |
| iMLt-3 | External verifier not described | Algorithm 2 (exact replay), Algorithm 3 (statistical), reason codes, separate worker | §4.3 |
| iMLt-4 | Skill extraction not described | Algorithm 1 and its limits | §4.2 |
| iMLt-5 | Toy tasks solvable out of the box; results saturated | Models write candidates and frequently produce faults that pass one example (Table 4); stochastic tools give non-saturated gate behaviour (Table 6); larger libraries (Table 8); model-written skills on the ALFWorld benchmark, where 1 of 19 working programs was adequate and only replay admitted it without admitting an inadequate one (Table 11). The other tasks remain small and synthetic, stated as a limitation | §6.2–6.5, §6.8, §7 |
| iMLt-6 | No baseline comparisons | First success, LLM self-verification, SmartSnap-style evidence verification, in-process checking, independent replay, replay with composition checks, flat library, end-to-end chain replay, four selection rules | Tables 3, 5, 7, 8 |
| iMLt-7 | Figures 1 and 2 identical; Figure 4 lacks detail | All figures redrawn with distinct content: architecture, interface propagation, lifecycle | Figures 1–3 |
| iMLt-8 | Undefined terms (evidence-store integrity, evidence-based checks, canonical programs) | Defined in Table 2 and §4.6; vague terms removed | §4.1, §4.6 |
| iMLt-9 | Continual learning, forgetting, reward hacking discussed but not tested | Retention experiment under a tool upgrade (Table 10); reward hacking is no longer claimed as tested; memory and reward schedule marked design-only | §6.7, App. C |
| iMLt-10 | Related work too brief; novelty unclear | See AC-2; the contribution is stated as admission control with recorded evidence | §1, §2 |

## Reviewer Df4u

| ID | Comment | Response | Where |
|---|---|---|---|
| Df4u-1 | Only three small deterministic tasks | Stochastic and partially observable tool experiment; model-written candidates; ALFWorld, an established partially observable text benchmark | §6.2, §6.3, §6.8 |
| Df4u-2 | No comparison with SmartSnap, SuperIntelliAgent | Both have public code (verified) but target Android GUI control and diffusion training. Their admission rule (LLM verifier over evidence, proposer-curated in SmartSnap) was re-implemented in the same harness and compared | §2, §5, Table 5 |
| Df4u-3 | Missing literature on self-verification and continual adaptation | Reflexion, CRITIC, Huang et al., SmartSnap, SuperIntelliAgent, SWE-Bench-CL, MEM1, Memory-R1 | §2 |
| Df4u-4 | Claims of scalability and governance aspirational | Claims narrowed; scope box; limitations list | §3, §7 |
| Df4u-5 | No variance analysis | Five independent seeds with bootstrap intervals; ranges over seeds; 500 gate trials per candidate in §6.3 | Tables 3, 5, 6, 8 |
| Df4u-6 | Verification under uncertainty | Sequential Clopper–Pearson gate with tau fixed in advance; compared with exact replay | Alg. 3, Table 6 |
| Df4u-7 | Verifier manipulation, poisoned skills, tampered evidence | Attack matrix across hashes, anchor, HMAC signature, and re-execution; latent poisoned skill with coverage and revocation defences | §6.6, Table 9 |
| Df4u-8 | Compositional scale | Chains of length 2 to 5 over two declared tags, held out from primitive tests | §6.4, Table 7 |
| Df4u-9 | Formalize skill interface and evidence semantics | Interfaces as Pre/Post tag assignments with propagation; bundle contents and audit criterion | §4.1, §4.4, §4.6 |

## Reviewer EXwR

| ID | Comment | Response | Where |
|---|---|---|---|
| EXwR-W1 | Very limited experimental setting | See Df4u-1 and iMLt-5 | §6 |
| EXwR-W2 | Skill selection among many skills; large graphs; exploration versus reuse | Four selection rules, libraries of 10, 50, 200 skills; a UCB rule that learns from runtime-observable outcomes shows that runtime feedback cannot detect silent faults | §6.5, Table 8 |
| EXwR-W3 / C1 | Manually chosen reward weights may invite gaming | Results use a fixed success metric; the schedule is moved to an appendix as an unevaluated design; the gate never reads weighted reward | §4.7, App. C |
| EXwR-C2 | Overlapping arrows in figures | Figures redrawn with at most 8 shapes and no crossing arrows; rendered at 5000 px and inspected | Figures 1–3 |
| EXwR-C3 | Note instructing reviewers to include phrases | Not present in the revision; the text extraction of the PDF contains no such note | whole paper |

## PAT feedback

| ID | Comment | Response | Where |
|---|---|---|---|
| PAT-1 | Distinguish capability drift from utilization drift | Admission governs which skills exist; selection results show utilization effects | §4.5, §6.5 |
| PAT-2 | State determinism assumptions early | Stated in §3 with the statistical relaxation | §3 |
| PAT-3 | Agent as adversary | Third adversary tier; poisoning experiment | §3, §6.6 |
| PAT-4 | Verification used for both rewards and gating | Rewards are stored for reconstruction; gating is the evaluated use | §1, §4.7 |
| PAT-5 | Skill compiler and interface synthesis underspecified | Algorithm 1; interfaces come from declared contracts, not inferred | §4.2 |
| PAT-6 | Reward weights, scaling constants, phase schedule undefined | Fixed metric in all results; schedule in App. C marked unevaluated | §4.7, App. C |
| PAT-7 | composition-ok predicate undefined | Defined via tag propagation; Figure 2 | §4.1, §4.4 |
| PAT-8 | Guarded fallbacks not defined | Removed from the evaluated method; listed as a design proposal in the scope box | §3 |
| PAT-9 | Only atomic skills validated | Compiler now handles multi-step trajectories; composition up to length 5 | §4.2, §6.4 |
| PAT-10 | Learned versus scripted policy unclear | Frozen policies only, stated throughout | §3, §5 |
| PAT-11 | Zero variance versus t-tests | No p-values; bootstrap intervals over seeds | §5, Tables |
| PAT-12 | End-to-end RL baseline underspecified | The RL baseline is removed with the RL claim; a fresh-sample no-library baseline is reported | Table 4 |
| PAT-13 | "Added" column inconsistent | Table removed; unique insertions counted separately in the code | §6.1 |
| PAT-14 | 64-bit truncated hash | Full SHA-256 everywhere, plus optional HMAC | §4.6 |
| PAT-15 | Typos (¡ for <, P for Σ, wrong section reference) | Text rewritten; no such characters remain | whole paper |

## Corrections found during revision

- The previous draft stated that SmartSnap and SuperIntelliAgent have no public implementations. Both do (github.com/TencentYoutuResearch/SmartSnap, github.com/jianzhelin/SuperIntelliAgent). The statement was removed.
- Reference metadata was re-checked against DataCite, arxiv.org, Crossref, and publisher pages. Corrected entries: Wu et al. (2025) (previously listed under Tu), Su et al. (2025) (previously listed under Zhuang, title updated), Chen, M. F. et al. (2023) for Skill-it (previously listed with other authors), and ALFWorld (ICLR 2021).
- The previous draft attributed results to a GPT-4o-mini policy served by Ollama and trained with REINFORCE. No such experiment exists in the repository; those claims were removed.
- The first revision reported one candidate pool as written by qwen3.5-4b. The model name recorded with every response shows that the Mac server answered all 120 of those requests with its loaded model, clover-35b-a3b. The pool is now reported as a second Clover run, and the client rejects responses whose model differs from the request.
