# ASG-SI: Audited Skill-Graph Self-Improvement

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A framework for governable self-improvement in agentic AI systems through verifier-gated skill promotion and auditable evidence trails.

---

## 🎯 Overview

**ASG-SI (Audited Skill-Graph Self-Improvement)** is a research framework that enables AI agents to self-improve during deployment while maintaining transparency, auditability, and measurable progress. Unlike traditional self-improving systems where capability changes are implicit in parameter updates, ASG-SI structures improvement as the accumulation of verified, reusable skills with explicit interfaces and evidence-backed validation.

### The Problem

Self-improving AI systems deployed in real-world applications face critical challenges:

- **Reward Hacking**: Agents may discover shortcuts that maximize metrics while violating intended constraints
- **Undetectable Drift**: Behavioral changes during deployment can be opaque and hard to govern
- **Non-Modular Improvement**: Aggregate performance gains don't reveal which specific capabilities improved
- **Lack of Auditability**: No way to inspect what the agent learned or why a capability was added

These issues are especially critical in **security-sensitive deployments** (healthcare, finance, infrastructure) where unexplained capability changes can violate compliance requirements or introduce unacceptable risks.

### The Solution

ASG-SI addresses these challenges through three core innovations:

1. **Verifier-Auditor Architecture**: Skills are promoted only after passing replay-based verification on held-out tasks, creating a governance boundary between candidate behaviors and deployed capabilities.

2. **Evidence-Reconstructible Rewards**: All reward components derive from verifier-produced evidence bundles (tool schemas, test outcomes, contract checks), enabling third-party audit and reward reconstruction.

3. **Auditable Skill Graph**: Improvement accumulates as a directed graph of verified skills with explicit interfaces, making capability changes inspectable, modular, and reversible.

---

## 🚀 Why This Matters

### Impact on AI Safety & Governance

**Transparency**: Operators can understand exactly what capabilities an agent acquired, how they were validated, and what evidence supports their behavior.

**Fine-Grained Control**: Skills can be inspected, disabled, or rolled back individually without retraining the entire model.

**Compliance**: Audit trails become part of regulatory compliance documentation, with cryptographically signed evidence bundles proving capability provenance.

**Reproducibility**: Zero-variance results across seeds demonstrate that verifier-gated promotion creates stable, reproducible learning dynamics.

### Real-World Applications

- **Healthcare AI**: Medical decision agents that self-improve while maintaining full audit trails for FDA compliance
- **Financial Systems**: Trading agents that adapt to market conditions with verifiable skill validation
- **Critical Infrastructure**: Control systems that learn from experience while ensuring safety-critical skills are rigorously tested
- **Enterprise Automation**: Business process agents that improve over time with transparency for stakeholders

---

## 🔬 Key Innovations

### 1. Replay-Based Verification

Unlike prior work that uses deterministic evaluators only for training signals, ASG-SI uses verification to **gate capability accumulation**:

```
Candidate Skill → Verifier Replay (held-out tasks) → Evidence Bundle → Pass/Fail → Promotion Decision
```

- **100% pass rate required** on held-out verification suite
- **Deterministic replay** ensures skills behave identically under test
- **Cryptographic hashing** of programs and artifacts for tamper-evidence

### 2. Decomposed Evidence-Backed Rewards

Five reward components, each reconstructible from evidence:

| Component | Evidence Source | Purpose |
|-----------|----------------|---------|
| Tool Validity | Schema validation logs | Ensure well-formed tool calls |
| Outcome Verification | Deterministic test results | Confirm task correctness |
| Skill Reuse | Contract satisfaction checks | Reward verified skill invocation |
| Composition Integrity | Interface contract checks | Validate multi-skill chains |
| Memory Discipline | Bounded context metrics | Penalize unbounded growth |

### 3. Audited Skill Graph

Skills as first-class artifacts with:
- **Explicit interfaces** (preconditions, postconditions, input/output types)
- **Canonical program representations** (tool call templates, parameter mappings)
- **Verification reports** (pass rates, test outcomes, timestamps)
- **Cryptographic provenance** (hashes of programs, evidence bundles, verifier versions)

Graph edges encode:
- Composition constraints (required intermediate types)
- Ordering dependencies (skill A must precede skill B)
- Guarded fallbacks (if primary fails, invoke alternative)

---

## 📊 Demonstrated Results

### Performance Gains

```
Metric                  Iteration 0  →  Iteration 5  Improvement
─────────────────────────────────────────────────────────────────
Success Rate            97.5%       →  100%          +2.5%
Average Reward          2.74        →  3.34          +21.9%
Skill Reuse Adoption    0%          →  77.5%         +77.5 pp
Direct Solving (Error)  18.8%       →  5.0%          -13.8 pp
```

### Learning Dynamics

- **Iteration 0**: Agent relies on direct tool calls (81.2% tool actions, 18.8% direct)
- **Iteration 1**: Rapid skill adoption after first verification (80% skill reuse)
- **Iterations 2-5**: Stable performance with 75-77.5% skill reuse
- **Final state**: Expert agent using verified skills, minimizing error-prone direct solving

### Auditability & Reproducibility

- **136 verification traces** generated with full audit logs
- **100% pass rate** maintained by all promoted skills on held-out suites
- **Zero variance** across 5 random seeds (deterministic learning dynamics)
- **33% verification overhead** (acceptable for governance benefits)

### Statistical Significance

Improvements over baselines are statistically significant (p<0.001, paired t-test):
- **+4.3%** over end-to-end RL (3.34 vs 3.20)
- **+14.7%** over unverified skill library (3.34 vs 2.91)
- **+21.9%** over tool-only baseline (3.34 vs 2.74)

---

## 🏗️ Architecture

### System Components

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Task Stream │ ──> │ Policy Agent │ <─> │  Tool APIs  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │ trajectories
                           ▼
                    ┌──────────────┐
                    │ Memory Store │
                    │   Compiler   │
                    └──────┬───────┘
                           │ candidates
                           ▼
                    ┌──────────────┐      ┌──────────────┐
                    │  Verifier &  │ ───> │   Audited    │
                    │   Auditor    │      │ Skill Graph  │
                    └──────────────┘      └───────┬──────┘
                                                   │ reuse
                                                   └──────┐
                                                         ▼
                                               (back to Policy Agent)
```

### Core Subsystems

1. **Policy Runtime**: Interacts with tasks using direct actions, tool calls, skill invocations, and memory operations. Logs full trajectories with tool transcripts and intermediate artifacts.

2. **Skill Compiler**: Extracts reusable patterns from successful trajectories, normalizes into canonical programs, and assigns explicit interfaces with input/output types.

3. **Verifier-Auditor**: Replays candidate skills on held-out tasks under controlled harnesses. Emits evidence bundles with test outcomes, schema checks, and cryptographic hashes.

4. **Skill Graph**: Directed multigraph storing verified skills as nodes with explicit interfaces. Edges encode composition constraints and guarded fallbacks.

5. **Memory Manager**: Bounded storage (capacity=8) retaining compact task outcome notes, providing summarized context to the policy.

---

## 💻 Installation & Setup

### Prerequisites

- **Python 3.8+**
- **Optional**: [Ollama](https://ollama.ai/) for model-based policy (not required for baseline)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/kenhuangus/ASG-SI.git
   cd ASG-SI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # or minimal install:
   pip install python-dotenv requests
   ```

3. **Run baseline (stochastic policy)**
   ```bash
   python asg_si_demo.py
   ```

   This runs the system with a built-in stochastic policy (no model required).

4. **Optional: Run with local LLM**
   ```bash
   # Start Ollama and pull a model
   ollama serve
   ollama pull qwen2.5:7b

   # Configure environment
   export OLLAMA_BASE_URL=http://localhost:11434
   export OLLAMA_MODEL=qwen2.5:7b

   # Run with model
   python asg_si_demo.py --model
   ```

---

## 🎮 Usage

### Basic Commands

```bash
# Run with stochastic policy (baseline, no model needed)
python asg_si_demo.py

# Run with local LLM policy
python asg_si_demo.py --model

# Run multi-seed experiments for statistical analysis
python run_multi_seed.py

# Analyze multi-seed results with error bars
python analyze_results.py
```

### Expected Output

```
============================================================
ASG-SI Demo Results
============================================================

iter=0  success=0.975  avg_reward=2.737  actions(skill/tool/direct)=0/65/15
iter=1  success=0.988  avg_reward=3.300  actions(skill/tool/direct)=64/11/5
iter=2  success=0.988  avg_reward=3.288  actions(skill/tool/direct)=60/16/4
iter=3  success=0.975  avg_reward=3.244  actions(skill/tool/direct)=61/14/5
iter=4  success=1.000  avg_reward=3.362  actions(skill/tool/direct)=62/16/2
iter=5  success=1.000  avg_reward=3.337  actions(skill/tool/direct)=62/14/4

============================================================
AGENT IMPROVEMENT SUMMARY
============================================================

📊 Performance Metrics:
   Success rate: 0.975 → 1.000 (+0.025)
   Average reward: 2.737 → 3.337 (+0.600, +21.9%)
   Skill actions: 0 → 62 (+62, 77.5% of tasks)
   Direct actions: 15 → 4 (-11, -73.3%)

🔧 Skill Development:
   Initial skills compiled: 65
   Final skill graph size: 3 (one per task type)
   Total verified skills added: 136

✅ Key Achievements:
   • Learned and verified reusable skills for all task types
   • Improved from direct tool usage to intelligent skill reuse
   • Maintained 100% success rate in final iterations
   • Built audited skill graph with verified capabilities
```

### Configuration

Create `.env` file for model-based policy:

```bash
# Ollama server URL
OLLAMA_BASE_URL=http://localhost:11434

# Model to use (any Ollama-supported model)
OLLAMA_MODEL=qwen2.5:7b
```

---

## 🔍 How It Works

### Learning Loop

```python
for iteration in range(num_iterations):
    # 1. Online Interaction
    trajectories = agent.solve_tasks(tasks, skill_graph)

    # 2. Skill Compilation
    candidates = compiler.extract_skills(trajectories)

    # 3. Verification
    for candidate in candidates:
        evidence = verifier.replay(candidate, heldout_tasks)
        if evidence.pass_rate == 1.0:  # 100% required
            skill_graph.add(candidate, evidence)
            audit_log.record(evidence)

    # 4. Reward Shaping
    rewards = compute_shaped_rewards(trajectories, evidence_bundles)

    # 5. Policy Update (optional, for RL-based policies)
    policy.update(trajectories, rewards)
```

### Verification Protocol

1. **Candidate Extraction**: Successful trajectory with single tool call → skill candidate
2. **Interface Construction**: Map tool arguments to task input keys, assign I/O types
3. **Held-Out Testing**: Replay on 3-10 fresh tasks of same kind
4. **Pass Criteria**: 100% success rate required for promotion
5. **Evidence Generation**: JSON audit trace with outcomes, hashes, timestamps
6. **Skill Promotion**: Add to graph with verification report

### Evidence Bundle Structure

```json
{
  "verified": true,
  "kind": "calc",
  "num_tests": 10,
  "pass_rate": 1.0,
  "results": [
    {"task_id": "abc123", "ok": true, "msg": "pass"}
  ],
  "checked_at": "2026-01-13T10:30:00Z",
  "skill_program_hash": "2a3f9c8d4e1b5a6c",
  "verifier_version": "1.0"
}
```

---

## 📈 Evaluation & Metrics

### Success Metrics

- **Success Rate**: % of tasks solved correctly (exact-match verification)
- **Average Reward**: Decomposed reward summing tool validity, outcome, reuse, composition
- **Skill Reuse %**: % of actions using verified skills (vs. direct tool calls or fallback)
- **Skill Graph Size**: Number of distinct verified skills
- **Audit Coverage**: Number of verification traces with full provenance

### Ablation Studies

| Configuration | Success | Reward | Skill Reuse | Audit Traces |
|--------------|---------|--------|-------------|--------------|
| Full ASG-SI | 100% | 3.34 | 77.5% | 136 |
| No Verification | 95% | 2.91 | 72.5% | 0 |
| No Shaped Rewards | 97.5% | 3.15 | 68.8% | 136 |
| No Memory | 98.8% | 3.28 | 75.0% | 136 |
| No Skills (baseline) | 97.5% | 2.74 | 0% | 0 |

**Key Findings**:
- Verification is critical for reliability (95% → 100% success)
- Shaped rewards guide toward auditable patterns (68.8% → 77.5% reuse)
- Memory provides modest gains in short-horizon tasks
- Skill graph enables all improvement (0% → 77.5% reuse)

---

## 🛠️ Technical Details

### Verifiable Task Environment

Three deterministic task types:

```python
# Arithmetic calculation
Task(kind="calc", input={"expr": "5 + 3"}, expected=8)

# String reversal
Task(kind="reverse", input={"text": "hello"}, expected="olleh")

# String concatenation
Task(kind="concat", input={"a": "foo", "b": "bar"}, expected="foobar")
```

### Tool Registry

Pure functions with explicit schemas (implementation uses safe evaluation with restricted character sets):

```python
tools = {
    "calc": {"expr": "str"},      # Arithmetic expression
    "reverse": {"text": "str"},   # String to reverse
    "concat": {"a": "str", "b": "str"}  # Strings to concatenate
}
```

### Skill Representation

```python
skill = {
    "skill_id": "calc_skill_1",
    "kind": "calc",
    "interface": {
        "inputs": ["expr"],
        "output_type": "int"
    },
    "program": {
        "type": "TOOL_CALL",
        "tool": "calc",
        "argmap": {"expr": "expr"}  # Maps task input to tool arg
    },
    "verified": True,
    "verifier_report": {
        "pass_rate": 1.0,
        "num_tests": 10,
        "skill_program_hash": "a3f8d2c1"
    }
}
```

### Reward Decomposition

```python
reward = (
    tool_validity(step)      # +1.0 if tool/skill action
    + reuse_bonus(step)      # +0.5 if verified skill used
    + outcome(success)       # +2.0 success, -1.0 failure
    + simplicity(steps)      # -0.05 per extra step
)
```

---

## 📚 Project Structure

```
ASG-SI/
├── asg_si_demo.py          # Main implementation (single-file prototype)
├── run_multi_seed.py       # Multi-seed experiments for statistics
├── analyze_results.py      # Statistical analysis with error bars
├── main.tex                # Research paper (LaTeX)
├── references.bib          # Bibliography
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment configuration
├── audit_logs/             # Generated verification traces (JSON)
└── multi_seed_results.json # Experimental results (generated)
```

---

## 🔬 Research Context

### Related Work

**Agentic RL Frameworks**: AgentRL, Agent Lightning provide scalable infrastructures for multi-turn RL. ASG-SI adds verification and audit layers for governable self-improvement.

**Reward Shaping**: Progressive reward shaping (PRS) provides stage-wise feedback. ASG-SI makes rewards evidence-reconstructible for auditability.

**Verifiable Rewards**: Work on RL with deterministic evaluators improves sample efficiency. ASG-SI addresses measurement gaps through third-party auditable evidence bundles.

**Skill Libraries**: Prior work accumulates skills heuristically. ASG-SI introduces verification gates preventing accumulation of incorrect skills.

**Memory & Continual Learning**: Recent work treats memory as learnable. ASG-SI incorporates bounded memory in the auditable control loop.

### Limitations

- **Simple Environment**: Current prototype uses 3 deterministic tool-call tasks. Extending to code generation, multi-step reasoning, or nondeterministic tools is future work.
- **Verification Overhead**: 33% overhead in prototype. Scaling to thousands of skills requires optimized verification strategies.
- **Interface Specification**: Manual interface design. Future work should explore automated interface inference.
- **Compositional Verification**: Verifying skill chains requires richer formal specifications.

### Future Directions

1. **Complex Environments**: Code generation (HumanEval, MBPP), web agents (WebArena), software engineering (SWE-bench)
2. **Nondeterministic Tools**: Statistical verification, record-replay mechanisms
3. **Compositional Reasoning**: Formal verification of multi-skill chains
4. **Continual Learning**: Long-term retention testing on evolving task streams
5. **Adversarial Robustness**: Testing under distribution shift and adversarial inputs

---

## 🤝 Contributing

This is a research prototype. Contributions welcome in:

- **Scaling**: Optimizing verification for large skill graphs
- **Environments**: Integrating with real-world benchmarks (SWE-bench, WebArena, etc.)
- **Verification**: Advanced verification strategies (formal methods, statistical testing)
- **Compositional Skills**: Multi-step skill chains with interface typing
- **Visualization**: Tools for inspecting skill graphs and audit trails

Please open issues for bugs or feature requests, and PRs for contributions.

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{huang2026asgsi,
  title={Audited Skill-Graph Self-Improvement for Agentic AI Systems},
  author={Huang, Ken and Huang, Jerry},
  year={2026},
  note={Implementation available at https://github.com/kenhuangus/ASG-SI}
}
```

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

This work builds on research in agentic RL, verifiable rewards, and AI safety. Thanks to the open-source community for tools enabling reproducible research: Ollama for local LLM serving, Python scientific stack, and LaTeX for documentation.

---

## 📞 Contact

- **Ken Huang**: ken.huang@owasp.org
- **Project**: https://github.com/kenhuangus/ASG-SI
- **Issues**: https://github.com/kenhuangus/ASG-SI/issues

---

**Building trustworthy AI through transparent, auditable self-improvement.**
