"""
Audited Skill-Graph Self-Improvement (ASG-SI) — Minimal Working Prototype

This is a fully runnable, single-file system implementing a simplified version of:
- Tool-augmented environment with verifiable rewards
- Agent policy with skill reuse + direct solving
- Skill compiler that extracts reusable skills from successful trajectories
- Verifier-Auditor that tests skills on held-out tasks and logs evidence
- Bounded memory manager
- Iterative self-improvement loop that grows an audited skill graph

Run:
  python asg_si_demo.py

This compatibility demo does not optimize policy parameters.
Use experiments.run for paper experiments; faults are explicit simulation controls.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import time
import copy
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use defaults


# ----------------------------
# Utilities
# ----------------------------

def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ----------------------------
# Local Model Client
# ----------------------------

class ModelClient:
    """
    Client for interacting with Ollama models via HTTP API.
    """
    def __init__(self, base_url: str = None, model: str = None) -> None:
        # Read from environment variables with defaults
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = self.base_url.rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        # print(f"ModelClient initialized: base_url={self.base_url}, model={self.model}")

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generate text using the Ollama model.
        """
        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }

        request = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        generated_text = result.get("response", "").strip()
        if not generated_text:
            raise RuntimeError("empty_model_response")
        return generated_text


# ----------------------------
# Toy Tool System
# ----------------------------

@dataclasses.dataclass(frozen=True)
class ToolCall:
    name: str
    args: Dict[str, Any]


class ToolRegistry:
    """
    Simple tool registry. Tools are pure functions to keep verification easy.
    """
    def __init__(self) -> None:
        self._tools = {
            "calc": self._tool_calc,
            "reverse": self._tool_reverse,
            "concat": self._tool_concat,
        }

    def list_tools(self) -> List[str]:
        return sorted(self._tools.keys())

    def schema(self, tool_name: str) -> Dict[str, Any]:
        # Very small schema: required args and their types
        schemas = {
            "calc": {"expr": "str"},
            "reverse": {"text": "str"},
            "concat": {"a": "str", "b": "str"},
        }
        return schemas.get(tool_name, {})

    def call(self, tool_call: ToolCall) -> Any:
        if tool_call.name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_call.name}")
        return self._tools[tool_call.name](tool_call.args)

    @staticmethod
    def _tool_calc(args: Dict[str, Any]) -> Any:
        expr = args.get("expr")
        if not isinstance(expr, str):
            raise TypeError("calc expects expr:str")
        # Safe-ish eval for simple arithmetic: digits, operators, whitespace, parentheses, dot
        allowed = set("0123456789+-*/(). %")
        if any(ch not in allowed for ch in expr):
            raise ValueError("calc expr contains disallowed characters")
        # Evaluate expression
        return eval(expr, {"__builtins__": {}})

    @staticmethod
    def _tool_reverse(args: Dict[str, Any]) -> str:
        text = args.get("text")
        if not isinstance(text, str):
            raise TypeError("reverse expects text:str")
        return text[::-1]

    @staticmethod
    def _tool_concat(args: Dict[str, Any]) -> str:
        a = args.get("a")
        b = args.get("b")
        if not isinstance(a, str) or not isinstance(b, str):
            raise TypeError("concat expects a:str, b:str")
        return a + b


# ----------------------------
# Environment: Verifiable Tasks
# ----------------------------

@dataclasses.dataclass(frozen=True)
class Task:
    task_id: str
    kind: str  # "calc", "reverse", "concat"
    input_data: Dict[str, Any]
    expected: Any

    def observation(self):
        return TaskObservation(self.task_id, self.kind, copy.deepcopy(self.input_data))


@dataclasses.dataclass(frozen=True)
class TaskObservation:
    """Public fields only: no evaluator answer is provided to the policy."""
    task_id: str
    kind: str
    input_data: Dict[str, Any]


class VerifiableEnv:
    """
    Generates tasks with ground-truth expected outputs. The agent may use tools.
    Verification is exact-match to keep it deterministic and verifiable.
    """
    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)

    def sample_task(self) -> Task:
        kind = self.rng.choice(["calc", "reverse", "concat"])
        if kind == "calc":
            a = self.rng.randint(1, 30)
            b = self.rng.randint(1, 30)
            op = self.rng.choice(["+", "-", "*"])
            expr = f"{a} {op} {b}"
            expected = eval(expr, {"__builtins__": {}})
            inp = {"expr": expr}
        elif kind == "reverse":
            length = self.rng.randint(4, 10)
            text = "".join(self.rng.choice("abcdefgxyz") for _ in range(length))
            expected = text[::-1]
            inp = {"text": text}
        else:
            a = "".join(self.rng.choice("abcxyz") for _ in range(self.rng.randint(2, 5)))
            b = "".join(self.rng.choice("defuvw") for _ in range(self.rng.randint(2, 5)))
            expected = a + b
            inp = {"a": a, "b": b}

        tid = stable_hash({"kind": kind, "input": inp, "expected": expected, "nonce": self.rng.random()})
        return Task(task_id=tid, kind=kind, input_data=inp, expected=expected)

    @staticmethod
    def verify(task: Task, answer: Any) -> Tuple[bool, str]:
        ok = answer == task.expected
        return ok, ("pass" if ok else f"fail(expected={task.expected!r}, got={answer!r})")


# ----------------------------
# Memory Manager (bounded)
# ----------------------------

class BoundedMemory:
    """
    Keeps a bounded list of compact notes. In a real system, these would be learned operations.
    Here we implement deterministic policy: keep only recent distinct notes.
    """
    def __init__(self, capacity: int = 8) -> None:
        self.capacity = capacity
        self.notes: List[str] = []

    def add(self, note: str) -> None:
        note = note.strip()
        if not note:
            return
        if note in self.notes:
            self.notes.remove(note)
        self.notes.append(note)
        while len(self.notes) > self.capacity:
            self.notes.pop(0)

    def summarize(self) -> str:
        if not self.notes:
            return ""
        return " | ".join(self.notes[-min(4, len(self.notes)):])


# ----------------------------
# Trajectory and Audit Structures
# ----------------------------

@dataclasses.dataclass
class Step:
    t: int
    action_type: str  # "TOOL" or "SKILL" or "DIRECT"
    payload: Dict[str, Any]
    output: Any


@dataclasses.dataclass
class Trajectory:
    task: Task
    steps: List[Step]
    final_answer: Any
    success: bool
    reward: float
    evidence: Dict[str, Any]


# ----------------------------
# Skill Graph
# ----------------------------

@dataclasses.dataclass
class Skill:
    skill_id: str
    name: str
    kind: str
    interface: Dict[str, Any]         # input keys + output type
    program: Dict[str, Any]           # canonical program representation
    created_at: str
    verified: bool = False
    verifier_report: Optional[Dict[str, Any]] = None


class SkillGraph:
    def __init__(self) -> None:
        self.skills: Dict[str, Skill] = {}
        # Index by kind for fast lookup
        self.by_kind: Dict[str, List[str]] = {"calc": [], "reverse": [], "concat": []}

    def add_skill(self, skill: Skill) -> bool:
        if skill.skill_id in self.skills:
            return False
        self.skills[skill.skill_id] = skill
        self.by_kind.setdefault(skill.kind, []).append(skill.skill_id)
        return True

    def get_verified_skills(self, kind: str) -> List[Skill]:
        ids = self.by_kind.get(kind, [])
        return [self.skills[sid] for sid in ids if self.skills[sid].verified]

    def size(self) -> int:
        return len(self.skills)


# ----------------------------
# Verifier-Auditor
# ----------------------------

class VerifierAuditor:
    """
    Verifies skill programs against held-out tests and produces auditable traces.
    """
    def __init__(self, tool_registry: ToolRegistry, audit_dir: str = "audit_logs") -> None:
        self.tools = tool_registry
        self.audit_dir = audit_dir
        os.makedirs(self.audit_dir, exist_ok=True)

    def verify_skill(self, skill: Skill, heldout_tasks: List[Task]) -> Tuple[bool, Dict[str, Any]]:
        # Verify that skill program solves tasks of matching kind for multiple inputs.
        # For this toy prototype, a skill program is a single tool invocation template.
        kind = skill.kind
        relevant = [t for t in heldout_tasks if t.kind == kind]
        # Require at least a few tests; if not available, pass with caution.
        tests = relevant[: max(3, min(10, len(relevant)))]
        results = []

        for t in tests:
            try:
                out = self.execute_skill(skill, t.input_data)
                ok, msg = VerifiableEnv.verify(t, out)
                results.append({"task_id": t.task_id, "ok": ok, "msg": msg})
            except Exception as e:
                results.append({"task_id": t.task_id, "ok": False, "msg": f"exception:{type(e).__name__}:{e}"})

        pass_rate = 0.0 if not results else sum(1 for r in results if r["ok"]) / len(results)
        verified = len(results) >= 3 and pass_rate >= 1.0
        report = {
            "status": "ACCEPT" if verified else ("INSUFFICIENT_EVIDENCE" if len(results) < 3 else "REJECT"),
            "verified": verified,
            "kind": kind,
            "num_tests": len(results),
            "pass_rate": pass_rate,
            "results": results,
            "checked_at": now_iso(),
            "skill_program_hash": stable_hash(skill.program),
        }
        return verified, report

    def execute_skill(self, skill: Skill, inputs: Dict[str, Any]) -> Any:
        prog = skill.program
        if prog.get("type") != "TOOL_CALL":
            raise ValueError("Unknown program type")
        tool_name = prog["tool"]
        # Map tool args from inputs based on argument map in program
        argmap = prog["argmap"]
        call_args: Dict[str, Any] = {}
        for tool_arg, input_key in argmap.items():
            call_args[tool_arg] = inputs[input_key]
        return self.tools.call(ToolCall(name=tool_name, args=call_args))

    def log_audit_trace(self, trace: Dict[str, Any]) -> str:
        trace_id = stable_hash(trace)
        path = os.path.join(self.audit_dir, f"audit_{trace_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
        return path


# ----------------------------
# Skill Compiler
# ----------------------------

class SkillCompiler:
    """
    Mines successful trajectories into candidate skills. For the demo, we compile
    a skill when a trajectory used a single tool call to solve the task correctly,
    producing a canonical skill program that can be reused for that task kind.
    """
    def __init__(self) -> None:
        self.counter = 0

    def compile_from_trajectory(self, traj: Trajectory) -> Optional[Skill]:
        if not traj.success:
            return None
        # Find if any step is a tool call matching task kind; in this toy env, it will be.
        tool_steps = [s for s in traj.steps if s.action_type == "TOOL"]
        if len(tool_steps) != 1:
            return None

        s = tool_steps[0]
        tool_name = s.payload.get("tool_name")
        tool_args = s.payload.get("tool_args", {})
        # We compile a generalizable template: map tool args to task input keys.
        # Since our env kinds map 1:1 to tools, this produces a reusable skill.
        # Determine argmap by matching values to input_data keys where possible.
        argmap: Dict[str, str] = {}
        for k, v in tool_args.items():
            if k in traj.task.input_data and traj.task.input_data[k] == v:
                argmap[k] = k
                continue
            matches = [key for key, value in traj.task.input_data.items() if value == v]
            if len(matches) != 1:
                return None
            argmap[k] = matches[0]

        self.counter += 1
        program = {
            "type": "TOOL_CALL",
            "tool": tool_name,
            "argmap": argmap,
        }
        interface = {
            "inputs": sorted(list(traj.task.input_data.keys())),
            "output_type": type(traj.final_answer).__name__,
        }
        skill_id = stable_hash({"kind": traj.task.kind, "program": program})
        return Skill(
            skill_id=skill_id,
            name=f"{traj.task.kind}_tool_skill_{self.counter}",
            kind=traj.task.kind,
            interface=interface,
            program=program,
            created_at=now_iso(),
        )


# ----------------------------
# Agent Policy
# ----------------------------

class AgentPolicy:
    """
    Simple stochastic policy:
    - If there is a verified skill for the task kind, try to use it with probability p_use_skill
    - Otherwise call the tool directly with probability p_use_tool
    - Otherwise try a direct (sometimes wrong) heuristic to illustrate improvement
    """
    def __init__(
        self,
        tool_registry: ToolRegistry,
        rng_seed: int = 11,
        p_use_skill: float = 0.75,
        p_use_tool: float = 0.80,
        p_error_direct: float = 0.0,
        fault_seed: int = 101,
    ) -> None:
        self.tools = tool_registry
        self.rng = random.Random(rng_seed)
        self.fault_rng = random.Random(fault_seed)
        self.p_use_skill = p_use_skill
        self.p_use_tool = p_use_tool
        self.p_error_direct = p_error_direct

    def act(
        self,
        task: Task,
        skill_graph: SkillGraph,
        verifier: VerifierAuditor,
        memory: BoundedMemory,
    ) -> Tuple[List[Step], Any]:
        steps: List[Step] = []
        t = 0

        mem_summary = memory.summarize()
        if mem_summary:
            memory_hint = f"mem:{mem_summary}"
        else:
            memory_hint = "mem:empty"

        # Attempt skill reuse
        verified_skills = skill_graph.get_verified_skills(task.kind)
        if verified_skills and self.rng.random() < self.p_use_skill:
            skill = self.rng.choice(verified_skills)
            out = verifier.execute_skill(skill, task.input_data)
            steps.append(Step(t=t, action_type="SKILL", payload={"skill_id": skill.skill_id, "hint": memory_hint}, output=out))
            return steps, out

        # Attempt direct tool call
        if self.rng.random() < self.p_use_tool:
            tool_call = self._tool_call_for_task(task)
            out = self.tools.call(tool_call)
            steps.append(Step(t=t, action_type="TOOL", payload={"tool_name": tool_call.name, "tool_args": tool_call.args, "hint": memory_hint}, output=out))
            return steps, out

        # Otherwise direct solve (sometimes wrong to demonstrate why skill learning matters)
        out = self._direct_solve(task)
        steps.append(Step(t=t, action_type="DIRECT", payload={"method": "heuristic", "hint": memory_hint}, output=out))
        return steps, out

    def _tool_call_for_task(self, task: Task) -> ToolCall:
        if task.kind == "calc":
            return ToolCall(name="calc", args={"expr": task.input_data["expr"]})
        if task.kind == "reverse":
            return ToolCall(name="reverse", args={"text": task.input_data["text"]})
        if task.kind == "concat":
            return ToolCall(name="concat", args={"a": task.input_data["a"], "b": task.input_data["b"]})
        raise ValueError("Unknown task kind")

    def _direct_solve(self, task: Task) -> Any:
        # Intentionally imperfect.
        if task.kind == "reverse":
            text = task.input_data["text"]
            if self.fault_rng.random() < self.p_error_direct:
                return text  # wrong
            return text[::-1]
        if task.kind == "concat":
            a, b = task.input_data["a"], task.input_data["b"]
            if self.fault_rng.random() < self.p_error_direct:
                return b + a  # wrong
            return a + b
        if task.kind == "calc":
            expr = task.input_data["expr"]
            if self.fault_rng.random() < self.p_error_direct:
                # wrong on purpose: drop spaces and mis-parse occasionally
                return 0
            return eval(expr, {"__builtins__": {}})
        raise ValueError("Unknown task kind")


class ModelBasedAgentPolicy:
    """Frozen Ollama actions; errors are explicit, without injected faults or fallback."""
    def __init__(self, tool_registry, model_client):
        self.tools, self.model = tool_registry, model_client
        self.parameters_updated = 0

    def act(self, task, skill_graph, verifier, memory):
        skills = skill_graph.get_verified_skills(task.kind)
        prompt = json.dumps({"instruction": "Return JSON: {type: TOOL, tool: name, args: object}, "
                              "{type: SKILL, skill_id: id}, or {type: DIRECT, answer: value}.",
                             "task": {"kind": task.kind, "inputs": task.input_data},
                             "tools": {name: self.tools.schema(name) for name in self.tools.list_tools()},
                             "skills": [{"skill_id": skill.skill_id, "name": skill.name} for skill in skills],
                             "memory": memory.summarize()})
        raw, phase = None, "provider_error"
        try:
            raw = self.model.generate(prompt, max_tokens=300, temperature=0.0)
            phase = "parse_error"
            action = json.loads(raw)
            if not isinstance(action, dict):
                raise ValueError("model action must be an object")
            phase = "execution_error"
            if action.get("type") == "TOOL":
                call = ToolCall(action["tool"], action["args"])
                answer = self.tools.call(call)
                payload = {"tool_name": call.name, "tool_args": call.args}
            elif action.get("type") == "SKILL":
                selected = next((skill for skill in skills if skill.skill_id == action["skill_id"]), None)
                if selected is None:
                    raise ValueError("model selected an unavailable skill")
                answer = verifier.execute_skill(selected, task.input_data)
                payload = {"skill_id": selected.skill_id}
            elif action.get("type") == "DIRECT":
                answer, payload = action["answer"], {"method": "frozen_model_direct"}
            else:
                raise ValueError("unsupported action type")
            payload.update(model_response=raw, backend=self.model.base_url, model=self.model.model, fallback=False)
            return [Step(0, action["type"], payload, answer)], answer
        except Exception as error:
            payload = {"error_type": phase, "error": f"{type(error).__name__}:{error}", "model_response": raw,
                       "backend": self.model.base_url, "model": self.model.model, "fallback": False}
            return [Step(0, "ERROR", payload, None)], None


# ----------------------------
# Reward Shaping (Verifier-backed)
# ----------------------------

def compute_shaped_reward(task: Task, steps: List[Step], final_answer: Any, success: bool) -> Tuple[float, Dict[str, Any]]:
    """
    Progressive shaped reward components:
    - tool_validity: did tool/skill call exist and appear well-formed
    - outcome: exact verification against ground truth
    - reuse_bonus: if skill reused
    - simplicity: small penalty for too many steps (here always 1)
    """
    tool_validity = 0.0
    reuse_bonus = 0.0
    if steps:
        if steps[0].action_type in ("TOOL", "SKILL"):
            tool_validity = 1.0
        if steps[0].action_type == "SKILL":
            reuse_bonus = 0.5

    outcome = 2.0 if success else -1.0
    simplicity = -0.05 * max(0, len(steps) - 1)

    reward = tool_validity + reuse_bonus + outcome + simplicity
    evidence = {
        "tool_validity": tool_validity,
        "reuse_bonus": reuse_bonus,
        "outcome_component": outcome,
        "simplicity_component": simplicity,
        "reward_total": reward,
    }
    return reward, evidence


# ----------------------------
# Trainer / Self-Improvement Loop
# ----------------------------

class ASGSISystem:
    def __init__(
        self,
        seed: int = 7,
        audit_dir: str = "audit_logs",
        memory_capacity: int = 8,
        use_model: bool = False,
        task_seed: Optional[int] = None,
        policy_seed: Optional[int] = None,
        verifier_seed: Optional[int] = None,
        fault_seed: Optional[int] = None,
        fault_error_rate: float = 0.0,
    ) -> None:
        from asg_si.splits import derive_seed
        self.seeds = {"task": task_seed if task_seed is not None else derive_seed(seed, "task"),
                      "policy": policy_seed if policy_seed is not None else derive_seed(seed, "policy"),
                      "verifier": verifier_seed if verifier_seed is not None else derive_seed(seed, "verifier"),
                      "fault": fault_seed if fault_seed is not None else derive_seed(seed, "fault")}
        self.env = VerifiableEnv(self.seeds["task"])
        self.promotion_env = VerifiableEnv(self.seeds["verifier"])
        self.tools = ToolRegistry()
        self.skill_graph = SkillGraph()
        self.memory = BoundedMemory(capacity=memory_capacity)
        self.auditor = VerifierAuditor(self.tools, audit_dir=audit_dir)
        self.compiler = SkillCompiler()

        if use_model:
            # Use None to let ModelClient read from environment variables
            print("Initializing model client...")
            model_client = ModelClient()
            self.agent = ModelBasedAgentPolicy(self.tools, model_client)
            print("Model-based agent initialized")
        else:
            self.agent = AgentPolicy(self.tools, rng_seed=self.seeds["policy"], p_error_direct=fault_error_rate,
                                     fault_seed=self.seeds["fault"])
        self.metadata = {"mechanism": "frozen_ollama" if use_model else "scripted_simulation", "parameters_updated": 0,
                         "seeds": self.seeds, "fault_error_rate": 0.0 if use_model else fault_error_rate}

        self.rng = random.Random(seed + 999)

    def run_iteration(
        self,
        iteration: int,
        num_tasks: int = 60,
        heldout_size: int = 30,
    ) -> Dict[str, Any]:
        # Prepare held-out tasks for verification
        heldout_tasks = [self.promotion_env.sample_task() for _ in range(heldout_size)]

        trajectories: List[Trajectory] = []
        for _ in range(num_tasks):
            task = self.env.sample_task()
            steps, answer = self.agent.act(task.observation(), self.skill_graph, self.auditor, self.memory)
            success, verify_msg = self.env.verify(task, answer)
            reward, evidence = compute_shaped_reward(task, steps, answer, success)

            # Memory update: store compact, verifiable note
            note = f"{task.kind}:{stable_hash(task.input_data)}:{'ok' if success else 'bad'}"
            self.memory.add(note)

            traj = Trajectory(
                task=task,
                steps=steps,
                final_answer=answer,
                success=success,
                reward=reward,
                evidence={"verify": verify_msg, **evidence},
            )
            trajectories.append(traj)

        # Compile skills from successful trajectories
        compiled = 0
        verified_added = 0
        for traj in trajectories:
            candidate = self.compiler.compile_from_trajectory(traj)
            if candidate is None:
                continue
            compiled += 1

            verified, report = self.auditor.verify_skill(candidate, heldout_tasks)
            candidate.verified = verified
            candidate.verifier_report = report

            # Log audit trace regardless of pass/fail
            trace = {
                "ts": now_iso(),
                "iteration": iteration,
                "event": "skill_candidate_verification",
                "task_kind": candidate.kind,
                "skill_id": candidate.skill_id,
                "skill_name": candidate.name,
                "interface": candidate.interface,
                "program": candidate.program,
                "verification_report": report,
            }
            audit_path = self.auditor.log_audit_trace(trace)

            if verified:
                verified_added += int(self.skill_graph.add_skill(candidate))

        # Metrics
        success_rate = sum(1 for tr in trajectories if tr.success) / len(trajectories)
        used_skill = sum(1 for tr in trajectories if tr.steps and tr.steps[0].action_type == "SKILL")
        used_tool = sum(1 for tr in trajectories if tr.steps and tr.steps[0].action_type == "TOOL")
        used_direct = sum(1 for tr in trajectories if tr.steps and tr.steps[0].action_type == "DIRECT")
        avg_reward = sum(tr.reward for tr in trajectories) / len(trajectories)

        summary = {
            "iteration": iteration,
            "num_tasks": num_tasks,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "actions": {"skill": used_skill, "tool": used_tool, "direct": used_direct},
            "compiled_candidates": compiled,
            "verified_skills_added": verified_added,
            "skill_graph_size": self.skill_graph.size(),
            "memory_summary": self.memory.summarize(),
            "audit_dir": self.auditor.audit_dir,
            "policy_metadata": self.metadata,
            "verification_events": compiled,
        }
        return summary

    def train(self, iters: int = 5, num_tasks: int = 60) -> List[Dict[str, Any]]:
        all_summaries: List[Dict[str, Any]] = []
        for i in range(iters):
            s = self.run_iteration(iteration=i, num_tasks=num_tasks, heldout_size=30)
            all_summaries.append(s)
        return all_summaries

    def evaluate(self, tasks, policy_seed=0):
        """Copy mutable state for frozen-checkpoint evaluation."""
        agent, graph, memory = copy.deepcopy(self.agent), copy.deepcopy(self.skill_graph), copy.deepcopy(self.memory)
        if isinstance(agent, AgentPolicy):
            agent.rng = random.Random(policy_seed)
        records = []
        for task in tasks:
            steps, answer = agent.act(task.observation(), graph, self.auditor, memory)
            success, message = self.env.verify(task, answer)
            reward, evidence = compute_shaped_reward(task, steps, answer, success)
            records.append({"task_id": task.task_id, "success": success, "reward": reward,
                            "steps": [dataclasses.asdict(step) for step in steps], "evidence": evidence})
        n = len(records)
        return {"num_tasks": n, "success_rate": sum(r["success"] for r in records)/n if n else None,
                "avg_reward": sum(r["reward"] for r in records)/n if n else None,
                "actions": {name.lower(): sum(r["steps"][0]["action_type"] == name for r in records)
                            for name in ("SKILL", "TOOL", "DIRECT", "ERROR")}, "records": records}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compatibility simulation; no policy-gradient updates")
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--num-tasks", type=int, default=80)
    parser.add_argument("--audit-dir", default="audit_logs")
    parser.add_argument("--fault-error-rate", type=float, default=0.0)
    args = parser.parse_args()
    if args.iters < 1 or args.num_tasks < 1 or not 0 <= args.fault_error_rate <= 1:
        parser.error("use positive task/iteration counts and a fault rate in [0, 1]")
    system = ASGSISystem(seed=args.seed, audit_dir=args.audit_dir, use_model=args.model, fault_error_rate=args.fault_error_rate)
    print(json.dumps({"metadata": system.metadata, "collection_summaries": system.train(args.iters, args.num_tasks)}, indent=2))

if __name__ == "__main__":
    main()
