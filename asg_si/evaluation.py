"""Hidden-suite candidate adjudication and immutable-checkpoint task evaluation."""
from .executor import execute
from .oracle import assess
from .policies import ScriptedPolicy
from .rewards import DEFAULT_WEIGHTS, compute_reward
from .splits import BOUNDARIES
from .tools import ToolRegistry
from .types import ExecutionResult

def adjudicate_candidate(candidate, hidden_cases):
    registry = ToolRegistry()
    adequate = len(hidden_cases) >= len(BOUNDARIES) and not set(BOUNDARIES)-{c.boundary for c in hidden_cases}
    metadata_ok = candidate.claimed_digest == candidate.skill.skill_id
    called = {s["tool"] for s in candidate.skill.steps if "tool" in s}
    metadata_ok = metadata_ok and called == set(candidate.skill.tool_versions)
    try:
        metadata_ok = metadata_ok and all(registry.version(k) == v for k, v in candidate.skill.tool_versions.items())
    except ValueError:
        metadata_ok = False
    rows = []
    for case in hidden_cases:
        result = execute(candidate.skill, case.observation, registry, enforce_pins=False)
        rows.append({"case_hash": case.observation.content_hash, **assess(case, result), "execution": result.to_dict()})
    valid = metadata_ok and all(r["success"] for r in rows)
    return {"submission_id": candidate.submission_id, "valid": valid if adequate else None, "metadata_valid": metadata_ok,
            "label_scope": "hidden suite; not universal correctness", "num_tests": len(rows), "results": rows}

def evaluate_checkpoint(graph, cases, *, policy_seed, composition=False, weights=None):
    before = graph.state_hash
    policy, available, rows = ScriptedPolicy(policy_seed), graph.available(), []
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    for case in cases:
        skill = policy.select(case.observation, available)
        result = execute(skill, case.observation, graph.registry, composition=composition, graph=graph) if skill else ExecutionResult(error="no_admitted_skill")
        grade = assess(case, result)
        rows.append({"task_id": case.observation.task_id, "case_hash": case.observation.content_hash,
                     "observation": case.observation.to_dict(), "expected": case.expected, "boundary": case.boundary,
                     "skill_id": skill.skill_id if skill else None, "policy_mechanism": policy.mechanism, "parameters_updated": 0,
                     "execution": result.to_dict(), "grade": grade, "reward": compute_reward(result, grade, bool(skill), weights)})
    if graph.state_hash != before:
        raise RuntimeError("evaluation mutated the skill library")
    return rows
