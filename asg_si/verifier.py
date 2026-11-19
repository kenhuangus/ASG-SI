"""Three-way admission with coverage requirements and a separate replay worker."""
from dataclasses import asdict, dataclass
import json
import subprocess
import sys
from time import perf_counter
from .executor import execute
from .oracle import assess
from .rewards import compute_reward
from .splits import BOUNDARIES
from .tools import ToolRegistry
from .types import ACCEPT, REJECT, INSUFFICIENT_EVIDENCE, digest

METHODS = ("first_success", "self_verification", "independent_replay", "independent_composition")
VERIFIER_VERSION = "replay-gate-1"

@dataclass(frozen=True)
class GateConfig:
    method: str = "independent_replay"
    min_cases: int = 5
    max_cases: int = 15
    required_boundaries: tuple[str, ...] = BOUNDARIES
    worker_timeout_seconds: float = 15.0

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError("unknown admission method")
        if self.min_cases < 1 or self.max_cases < self.min_cases or self.worker_timeout_seconds <= 0:
            raise ValueError("invalid test budget or worker timeout")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["required_boundaries"] = tuple(value.get("required_boundaries", BOUNDARIES))
        return cls(**value)

def verify_in_process(candidate, cases, config):
    registry = ToolRegistry()
    first = config.method == "first_success"
    composition = config.method == "independent_composition"
    selected = list(candidate.discovery[:1]) if first else [c for c in cases if c.observation.kind == candidate.skill.kind][:config.max_cases]
    coverage = sorted({c.boundary for c in selected})
    decision = {"status": INSUFFICIENT_EVIDENCE, "reasons": [], "method": config.method,
                "verifier_version": VERIFIER_VERSION, "config_hash": digest(config.to_dict()),
                "skill_id": candidate.skill.skill_id, "submission_id": candidate.submission_id,
                "composition_checked": composition, "coverage": coverage, "num_tests": 0,
                "unique_tests": len({c.observation.content_hash for c in selected}), "tool_calls": 0, "results": []}
    if not first:
        if candidate.claimed_digest != candidate.skill.skill_id:
            decision.update(status=REJECT, reasons=["artifact_digest_mismatch"])
            return decision
        called = {s["tool"] for s in candidate.skill.steps if "tool" in s}
        if called != set(candidate.skill.tool_versions):
            decision.update(status=REJECT, reasons=["incomplete_dependency_pins"])
            return decision
        try:
            changed = any(registry.version(k) != v for k, v in candidate.skill.tool_versions.items())
        except ValueError:
            changed = True
        if changed:
            decision.update(status=REJECT, reasons=["dependency_version_mismatch"])
            return decision
        if decision["unique_tests"] != len(selected):
            decision["reasons"] = ["duplicate_test_content"]
            return decision
        if len(selected) < config.min_cases or set(config.required_boundaries) - set(coverage):
            decision["reasons"] = ["insufficient_test_count_or_boundary_coverage"]
            return decision
    if not selected:
        decision["reasons"] = ["no_relevant_tests"]
        return decision
    for case in selected:
        result = execute(candidate.skill, case.observation, registry, composition=composition, enforce_pins=not first)
        grade = assess(case, result)
        decision["results"].append({"case_hash": case.observation.content_hash,
                                    "passed": grade["outcome_correct"] if first else grade["success"],
                                    "execution": result.to_dict(), "grade": grade, "reward": compute_reward(result, grade, False)})
        decision["tool_calls"] += result.tool_calls
    decision["num_tests"] = len(decision["results"])
    decision["status"] = ACCEPT if all(r["passed"] for r in decision["results"]) else REJECT
    decision["reasons"] = ["all_required_checks_passed" if decision["status"] == ACCEPT else "replay_check_failed"]
    return decision

def verify(candidate, cases, config):
    start = perf_counter()
    if config.method in ("first_success", "self_verification"):
        result = verify_in_process(candidate, cases, config)
        worker = "in_process"
    else:
        # Process separation is not an OS-enforced adversarial sandbox.
        payload = {"candidate": candidate.to_dict(), "cases": [c.to_dict() for c in cases], "config": config.to_dict()}
        try:
            completed = subprocess.run([sys.executable, "-m", "asg_si.verifier_worker"], input=json.dumps(payload, allow_nan=False),
                                       capture_output=True, text=True, timeout=config.worker_timeout_seconds, check=True)
            result = json.loads(completed.stdout)
            if result.get("status") not in (ACCEPT, REJECT, INSUFFICIENT_EVIDENCE):
                raise ValueError("invalid worker response")
        except (subprocess.SubprocessError, ValueError) as error:
            result = {"status": INSUFFICIENT_EVIDENCE, "reasons": ["worker_failure"], "error": type(error).__name__,
                      "method": config.method, "submission_id": candidate.submission_id, "skill_id": candidate.skill.skill_id,
                      "num_tests": 0, "tool_calls": 0, "results": [], "composition_checked": False}
        worker = "separate_python_process"
    return {**result, "worker": worker, "verification_ms": (perf_counter()-start)*1000}
