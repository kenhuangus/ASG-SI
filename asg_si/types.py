"""Serializable programs and public observations, separated from evaluator labels."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any

def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

def digest(value: Any) -> str:
    return sha256(canonical_json(value).encode()).hexdigest()

def clone(value: Any) -> Any:
    return json.loads(canonical_json(value))

@dataclass(frozen=True)
class TaskObservation:
    task_id: str
    kind: str
    inputs: dict[str, Any]

    @property
    def content_hash(self) -> str:
        return digest({"kind": self.kind, "inputs": self.inputs})

    def to_dict(self):
        return asdict(self)

@dataclass(frozen=True)
class EvaluationCase:
    observation: TaskObservation
    expected: Any
    boundary: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(TaskObservation(**value["observation"]), value["expected"], value["boundary"])

@dataclass(frozen=True)
class Skill:
    name: str
    steps: tuple[dict, ...]
    output: str
    tool_versions: dict[str, str]
    input_keys: tuple[str, ...] = ("records", "destination", "allowed_destination")
    dependencies: dict[str, str] = field(default_factory=dict)
    kind: str = "volume_report"

    @property
    def skill_id(self) -> str:
        value = self.to_dict()
        value.pop("name")
        value["input_keys"] = sorted(value["input_keys"])
        return digest(value)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = clone(value)
        value["steps"] = tuple(value["steps"])
        value["input_keys"] = tuple(value.get("input_keys", ()))
        return cls(**value)

@dataclass(frozen=True)
class Candidate:
    submission_id: str
    skill: Skill
    claimed_digest: str
    discovery: tuple[EvaluationCase, ...]
    provenance: str

    def to_dict(self):
        return {"submission_id": self.submission_id, "skill": self.skill.to_dict(),
                "claimed_digest": self.claimed_digest, "discovery": [c.to_dict() for c in self.discovery],
                "provenance": self.provenance}

    @classmethod
    def from_dict(cls, value):
        return cls(value["submission_id"], Skill.from_dict(value["skill"]), value["claimed_digest"],
                   tuple(EvaluationCase.from_dict(c) for c in value["discovery"]), value["provenance"])

@dataclass
class ExecutionResult:
    output: Any = None
    events: list[dict] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    error: str | None = None
    tool_calls: int = 0
    composition_checks: int = 0

    def to_dict(self):
        return asdict(self)

ACCEPT = "ACCEPT"
REJECT = "REJECT"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
