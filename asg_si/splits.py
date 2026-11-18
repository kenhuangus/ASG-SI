"""Separate seeded streams and overlap checks on canonical task content, not IDs."""
import random
from .oracle import reference_output
from .types import EvaluationCase, TaskObservation, digest
BOUNDARIES = ("ordinary", "mixed_units", "invalid", "negative", "empty")

def derive_seed(seed, stream):
    return int(digest({"seed": seed, "stream": stream})[:16], 16)

def make_case(inputs, boundary):
    observation = TaskObservation(digest({"kind": "volume_report", "inputs": inputs}), "volume_report", inputs)
    return EvaluationCase(observation, reference_output(inputs), boundary)

def make_splits(seed, *, promotion_size=15, evaluation_size=30, stress_size=20):
    discovery = make_case({"records": [{"value": 1, "unit": "L", "valid": True}, {"value": 1, "unit": "L", "valid": True}],
                           "destination": "report", "allowed_destination": "report"}, "ordinary")
    zero = make_case({"records": [], "destination": "report", "allowed_destination": "report"}, "empty")
    splits = {"discovery": [discovery, zero]}
    used = {c.observation.content_hash for c in splits["discovery"]}
    for split, size in (("promotion", promotion_size), ("evaluation", evaluation_size), ("stress", stress_size)):
        if size < 1:
            raise ValueError("split sizes must be positive")
        rng = random.Random(derive_seed(seed, split))
        cases = []
        while len(cases) < size:
            boundary = BOUNDARIES[len(cases) % len(BOUNDARIES)]
            scale = 1000 if split == "stress" else 1
            records = [{"value": rng.randint(1, 1000000) / 1000 * scale,
                        "unit": "mL" if rng.random() < .5 else "L", "valid": True}
                       for _ in range(rng.randint(2, 6))]
            if boundary == "mixed_units":
                records[0]["unit"], records[1]["unit"] = "mL", "L"
            elif boundary == "invalid":
                records[0]["valid"] = False
            elif boundary == "negative":
                records[0]["value"] *= -1
            elif boundary == "empty":
                for row in records:
                    row["valid"] = False
            case = make_case({"records": records, "destination": "report", "allowed_destination": "report"}, boundary)
            if case.observation.content_hash not in used:
                used.add(case.observation.content_hash)
                cases.append(case)
        splits[split] = cases
    assert_disjoint(splits)
    return splits

def assert_disjoint(splits):
    seen = {}
    for name, cases in splits.items():
        local = set()
        for case in cases:
            identity = case.observation.content_hash
            if identity in local or identity in seen:
                raise ValueError(f"canonical task overlap in {name}; previous split {seen.get(identity, name)}")
            local.add(identity)
        seen.update({key: name for key in local})

def split_manifest(splits):
    assert_disjoint(splits)
    return {name: {"count": len(cases), "content_hashes": [c.observation.content_hash for c in cases],
                   "boundaries": sorted({c.boundary for c in cases})} for name, cases in splits.items()}
