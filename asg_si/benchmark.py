"""Frozen trajectory-compiled candidates and explicitly controlled faulty variants."""
from dataclasses import replace
from .compiler import compile_trajectory
from .executor import execute
from .oracle import assess
from .tools import ToolRegistry
from .types import Candidate, Skill, clone, digest

def workflow(name="volume_workflow", *, redundant_validation=False, redundant_read=False):
    steps = [{"id": "read", "tool": "read_records", "args": {"records": {"input": "records"}}}]
    source = "read"
    if redundant_read:
        steps.append({"id": "reread", "tool": "read_records", "args": {"records": {"step": source}}})
        source = "reread"
    steps.append({"id": "validate", "tool": "validate_fields", "args": {"records": {"step": source}}})
    source = "validate"
    if redundant_validation:
        steps.append({"id": "revalidate", "tool": "validate_fields", "args": {"records": {"step": source}}})
        source = "revalidate"
    steps.extend([
        {"id": "convert", "tool": "convert_liters", "args": {"records": {"step": source}}},
        {"id": "aggregate", "tool": "aggregate", "args": {"batch": {"step": "convert"}}},
        {"id": "write", "tool": "write_output", "args": {"report": {"step": "aggregate"}, "destination": {"input": "destination"}}},
    ])
    return Skill(name, tuple(steps), "write", {s["tool"]: "1.0.0" for s in steps})

def make_candidates(discovery_cases):
    """Generation sees discovery only; fault annotations are analysis-only."""
    sample, zero = discovery_cases
    candidates, annotations = [], {}
    def add(skill, fault, case=sample, claimed=None, provenance="controlled_mutation"):
        submission = digest({"skill": skill.to_dict(), "fault": fault, "n": len(candidates)})
        candidates.append(Candidate(submission, skill, claimed or skill.skill_id, (case,), provenance))
        annotations[submission] = fault
    for name, options in (("compiled_standard", {}), ("compiled_revalidate", {"redundant_validation": True}),
                          ("compiled_reread", {"redundant_read": True})):
        program = workflow(name, **options)
        result = execute(program, sample.observation)
        compiled = compile_trajectory(name, sample.observation, result, successful=assess(sample, result)["success"])
        if compiled is None:
            raise RuntimeError("successful discovery trajectory did not compile")
        add(compiled, "none", provenance="trajectory_compiled")
    base = workflow()
    constant = Skill("memorized_answer", ({"id": "constant", "tool": "constant_report", "args": {"value": {"literal": sample.expected["total"]}}},), "constant", {"constant_report": "1.0.0"})
    add(constant, "memorized_answer")
    steps = clone(base.steps)
    steps.insert(-2, {"id": "wrong_binding", "tool": "duplicate_first", "args": {"batch": {"step": "convert"}}})
    steps[-2]["args"]["batch"] = {"step": "wrong_binding"}
    add(replace(base, name="incorrect_binding", steps=tuple(steps), tool_versions={**base.tool_versions, "duplicate_first": "1.0.0"}), "incorrect_binding")
    steps = clone(base.steps)
    steps[2]["tool"] = "convert_milliliters"
    pins = dict(base.tool_versions)
    pins.pop("convert_liters")
    pins["convert_milliliters"] = "1.0.0"
    add(replace(base, name="unit_mismatch", steps=tuple(steps), tool_versions=pins), "unit_mismatch", zero)
    steps = clone(base.steps)
    steps[-1]["args"]["destination"] = {"literal": "protected"}
    add(replace(base, name="prohibited_write", steps=tuple(steps)), "constraint_violation")
    add(replace(base, name="modified_artifact"), "modified_artifact", claimed="0"*64)
    add(replace(base, name="stale_dependency", tool_versions={**base.tool_versions, "aggregate": "0.9.0"}), "changed_dependency")
    steps = clone(base.steps)
    steps.pop(1)
    steps[1]["args"]["records"] = {"step": "read"}
    pins = dict(base.tool_versions)
    pins.pop("validate_fields")
    add(replace(base, name="skipped_validation", steps=tuple(steps), tool_versions=pins), "skipped_validation")
    return candidates, annotations
