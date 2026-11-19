"""Extract explicit parameter bindings from successful tool trajectories."""
from .types import Skill, clone

def compile_trajectory(name, observation, result, *, successful):
    if not successful or result.error or not result.events:
        return None
    if any("tool" not in e or "bindings" not in e for e in result.events):
        return None
    steps = tuple({"id": e["step_id"], "tool": e["tool"], "args": clone(e["bindings"])} for e in result.events)
    return Skill(name, steps, steps[-1]["id"], {e["tool"]: e["version"] for e in result.events},
                 tuple(sorted(observation.inputs)))
