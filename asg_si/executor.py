"""Bounded interpreter: task inputs, previous outputs, constants, and pinned skills."""
from .contracts import ContractError, declared_edges
from .tools import ToolRegistry
from .types import ExecutionResult, clone

def _resolve(binding, inputs, outputs):
    if not isinstance(binding, dict) or len(binding) != 1:
        raise ContractError("binding must have exactly one of input, step, literal")
    kind, value = next(iter(binding.items()))
    if kind == "input":
        return clone(inputs[value])
    if kind == "step":
        return clone(outputs[value])
    if kind == "literal":
        return clone(value)
    raise ContractError(f"unknown binding: {kind}")

def execute(skill, observation, registry=None, *, composition=False, enforce_pins=True, graph=None, max_steps=32):
    registry = registry or ToolRegistry()
    result = ExecutionResult()
    budget = [max_steps]

    def run(program, inputs, ancestry):
        if program.skill_id in ancestry:
            raise ContractError("cyclic skill dependency")
        ancestry = (*ancestry, program.skill_id)
        if set(program.input_keys) - inputs.keys():
            raise ContractError("missing declared skill inputs")
        tools = {s["tool"] for s in program.steps if "tool" in s}
        dependencies = {s["skill"] for s in program.steps if "skill" in s}
        if enforce_pins:
            if set(program.tool_versions) != tools or set(program.dependencies) != dependencies:
                raise ContractError("dependencies must be pinned exactly")
            for tool, version in program.tool_versions.items():
                if registry.version(tool) != version:
                    raise ContractError(f"dependency version changed: {tool}")
        if composition and any(not e["declared_compatible"] for e in declared_edges(program, registry)):
            raise ContractError("Post_A does not imply Pre_B under declared contracts")
        outputs = {}
        for step in program.steps:
            budget[0] -= 1
            if budget[0] < 0:
                raise ContractError("step budget exceeded")
            step_id = step["id"]
            if step_id in outputs:
                raise ContractError("duplicate step id")
            if ("tool" in step) == ("skill" in step):
                raise ContractError("step must select exactly one tool or skill")
            arguments = {k: _resolve(v, inputs, outputs) for k, v in step["args"].items()}
            event = {"step_id": step_id, "program_skill_id": program.skill_id, "call_depth": len(ancestry)-1,
                     "bindings": clone(step["args"]), "arguments": arguments}
            if "tool" in step:
                event.update(tool=step["tool"], version=registry.version(step["tool"]))
                result.tool_calls += 1
                try:
                    output, violations, checks = registry.call(step["tool"], arguments, observation, composition)
                    result.violations.extend(violations)
                    result.composition_checks += checks
                    event.update(output=clone(output), violations=violations, composition_checks=checks)
                except (ValueError, TypeError, KeyError, OverflowError) as error:
                    event["error"] = f"{type(error).__name__}:{error}"
                    result.events.append(event)
                    raise
            else:
                if graph is None:
                    raise ContractError("skill call requires an admitted dependency graph")
                dependency = graph.get(step["skill"])
                if dependency is None or dependency.skill_id != program.dependencies.get(step["skill"]):
                    raise ContractError("missing, revoked, or changed skill dependency")
                output = run(dependency, arguments, ancestry)
                event.update(skill=dependency.skill_id, output=clone(output))
            outputs[step_id] = output
            result.events.append(event)
        return clone(outputs[program.output])
    try:
        result.output = run(skill, clone(observation.inputs), ())
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        result.error = f"{type(error).__name__}:{error}"
    return result
