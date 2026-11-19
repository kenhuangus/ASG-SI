"""Declared contracts and empirical transition checks, not universal proofs."""
from dataclasses import asdict, dataclass

class ContractError(ValueError):
    pass

@dataclass(frozen=True)
class ToolContract:
    arguments: dict[str, str]
    output_type: str
    input_units: dict[str, str] | None = None
    output_unit: str | None = None

    def to_dict(self):
        return asdict(self)

def matches(value, name):
    return {"list": isinstance(value, list), "dict": isinstance(value, dict), "str": isinstance(value, str),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool)}.get(name, False)

def check_arguments(contract, arguments, semantic):
    if set(arguments) != set(contract.arguments):
        raise ContractError("argument names do not match declared schema")
    for name, typename in contract.arguments.items():
        if not matches(arguments[name], typename):
            raise ContractError(f"{name} must have type {typename}")
    checks = 0
    if semantic:
        for name, unit in (contract.input_units or {}).items():
            checks += 1
            if arguments[name].get("unit") != unit:
                raise ContractError(f"{name} requires {unit}, received {arguments[name].get('unit')}")
    return checks

def check_output(contract, output, semantic):
    if not matches(output, contract.output_type):
        raise ContractError("output type does not match declared schema")
    if semantic and contract.output_unit and output.get("unit") != contract.output_unit:
        raise ContractError("output unit does not match declared postcondition")

def declared_edges(skill, registry):
    """Known disagreements refute Post_A => Pre_B; replay does not prove implication."""
    steps = {step["id"]: step for step in skill.steps}
    edges = []
    for step in skill.steps:
        for argument, binding in step.get("args", {}).items():
            if not isinstance(binding, dict) or "step" not in binding:
                continue
            source = steps.get(binding["step"])
            if source is None:
                raise ContractError("transition references an unknown step")
            previous = registry.contract(source["tool"]) if "tool" in source else None
            following = registry.contract(step["tool"]) if "tool" in step else None
            expected_type = following.arguments.get(argument) if following else None
            required_unit = (following.input_units or {}).get(argument) if following else None
            compatible = not (previous and expected_type and previous.output_type != expected_type)
            if previous and previous.output_unit and required_unit:
                compatible = compatible and previous.output_unit == required_unit
            edges.append({"source": source["id"], "target": step["id"], "argument": argument,
                          "binding": binding, "declared_compatible": bool(compatible),
                          "required_unit": required_unit, "scope": "declared contracts; empirical replay required"})
    return edges
