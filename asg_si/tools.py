"""Pure sensor-workflow tools. Writes are simulated; violations are recorded."""
from math import isfinite
from .contracts import ContractError, ToolContract, check_arguments, check_output
from .types import clone

class ToolRegistry:
    VERSION = "volume-tools-1"

    def __init__(self):
        self.contracts = {
            "read_records": ToolContract({"records": "list"}, "list"),
            "validate_fields": ToolContract({"records": "list"}, "list"),
            "convert_liters": ToolContract({"records": "list"}, "dict", output_unit="L"),
            "convert_milliliters": ToolContract({"records": "list"}, "dict", output_unit="mL"),
            "duplicate_first": ToolContract({"batch": "dict"}, "dict", {"batch": "L"}, "L"),
            "aggregate": ToolContract({"batch": "dict"}, "dict", {"batch": "L"}, "L"),
            "write_output": ToolContract({"report": "dict", "destination": "str"}, "dict", {"report": "L"}, "L"),
            "constant_report": ToolContract({"value": "number"}, "dict", output_unit="L"),
            # Plausible shortcut: sums raw values as given, ignoring units and validity flags.
            "sum_values": ToolContract({"records": "list"}, "dict", output_unit="L"),
        }

    def contract(self, name):
        if name not in self.contracts:
            raise ContractError(f"unknown tool: {name}")
        return self.contracts[name]

    def version(self, name):
        self.contract(name)
        return "1.0.0"

    def snapshot(self):
        return {name: {"version": self.version(name), "contract": c.to_dict()}
                for name, c in sorted(self.contracts.items())}

    def call(self, name, arguments, observation, semantic):
        contract = self.contract(name)
        checks = check_arguments(contract, arguments, semantic)
        violations = []
        if name == "read_records":
            output = clone(arguments["records"])
        elif name == "validate_fields":
            output = []
            for row in arguments["records"]:
                if not isinstance(row, dict):
                    raise ContractError("record must be an object")
                if not row.get("valid", True):
                    continue
                value = row.get("value")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                    raise ContractError("record value must be finite and numeric")
                if row.get("unit") not in ("L", "mL"):
                    raise ContractError("unknown record unit")
                output.append(clone(row))
        elif name in ("convert_liters", "convert_milliliters"):
            values = [r["value"] * (1 if r["unit"] == "L" else .001) for r in arguments["records"]]
            if name == "convert_milliliters":
                values = [v * 1000 for v in values]
            output = {"values": values, "unit": "L" if name == "convert_liters" else "mL"}
        elif name == "duplicate_first":
            batch = arguments["batch"]
            values = batch["values"]
            output = {"values": [values[0]] * len(values) if values else [], "unit": batch["unit"]}
        elif name == "aggregate":
            output = {"total": sum(arguments["batch"]["values"]), "unit": "L"}
        elif name == "sum_values":
            values = [r.get("value", 0) for r in arguments["records"] if isinstance(r, dict)]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
                raise ContractError("record value must be numeric")
            output = {"total": sum(values), "unit": "L"}
        elif name == "constant_report":
            output = {"total": arguments["value"], "unit": "L"}
        elif name == "write_output":
            if arguments["destination"] != observation.inputs["allowed_destination"]:
                violations.append("prohibited_destination")
            output = clone(arguments["report"])
        else:
            raise ContractError(f"unimplemented tool: {name}")
        check_output(contract, output, semantic)
        return output, violations, checks
