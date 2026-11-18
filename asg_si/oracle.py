"""Independent Decimal reference specification; never derives labels from candidate tools."""
from decimal import Decimal
import math

def reference_output(inputs):
    total = Decimal(0)
    for record in inputs["records"]:
        if not record.get("valid", True):
            continue
        value = Decimal(str(record["value"]))
        if record["unit"] == "mL":
            value /= Decimal(1000)
        elif record["unit"] != "L":
            raise ValueError("unsupported reference unit")
        total += value
    return {"total": float(total), "unit": "L"}

def output_correct(expected, actual):
    if not isinstance(actual, dict) or set(actual) != {"total", "unit"}:
        return False
    value = actual["total"]
    return (actual["unit"] == expected["unit"] and isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(value)
            and math.isclose(expected["total"], value, rel_tol=1e-10, abs_tol=1e-10))

def assess(case, result):
    outcome = result.error is None and output_correct(case.expected, result.output)
    return {"outcome_correct": outcome, "constraint_violation": bool(result.violations),
            "success": outcome and not result.violations}
