"""Fixed evaluation metric reconstructed from execution events and independent grading."""
from .types import digest
DEFAULT_WEIGHTS = {"outcome": 1.0, "tool_validity": 0.0, "reuse": 0.0, "composition": 0.0, "extra_steps": 0.0}

def compute_reward(result, grade, reused, weights=None):
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError("all reward weights must be explicit and fixed across checkpoints")
    components = {"outcome": float(grade["success"]),
                  "tool_validity": float(bool(result.events) and result.error is None), "reuse": float(reused),
                  "composition": float(result.composition_checks > 0 and result.error is None),
                  "extra_steps": float(max(0, result.tool_calls-1))}
    return {"components": components, "weights": weights, "metric_hash": digest(weights),
            "total": sum(weights[k] * v for k, v in components.items())}
