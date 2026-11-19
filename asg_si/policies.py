"""Frozen selection mechanisms; policies accept observations, never evaluator cases."""
import json
import random
from .types import TaskObservation

class PolicyError(RuntimeError):
    pass

class ScriptedPolicy:
    mechanism = "scripted_uniform_library_selection"
    parameters_updated = 0

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def select(self, observation, available):
        if not isinstance(observation, TaskObservation):
            raise TypeError("policy accepts only TaskObservation")
        skills = [s for s in available if s.kind == observation.kind]
        return self.rng.choice(skills) if skills else None

class FrozenLLMPolicy:
    """Provider-injected JSON router, without fixed reuse, injected faults, or fallback."""
    mechanism = "frozen_llm_router"
    parameters_updated = 0

    def __init__(self, provider, *, model, backend):
        self.provider, self.model, self.backend = provider, model, backend
        self.events = []

    def select(self, observation, available):
        if not isinstance(observation, TaskObservation):
            raise TypeError("policy accepts only TaskObservation")
        request = {"observation": observation.to_dict(), "skills": [{"skill_id": s.skill_id, "name": s.name} for s in available],
                   "model": self.model, "backend": self.backend}
        try:
            raw = self.provider(request)
        except Exception as error:
            self.events.append({"error_type": "provider_error", "message": str(error), "fallback": False})
            raise PolicyError("provider_error") from error
        try:
            action = json.loads(raw)
            if not isinstance(action, dict) or set(action) != {"skill_id"}:
                raise ValueError("expected exactly skill_id")
            selected = next(s for s in available if s.skill_id == action["skill_id"])
        except (ValueError, TypeError, StopIteration) as error:
            self.events.append({"error_type": "parse_or_selection_error", "response": raw, "fallback": False})
            raise PolicyError("parse_or_selection_error") from error
        self.events.append({"response": raw, "selected": selected.skill_id, "fallback": False})
        return selected
