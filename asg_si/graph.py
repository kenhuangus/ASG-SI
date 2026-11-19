"""Versioned registry, composition bindings, and transitive dependency revocation."""
from .contracts import declared_edges
from .tools import ToolRegistry
from .types import ACCEPT, Skill, clone, digest

class SkillGraph:
    def __init__(self, registry=None):
        self.registry = registry or ToolRegistry()
        self._skills, self._revoked, self.decisions, self.edges = {}, {}, {}, {}

    def add_skill(self, skill, decision):
        if decision["status"] != ACCEPT:
            return False
        identity = skill.skill_id
        if decision.get("skill_id", identity) != identity:
            raise ValueError("decision belongs to another executable")
        if identity in self._skills:
            return False
        for dep, version in skill.dependencies.items():
            if self.get(dep) is None or dep != version:
                raise ValueError("cannot admit missing, revoked, or changed dependency")
        self._skills[identity] = Skill.from_dict(skill.to_dict())
        self.decisions[identity] = clone(decision)
        self.edges[identity] = [dict(e, checked=bool(decision.get("composition_checked", False))) for e in declared_edges(skill, self.registry)]
        return True

    def get(self, identity):
        skill = self._skills.get(identity)
        if skill is None or identity in self._revoked or identity != skill.skill_id:
            return None
        if any(dep != version or self.get(dep) is None for dep, version in skill.dependencies.items()):
            return None
        return Skill.from_dict(skill.to_dict())

    def available(self):
        return [s for identity in sorted(self._skills) if (s := self.get(identity)) is not None]

    def revoke(self, identity, reason):
        if identity not in self._skills:
            return []
        revoked, pending = [], [identity]
        while pending:
            current = pending.pop()
            if current in self._revoked:
                continue
            self._revoked[current] = reason
            revoked.append(current)
            pending.extend(key for key, skill in self._skills.items() if current in skill.dependencies)
        return sorted(revoked)

    def invalidate_changed_tools(self):
        revoked = []
        for identity, skill in self._skills.items():
            if any(self.registry.version(tool) != version for tool, version in skill.tool_versions.items()):
                revoked.extend(self.revoke(identity, "tool version changed; re-verification required"))
        return sorted(set(revoked))

    def snapshot(self):
        return {"skills": {key: value.to_dict() for key, value in sorted(self._skills.items())},
                "revoked": clone(self._revoked), "decisions": clone(self.decisions), "edges": clone(self.edges)}

    @classmethod
    def from_snapshot(cls, snapshot):
        graph = cls()
        for identity, value in snapshot["skills"].items():
            skill = Skill.from_dict(value)
            if skill.skill_id != identity:
                raise ValueError("snapshot executable digest mismatch")
            graph._skills[identity] = skill
        graph._revoked = clone(snapshot["revoked"])
        graph.decisions = clone(snapshot["decisions"])
        graph.edges = clone(snapshot["edges"])
        return graph

    @property
    def state_hash(self):
        return digest(self.snapshot())
