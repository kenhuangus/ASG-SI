"""Retention under a silent tool change, with and without the library lifecycle.

A stream of 10 checkpoints (30 evaluation tasks each, fresh seed per checkpoint). Before checkpoint 4
the tool provider upgrades validate_fields to 2.0.0: it still type-checks records but no longer drops
records whose "valid" flag is false, and a new tool filter_valid is published. Conditions:
  unpinned          skills keep running against whatever tool version is installed
  pinned            version pins fail closed (the call errors) but nothing re-verifies
  pinned_lifecycle  at every checkpoint boundary, changed tools invalidate dependent skills
                    (transitive revocation); the candidate pool is re-verified against the new
                    registry, so stale programs re-pinned to 2.0.0 are rejected and an adapted
                    program that calls filter_valid can be admitted
"""
import argparse
from dataclasses import replace
from pathlib import Path

from asg_si.benchmark import workflow
from asg_si.contracts import ToolContract, check_arguments
from asg_si.evaluation import evaluate_checkpoint
from asg_si.evidence import write_json
from asg_si.graph import SkillGraph
from asg_si.splits import make_splits
from asg_si.tools import ToolRegistry
from asg_si.types import ACCEPT, Candidate, Skill, clone, digest
from asg_si.verifier import GateConfig, verify_in_process

CHANGE_AT = 4


class RegistryV2(ToolRegistry):
    VERSION = "volume-tools-2"

    def __init__(self):
        super().__init__()
        self.contracts["filter_valid"] = ToolContract({"records": "list"}, "list")

    def version(self, name):
        self.contract(name)
        return "2.0.0" if name == "validate_fields" else "1.0.0"

    def call(self, name, arguments, observation, semantic):
        if name == "validate_fields":
            check_arguments(self.contracts[name], arguments, semantic)
            rows = [clone(r) for r in arguments["records"]]
            for r in rows:   # 2.0.0 keeps invalid records; type checks unchanged
                if r.get("unit") not in ("L", "mL"):
                    raise ValueError("unknown record unit")
            return rows, [], 0
        if name == "filter_valid":
            check_arguments(self.contracts[name], arguments, semantic)
            return [clone(r) for r in arguments["records"] if r.get("valid", True)], [], 0
        return super().call(name, arguments, observation, semantic)


def adapted_program():
    base = workflow("adapted_filter_valid")
    steps = list(clone(base.steps))
    i = [s["id"] for s in steps].index("validate")
    steps.insert(i + 1, {"id": "filter", "tool": "filter_valid", "args": {"records": {"step": "validate"}}})
    steps[i + 2]["args"]["records"] = {"step": "filter"}
    return Skill("adapted_filter_valid", tuple(steps), "write", {s["tool"]: "1.0.0" for s in steps})


def repin(skill, registry):
    return replace(skill, tool_versions={t: registry.version(t) for t in skill.tool_versions})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoints", type=int, default=10)
    a = p.parse_args()
    v1, v2 = ToolRegistry(), RegistryV2()
    original = workflow("compiled_standard")
    pool = [original, adapted_program()]   # the adapted program only becomes executable once filter_valid exists
    results = {}
    for condition in ("unpinned", "pinned", "pinned_lifecycle"):
        graph = SkillGraph(v1)
        graph.add_skill(original, {"status": ACCEPT, "skill_id": original.skill_id})
        series, events = [], []
        for cp in range(a.checkpoints):
            registry = v2 if cp >= CHANGE_AT else v1
            if condition == "pinned_lifecycle" and cp > 0:
                graph.registry = registry
                revoked = graph.invalidate_changed_tools()
                if revoked:
                    events.append({"checkpoint": cp, "revoked": len(revoked)})
                    promo = make_splits(1000 + cp)["promotion"]
                    for skill in pool:
                        cand_skill = repin(skill, registry)
                        try:
                            [registry.version(t) for t in cand_skill.tool_versions]
                        except ValueError:
                            continue
                        cand = Candidate(digest({"cp": cp, "s": cand_skill.skill_id}), cand_skill, cand_skill.skill_id, (), "re-verification")
                        d = verify_in_process(cand, promo, GateConfig(method="self_verification"), registry=registry)
                        graph.add_skill(cand_skill, d)
                        events.append({"checkpoint": cp, "skill": skill.name, "decision": d["status"]})
            graph.registry = registry
            cases = make_splits(2000 + cp)["evaluation"]
            if condition == "unpinned":
                from asg_si.executor import execute
                from asg_si.oracle import assess
                ok = sum(assess(c, execute(s, c.observation, registry, enforce_pins=False))["success"]
                         for c in cases for s in graph.available()[:1]) / len(cases)
                errors = 0
            else:
                rows = evaluate_checkpoint(graph, cases, policy_seed=cp)
                ok = sum(r["grade"]["success"] for r in rows) / len(rows)
                errors = sum(r["execution"]["error"] is not None for r in rows)
            series.append({"checkpoint": cp, "tool_version": registry.version("validate_fields"), "success": ok,
                           "observable_errors": errors, "library": [s.name for s in graph.available()]})
        results[condition] = {"series": series, "lifecycle_events": events}
    write_json(Path(a.output), {"change_at_checkpoint": CHANGE_AT, "results": results,
                                "scope": "one controlled tool change; re-verification draws from a fixed candidate pool"})
    for c, r in results.items():
        print(c, [round(s["success"], 2) for s in r["series"]], r["lifecycle_events"])


if __name__ == "__main__":
    main()
