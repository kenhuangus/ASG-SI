"""Composition at depth: chains of 2-5 admitted primitive skills over two contract tags.

Primitives (each passes its own fixture tests before any chain is formed):
  prep_<u>      convert validated records to unit u            Post: unit=u, validated=True
  rawprep_<u>   convert all records, including invalid ones     Post: unit=u, validated=False
  rescale_<a>_<b>  rescale a batch from unit a to unit b        Pre: unit=a   Post: unit=b, validated=inherit
  render_liters total a batch                                  Pre: unit=L, validated=True
Every chain is prep -> 0..3 rescales -> render; all chains are held out from primitive testing.
Conditions:
  flat            execute any chain (no edge checks)
  chain_replay_5  admit a chain only if it passes 5 end-to-end tasks, then deploy it
  checked_graph   admit a chain only if the interface check passes (no tool calls needed)
"""
import argparse
import itertools
import random
from pathlib import Path

from asg_si.contracts import ContractError, ToolContract, check_arguments, check_output
from asg_si.evidence import write_json
from asg_si.executor import execute
from asg_si.graph import SkillGraph
from asg_si.interfaces import INHERIT, check_chain, make_interface
from asg_si.oracle import assess
from asg_si.tools import ToolRegistry
from asg_si.types import ACCEPT, Skill
from experiments.stochastic import make_cases

UNITS = {"L": 1.0, "mL": 1000.0, "cL": 100.0}


class UnitRegistry(ToolRegistry):
    VERSION = "unit-tools-1"

    def __init__(self):
        super().__init__()
        for u in UNITS:
            self.contracts[f"prep_{u}"] = ToolContract({"records": "list"}, "dict", output_unit=u)
            self.contracts[f"rawprep_{u}"] = ToolContract({"records": "list"}, "dict", output_unit=u)
            for v in UNITS:
                if u != v:
                    self.contracts[f"rescale_{u}_{v}"] = ToolContract({"batch": "dict"}, "dict", {"batch": u}, v)
        self.contracts["render_liters"] = ToolContract({"batch": "dict"}, "dict", {"batch": "L"}, "L")

    def call(self, name, arguments, observation, semantic):
        contract = self.contract(name)
        if not (name.startswith(("prep_", "rawprep_", "rescale_")) or name == "render_liters"):
            return super().call(name, arguments, observation, semantic)
        checks = check_arguments(contract, arguments, semantic)
        if name.startswith(("prep_", "rawprep_")):
            unit = name.split("_")[1]
            keep = [r for r in arguments["records"] if name.startswith("rawprep_") or r.get("valid", True)]
            liters = [r["value"] * (1 if r["unit"] == "L" else .001) for r in keep]
            out = {"values": [x * UNITS[unit] for x in liters], "unit": unit, "validated": name.startswith("prep_")}
        elif name.startswith("rescale_"):
            _, a, b = name.split("_")
            batch = arguments["batch"]
            out = {"values": [x / UNITS[a] * UNITS[b] for x in batch["values"]], "unit": b, "validated": batch.get("validated")}
        else:
            out = {"total": sum(arguments["batch"]["values"]), "unit": "L"}
        check_output(contract, out, semantic)
        return out, [], checks


def primitive(tool, arg, kind, iface):
    skill = Skill(tool, ({"id": "s", "tool": tool, "args": {arg: {"input": arg}}},), "s", {tool: "1.0.0"}, (arg,), kind=kind)
    return skill, iface


def primitives():
    prims = {}
    for u in UNITS:
        prims[f"prep_{u}"] = primitive(f"prep_{u}", "records", "prepare", make_interface({}, {"unit": u, "validated": True}))
        prims[f"rawprep_{u}"] = primitive(f"rawprep_{u}", "records", "prepare", make_interface({}, {"unit": u, "validated": False}))
        for v in UNITS:
            if u != v:
                prims[f"rescale_{u}_{v}"] = primitive(f"rescale_{u}_{v}", "batch", "rescale", make_interface({"unit": u}, {"unit": v, "validated": INHERIT}))
    prims["render_liters"] = primitive("render_liters", "batch", "render", make_interface({"unit": "L", "validated": True}, {}))
    return prims


def chain_skill(names, prims):
    steps, src = [], None
    for i, name in enumerate(names):
        skill = prims[name][0]
        arg = "records" if i == 0 else "batch"
        steps.append({"id": f"c{i}", "skill": skill.skill_id, "args": {arg: {"input": "records"} if i == 0 else {"step": src}}})
        src = f"c{i}"
    return Skill("+".join(names), tuple(steps), src, {}, ("records",),
                 dependencies={prims[n][0].skill_id: prims[n][0].skill_id for n in names}, kind="chain")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--chains-per-depth", type=int, default=120)
    p.add_argument("--tasks", type=int, default=20)
    a = p.parse_args()
    registry, prims = UnitRegistry(), primitives()
    graph = SkillGraph(registry)
    fixtures = make_cases(77, 15)
    for name, (skill, iface) in prims.items():  # primitive fixture tests: each runs without error on its own input type
        for case in fixtures:
            inputs = {"records": case.observation.inputs["records"]} if skill.kind == "prepare" else \
                     {"batch": {"values": [1.0, 2.0], "unit": iface["pre"].get("unit", "L"), "validated": True}}
            obs = type(case.observation)(case.observation.task_id, case.observation.kind, inputs)
            if execute(skill, obs, registry).error:
                raise RuntimeError(f"primitive {name} failed its fixture")
        graph.add_skill(skill, {"status": ACCEPT, "skill_id": skill.skill_id, "composition_checked": True})
    rng = random.Random(7)
    preps = [n for n in prims if prims[n][0].kind == "prepare"]
    rescales = [n for n in prims if prims[n][0].kind == "rescale"]
    eval_cases, promo_cases = make_cases(2024, a.tasks), make_cases(4048, 5)
    rows = []
    for depth in range(4):
        space = [(pr, *rs, "render_liters") for pr in preps for rs in itertools.product(rescales, repeat=depth)]
        ok = [s for s in space if check_chain([prims[n][1] for n in s])["ok"]]
        bad = [s for s in space if s not in set(ok)]
        half = a.chains_per_depth // 2
        chosen = (ok if len(ok) <= half else rng.sample(ok, half)) + (bad if len(bad) <= half else rng.sample(bad, half))
        for names in chosen:
            skill = chain_skill(names, prims)
            compatible = check_chain([prims[n][1] for n in names])["ok"]
            flat = [assess(c, execute(skill, c.observation, registry, graph=graph)) for c in eval_cases]
            flat_calls = sum(execute(skill, c.observation, registry, graph=graph).tool_calls for c in eval_cases)
            promo = [execute(skill, c.observation, registry, graph=graph) for c in promo_cases]
            replay_ok = all(assess(c, r)["success"] for c, r in zip(promo_cases, promo))
            rows.append({"chain": list(names), "length": len(names), "compatible": compatible,
                         "flat_incorrect_tasks": sum(not g["success"] for g in flat), "flat_tool_calls": flat_calls,
                         "replay5_admitted": replay_ok, "replay5_test_calls": sum(r.tool_calls for r in promo),
                         "replay5_incorrect_tasks": sum(not g["success"] for g in flat) if replay_ok else 0,
                         "checked_admitted": compatible,
                         "checked_incorrect_tasks": sum(not g["success"] for g in flat) if compatible else 0})
    summary = []
    for length in sorted({r["length"] for r in rows}):
        rs = [r for r in rows if r["length"] == length]
        inc = [r for r in rs if not r["compatible"]]
        summary.append({
            "chain_length": length, "chains": len(rs), "incompatible": len(inc),
            "tasks_per_chain": a.tasks,
            "flat_incorrect_task_rate": sum(r["flat_incorrect_tasks"] for r in rs) / (len(rs) * a.tasks),
            "incompatible_chains_with_some_correct_tasks": sum(r["flat_incorrect_tasks"] < a.tasks for r in inc),
            "incompatible_chains_correct_on_all_tasks": sum(r["flat_incorrect_tasks"] == 0 for r in inc),
            "compatible_chains_with_incorrect_tasks": sum(r["flat_incorrect_tasks"] > 0 for r in rs if r["compatible"]),
            "replay5_incompatible_admitted": sum(r["replay5_admitted"] for r in inc),
            "replay5_compatible_rejected": sum(not r["replay5_admitted"] for r in rs if r["compatible"]),
            "replay5_incorrect_task_rate": sum(r["replay5_incorrect_tasks"] for r in rs) / (len(rs) * a.tasks),
            "replay5_test_calls_per_chain": sum(r["replay5_test_calls"] for r in rs) / len(rs),
            "checked_incompatible_admitted": sum(r["checked_admitted"] for r in inc),
            "checked_compatible_rejected": sum(not r["checked_admitted"] for r in rs if r["compatible"]),
            "checked_incorrect_task_rate": sum(r["checked_incorrect_tasks"] for r in rs) / (len(rs) * a.tasks)})
    write_json(Path(a.output), {"summary": summary, "rows": rows,
                                "scope": "declared unit and validation tags; chains held out from primitive tests; not a proof for undeclared properties"})
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
