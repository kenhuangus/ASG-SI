"""Controlled held-out pairings of individually tested declared primitive skills."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from asg_si.evidence import write_json
from asg_si.executor import execute
from asg_si.graph import SkillGraph
from asg_si.oracle import assess, output_correct
from asg_si.splits import make_splits
from asg_si.types import ACCEPT, Skill

def primitives():
    programs = []
    for unit, converter in (("L", "convert_liters"), ("mL", "convert_milliliters")):
        steps = ({"id": "read", "tool": "read_records", "args": {"records": {"input": "records"}}},
                 {"id": "validate", "tool": "validate_fields", "args": {"records": {"step": "read"}}},
                 {"id": "convert", "tool": converter, "args": {"records": {"step": "validate"}}})
        programs.append(Skill(f"prepare_{unit}", steps, "convert", {s["tool"]: "1.0.0" for s in steps}, ("records",), kind="prepare_batch"))
    steps = ({"id": "aggregate", "tool": "aggregate", "args": {"batch": {"input": "batch"}}},
             {"id": "write", "tool": "write_output", "args": {"report": {"step": "aggregate"}, "destination": {"input": "destination"}}})
    programs.append(Skill("render_liters", steps, "write", {"aggregate": "1.0.0", "write_output": "1.0.0"}, ("batch", "destination"), kind="render_report"))
    return programs

def compose(prepare, render):
    steps = ({"id": "prepare", "skill": prepare.skill_id, "args": {"records": {"input": "records"}}},
             {"id": "render", "skill": render.skill_id, "args": {"batch": {"step": "prepare"}, "destination": {"input": "destination"}}})
    return Skill(f"{prepare.name}_then_{render.name}", steps, "render", {},
                 dependencies={prepare.skill_id: prepare.skill_id, render.skill_id: render.skill_id})

def prepare_graph(cases):
    graph, reports = SkillGraph(), []
    liters, milliliters, render = primitives()
    for skill in (liters, milliliters):
        checks = []
        factor = 1 if skill.name == "prepare_L" else 1000
        for case in cases:
            result = execute(skill, case.observation, composition=True)
            ok = result.error is None and output_correct({"total": case.expected["total"]*factor, "unit": "L"},
                 {"total": sum(result.output["values"]) if result.output else None, "unit": "L"})
            ok = ok and result.output["unit"] == ("L" if factor == 1 else "mL")
            checks.append({"case_hash": case.observation.content_hash, "passed": ok, "execution": result.to_dict()})
        if not all(c["passed"] for c in checks):
            raise RuntimeError("primitive failed its individual contract tests")
        decision = {"status": ACCEPT, "composition_checked": True, "basis": "individual_primitive_fixture_tests"}
        graph.add_skill(skill, decision)
        reports.append({"skill": skill.to_dict(), "tests": checks, "decision": decision})
    checks = []
    for case in cases:
        observation = replace(case.observation, inputs={**case.observation.inputs, "batch": {"values": [case.expected["total"]], "unit": "L"}})
        result = execute(render, observation, composition=True)
        checks.append({"case_hash": case.observation.content_hash, "passed": assess(case, result)["success"],
                       "observation": observation.to_dict(), "execution": result.to_dict()})
    if not all(c["passed"] for c in checks):
        raise RuntimeError("render primitive failed individual tests")
    decision = {"status": ACCEPT, "composition_checked": True, "basis": "individual_primitive_fixture_tests"}
    graph.add_skill(render, decision)
    reports.append({"skill": render.to_dict(), "tests": checks, "decision": decision})
    return graph, reports

def run_composition(seed=7):
    splits = make_splits(seed)
    graph, primitive_reports = prepare_graph(splits["promotion"])
    liters, milliliters, render = primitives()
    rows = []
    for prepare, compatible in ((liters, True), (milliliters, False)):
        program = compose(prepare, render)
        for mode in ("flat_verified_library", "checked_graph"):
            for case in splits["evaluation"]:
                result = execute(program, case.observation, graph=graph, composition=mode == "checked_graph")
                rows.append({"seed": seed, "mode": mode, "compatible": compatible, "unseen_pairing": True,
                             "composition_depth": 2, "tool_depth": 5, "program": program.to_dict(),
                             "observation": case.observation.to_dict(), "expected": case.expected,
                             "case_hash": case.observation.content_hash, "grade": assess(case, result), "execution": result.to_dict()})
    summary = []
    for mode in ("flat_verified_library", "checked_graph"):
        for compatible in (True, False):
            selected = [r for r in rows if r["mode"] == mode and r["compatible"] == compatible]
            summary.append({"mode": mode, "compatible": compatible, "num_tasks": len(selected),
                            "successes": sum(r["grade"]["success"] for r in selected),
                            "blocked": sum(r["execution"]["error"] is not None for r in selected),
                            "incorrect_completed": sum(not r["grade"]["success"] and r["execution"]["error"] is None for r in selected),
                            "tool_calls": sum(r["execution"]["tool_calls"] for r in selected)})
    return {"seed": seed, "scope": "controlled two-skill pairings withheld from primitive tests; not broad compositional generalization",
            "primitive_reports": primitive_reports, "summary": summary, "rows": rows}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("output already exists")
    result = run_composition(args.seed)
    write_json(Path(args.output), result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "primitive_reports")}, indent=2))

if __name__ == "__main__":
    main()
