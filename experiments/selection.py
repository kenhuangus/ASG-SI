"""Skill selection in larger libraries, with and without admission gating.

Libraries of N skills (40 % valid variants, 60 % faulty variants drawn from six fault templates,
each with redundant-step variants) are either admitted wholesale (ungated) or filtered by the replay
gate (in-process checking; identical decisions to independent replay for these pure tools).
Selection rules:
  uniform          - uniform random among applicable skills (the rule used in Section 13.1)
  fewest_steps     - cheapest program first (cost-greedy)
  evidence_ranked  - most passed admission tests, ties broken by fewest steps
  runtime_feedback - Beta-Bernoulli UCB over runtime-observable outcomes (no error and no constraint
                     violation). Wrong totals are NOT observable at runtime, which is the point.
"""
import argparse
import math
import random
from pathlib import Path

from asg_si.evidence import write_json
from asg_si.executor import execute
from asg_si.oracle import assess
from asg_si.splits import make_splits
from asg_si.types import Candidate, Skill, digest
from asg_si.verifier import GateConfig, verify


def base_steps(extra_reads, extra_validations):
    steps = [{"id": "read", "tool": "read_records", "args": {"records": {"input": "records"}}}]
    src = "read"
    for i in range(extra_reads):
        steps.append({"id": f"reread{i}", "tool": "read_records", "args": {"records": {"step": src}}})
        src = f"reread{i}"
    steps.append({"id": "validate", "tool": "validate_fields", "args": {"records": {"step": src}}})
    src = "validate"
    for i in range(extra_validations):
        steps.append({"id": f"revalidate{i}", "tool": "validate_fields", "args": {"records": {"step": src}}})
        src = f"revalidate{i}"
    steps += [{"id": "convert", "tool": "convert_liters", "args": {"records": {"step": src}}},
              {"id": "aggregate", "tool": "aggregate", "args": {"batch": {"step": "convert"}}},
              {"id": "write", "tool": "write_output", "args": {"report": {"step": "aggregate"}, "destination": {"input": "destination"}}}]
    return steps


def mutate(steps, fault):
    steps = [dict(s, args=dict(s["args"])) for s in steps]
    ids = [s["id"] for s in steps]
    if fault == "unit_mismatch":
        steps[ids.index("convert")]["tool"] = "convert_milliliters"
    elif fault == "skipped_validation":
        v = ids.index("validate")
        prev = steps[v]["args"]["records"]
        steps = [s for s in steps if s["tool"] != "validate_fields"]
        steps[[s["id"] for s in steps].index("convert")]["args"]["records"] = prev
    elif fault == "prohibited_write":
        steps[-1]["args"]["destination"] = {"literal": "protected"}
    elif fault == "incorrect_binding":
        a = ids.index("aggregate")
        steps.insert(a, {"id": "dup", "tool": "duplicate_first", "args": {"batch": {"step": "convert"}}})
        steps[a + 1]["args"]["batch"] = {"step": "dup"}
    elif fault == "shortcut_sum":
        v = ids.index("convert")
        src = steps[v]["args"]["records"]
        steps = steps[:v] + [{"id": "total", "tool": "sum_values", "args": {"records": src}},
                             {"id": "write", "tool": "write_output", "args": {"report": {"step": "total"}, "destination": {"input": "destination"}}}]
    return steps


FAULTS = ("unit_mismatch", "skipped_validation", "prohibited_write", "incorrect_binding", "shortcut_sum", "memorized_answer")


def make_library(n, rng, discovery):
    skills = []
    n_valid = round(0.4 * n)
    for i in range(n):
        er, ev = rng.randint(0, 3), rng.randint(0, 3)
        if i < n_valid:
            steps, fault = base_steps(er, ev), "none"
        else:
            fault = FAULTS[(i - n_valid) % len(FAULTS)]
            if fault == "memorized_answer":
                steps = [{"id": "const", "tool": "constant_report", "args": {"value": {"literal": discovery.expected["total"] + 0.001 * i}}}]
            else:
                steps = mutate(base_steps(er, ev), fault)
        skill = Skill(f"s{i}", tuple(steps), steps[-1]["id"], {s["tool"]: "1.0.0" for s in steps})
        skills.append((skill, fault))
    rng.shuffle(skills)
    return skills


class Selector:
    def __init__(self, rule, skills, passed, rng):
        self.rule, self.skills, self.passed, self.rng = rule, skills, passed, rng
        self.stats = {s.skill_id: [1, 1] for s in skills}  # Beta(1, 1) over observable success
        self.t = 0

    def pick(self):
        self.t += 1
        if not self.skills:
            return None
        if self.rule == "uniform":
            return self.rng.choice(self.skills)
        if self.rule == "fewest_steps":
            return min(self.skills, key=lambda s: (len(s.steps), s.skill_id))
        if self.rule == "evidence_ranked":
            return min(self.skills, key=lambda s: (-self.passed.get(s.skill_id, 0), len(s.steps), s.skill_id))
        if self.rule == "runtime_feedback":
            def ucb(s):
                a, b = self.stats[s.skill_id]
                return a / (a + b) + math.sqrt(2 * math.log(self.t + 1) / (a + b))
            return max(self.skills, key=lambda s: (ucb(s), -len(s.steps), s.skill_id))
        raise ValueError(self.rule)

    def update(self, skill, observable_ok):
        self.stats[skill.skill_id][0 if observable_ok else 1] += 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 200])
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123, 456, 789])
    p.add_argument("--tasks", type=int, default=200)
    a = p.parse_args()
    rows = []
    for n in a.sizes:
        for seed in a.seeds:
            splits = make_splits(seed, evaluation_size=a.tasks)
            discovery = splits["discovery"][0]
            library = make_library(n, random.Random(seed * 1000 + n), discovery)
            gated, passed = [], {}
            for skill, fault in library:
                cand = Candidate(digest({"n": n, "seed": seed, "s": skill.skill_id}), skill, skill.skill_id, (discovery,), "controlled")
                d = verify(cand, splits["promotion"], GateConfig(method="self_verification"))
                passed[skill.skill_id] = sum(r["passed"] for r in d["results"])
                if d["status"] == "ACCEPT":
                    gated.append(skill)
            pools = {"ungated": [s for s, _ in library], "gated": gated}
            faults = {s.skill_id: f for s, f in library}
            for admission, pool in pools.items():
                for rule in ("uniform", "fewest_steps", "evidence_ranked", "runtime_feedback"):
                    sel = Selector(rule, pool, passed, random.Random(seed))
                    ok = viol = calls = 0
                    for case in splits["evaluation"]:
                        skill = sel.pick()
                        if skill is None:
                            continue
                        result = execute(skill, case.observation)
                        g = assess(case, result)
                        sel.update(skill, result.error is None and not result.violations)
                        ok += g["success"]
                        viol += g["constraint_violation"]
                        calls += result.tool_calls
                    m = len(splits["evaluation"])
                    rows.append({"library_size": n, "seed": seed, "admission": admission, "rule": rule,
                                 "admitted": len(pool), "admitted_faulty": sum(faults[s.skill_id] != "none" for s in pool),
                                 "success": ok / m, "violation_rate": viol / m, "tool_calls_per_task": calls / m})
    summary = []
    for n in a.sizes:
        for admission in ("ungated", "gated"):
            for rule in ("uniform", "fewest_steps", "evidence_ranked", "runtime_feedback"):
                rs = [r for r in rows if r["library_size"] == n and r["admission"] == admission and r["rule"] == rule]
                summary.append({"library_size": n, "admission": admission, "rule": rule,
                                **{k: sum(r[k] for r in rs) / len(rs) for k in ("admitted", "admitted_faulty", "success", "violation_rate", "tool_calls_per_task")},
                                "success_min": min(r["success"] for r in rs), "success_max": max(r["success"] for r in rs)})
    write_json(Path(a.output), {"rows": rows, "summary": summary,
                                "scope": "controlled variants; 40 % valid; selection observes only errors and constraint violations at runtime"})
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
