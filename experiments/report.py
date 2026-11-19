"""Descriptive paper tables derived from per-task and per-decision raw events."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
from statistics import mean
from asg_si.evidence import write_json
from asg_si.rewards import DEFAULT_WEIGHTS, compute_reward
from asg_si.types import ACCEPT, REJECT, INSUFFICIENT_EVIDENCE, ExecutionResult

def load_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]

def bootstrap_ci(values, *, seed=91, repetitions=2000):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "ci95": None, "n_runs": 0}
    if len(values) == 1:
        return {"mean": values[0], "ci95": None, "n_runs": 1}
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(repetitions))
    return {"mean": mean(values), "ci95": [means[int(.025*repetitions)], means[int(.975*repetitions)]],
            "n_runs": len(values), "statistical_unit": "independent run seed", "ci_method": "percentile run bootstrap"}

def ratio(n, d):
    return n/d if d else None

def generate_report(root):
    root = Path(root)
    decisions, tasks = load_jsonl(root/"decisions.jsonl"), load_jsonl(root/"tasks.jsonl")
    gd, gt = defaultdict(list), defaultdict(list)
    for row in decisions:
        gd[row["seed"], row["phase"], row["method"]].append(row)
    for row in tasks:
        gt[row["seed"], row["phase"], row["method"], row["checkpoint"], row["split"]].append(row)
    per_run, conditions = [], defaultdict(list)
    for (seed, phase, method), rows in gd.items():
        faulty, valid = [r for r in rows if r["candidate_valid"] is False], [r for r in rows if r["candidate_valid"] is True]
        final = max(r["checkpoint"] for r in rows)
        evaluation, stress = gt[seed, phase, method, final, "evaluation"], gt[seed, phase, method, final, "stress"]
        m = {"seed": seed, "phase": phase, "method": method,
             "faulty_admission_rate": ratio(sum(r["decision"]["status"] == ACCEPT for r in faulty), len(faulty)),
             "valid_rejection_rate": ratio(sum(r["decision"]["status"] == REJECT for r in valid), len(valid)),
             "valid_non_admission_rate": ratio(sum(r["decision"]["status"] != ACCEPT for r in valid), len(valid)),
             "downstream_success": ratio(sum(r["grade"]["success"] for r in evaluation), len(evaluation)),
             "stress_success": ratio(sum(r["grade"]["success"] for r in stress), len(stress)),
             "constraint_violation_rate": ratio(sum(r["grade"]["constraint_violation"] for r in evaluation), len(evaluation)),
             "verification_tool_calls": sum(r["decision"]["tool_calls"] for r in rows),
             "verification_ms": sum(r["decision"]["verification_ms"] for r in rows), "submissions": len(rows),
             "faulty_candidates": len(faulty), "valid_candidates": len(valid), "unknown_validity": sum(r["candidate_valid"] is None for r in rows),
             "accepted_decisions": sum(r["decision"]["status"] == ACCEPT for r in rows),
             "insufficient_decisions": sum(r["decision"]["status"] == INSUFFICIENT_EVIDENCE for r in rows),
             "unique_promotions": sum(r["unique_insertion"] for r in rows), "replay_events": sum(r["decision"]["num_tests"] for r in rows),
             "evidence_bytes": sum(r["evidence_bytes"] for r in rows)}
        per_run.append(m)
        conditions[phase, method].append(m)
    names = ["faulty_admission_rate", "valid_rejection_rate", "valid_non_admission_rate", "downstream_success", "stress_success",
             "constraint_violation_rate", "verification_tool_calls", "verification_ms", "unique_promotions", "accepted_decisions", "replay_events", "evidence_bytes"]
    summary = [{"phase": phase, "method": method, "metrics": {k: bootstrap_ci([r[k] for r in rows]) for k in names}}
               for (phase, method), rows in sorted(conditions.items())]
    effects = []
    for phase in sorted({r["phase"] for r in per_run}):
        baseline = {r["seed"]: r for r in conditions.get((phase, "first_success"), [])}
        for method in sorted({r["method"] for r in per_run}-{"first_success"}):
            pairs = [(r, baseline[r["seed"]]) for r in conditions.get((phase, method), []) if r["seed"] in baseline]
            effects.append({"phase": phase, "method": method, "reference": "first_success", "paired_run_differences":
                            {k: bootstrap_ci([r[k]-b[k] for r, b in pairs if r[k] is not None and b[k] is not None]) for k in names}})
    cg, fg, diagnostics = defaultdict(list), defaultdict(list), []
    for (seed, phase, method, checkpoint, split), rows in sorted(gt.items()):
        cg[phase, method, checkpoint, split].append(mean(float(r["grade"]["success"]) for r in rows))
        rewards = [compute_reward(ExecutionResult(**r["execution"]), r["grade"], bool(r["skill_id"]), {**DEFAULT_WEIGHTS, "reuse": .5})["total"] for r in rows]
        diagnostics.append({"seed": seed, "phase": phase, "method": method, "checkpoint": checkpoint, "split": split,
                            "without_reuse_bonus": mean(r["reward"]["components"]["outcome"] for r in rows), "with_reuse_bonus": mean(rewards),
                            "interpretation": "post-hoc metric sensitivity on identical actions; not a training ablation"})
    curves = [dict(zip(("phase", "method", "checkpoint", "split"), key), success=bootstrap_ci(values)) for key, values in sorted(cg.items())]
    for row in decisions:
        fg[row["phase"], row["method"], row["fault_type"]].append(row)
    faults = [{"phase": p, "method": m, "fault_type": f, "submissions": len(rows), "accepted": sum(r["decision"]["status"] == ACCEPT for r in rows)}
              for (p, m, f), rows in sorted(fg.items())]
    report = {"setting": "deterministic synthetic benchmark; no LLM training",
              "statistics": "Descriptive run-bootstrap CIs conditional on this frozen candidate pool; no p-values or equivalence claims.",
              "summary": summary, "paired_effects": effects, "per_run": per_run, "checkpoint_curves": curves, "by_fault": faults, "reward_diagnostics": diagnostics}
    write_json(root/"report.json", report)
    with (root/"per_run.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_run[0]) if per_run else [])
        writer.writeheader()
        writer.writerows(per_run)
    lines = ["# Deterministic admission benchmark", "", "Means across independent run seeds; controlled synthetic results.", "",
             "| Phase | Admission method | Faulty admitted | Valid rejected | Downstream success | Violations | Verification calls |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        m = row["metrics"]
        def pct(k):
            return "unknown" if m[k]["mean"] is None else f"{m[k]['mean']:.1%}"
        lines.append(f"| {row['phase']} | {row['method']} | {pct('faulty_admission_rate')} | {pct('valid_rejection_rate')} | {pct('downstream_success')} | {pct('constraint_violation_rate')} | {m['verification_tool_calls']['mean']:.0f} |")
    lines += ["", "95% descriptive run-bootstrap intervals, paired effect sizes, exact denominators, fault-specific results, and checkpoint curves are in report.json and per_run.csv.",
              "Intervals are conditional on this small frozen candidate pool. Zero-width intervals do not establish general reliability.",
              "Self-verification is an in-process checking surrogate, not an LLM self-critique experiment. It shares inputs and checks with independent replay; process separation alone should not change decisions here.",
              "Budgets are matched by allocated maximum tests; actual work and process overhead are reported separately.",
              "Online results use staged submissions from the frozen pool, not adaptive candidate generation.",
              "Model tokens and provider costs are zero. Reward sensitivity is post-hoc, not evidence of reward-driven learning."]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    args = parser.parse_args()
    generate_report(args.run_directory)
    print((Path(args.run_directory)/"report.md").read_text())

if __name__ == "__main__":
    main()
