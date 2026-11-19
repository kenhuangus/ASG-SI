"""Analyze recorded denominators and checkpoints without inferring missing experiments."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from experiments.report import bootstrap_ci


def analyze(payload):
    runs = payload.get("runs", payload)
    grouped = defaultdict(list)
    for seed, rows in runs.items():
        for row in rows:
            n = row["num_tasks"]
            if n <= 0:
                raise ValueError("invalid recorded denominator")
            grouped[row["iteration"]].append({"success": row["success_rate"], "reward": row["avg_reward"],
                "skill_pct": 100*row["actions"]["skill"]/n, "direct_pct": 100*row["actions"]["direct"]/n})
    return {"schema_version": payload.get("schema_version", 1),
            "historical_data_warning": None if payload.get("schema_version") == 2 else "Legacy summaries do not establish independent policy seeds or frozen evaluation.",
            "statistical_unit": "run seed; descriptive bootstrap intervals",
            "checkpoints": [{"iteration": i, "metrics": {k: bootstrap_ci([r[k] for r in rows]) for k in ("success", "reward", "skill_pct", "direct_pct")}}
                            for i, rows in sorted(grouped.items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="multi_seed_results.json")
    parser.add_argument("--output", default="legacy_analysis.json")
    args = parser.parse_args()
    report = analyze(json.loads(Path(args.input).read_text()))
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
