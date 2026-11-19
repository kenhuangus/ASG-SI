"""Compatibility runner with independent seeds and fixed checkpoint evaluation."""
import argparse
import json
from pathlib import Path
from asg_si.splits import derive_seed
from asg_si_demo import ASGSISystem, VerifiableEnv


def run_experiment(seed, iters=6, num_tasks=80, evaluation_tasks=80, output="legacy_runs", fault_error_rate=0.0):
    if min(iters, num_tasks, evaluation_tasks) < 1:
        raise ValueError("counts must be positive")
    system = ASGSISystem(seed=seed, audit_dir=str(Path(output)/f"seed_{seed}"/"audit"), fault_error_rate=fault_error_rate)
    env = VerifiableEnv(derive_seed(seed, "final_evaluation"))
    tasks = [env.sample_task() for _ in range(evaluation_tasks)]
    summaries = []
    for checkpoint in range(iters+1):
        collection = None if checkpoint == 0 else system.run_iteration(checkpoint-1, num_tasks)
        summaries.append({"iteration": checkpoint, **system.evaluate(tasks, derive_seed(seed, "evaluation_policy")),
                          "collection": collection, "policy_metadata": system.metadata, "skill_graph_size": system.skill_graph.size()})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123, 456, 789])
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--num-tasks", type=int, default=80)
    parser.add_argument("--evaluation-tasks", type=int, default=80)
    parser.add_argument("--output", default="legacy_runs")
    parser.add_argument("--fault-error-rate", type=float, default=0.0)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.iters, args.num_tasks, args.evaluation_tasks) < 1:
        parser.error("use distinct seeds and positive counts")
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        parser.error("output directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    results = {str(seed): run_experiment(seed, args.iters, args.num_tasks, args.evaluation_tasks, root, args.fault_error_rate) for seed in args.seeds}
    target = root/"multi_seed_results.json"
    target.write_text(json.dumps({"schema_version": 2, "mechanism": "scripted_simulation", "parameters_updated": 0,
                                  "config": vars(args), "runs": results}, indent=2))
    print(f"Saved fixed-checkpoint results for {len(results)} runs to {target}")

if __name__ == "__main__":
    main()
