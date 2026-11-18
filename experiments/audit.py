"""Replay all admissions and downstream tasks without consulting summary tables."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from asg_si.audit_replay import audit_replay
from asg_si.evaluation import evaluate_checkpoint
from asg_si.evidence import EvidenceError, environment_snapshot, read_bundle, write_json
from asg_si.graph import SkillGraph
from asg_si.oracle import output_correct, reference_output
from asg_si.splits import derive_seed
from asg_si.types import EvaluationCase, digest
from .report import load_jsonl

def audit_run(root):
    root = Path(root)
    manifest = json.loads((root/"run.json").read_text())
    if manifest["environment"] != environment_snapshot():
        raise EvidenceError("environment_mismatch: restore recorded runtime and source")
    bundles = candidate_tests = 0
    for record in load_jsonl(root/"decisions.jsonl"):
        path = root/record["bundle"]
        result = audit_replay(path, expected_manifest_sha256=record["manifest_sha256"])
        stored = read_bundle(path, expected_manifest_sha256=record["manifest_sha256"])
        if digest(stored["artifacts"]["decision.json"]) != digest(record["decision"]):
            raise EvidenceError("decision index differs from its evidence artifact")
        bundles += 1
        candidate_tests += result["replayed_tests"]
    checkpoints = {(r["seed"], r["phase"], r["method"], r["checkpoint"]): r for r in load_jsonl(root/"checkpoints.jsonl")}
    grouped = defaultdict(list)
    for row in load_jsonl(root/"tasks.jsonl"):
        grouped[row["seed"], row["phase"], row["method"], row["checkpoint"], row["split"]].append(row)
    task_count = 0
    for (seed, phase, method, checkpoint, split), rows in grouped.items():
        graph = SkillGraph() if checkpoint == 0 else SkillGraph.from_snapshot(json.loads(
            (root/f"seed_{seed}"/phase/method/f"checkpoint_{checkpoint}"/"library.json").read_text()))
        if checkpoint and graph.state_hash != checkpoints[seed, phase, method, checkpoint]["library_hash"]:
            raise EvidenceError("library snapshot differs from recorded checkpoint digest")
        cases = [EvaluationCase.from_dict({k: r[k] for k in ("observation", "expected", "boundary")}) for r in rows]
        if any(not output_correct(reference_output(c.observation.inputs), c.expected) for c in cases):
            raise EvidenceError("stored evaluation label violates reference specification")
        actual = evaluate_checkpoint(graph, cases, policy_seed=derive_seed(seed, "policy_"+split),
                                     composition=method == "independent_composition", weights=manifest["evaluation_weights"])
        for original, reproduced in zip(rows, actual):
            original = {k: v for k, v in original.items() if k not in ("seed", "phase", "method", "checkpoint", "split")}
            if digest(original) != digest(reproduced):
                raise EvidenceError("task_replay_mismatch: policy selection, output, grading, or reward differs")
            task_count += 1
    return {"status": "REPRODUCED", "admission_bundles": bundles, "candidate_test_executions": candidate_tests,
            "downstream_tasks": task_count, "downstream_rewards_reconstructed": task_count,
            "integrity_scope": "recorded SHA-256 roots; retain an external checkpoint to detect whole-run replacement"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    args = parser.parse_args()
    try:
        result = audit_run(args.run_directory)
        write_json(Path(args.run_directory)/"audit_report.json", result)
        print(json.dumps(result, indent=2))
    except (EvidenceError, ValueError, KeyError, OSError) as error:
        parser.exit(1, str(error)+"\n")

if __name__ == "__main__":
    main()
