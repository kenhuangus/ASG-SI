"""Static admission and staged online accumulation on a frozen candidate pool."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import subprocess
from asg_si.benchmark import make_candidates
from asg_si.evaluation import adjudicate_candidate, evaluate_checkpoint
from asg_si.evidence import environment_snapshot, write_bundle, write_json
from asg_si.graph import SkillGraph
from asg_si.rewards import DEFAULT_WEIGHTS
from asg_si.splits import derive_seed, make_splits, split_manifest
from asg_si.types import ACCEPT, canonical_json, digest
from asg_si.verifier import GateConfig, METHODS, verify

def _append(path, value):
    with Path(path).open("a") as handle:
        handle.write(canonical_json(value)+"\n")

def _revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None

def run(config, output):
    seeds = config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("independent runs require distinct seeds")
    methods = config.get("methods", list(METHODS))
    if not methods or len(set(methods)) != len(methods) or set(methods)-set(METHODS):
        raise ValueError("invalid admission configurations")
    phases = config.get("phases", ["static", "online"])
    if not phases or len(set(phases)) != len(phases) or set(phases)-{"static", "online"}:
        raise ValueError("invalid experiment phases")
    size = config.get("online_batch_size", 2)
    if size < 1:
        raise ValueError("online batch size must be positive")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; previous results are never silently overwritten")
    gate_base = GateConfig.from_dict(config.get("gate", {}))
    weights = dict(config.get("evaluation_weights", DEFAULT_WEIGHTS))
    if set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError("incomplete fixed evaluation metric")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "config_hash": digest(config), "base_commit": _revision(),
                "environment": environment_snapshot(), "policy": "scripted_uniform_library_selection", "parameters_updated": 0,
                "evaluation_weights": weights, "evaluation_metric_hash": digest(weights),
                "setting": "controlled deterministic candidate-admission simulation",
                "budget": "same allocated max_cases; actual work and process overhead are reported",
                "self_vs_independent": "identical checks and inputs; only process separation differs"}
    write_json(output/"run.json", manifest)
    for seed in seeds:
        splits = make_splits(seed, **config.get("splits", {}))
        candidates, annotations = make_candidates(splits["discovery"])
        root = output/f"seed_{seed}"
        write_json(root/"split_manifest.json", split_manifest(splits))
        write_json(root/"seeds.json", {stream: derive_seed(seed, stream) for stream in
                   ("promotion", "evaluation", "stress", "candidate_order", "policy_evaluation", "policy_stress")})
        write_json(root/"splits.json", {name: [c.to_dict() for c in cases] for name, cases in splits.items()})
        write_json(root/"candidates.json", [c.to_dict() for c in candidates])
        write_json(root/"analysis_annotations.json", annotations)
        labels = {}
        for candidate in candidates:
            label = adjudicate_candidate(candidate, splits["evaluation"]+splits["stress"])
            labels[candidate.submission_id] = label["valid"]
            write_json(root/"hidden_adjudication"/f"{candidate.submission_id}.json", label)
        order = list(candidates)
        random.Random(derive_seed(seed, "candidate_order")).shuffle(order)
        for method in methods:
            gate = replace(gate_base, method=method)
            for phase in phases:
                graph, cumulative_cost = SkillGraph(), 0
                batches = [order] if phase == "static" else [order[i:i+size] for i in range(0, len(order), size)]
                if phase == "online":
                    _evaluate(output, seed, phase, method, 0, graph, splits, weights)
                for checkpoint, batch in enumerate(batches, 1):
                    accepted = inserted = 0
                    for candidate in batch:
                        decision = verify(candidate, splits["promotion"], gate)
                        reference = write_bundle(root/phase/method/f"checkpoint_{checkpoint}"/candidate.submission_id,
                                                 candidate, splits["promotion"], gate, decision)
                        reference["bundle"] = str(Path(reference["bundle"]).relative_to(output))
                        unique = graph.add_skill(candidate.skill, decision)
                        accepted += decision["status"] == ACCEPT
                        inserted += unique
                        cumulative_cost += decision["tool_calls"]
                        _append(output/"decisions.jsonl", {"seed": seed, "phase": phase, "method": method,
                            "checkpoint": checkpoint, "submission_id": candidate.submission_id, "skill_id": candidate.skill.skill_id,
                            "candidate_valid": labels[candidate.submission_id], "fault_type": annotations[candidate.submission_id],
                            "unique_insertion": unique, **reference, "decision": decision})
                    write_json(root/phase/method/f"checkpoint_{checkpoint}"/"library.json", graph.snapshot())
                    _append(output/"checkpoints.jsonl", {"seed": seed, "phase": phase, "method": method,
                            "checkpoint": checkpoint, "submissions": len(batch), "accepted_decisions": accepted,
                            "unique_insertions": inserted, "active_skills": len(graph.available()),
                            "cumulative_verification_tool_calls": cumulative_cost, "library_hash": graph.state_hash})
                    _evaluate(output, seed, phase, method, checkpoint, graph, splits, weights)
            print(f"seed={seed} method={method}: completed {', '.join(phases)}", flush=True)
    return manifest

def _evaluate(output, seed, phase, method, checkpoint, graph, splits, weights):
    for split in ("evaluation", "stress"):
        rows = evaluate_checkpoint(graph, splits[split], policy_seed=derive_seed(seed, "policy_"+split),
                                   composition=method == "independent_composition", weights=weights)
        for row in rows:
            _append(output/"tasks.jsonl", {"seed": seed, "phase": phase, "method": method,
                    "checkpoint": checkpoint, "split": split, **row})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        run(json.loads(Path(args.config).read_text()), args.output)
    except ValueError as error:
        parser.exit(1, str(error)+"\n")

if __name__ == "__main__":
    main()
