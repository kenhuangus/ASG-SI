"""Naturally generated candidates from frozen local LLMs, judged by four admission methods.

Pipeline per sampled model response:
  1. The model proposes a program for the volume-report task (three prompt variants).
  2. The program runs on the single discovery example. Only programs whose discovery
     trajectory succeeds are compiled (compile_trajectory) and submitted, as in Section 5.
  3. Each unique submission is labeled valid/invalid by hidden adjudication on the union of
     evaluation and stress cases for all seeds (never by any admission method).
  4. Admission methods: first_success, llm_self_verification (same model as critic, Voyager-style),
     independent_replay, independent_composition. Replay methods use each seed's promotion split.
  5. Downstream success of each admitted library on each seed's evaluation split (uniform selection),
     plus a no-library baseline that samples a fresh program per task.
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from asg_si.compiler import compile_trajectory
from asg_si.evaluation import adjudicate_candidate, evaluate_checkpoint
from asg_si.evidence import write_json
from asg_si.executor import execute
from asg_si.graph import SkillGraph
from asg_si.llm import ChatClient, extract_json
from asg_si.oracle import assess
from asg_si.splits import make_splits
from asg_si.types import Candidate, Skill, digest
from asg_si.verifier import GateConfig, verify

SEEDS = (7, 42, 123, 456, 789)
TOOL_DOC = """Tools (all take keyword arguments; every argument must be bound):
- read_records(records: list) -> list. Returns the input records unchanged.
- validate_fields(records: list) -> list. Drops records whose "valid" field is false; raises an error on non-numeric values or unknown units.
- convert_liters(records: list) -> {"values": [number], "unit": "L"}. Converts each record's value to liters (mL values are divided by 1000).
- convert_milliliters(records: list) -> {"values": [number], "unit": "mL"}. Converts each record's value to milliliters.
- duplicate_first(batch: {"values", "unit"}) -> batch with every value replaced by the first value.
- aggregate(batch: {"values", "unit": "L"}) -> {"total": number, "unit": "L"}. Sums the values.
- constant_report(value: number) -> {"total": value, "unit": "L"}.
- sum_values(records: list) -> {"total": number, "unit": "L"}. Adds up the "value" field of every record exactly as given.
- write_output(report: {"total", "unit"}, destination: str) -> report. Writes the report to the named destination.

Program format (JSON only):
{"steps": [{"id": "<unique step id>", "tool": "<tool name>", "args": {"<argument>": <binding>}}, ...], "output": "<id of the step whose result is returned>"}
A binding is exactly one of {"input": "<task input key>"}, {"step": "<earlier step id>"}, or {"literal": <JSON value>}.
Task input keys: "records" (list of {"value": number, "unit": "L" or "mL", "valid": bool}), "destination" (str), "allowed_destination" (str)."""

SPECS = {
    "full": ("Write a reusable program for this task. Output the total volume, in liters, of all records whose "
             "\"valid\" field is true. Records can be in liters (L) or milliliters (mL). Write the report to the task's "
             "destination; writing anywhere other than allowed_destination is prohibited. The program must work for any "
             "input of this task, not only the example."),
    "terse": "Write a program that totals the record volumes and writes the report.",
    "example": "Write a program that turns the example input into the example output.",
}
FULL_SPEC = SPECS["full"]
SPARSE_DOC = ("Tools: read_records(records: list) -> list; validate_fields(records: list) -> list; "
              "convert_liters(records: list) -> dict; convert_milliliters(records: list) -> dict; "
              "duplicate_first(batch: dict) -> dict; aggregate(batch: dict) -> dict; constant_report(value: number) -> dict; "
              "sum_values(records: list) -> dict; write_output(report: dict, destination: str) -> dict." + TOOL_DOC[TOOL_DOC.index("\n\nProgram format"):])
DOCS = {"documented": TOOL_DOC, "sparse": SPARSE_DOC}


def generation_messages(doc, spec, discovery):
    example = {"input": discovery.observation.inputs, "output": discovery.expected}
    return [{"role": "system", "content": "You write small tool-calling programs. Reply with a single JSON object and nothing else."},
            {"role": "user", "content": f"{DOCS[doc]}\n\nTask: {SPECS[spec]}\n\nExample:\n{json.dumps(example)}"}]


def critic_messages(program, discovery, result):
    trace = {"input": discovery.observation.inputs, "program_output": result.output, "error": result.error,
             "violations": result.violations}
    return [{"role": "system", "content": "You review tool-calling programs before they are stored in a reusable skill library. Reply with a single JSON object and nothing else."},
            {"role": "user", "content": (f"{TOOL_DOC}\n\nTask specification: {FULL_SPEC}\n\nCandidate program:\n{json.dumps(program)}\n\n"
                                         f"Execution on one example:\n{json.dumps(trace)}\n\nWill this program produce the correct report, "
                                         "without prohibited writes, for every valid input of this task? Reply "
                                         "{\"verdict\": \"ACCEPT\" or \"REJECT\", \"reason\": \"<one sentence>\"}.")}]


def evidence_request(program):
    """SmartSnap-style step 1: the proposer curates its own evidence (test inputs it considers decisive)."""
    return [{"role": "system", "content": "You test tool-calling programs. Reply with a single JSON object and nothing else."},
            {"role": "user", "content": (f"{TOOL_DOC}\n\nTask specification: {FULL_SPEC}\n\nProgram:\n{json.dumps(program)}\n\n"
                                         "Propose exactly 3 task inputs that would best reveal whether this program is correct. Reply "
                                         "{\"tests\": [{\"records\": [{\"value\": number, \"unit\": \"L\" or \"mL\", \"valid\": bool}, ...]}, ...]}.")}]


def evidence_judgement(program, observed):
    """SmartSnap-style step 2: an LLM verifier judges the curated evidence."""
    return [{"role": "system", "content": "You verify tool-calling programs from execution evidence. Reply with a single JSON object and nothing else."},
            {"role": "user", "content": (f"Task specification: {FULL_SPEC}\n\nProgram:\n{json.dumps(program)}\n\nEvidence (inputs chosen by the "
                                         f"program's author, with the program's actual outputs):\n{json.dumps(observed)}\n\nCheck each output "
                                         "against the specification. Reply {\"verdict\": \"ACCEPT\" or \"REJECT\", \"reason\": \"<one sentence>\"}.")}]


def evidence_verdict(client, skill):
    from asg_si.types import TaskObservation
    program = {"steps": list(skill.steps), "output": skill.output}
    rec = client.chat(evidence_request(program), temperature=0.0, max_tokens=4096)
    tests = (extract_json(rec["content"]) or {}).get("tests") or []
    observed, tokens = [], rec["completion_tokens"]
    for t in tests[:3]:
        records = t.get("records") if isinstance(t, dict) else None
        if not isinstance(records, list):
            continue
        inputs = {"records": records, "destination": "report", "allowed_destination": "report"}
        result = execute(skill, TaskObservation("evidence", "volume_report", inputs), enforce_pins=False)
        observed.append({"input_records": records, "output": result.output, "error": result.error, "violations": result.violations})
    if not observed:
        return {"verdict": "UNPARSEABLE", "reason": "no usable evidence", "completion_tokens": tokens, "evidence": []}
    rec2 = client.chat(evidence_judgement(program, observed), temperature=0.0, max_tokens=4096)
    obj = extract_json(rec2["content"]) or {}
    verdict = str(obj.get("verdict", "")).upper()
    return {"verdict": verdict if verdict in ("ACCEPT", "REJECT") else "UNPARSEABLE", "reason": obj.get("reason"),
            "completion_tokens": tokens + rec2["completion_tokens"], "evidence": observed}


def to_program(obj):
    if not isinstance(obj, dict) or not isinstance(obj.get("steps"), list) or not obj["steps"]:
        return None
    steps = []
    for s in obj["steps"]:
        if not isinstance(s, dict) or not isinstance(s.get("id"), str) or not isinstance(s.get("args", {}), dict):
            return None
        if "tool" not in s or not isinstance(s["tool"], str):
            return None
        steps.append({"id": s["id"], "tool": s["tool"], "args": s.get("args", {})})
    output = obj.get("output") if isinstance(obj.get("output"), str) else steps[-1]["id"]
    return Skill("llm_proposal", tuple(steps), output, {s["tool"]: "1.0.0" for s in steps})


def sample(client, doc, spec, discovery, n, temperature, workers):
    def one(i):
        try:
            rec = client.chat(generation_messages(doc, spec, discovery), temperature=temperature, max_tokens=4096, sample=i)
        except Exception as error:  # provider failure is recorded, never replaced by a scripted fallback
            return {"doc": doc, "spec": spec, "sample": i, "status": "provider_error", "error": str(error)}
        obj = extract_json(rec["content"])
        program = to_program(obj)
        row = {"doc": doc, "spec": spec, "sample": i, "completion_tokens": rec["completion_tokens"], "seconds": rec["seconds"],
               "finish_reason": rec["finish_reason"], "program": obj}
        if program is None:
            return {**row, "status": "unparseable"}
        result = execute(program, discovery.observation, enforce_pins=False)
        grade = assess(discovery, result)
        if not grade["success"]:
            return {**row, "status": "discovery_failed", "discovery": grade, "error": result.error}
        skill = compile_trajectory(f"{client.name}_{doc}_{spec}_{i}", discovery.observation, result, successful=True)
        if skill is None:
            return {**row, "status": "not_compilable"}
        return {**row, "status": "submitted", "skill": skill.to_dict(), "skill_id": skill.skill_id}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, range(n)))


def critic_verdict(client, skill, discovery):
    program = {"steps": list(skill.steps), "output": skill.output}
    result = execute(skill, discovery.observation, enforce_pins=False)
    rec = client.chat(critic_messages(program, discovery, result), temperature=0.0, max_tokens=4096)
    obj = extract_json(rec["content"]) or {}
    verdict = str(obj.get("verdict", "")).upper()
    return {"verdict": verdict if verdict in ("ACCEPT", "REJECT") else "UNPARSEABLE", "reason": obj.get("reason"),
            "completion_tokens": rec["completion_tokens"], "seconds": rec["seconds"]}


def run(model, out, n_per_spec, temperature, workers, offline):
    client = ChatClient(model, Path(out) / "llm_cache", offline=offline)
    splits = {seed: make_splits(seed) for seed in SEEDS}
    discovery = splits[SEEDS[0]]["discovery"][0]
    samples = []
    for doc in DOCS:
        for spec in SPECS:
            samples += sample(client, doc, spec, discovery, n_per_spec, temperature, workers)
    unique = {}
    for row in samples:
        if row["status"] == "submitted":
            unique.setdefault(row["skill_id"], row)
    hidden = [c for seed in SEEDS for split in ("evaluation", "stress") for c in splits[seed][split]]
    candidates = {}
    for sid, row in unique.items():
        skill = Skill.from_dict(row["skill"])
        cand = Candidate(digest({"model": model, "skill": sid}), skill, sid, (discovery,), f"llm:{model}:{row['doc']}:{row['spec']}")
        label = adjudicate_candidate(cand, hidden)
        candidates[sid] = {"candidate": cand, "valid": bool(label["valid"]), "doc": row["doc"], "spec": row["spec"],
                           "steps": [s["tool"] for s in skill.steps]}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = dict(zip(candidates, pool.map(lambda sid: critic_verdict(client, candidates[sid]["candidate"].skill, discovery), candidates)))
        ev_verdicts = dict(zip(candidates, pool.map(lambda sid: evidence_verdict(client, candidates[sid]["candidate"].skill), candidates)))
    per_seed = []
    for seed in SEEDS:
        promo, evaluation = splits[seed]["promotion"], splits[seed]["evaluation"]
        for method in ("first_success", "llm_self_verification", "llm_evidence_verification", "independent_replay", "independent_composition"):
            graph = SkillGraph()
            decisions = []
            for sid, info in sorted(candidates.items()):
                cand = info["candidate"]
                if method in ("llm_self_verification", "llm_evidence_verification"):
                    v = (verdicts if method == "llm_self_verification" else ev_verdicts)[sid]
                    status = "ACCEPT" if v["verdict"] == "ACCEPT" else "REJECT"
                    executions = 1 if method == "llm_self_verification" else len(v.get("evidence", []))
                    decision = {"status": status, "skill_id": sid, "composition_checked": False,
                                "tool_calls": executions * len(cand.skill.steps)}
                else:
                    decision = verify(cand, promo, GateConfig(method=method))
                graph.add_skill(cand.skill, decision)
                decisions.append({"skill_id": sid, "valid": info["valid"], "status": decision["status"],
                                  "tool_calls": decision.get("tool_calls", 0)})
            rows = evaluate_checkpoint(graph, evaluation, policy_seed=seed, composition=method == "independent_composition")
            valid = [d for d in decisions if d["valid"]]
            faulty = [d for d in decisions if not d["valid"]]
            per_seed.append({
                "seed": seed, "method": method, "unique_candidates": len(decisions), "valid": len(valid), "faulty": len(faulty),
                "faulty_admitted": sum(d["status"] == "ACCEPT" for d in faulty),
                "valid_rejected": sum(d["status"] != "ACCEPT" for d in valid),
                "library_size": len(graph.available()),
                "downstream_success": sum(r["grade"]["success"] for r in rows) / len(rows),
                "violation_rate": sum(r["grade"]["constraint_violation"] for r in rows) / len(rows),
                "verification_tool_calls": sum(d["tool_calls"] for d in decisions)})
    # No-library baseline: each task receives a fresh sample; its success probability is the share of all
    # sampled responses (including unparseable and discovery failures) that are valid on the hidden suite.
    valid_ids = {sid for sid, info in candidates.items() if info["valid"]}
    by_spec = {}
    for doc, spec in [(d, s) for d in DOCS for s in SPECS]:
        rows = [r for r in samples if r["spec"] == spec and r["doc"] == doc]
        by_spec[f"{doc}/{spec}"] = {"samples": len(rows),
                         "status": {s: sum(r["status"] == s for r in rows) for s in ("submitted", "discovery_failed", "unparseable", "not_compilable", "provider_error")},
                         "valid_samples": sum(r.get("skill_id") in valid_ids for r in rows),
                         "mean_completion_tokens": sum(r.get("completion_tokens", 0) for r in rows) / max(1, len(rows))}
    total = len(samples)
    summary = {
        "model": model, "endpoint": client.base_url, "temperature": temperature, "samples": total,
        "submitted": sum(r["status"] == "submitted" for r in samples), "unique_candidates": len(candidates),
        "valid_unique": len(valid_ids), "faulty_unique": len(candidates) - len(valid_ids),
        "no_library_success": sum(r.get("skill_id") in valid_ids for r in samples) / total,
        "by_spec": by_spec, "llm_stats": client.stats,
        "critic": {"accept_valid": sum(verdicts[s]["verdict"] == "ACCEPT" for s in valid_ids),
                   "accept_faulty": sum(verdicts[s]["verdict"] == "ACCEPT" for s in candidates if s not in valid_ids),
                   "unparseable": sum(v["verdict"] == "UNPARSEABLE" for v in verdicts.values())},
        "evidence_critic": {"accept_valid": sum(ev_verdicts[s]["verdict"] == "ACCEPT" for s in valid_ids),
                            "accept_faulty": sum(ev_verdicts[s]["verdict"] == "ACCEPT" for s in candidates if s not in valid_ids),
                            "unparseable": sum(v["verdict"] == "UNPARSEABLE" for v in ev_verdicts.values())},
        "llm_verification_tokens": {"self": sum(v["completion_tokens"] for v in verdicts.values()),
                                    "evidence": sum(v["completion_tokens"] for v in ev_verdicts.values())},
    }
    write_json(Path(out) / f"{model}_samples.json", samples)
    write_json(Path(out) / f"{model}_candidates.json", {sid: {k: v for k, v in info.items() if k != "candidate"} | {"critic": verdicts[sid], "evidence_critic": ev_verdicts[sid], "skill": info["candidate"].skill.to_dict()} for sid, info in candidates.items()})
    write_json(Path(out) / f"{model}_per_seed.json", per_seed)
    write_json(Path(out) / f"{model}_summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("clover", "deepseek", "qwen4b"), required=True,
                   help="qwen4b replays Clover run 2 (requested as qwen3.5-4b; served by clover-35b-a3b)")
    p.add_argument("--output", required=True)
    p.add_argument("--n-per-spec", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--offline", action="store_true", help="replay recorded responses only")
    a = p.parse_args()
    print(json.dumps(run(a.model, a.output, a.n_per_spec, a.temperature, a.workers, a.offline), indent=2))


if __name__ == "__main__":
    main()
