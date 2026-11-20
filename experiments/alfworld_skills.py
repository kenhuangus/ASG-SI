"""Skill admission on ALFWorld (text) with candidates written by frozen local models.

Stages (results under --output, default results/review/alfworld):
  prepare  (Linux)   select disjoint train pools per task type and record expert discovery episodes
  generate           each model writes solve(env, goal) from one discovery episode and repairs it from
                     its own discovery episode, up to four programs in total
  execute            run every compiled candidate on its discovery game, the evidence pool, the
                     verification pool, and every validation game of its type (held-out adjudication)
Episodes run under Linux; from Windows they are run in WSL through stdin/stdout (run-batch).
  judge    (any)     same-model self-verification and SmartSnap-style evidence verification
  analyze  (any)     admission outcomes and library success per method, five seeds

No tool, environment rule, or task was added to ALFWorld; faults are whatever the models write.
"""
import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASK_TYPES = ("pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
              "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place")
KINDS = {
    "pick_and_place_simple": "find an object of the requested type and put it in or on a receptacle of the requested type",
    "look_at_obj_in_light": "find an object of the requested type and examine it under a lamp of the requested type by holding it and turning the lamp on",
    "pick_clean_then_place_in_recep": "find an object of the requested type, clean it with a sink basin, and put it in or on a receptacle of the requested type",
    "pick_heat_then_place_in_recep": "find an object of the requested type, heat it with a microwave, and put it in or on a receptacle of the requested type",
    "pick_cool_then_place_in_recep": "find an object of the requested type, cool it with a fridge, and put it in or on a receptacle of the requested type",
    "pick_two_obj_and_place": "find two objects of the requested type and put both in or on a receptacle of the requested type",
}
API_DOC = """Skill interface (Python 3):
  def solve(env, goal): ...
  goal: str                      the task text, for example "put a clean apple in fridge."
  env.observation: str           the most recent observation; before the first command it is the room
                                 description, which lists the receptacles in the room
  env.step(command: str) -> str  executes one text command and returns the feedback (also stored in env.observation)
  env.admissible() -> list[str]  the commands that are valid in the current state
The episode ends as soon as the task is completed. At most 50 commands may be executed. Only the modules
re, collections, itertools, math, and string may be imported. solve must not read or write files."""
ENV_DOC = """Environment rules:
  Commands: "go to X", "open X", "close X", "take O from X", "move O to X", "heat O with X", "cool O with X",
  "clean O with X", "use X", "examine X", "inventory", "look". Invalid commands return "Nothing happens."
  Objects and receptacles carry instance numbers ("apple 2", "drawer 3"); goals name only their types.
  An object may be on any receptacle, including inside closed ones (drawers, cabinets, fridge, microwave, safe);
  a closed receptacle must be opened to see and take its contents. The agent holds at most one object.
  Heating uses a microwave, cooling a fridge, cleaning a sink basin, and lamps are turned on with "use"."""
# In a pilot on two task types, clover-35b-a3b wrote working skills for 1 of 8 tasks with reasoning off and
# 2 of 8 with reasoning on, so Clover runs with reasoning on.
SAMPLES = {"clover": 4, "deepseek": 3}
NO_REASONING = {"chat_template_kwargs": {"enable_thinking": False}}  # honoured by all three servers
REASONING = {"clover": True, "deepseek": False}  # DeepSeek decodes about 18 tokens/s on the Spark


def llm_settings(model):
    """(max_tokens, extra request fields) per model: reasoning models get a larger output budget."""
    return (16384, None) if REASONING[model] else (4096, NO_REASONING)
DOCS = ("documented", "sparse")
MAX_ATTEMPTS = 4          # one program plus up to three repairs from discovery-episode feedback
REPLY = "Reply with one ```python code block that defines solve(env, goal), and nothing else."
VERIFY_THRESHOLD = 9      # replay gate: admit iff the candidate completes at least 9 of 10 verification games
ADEQUATE = 0.8            # held-out adjudication: a skill is adequate iff validation success >= 0.8
SEEDS = range(5)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def episode_text(steps, limit=None):
    lines = []
    for cmd, obs in steps:
        obs = " ".join(obs.split())
        lines.append(f"> {cmd}\n{obs[:limit] if limit else obs}")
    return "\n".join(lines)


# ---------------------------------------------------------------- prepare / execute (Linux)

def prepare(out):
    from experiments import alfworld_harness as h
    sel = h.select(seed=0)
    write_json(Path(out) / "prep.json", {"seed": 0, "step_budget": h.STEP_BUDGET, "pools": sel})
    print({tt: {k: len(v) for k, v in pools.items()} for tt, pools in sel.items()})


def _run(job):
    from experiments import alfworld_harness as h
    return h.run_skill(job["code"], job["game"])


MARK = "@@RESULTS@@"


def run_batch(workers):
    """Linux side: read {"jobs": [...]} from stdin, write results after a marker line on stdout."""
    from multiprocessing import Pool
    jobs = json.loads(sys.stdin.read())["jobs"]
    with Pool(workers, maxtasksperchild=50) as pool:
        results = pool.map(_run, jobs, chunksize=1)
    sys.stdout.write(MARK + "\n" + json.dumps(results) + "\n")


def run_jobs(jobs, workers=1):
    """Run skill episodes. On Windows the episodes run in WSL through stdin/stdout; nothing listens on a port."""
    import subprocess
    if sys.platform == "linux":
        from multiprocessing import Pool
        with Pool(workers) as pool:
            return pool.map(_run, jobs, chunksize=1)
    root = str(ROOT).replace("\\", "/")
    wsl_root = "/mnt/" + root[0].lower() + root[2:]
    cmd = (f"cd {wsl_root} && . ~/alfenv/bin/activate && export ALFWORLD_DATA=~/alfworld_data && "
           f"python -m experiments.alfworld_skills run-batch --workers {workers}")
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-lc", cmd], input=json.dumps({"jobs": jobs}),
                          capture_output=True, text=True, encoding="utf-8")
    if MARK not in proc.stdout:
        raise RuntimeError(f"episode runner failed: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout.split(MARK, 1)[1])


def execute(out, model, workers):
    prep = read_json(Path(out) / "prep.json")["pools"]
    cands = [c for c in read_json(Path(out) / f"candidates_{model}.json") if c["compiled"]]
    jobs = {}
    for c in cands:
        pools = prep[c["task_type"]]
        games = ([c["discovery_game"]] + [e["game"] for e in pools["evidence"]] + pools["verify"] + pools["eval"])
        for g in games:
            jobs[(c["code_sha"], g)] = {"code": c["code"], "game": g}
    keys = sorted(jobs)
    print(f"{model}: {len(cands)} compiled candidates, {len(keys)} episodes", flush=True)
    results = dict(zip(keys, run_jobs([jobs[k] for k in keys], workers)))
    record = {}
    for c in cands:
        pools, key = prep[c["task_type"]], c["code_sha"]
        slim = lambda r: {k: r[k] for k in ("game", "won", "steps", "end", "error")}  # noqa: E731
        record[c["id"]] = {
            "discovery": results[(key, c["discovery_game"])],
            "evidence": {e["game"]: results[(key, e["game"])] for e in pools["evidence"]},
            "verify": [slim(results[(key, g)]) for g in pools["verify"]],
            "eval": [slim(results[(key, g)]) for g in pools["eval"]],
        }
    write_json(Path(out) / f"exec_{model}.json", record)


# ---------------------------------------------------------------- generate / judge (LLM)

def generation_messages(task_type, docs, disc):
    parts = [API_DOC]
    if docs == "documented":
        parts += [ENV_DOC, f"The skill must handle every task of this kind: {KINDS[task_type]}. "
                           "Goal wording varies between tasks of the same kind."]
    parts.append("Below is a successful episode of one such task. Write solve(env, goal) so that it completes this "
                 "task and other tasks of the same kind in other rooms, with other objects and receptacles.\n\n"
                 f"Room: {disc['room']}\nGoal: {disc['goal']}\nEpisode:\n{episode_text(disc['steps'])}")
    return [{"role": "system", "content": "You write reusable Python skills for a text-based household environment. "
                                          "Reply with one ```python code block that defines solve(env, goal), and nothing else."},
            {"role": "user", "content": "\n\n".join(parts)}]


def extract_code(text):
    text = text.split("</think>")[-1]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    for b in reversed(blocks):
        if "def solve" in b:
            return b.strip() + "\n"
    if "def solve" in text and "```" not in text:
        return text.strip() + "\n"
    return None


def client_for(model, out, offline):
    from asg_si.llm import ChatClient
    return ChatClient(model, Path(out) / "cache", offline=offline)


def generate(out, model, offline, workers, temperature=0.8):
    from concurrent.futures import ThreadPoolExecutor
    prep = read_json(Path(out) / "prep.json")["pools"]
    client = client_for(model, out, offline)
    budget, extra = llm_settings(model)
    jobs = [(tt, d, i) for tt in TASK_TYPES for d in DOCS for i in range(SAMPLES[model])]

    def one(job):
        """Write a skill, run it on the discovery task, and repair it from the episode (Voyager-style),
        for at most MAX_ATTEMPTS programs. The candidate is the first program that completes the task."""
        tt, d, i = job
        disc = prep[tt]["discovery"][i % len(prep[tt]["discovery"])]
        messages = generation_messages(tt, d, disc)
        attempts, code, tokens = [], None, 0
        for _ in range(MAX_ATTEMPTS):
            rec = client.chat(messages, temperature=temperature, max_tokens=budget, sample=i, extra=extra)
            tokens += rec["completion_tokens"]
            code = extract_code(rec["content"])
            if not code:
                attempts.append({"code_sha": None, "finish_reason": rec["finish_reason"], "end": "no_code"})
                feedback = "No ```python code block defining solve(env, goal) was found. " + REPLY
            else:
                run = run_jobs([{"code": code, "game": disc["game"]}])[0]
                attempts.append({"code_sha": sha(code), "finish_reason": rec["finish_reason"], "end": run["end"],
                                 "steps": run["steps"], "error": run["error"]})
                if run["won"]:
                    break
                feedback = (f"The skill was run on the example task. The episode ended by: {run['end']}"
                            f"{' (' + run['error'] + ')' if run['error'] else ''}; commands used: {run['steps']}; "
                            f"the task was not completed.\nEpisode:\n{episode_text(run['trace'], 200) or '(no commands)'}"
                            f"\n\nFix the skill so that it completes this task and other tasks of the same kind. {REPLY}")
            messages = messages + [{"role": "assistant", "content": rec["content"]},
                                   {"role": "user", "content": feedback}]
        compiled = bool(attempts and attempts[-1]["end"] == "won")
        return {"model": model, "task_type": tt, "docs": d, "sample": i, "discovery_game": disc["game"],
                "code": code if compiled else None, "code_sha": sha(code) if compiled else None,
                "compiled": compiled, "attempts": attempts, "completion_tokens": tokens,
                "finish_reason": attempts[-1]["finish_reason"], "id": f"{model}-{tt}-{d}-{i}"}

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(one, jobs))
    write_json(Path(out) / f"candidates_{model}.json", rows)
    print(f"{model}: {sum(r['compiled'] for r in rows)}/{len(rows)} compiled; stats {client.stats}")


def critic_messages(c, disc_run):
    return [{"role": "system", "content": "You review skills before they are stored in a reusable skill library. "
                                          "Reply with a single JSON object and nothing else."},
            {"role": "user", "content": (f"{API_DOC}\n\n{ENV_DOC}\n\nTask kind: {KINDS[c['task_type']]}.\n\n"
                                         f"Candidate skill:\n```python\n{c['code']}```\n\nExecution on one task "
                                         f"(goal: {disc_run['goal']}):\n{episode_text(disc_run['trace'], 200)}\n\n"
                                         "Will this skill complete every task of this kind, in any room, within 50 "
                                         "commands? Reply {\"verdict\": \"ACCEPT\" or \"REJECT\", \"reason\": \"<one sentence>\"}.")}]


def evidence_request(c, pool):
    tasks = "\n".join(f"{i}: goal \"{e['goal']}\"; room: {e['room']}" for i, e in enumerate(pool))
    return [{"role": "system", "content": "You test skills. Reply with a single JSON object and nothing else."},
            {"role": "user", "content": (f"{API_DOC}\n\n{ENV_DOC}\n\nTask kind: {KINDS[c['task_type']]}.\n\n"
                                         f"Skill:\n```python\n{c['code']}```\n\nAvailable test tasks:\n{tasks}\n\n"
                                         "Choose exactly 3 test tasks that would best reveal whether this skill is "
                                         "correct. Reply {\"tests\": [three task numbers]}.")}]


def evidence_judgement(c, observed):
    ev = "\n\n".join(f"Task {o['index']} (goal: {o['goal']}); episode ended by: {o['ended']}; commands used: "
                     f"{o['steps']}\n{episode_text(o['trace'], 200)}" for o in observed)
    return [{"role": "system", "content": "You verify skills from execution evidence. Reply with a single JSON object and nothing else."},
            {"role": "user", "content": (f"{ENV_DOC}\n\nTask kind: {KINDS[c['task_type']]}.\n\nSkill:\n```python\n"
                                         f"{c['code']}```\n\nEvidence (tasks chosen by the skill's author, with the "
                                         f"skill's actual episodes):\n{ev}\n\nDid the skill complete each task? Should it "
                                         "be stored for reuse on every task of this kind? Reply {\"verdict\": \"ACCEPT\" or "
                                         "\"REJECT\", \"reason\": \"<one sentence>\"}.")}]


def _verdict(content):
    from asg_si.llm import extract_json
    obj = extract_json(content) or {}
    v = str(obj.get("verdict", "")).upper()
    return (v if v in ("ACCEPT", "REJECT") else "UNPARSEABLE"), obj.get("reason")


def judge(out, model, offline, workers):
    from concurrent.futures import ThreadPoolExecutor
    from asg_si.llm import extract_json
    prep = read_json(Path(out) / "prep.json")["pools"]
    cands = {c["id"]: c for c in read_json(Path(out) / f"candidates_{model}.json") if c["code"]}
    runs = read_json(Path(out) / f"exec_{model}.json")
    client = client_for(model, out, offline)
    budget, extra = llm_settings(model)
    todo = [cid for cid in sorted(cands) if runs[cid]["discovery"]["won"]]

    def one(cid):
        c, r = cands[cid], runs[cid]
        rec = client.chat(critic_messages(c, r["discovery"]), temperature=0.0, max_tokens=budget, extra=extra)
        sv, sv_reason = _verdict(rec["content"])
        tokens = {"self": rec["completion_tokens"]}
        pool = prep[c["task_type"]]["evidence"]
        req = client.chat(evidence_request(c, pool), temperature=0.0, max_tokens=budget, extra=extra)
        picks = (extract_json(req["content"]) or {}).get("tests") or []
        idx = []
        for p in picks:
            try:
                p = int(p)
            except (TypeError, ValueError):
                continue
            if 0 <= p < len(pool) and p not in idx:
                idx.append(p)
        idx = idx[:3]
        observed = [{"index": i, "goal": pool[i]["goal"], "ended": "task completed" if runs[cid]["evidence"][pool[i]["game"]]["won"]
                     else runs[cid]["evidence"][pool[i]["game"]]["end"], "steps": runs[cid]["evidence"][pool[i]["game"]]["steps"],
                     "trace": runs[cid]["evidence"][pool[i]["game"]]["trace"]} for i in idx]
        tokens["evidence"] = req["completion_tokens"]
        if observed:
            jr = client.chat(evidence_judgement(c, observed), temperature=0.0, max_tokens=budget, extra=extra)
            ev, ev_reason = _verdict(jr["content"])
            tokens["evidence"] += jr["completion_tokens"]
        else:
            ev, ev_reason = "UNPARSEABLE", "no usable test choice"
        return cid, {"self_verification": sv, "self_reason": sv_reason, "evidence_verification": ev,
                     "evidence_reason": ev_reason, "evidence_tasks": idx,
                     "evidence_all_completed": all(o["ended"] == "task completed" for o in observed) if observed else None,
                     "completion_tokens": tokens}

    with ThreadPoolExecutor(workers) as pool:
        verdicts = dict(pool.map(one, todo))
    write_json(Path(out) / f"judge_{model}.json", verdicts)
    print(f"{model}: judged {len(verdicts)}; stats {client.stats}")


# ---------------------------------------------------------------- analyze

def rate(xs):
    return sum(xs) / len(xs) if xs else 0.0


def analyze(out, model):
    prep = read_json(Path(out) / "prep.json")["pools"]
    cands = read_json(Path(out) / f"candidates_{model}.json")
    runs = read_json(Path(out) / f"exec_{model}.json")
    verdicts = read_json(Path(out) / f"judge_{model}.json")
    rows = []
    for c in cands:
        r = runs.get(c["id"])
        row = {"id": c["id"], "task_type": c["task_type"], "docs": c["docs"], "has_code": bool(c["code"]),
               "finish_reason": c["finish_reason"]}
        if r:
            ev = [e["won"] for e in r["eval"]]
            row.update(discovery_won=r["discovery"]["won"], verify_wins=sum(v["won"] for v in r["verify"]),
                       eval_success=rate(ev), eval_n=len(ev),
                       eval_ends={k: sum(e["end"] == k for e in r["eval"]) for k in sorted({e["end"] for e in r["eval"]})})
            row["adequate"] = row["eval_success"] >= ADEQUATE
        v = verdicts.get(c["id"], {})
        compiled = bool(r and r["discovery"]["won"])
        row["admit"] = {
            "first_success": compiled,
            "llm_self_verification": compiled and v.get("self_verification") == "ACCEPT",
            "llm_evidence_verification": compiled and v.get("evidence_verification") == "ACCEPT",
            "independent_replay": compiled and row.get("verify_wins", 0) >= VERIFY_THRESHOLD,
        }
        row["verdicts"] = {k: v.get(k) for k in ("self_verification", "evidence_verification", "evidence_all_completed")}
        rows.append(row)
    compiled = [r for r in rows if r.get("discovery_won")]
    methods = {}
    total_eval = sum(len(prep[tt]["eval"]) for tt in TASK_TYPES)
    for m in ("first_success", "llm_self_verification", "llm_evidence_verification", "independent_replay"):
        adm = [r for r in compiled if r["admit"][m]]
        lib = []
        for s in SEEDS:
            wins = 0
            for tt in TASK_TYPES:
                pool = [r for r in compiled if r["task_type"] == tt]
                random.Random(f"{s}-{tt}").shuffle(pool)
                pick = next((r for r in pool if r["admit"][m]), None)
                if pick:
                    wins += pick["eval_success"] * pick["eval_n"]
            lib.append(wins / total_eval)
        methods[m] = {"admitted": len(adm), "admitted_inadequate": sum(not r["adequate"] for r in adm),
                      "rejected_adequate": sum(r["adequate"] and not r["admit"][m] for r in compiled),
                      "mean_eval_success_admitted": rate([r["eval_success"] for r in adm]),
                      "library_success_by_seed": lib, "library_success_mean": statistics.mean(lib),
                      "library_success_range": [min(lib), max(lib)]}
    summary = {"model": model, "samples": len(rows), "with_code": sum(r["has_code"] for r in rows),
               "compiled": len(compiled), "adequate_compiled": sum(r["adequate"] for r in compiled),
               "inadequate_compiled": sum(not r["adequate"] for r in compiled),
               "eval_games": total_eval, "verify_threshold": f"{VERIFY_THRESHOLD}/10", "adequate_threshold": ADEQUATE,
               "by_type": {tt: {"compiled": sum(r["task_type"] == tt for r in compiled),
                                "adequate": sum(r["task_type"] == tt and r["adequate"] for r in compiled),
                                "best_eval_success": max([r["eval_success"] for r in compiled if r["task_type"] == tt], default=None)}
                           for tt in TASK_TYPES},
               "by_docs": {d: {"samples": sum(r["docs"] == d for r in rows), "compiled": sum(r["docs"] == d for r in compiled),
                               "adequate": sum(r["docs"] == d and r["adequate"] for r in compiled)} for d in DOCS},
               "methods": methods}
    write_json(Path(out) / f"summary_{model}.json", summary)
    write_json(Path(out) / f"rows_{model}.json", rows)
    print(json.dumps(summary, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("prepare", "generate", "execute", "judge", "analyze", "run-batch"))
    p.add_argument("--model", choices=tuple(SAMPLES))
    p.add_argument("--output", default=str(ROOT / "results" / "review" / "alfworld"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--offline", action="store_true", help="replay recorded model responses only")
    a = p.parse_args()
    if a.stage == "run-batch":
        run_batch(a.workers)
    elif a.stage == "prepare":
        prepare(a.output)
    elif a.stage == "generate":
        generate(a.output, a.model, a.offline, a.workers)
    elif a.stage == "execute":
        execute(a.output, a.model, a.workers)
    elif a.stage == "judge":
        judge(a.output, a.model, a.offline, a.workers)
    else:
        analyze(a.output, a.model)


if __name__ == "__main__":
    main()
