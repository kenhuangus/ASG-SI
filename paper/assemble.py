"""Fill paper_body.md placeholders from result files, expand citations, and enforce citation/reference parity."""
import json
import re
import sys
from pathlib import Path

import references as R

HERE = Path(__file__).parent
ROOT = HERE.parent
RES = ROOT / "results" / "review"


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def pct(x, d=1):
    return f"{100 * x:.{d}f} %"


def md_table(header, rows, widths=None, caption=""):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    if widths:
        out.append("Widths: " + ",".join(str(w) for w in widths))
    out.append("Table: " + caption)
    return "\n".join(out)


def ci(m, scale=100, d=1):
    lo, hi = m["ci95"]
    s = f"{scale * m['mean']:.{d}f}"
    return s if abs(hi - lo) < 1e-12 else f"{s} ({scale * lo:.{d}f}–{scale * hi:.{d}f})"


METHOD_NAMES = {"first_success": "Admit after first success", "self_verification": "In-process checking",
                "independent_replay": "Independent replay", "independent_composition": "Replay + composition checks",
                "llm_self_verification": "LLM self-verification", "llm_evidence_verification": "LLM evidence verification (SmartSnap-style)"}


def table_main():
    rep = load("main_report.json")
    rows = []
    for method in ("first_success", "self_verification", "independent_replay", "independent_composition"):
        m = next(s["metrics"] for s in rep["summary"] if s["method"] == method and s["phase"] == "static")
        rows.append([METHOD_NAMES[method], ci(m["faulty_admission_rate"]), ci(m["valid_rejection_rate"]),
                     ci(m["downstream_success"]), ci(m["stress_success"]), ci(m["constraint_violation_rate"]),
                     f"{m['verification_tool_calls']['mean']:.0f}"])
    return md_table(["Admission method", "Faulty admitted %", "Valid rejected %", "Eval. success %", "Stress success %", "Violations %", "Verif. tool calls"],
                    rows, [1.6, 0.75, 0.75, 0.9, 0.9, 0.85, 0.65],
                    "Table 3: Controlled fault benchmark, mean over five run seeds with 95 % run-bootstrap intervals where nonzero. Per method: 35 faulty and 15 valid decisions, 150 evaluation and 100 stress tasks")


# "qwen4b" is the cache and file key of Clover run 2: requested as qwen3.5-4b, served by clover-35b-a3b.
MODELS = (("clover", "Clover, run 1"), ("qwen4b", "Clover, run 2"), ("deepseek", "DeepSeek (748B)"))


def llm(model):
    p = RES / f"llm_{model}_summary.json"
    if not p.exists():
        return None
    return load(f"llm_{model}_summary.json"), load(f"llm_{model}_per_seed.json")


def table_llm_gen():
    rows = []
    for key, name in MODELS:
        d = llm(key)
        if not d:
            continue
        s = d[0]
        samples = load(f"llm_{key}_samples.json")
        length = sum(r.get("finish_reason") == "length" and r["status"] in ("unparseable", "provider_error") for r in samples)
        other = sum(r["status"] in ("unparseable", "provider_error", "not_compilable") for r in samples) - length
        failed = sum(r["status"] == "discovery_failed" for r in samples)
        sub = sum(r["status"] == "submitted" for r in samples)
        rows.append([name, s["samples"], length, other, failed, sub,
                     s["unique_candidates"], s["valid_unique"], s["faulty_unique"], pct(s["no_library_success"])])
    return md_table(["Model", "Samples", "Over token budget", "Other errors", "Failed example", "Submitted", "Unique", "Valid", "Faulty", "Fresh-sample success"],
                    rows, [1.1, 0.6, 0.62, 0.55, 0.6, 0.72, 0.55, 0.45, 0.5, 0.71],
                    "Table 4: Candidate generation by frozen models over six prompt conditions. Fresh-sample success is the share of all samples whose program is valid on the hidden suite, which equals expected success when every task receives a new sample and no library")


def table_llm_admit():
    rows = []
    for key, name in MODELS:
        d = llm(key)
        if not d:
            continue
        s, per = d
        for method in ("first_success", "llm_self_verification", "llm_evidence_verification", "independent_replay", "independent_composition"):
            rs = [r for r in per if r["method"] == method]
            if not rs:
                continue
            n = len(rs)
            fa = sum(r["faulty_admitted"] for r in rs) / n
            vr = sum(r["valid_rejected"] for r in rs) / n
            succ = [r["downstream_success"] for r in rs]
            calls = sum(r["verification_tool_calls"] for r in rs) / n
            tok = s["llm_verification_tokens"]["self" if method == "llm_self_verification" else "evidence"] if method.startswith("llm_") and "llm_verification_tokens" in s else 0
            rng = f"{pct(sum(succ) / n)}" + ("" if max(succ) - min(succ) < 1e-12 else f" ({100 * min(succ):.1f}–{100 * max(succ):.1f})")
            rows.append([name, METHOD_NAMES[method], f"{fa:g} / {rs[0]['faulty']}", f"{vr:g} / {rs[0]['valid']}", rng, f"{calls:.0f}", f"{tok:,}" if tok else "0"])
    return md_table(["Model", "Admission method", "Faulty admitted", "Valid rejected", "Eval. success (range over seeds)", "Verif. tool calls", "Verif. LLM tokens"],
                    rows, [1.0, 1.65, 0.7, 0.7, 1.0, 0.6, 0.75],
                    "Table 5: Admission of model-written candidates, mean over five seeds (LLM verdicts do not depend on the seed). Counts are unique compiled candidates; success uses each admitted library with uniform selection on 30 evaluation tasks per seed")


def table_stoch():
    s = load("stochastic.json")
    names = {"single_read": "one read", "two_reads": "two reads", "three_reads": "three reads", "unit_blind_three_reads": "three reads, ignores units"}
    rows = []
    for c in s["candidates"]:
        g = c["gates"]
        rows.append([names[c["candidate"]], f"{c['true_success']:.3f}", "yes" if c["meets_requirement"] else "no",
                     pct(g["exact_single_replay"]["acceptance_rate"]), pct(g["exact_all_of_4_replays"]["acceptance_rate"]),
                     pct(g["statistical_cp"]["acceptance_rate"]), f"{g['statistical_cp']['mean_executions']:.0f}"])
    return md_table(["Candidate", "True success", "Meets tau = 0.90", "Exact, 1 replay", "Exact, all of 4", "Statistical gate", "Statistical executions"],
                    rows, [1.55, 0.7, 0.75, 0.8, 0.8, 0.8, 0.9],
                    "Table 6: Acceptance rates over 500 gate trials per candidate with stochastic reads (p_drop = 0.1)")


def table_depth():
    s = load("composition_depth.json")["summary"]
    rows = [[r["chain_length"], r["chains"], r["incompatible"], pct(r["flat_incorrect_task_rate"]),
             r["incompatible_chains_correct_on_all_tasks"], r["replay5_incompatible_admitted"], f"{r['replay5_test_calls_per_chain']:.0f}",
             r["checked_incompatible_admitted"], r["checked_compatible_rejected"]] for r in s]
    return md_table(["Length", "Chains", "Incompat.", "Flat: wrong tasks", "Incompat., always correct", "Replay-5: incompat. admitted", "Replay-5: calls per chain", "Check: incompat. admitted", "Check: compat. rejected"],
                    rows, [0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.75, 0.75, 0.7],
                    "Table 7: Composition at depth, 20 evaluation tasks per chain. The interface check uses no tool calls")


def table_sel():
    s = load("selection.json")["summary"]
    names = {"uniform": "uniform", "fewest_steps": "fewest steps", "evidence_ranked": "evidence-ranked", "runtime_feedback": "runtime feedback (UCB)"}
    rows = []
    for r in s:
        rng = pct(r["success"]) if r["success_max"] - r["success_min"] < 1e-12 else f"{pct(r['success'])} ({100 * r['success_min']:.1f}–{100 * r['success_max']:.1f})"
        rows.append([r["library_size"], r["admission"], names[r["rule"]], f"{r['admitted']:.0f} ({r['admitted_faulty']:.0f})", rng, pct(r["violation_rate"]), f"{r['tool_calls_per_task']:.2f}"])
    return md_table(["Library size", "Admission", "Selection rule", "Admitted (faulty)", "Success (range over seeds)", "Violations", "Tool calls per task"],
                    rows, [0.65, 0.75, 1.35, 0.9, 1.3, 0.7, 0.75],
                    "Table 8: Selection rules in libraries of 10 to 200 skills, mean over five seeds with 200 tasks each")


def table_tamper():
    s = load("tamper.json")["evidence_attacks"]
    short = lambda v: "detected" if v.startswith("detected") else "not detected"
    order = ["no attack", "edit decision only", "edit decision and recompute manifest hashes", "delete test inputs",
             "compromised verifier writes forged decision", "compromised verifier records modified source"]
    assert set(order) == set(s), "attack names changed"
    rows = [[k, short(s[k]["hashes"]), short(s[k]["anchor"]), short(s[k]["signature"]), short(s[k]["replay"])] for k in order]
    return md_table(["Attack", "Artifact hashes", "External anchor", "HMAC signature", "Audit re-execution"],
                    rows, [2.4, 1.0, 1.0, 1.0, 1.0],
                    "Table 9: Detection of evidence attacks by each defence layer used alone")


def table_ret():
    s = load("retention.json")
    rows = []
    for cond, name in (("unpinned", "unpinned"), ("pinned", "pinned, no lifecycle"), ("pinned_lifecycle", "pinned with revocation and re-verification")):
        series = s["results"][cond]["series"]
        before = [x["success"] for x in series if x["checkpoint"] < s["change_at_checkpoint"]]
        after = [x["success"] for x in series if x["checkpoint"] >= s["change_at_checkpoint"]]
        errs = sum(x["observable_errors"] for x in series if x["checkpoint"] >= s["change_at_checkpoint"])
        rows.append([name, pct(sum(before) / len(before)), pct(sum(after) / len(after)), errs])
    return md_table(["Condition", "Success before change", "Success after change", "Observable errors after change"],
                    rows, [2.6, 1.2, 1.2, 1.4],
                    "Table 10: Retention across ten checkpoints of 30 tasks; the tool change takes effect at checkpoint 4")


def worker_overhead():
    rep = load("main_report.json")
    get = lambda m: next(s["metrics"]["verification_ms"]["mean"] for s in rep["summary"] if s["method"] == m and s["phase"] == "static")
    return f"about {round(get('independent_replay') - get('self_verification'), -1):.0f} ms"


def poison_violations():
    rows = load("tamper.json")["poisoning"]
    v = [r["deployed_without_monitor"]["violations"] for r in rows]
    return f"{min(v)} to {max(v)}"


def label(key):
    if key in R.MANUAL:
        manual = {"clopper1934": ("Clopper & Pearson", 1934), "necula1997pcc": ("Necula", 1997), "sha2001simplicity": ("Sha", 2001),
                  "torresarias2019intoto": ("Torres-Arias et al.", 2019)}
        return manual[key]
    aid, venue, *year = R.ARXIV[key]
    m = R.BY_ID[aid]
    fam = [a[0] for a in m["authors"]]
    who = fam[0] if len(fam) == 1 else (f"{fam[0]} & {fam[1]}" if len(fam) == 2 else f"{fam[0]} et al.")
    return who, (year[0] if year else m["year"])


def cite(match):
    keys = [k.strip().lstrip("@") for k in match.group(1).split(";")]
    cite.used.update(keys)
    return "(" + "; ".join(f"{a}, {y}" for a, y in map(label, keys)) + ")"


cite.used = set()


def main():
    body = (HERE / "paper_body.md").read_text(encoding="utf-8")
    texts = {}
    tp = HERE / "paper_llm_text.md"
    if tp.exists():
        for block in tp.read_text(encoding="utf-8").split("\n@@")[1:]:
            name, _, content = block.partition("\n")
            texts[name.strip()] = content.strip()
    fills = {"TABLE_MAIN": table_main, "TABLE_LLM_GEN": table_llm_gen, "TABLE_LLM_ADMIT": table_llm_admit,
             "TABLE_STOCH": table_stoch, "TABLE_DEPTH": table_depth, "TABLE_SEL": table_sel, "TABLE_TAMPER": table_tamper,
             "TABLE_RET": table_ret, "POISON_VIOL": poison_violations, "WORKER_OVERHEAD": worker_overhead}
    for k, fn in fills.items():
        body = body.replace("{{" + k + "}}", fn())
    for k, v in texts.items():
        body = body.replace("{{" + k + "}}", v)
    body = re.sub(r"\[(@[^\]]+)\]", cite, body)
    entries = R.all_entries()
    unused = set(entries) - cite.used
    unknown = cite.used - set(entries)
    if unused or unknown:
        sys.exit(f"citation parity failure: uncited={sorted(unused)} unknown={sorted(unknown)}")
    body = body.replace("{{REFERENCES}}", "\n".join("REF " + v for v in entries.values()))
    left = re.findall(r"\{\{[A-Z_]+\}\}", body)
    if left:
        print("WARNING unfilled:", sorted(set(left)))
    (HERE / "paper.md").write_text(body, encoding="utf-8")
    print("paper.md written;", len(cite.used), "references cited")


if __name__ == "__main__":
    main()
