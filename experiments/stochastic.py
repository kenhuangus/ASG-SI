"""Admission under stochastic, partially observable tools.

`read_records_noisy` returns each record independently with probability 1 - p_drop, so a single
call observes only part of the input. Candidate programs differ in how many independent reads they
reconcile. Exact single replay cannot distinguish an adequate skill from a lucky one; requiring every
repeated replay to pass rejects adequate skills. The statistical gate fixes a reliability requirement
tau in advance and accepts only when the one-sided Clopper-Pearson lower bound on the success rate is
at least tau (sequential: 60 executions, then up to 240 when the interval straddles tau).
Every execution is seeded, so evidence can be replayed exactly from the recorded seeds.
"""
import argparse
import json
import math
import random
from pathlib import Path

from asg_si.contracts import ContractError, ToolContract, check_arguments, check_output
from asg_si.evidence import write_json
from asg_si.executor import execute
from asg_si.oracle import assess
from asg_si.splits import BOUNDARIES, make_case
from asg_si.tools import ToolRegistry
from asg_si.types import Skill, clone

TAU = 0.90          # prespecified minimum acceptable success rate
ALPHA = 0.05        # one-sided error for the lower confidence bound
P_DROP = 0.10


class NoisyRegistry(ToolRegistry):
    VERSION = "volume-tools-noisy-1"

    def __init__(self, seed, p_drop=P_DROP):
        super().__init__()
        self.rng, self.p_drop = random.Random(seed), p_drop
        self.contracts["read_records_noisy"] = ToolContract({"records": "list"}, "list")
        self.contracts["reconcile2"] = ToolContract({"a": "list", "b": "list"}, "list")
        self.contracts["reconcile3"] = ToolContract({"a": "list", "b": "list", "c": "list"}, "list")

    def call(self, name, arguments, observation, semantic):
        if name == "read_records_noisy":
            check_arguments(self.contracts[name], arguments, semantic)
            out = [clone(r) for r in arguments["records"] if self.rng.random() >= self.p_drop]
            return out, [], 0
        if name in ("reconcile2", "reconcile3"):
            check_arguments(self.contracts[name], arguments, semantic)
            merged = {}
            for key in sorted(arguments):
                for row in arguments[key]:
                    merged.setdefault(row["id"], clone(row))
            out = [merged[k] for k in sorted(merged)]
            check_output(self.contracts[name], out, semantic)
            return out, [], 0
        return super().call(name, arguments, observation, semantic)


def program(name, reads):
    steps = [{"id": f"read{i}", "tool": "read_records_noisy", "args": {"records": {"input": "records"}}} for i in range(reads)]
    source = "read0"
    if reads > 1:
        args = {k: {"step": f"read{i}"} for i, k in enumerate("abc"[:reads])}
        steps.append({"id": "merge", "tool": f"reconcile{reads}", "args": args})
        source = "merge"
    steps += [{"id": "validate", "tool": "validate_fields", "args": {"records": {"step": source}}},
              {"id": "convert", "tool": "convert_liters", "args": {"records": {"step": "validate"}}},
              {"id": "aggregate", "tool": "aggregate", "args": {"batch": {"step": "convert"}}}]
    return Skill(name, tuple(steps), "aggregate", {s["tool"]: "1.0.0" for s in steps})


def unit_blind():
    steps = ({"id": "read0", "tool": "read_records_noisy", "args": {"records": {"input": "records"}}},
             {"id": "read1", "tool": "read_records_noisy", "args": {"records": {"input": "records"}}},
             {"id": "read2", "tool": "read_records_noisy", "args": {"records": {"input": "records"}}},
             {"id": "merge", "tool": "reconcile3", "args": {"a": {"step": "read0"}, "b": {"step": "read1"}, "c": {"step": "read2"}}},
             {"id": "validate", "tool": "validate_fields", "args": {"records": {"step": "merge"}}},
             {"id": "total", "tool": "sum_values", "args": {"records": {"step": "validate"}}})
    return Skill("unit_blind_three_reads", steps, "total", {s["tool"]: "1.0.0" for s in steps})


CANDIDATES = [program("single_read", 1), program("two_reads", 2), program("three_reads", 3), unit_blind()]


def make_cases(seed, n):
    rng, cases = random.Random(seed), []
    for i in range(n):
        boundary = BOUNDARIES[i % len(BOUNDARIES)]
        k = 0 if boundary == "empty" else rng.randint(2, 6)
        records = []
        for j in range(k):
            unit = "L" if boundary == "ordinary" else ("mL" if rng.random() < .5 else "L")
            value = rng.randint(1, 100000) / 1000 * (-1 if boundary == "negative" and j == 0 else 1)
            records.append({"id": j, "value": value, "unit": unit, "valid": not (boundary == "invalid" and j == 0)})
        cases.append(make_case({"records": records, "destination": "report", "allowed_destination": "report", "case": i}, boundary))
    return cases


def run_once(skill, case, seed):
    return assess(case, execute(skill, case.observation, NoisyRegistry(seed)))["success"]


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def cp_lower(k, n, alpha=ALPHA):
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if 1 - binom_cdf(k - 1, n, mid) < alpha:   # P(X >= k | mid) < alpha  -> mid too small
            lo = mid
        else:
            hi = mid
    return lo


def cp_upper(k, n, alpha=ALPHA):
    if k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) < alpha:            # P(X <= k | mid) < alpha  -> mid too large
            hi = mid
        else:
            lo = mid
    return hi


def gate(skill, cases, rng, kind):
    seeds = lambda m: [rng.getrandbits(48) for _ in range(m)]
    if kind == "exact_single_replay":
        runs = [run_once(skill, c, s) for c, s in zip(cases, seeds(len(cases)))]
        return ("ACCEPT" if all(runs) else "REJECT"), len(runs)
    if kind == "exact_all_of_4_replays":
        runs = [run_once(skill, c, s) for c in cases for s in seeds(4)]
        return ("ACCEPT" if all(runs) else "REJECT"), len(runs)
    if kind == "statistical_cp":
        k = n = 0
        for budget in (60, 240):
            while n < budget:
                c = cases[n % len(cases)]
                k += run_once(skill, c, rng.getrandbits(48))
                n += 1
            if cp_lower(k, n) >= TAU:
                return "ACCEPT", n
            if cp_upper(k, n) < TAU:
                return "REJECT", n
        return "REJECT", n                           # still inconclusive at the maximum budget
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--mc", type=int, default=20000)
    a = p.parse_args()
    eval_cases = make_cases(1001, 200)
    rows = []
    for skill in CANDIDATES:
        rng = random.Random(7)
        true_p = sum(run_once(skill, eval_cases[i % len(eval_cases)], rng.getrandbits(48)) for i in range(a.mc)) / a.mc
        row = {"candidate": skill.name, "true_success": true_p, "meets_requirement": true_p >= TAU, "gates": {}}
        for kind in ("exact_single_replay", "exact_all_of_4_replays", "statistical_cp"):
            decisions, cost = [], 0
            for t in range(a.trials):
                rng = random.Random(10_000 + t)
                promo = make_cases(5000 + t, 15)
                d, n = gate(skill, promo, rng, kind)
                decisions.append(d == "ACCEPT")
                cost += n
            row["gates"][kind] = {"acceptance_rate": sum(decisions) / a.trials, "mean_executions": cost / a.trials}
        rows.append(row)
        print(json.dumps(row))
    summary = {"tau": TAU, "alpha": ALPHA, "p_drop": P_DROP, "trials": a.trials, "monte_carlo_executions": a.mc,
               "evaluation_cases": len(eval_cases), "candidates": rows,
               "scope": "synthetic stochastic tool environment; true success estimated by Monte Carlo on a fixed evaluation distribution"}
    write_json(Path(a.output), summary)


if __name__ == "__main__":
    main()
