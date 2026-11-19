from contextlib import redirect_stdout
import copy
from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from asg_si.audit_replay import audit_replay
from asg_si.benchmark import make_candidates, workflow
from asg_si.compiler import compile_trajectory
from asg_si.evaluation import adjudicate_candidate, evaluate_checkpoint
from asg_si.evidence import EvidenceError, write_bundle, write_json
from asg_si.executor import execute
from asg_si.graph import SkillGraph
from asg_si.oracle import assess
from asg_si.policies import FrozenLLMPolicy, PolicyError, ScriptedPolicy
from asg_si.rewards import DEFAULT_WEIGHTS, compute_reward
from asg_si.splits import assert_disjoint, derive_seed, make_splits
from asg_si.types import ACCEPT, REJECT, INSUFFICIENT_EVIDENCE, digest
from asg_si.verifier import GateConfig, verify, verify_in_process
from asg_si_demo import ASGSISystem, ModelBasedAgentPolicy, SkillCompiler, Step, Task, ToolCall, Trajectory, VerifiableEnv
from analyze_results import analyze
from experiments.audit import audit_run
from experiments.composition import compose, prepare_graph, primitives, run_composition
from experiments.report import generate_report, load_jsonl
from experiments.run import run


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = make_splits(7)
        cls.candidates, cls.annotations = make_candidates(cls.splits["discovery"])

    def candidate(self, fault):
        return next(c for c in self.candidates if self.annotations[c.submission_id] == fault)

    def test_independent_hidden_labels(self):
        labels = [adjudicate_candidate(c, self.splits["evaluation"])["valid"] for c in self.candidates]
        self.assertEqual(labels, [True]*3+[False]*7)

    def test_first_success_accepts_all_but_replay_rejects_faults(self):
        for c in self.candidates:
            self.assertEqual(verify(c, [], GateConfig(method="first_success"))["status"], ACCEPT)
            decision = verify(c, self.splits["promotion"], GateConfig())
            self.assertEqual(decision["status"], ACCEPT if self.annotations[c.submission_id] == "none" else REJECT)
            self.assertEqual(decision["worker"], "separate_python_process")

    def test_identical_checks_in_separate_process_produce_identical_results(self):
        for c in self.candidates:
            a = verify(c, self.splits["promotion"], GateConfig(method="self_verification"))
            b = verify(c, self.splits["promotion"], GateConfig())
            for key in ("status", "reasons", "results", "num_tests", "tool_calls"):
                self.assertEqual(a[key], b[key])

    def test_insufficient_count_even_when_available_tests_pass(self):
        d = verify_in_process(self.candidates[0], self.splits["promotion"][:2], GateConfig())
        self.assertEqual(d["status"], INSUFFICIENT_EVIDENCE)
        self.assertEqual(d["num_tests"], 0)

    def test_insufficient_boundary_coverage(self):
        cases = [c for c in self.splits["promotion"] if c.boundary == "ordinary"]
        self.assertEqual(verify_in_process(self.candidates[0], cases, GateConfig(min_cases=2))["status"], INSUFFICIENT_EVIDENCE)

    def test_duplicate_content_with_different_ids_is_insufficient(self):
        case = self.splits["promotion"][0]
        cases = [replace(case, observation=replace(case.observation, task_id=str(i))) for i in range(10)]
        self.assertIn("duplicate_test_content", verify_in_process(self.candidates[0], cases, GateConfig())["reasons"])

    def test_digest_and_version_faults_reject_before_execution(self):
        for fault in ("modified_artifact", "changed_dependency"):
            d = verify_in_process(self.candidate(fault), self.splits["promotion"], GateConfig())
            self.assertEqual((d["status"], d["tool_calls"]), (REJECT, 0))

    def test_unit_mismatch_is_blocked_even_when_total_zero(self):
        c = self.candidate("unit_mismatch")
        flat = execute(c.skill, c.discovery[0].observation)
        checked = execute(c.skill, c.discovery[0].observation, composition=True)
        self.assertTrue(assess(c.discovery[0], flat)["success"])
        self.assertIn("Post_A", checked.error)
        self.assertEqual(checked.tool_calls, 0)

    def test_correct_output_can_violate_constraints(self):
        c = self.candidate("constraint_violation")
        grade = assess(c.discovery[0], execute(c.skill, c.discovery[0].observation))
        self.assertTrue(grade["outcome_correct"])
        self.assertTrue(grade["constraint_violation"])
        self.assertFalse(grade["success"])

    def test_step_budget_and_unavailable_binding_fail(self):
        case, skill = self.splits["promotion"][0], workflow()
        self.assertIn("budget", execute(skill, case.observation, max_steps=2).error)
        bad = replace(skill, steps=({"id": "bad", "tool": "aggregate", "args": {"batch": {"step": "future"}}},))
        self.assertIsNotNone(execute(bad, case.observation, enforce_pins=False).error)

    def test_policy_only_accepts_public_observations(self):
        case = self.splits["evaluation"][0]
        self.assertFalse(hasattr(case.observation, "expected"))
        with self.assertRaises(TypeError):
            ScriptedPolicy(1).select(case, [])

    def test_split_overlap_ignores_random_task_ids(self):
        case = self.splits["promotion"][0]
        renamed = replace(case, observation=replace(case.observation, task_id="different"))
        with self.assertRaises(ValueError):
            assert_disjoint({"discovery": [case], "evaluation": [renamed]})

    def test_random_streams_are_separate_and_reproducible(self):
        self.assertEqual(make_splits(7), make_splits(7))
        self.assertNotEqual(make_splits(7)["evaluation"], make_splits(42)["evaluation"])
        self.assertNotEqual(derive_seed(7, "policy"), derive_seed(42, "policy"))
        self.assertNotEqual(derive_seed(7, "promotion"), derive_seed(7, "evaluation"))

    def test_compiled_explicit_bindings_generalize(self):
        case = self.splits["discovery"][0]
        skill = workflow()
        compiled = compile_trajectory("compiled", case.observation, execute(skill, case.observation), successful=True)
        self.assertEqual(compiled.skill_id, skill.skill_id)
        self.assertTrue(all(assess(c, execute(compiled, c.observation))["success"] for c in self.splits["evaluation"]))

    def test_fixed_metric_has_no_composition_credit_without_checks(self):
        case = self.splits["promotion"][0]
        result = execute(workflow(), case.observation)
        reward = compute_reward(result, assess(case, result), True)
        self.assertEqual(reward["total"], 1)
        self.assertEqual(reward["components"]["composition"], 0)
        self.assertEqual(reward["metric_hash"], digest(DEFAULT_WEIGHTS))

    def test_inadequate_hidden_suite_gives_unknown_validity(self):
        self.assertIsNone(adjudicate_candidate(self.candidates[0], self.splits["evaluation"][:1])["valid"])


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.splits = make_splits(7)

    def test_duplicate_skill_insertion_is_not_a_new_promotion(self):
        graph, skill = SkillGraph(), workflow()
        self.assertTrue(graph.add_skill(skill, {"status": ACCEPT}))
        self.assertFalse(graph.add_skill(replace(skill, name="alias"), {"status": ACCEPT}))
        self.assertEqual(len(graph.available()), 1)

    def test_policy_cannot_mutate_stored_program(self):
        graph, skill = SkillGraph(), workflow()
        graph.add_skill(skill, {"status": ACCEPT})
        before = graph.state_hash
        graph.get(skill.skill_id).steps[0]["tool"] = "constant_report"
        self.assertEqual(graph.state_hash, before)

    def test_checkpoint_evaluation_is_immutable_and_repeatable(self):
        graph = SkillGraph()
        graph.add_skill(workflow(), {"status": ACCEPT})
        before = graph.state_hash
        a = evaluate_checkpoint(graph, self.splits["evaluation"], policy_seed=2)
        b = evaluate_checkpoint(graph, self.splits["evaluation"], policy_seed=2)
        self.assertEqual(a, b)
        self.assertEqual(graph.state_hash, before)

    def test_revocation_propagates_to_composed_dependents(self):
        graph, _ = prepare_graph(self.splits["promotion"])
        liters, _, render = primitives()
        skill = compose(liters, render)
        graph.add_skill(skill, {"status": ACCEPT})
        self.assertEqual(set(graph.revoke(liters.skill_id, "removed")), {liters.skill_id, skill.skill_id})
        self.assertIsNone(graph.get(skill.skill_id))

    def test_changed_tool_requires_reverification(self):
        graph, skill = SkillGraph(), workflow()
        graph.add_skill(skill, {"status": ACCEPT})
        graph.registry.version = lambda name: "2.0.0"
        self.assertIn(skill.skill_id, graph.invalidate_changed_tools())
        self.assertIsNone(graph.get(skill.skill_id))

    def test_individually_correct_skills_can_be_incompatible(self):
        rows = run_composition()["summary"]
        summary = {(r["mode"], r["compatible"]): r for r in rows}
        self.assertEqual(summary["checked_graph", True]["successes"], 30)
        self.assertEqual(summary["checked_graph", False]["blocked"], 30)
        self.assertGreater(summary["flat_verified_library", False]["incorrect_completed"], 0)

    def test_frozen_llm_exposes_provider_and_parsing_failures(self):
        obs, skill = self.splits["evaluation"][0].observation, workflow()
        good = FrozenLLMPolicy(lambda req: json.dumps({"skill_id": skill.skill_id}), model="test", backend="fake")
        self.assertEqual(good.select(obs, [skill]).skill_id, skill.skill_id)
        bad = FrozenLLMPolicy(lambda req: "invalid", model="test", backend="fake")
        with self.assertRaises(PolicyError):
            bad.select(obs, [skill])
        self.assertFalse(bad.events[0]["fallback"])
        def offline(req):
            raise ConnectionError("offline")
        failed = FrozenLLMPolicy(offline, model="test", backend="fake")
        with self.assertRaises(PolicyError):
            failed.select(obs, [skill])
        self.assertEqual(failed.events[0]["error_type"], "provider_error")


class AuditTests(unittest.TestCase):
    def bundle(self, directory):
        s = make_splits(7)
        candidate = make_candidates(s["discovery"])[0][0]
        config = GateConfig(method="self_verification")
        path = Path(directory)/"bundle"
        ref = write_bundle(path, candidate, s["promotion"], config, verify(candidate, s["promotion"], config))
        return path, ref

    def test_intact_evidence_reproduces_decisions_and_rewards(self):
        with TemporaryDirectory() as directory:
            path, ref = self.bundle(directory)
            result = audit_replay(path, expected_manifest_sha256=ref["manifest_sha256"])
            self.assertEqual((result["status"], result["reconstructed_rewards"]), ("REPRODUCED", 15))

    def test_missing_artifact_is_incomplete_evidence(self):
        with TemporaryDirectory() as directory:
            path, _ = self.bundle(directory)
            (path/"tests.json").unlink()
            with self.assertRaisesRegex(EvidenceError, "incomplete_evidence"):
                audit_replay(path)

    def test_modified_artifact_is_integrity_failure(self):
        with TemporaryDirectory() as directory:
            path, _ = self.bundle(directory)
            (path/"candidate.json").write_text("{}")
            with self.assertRaisesRegex(EvidenceError, "integrity_failure"):
                audit_replay(path)

    def test_external_digest_detects_manifest_rewriting(self):
        with TemporaryDirectory() as directory:
            path, ref = self.bundle(directory)
            manifest = json.loads((path/"manifest.json").read_text())
            manifest["environment"]["python"] = "different"
            write_json(path/"manifest.json", manifest)
            with self.assertRaisesRegex(EvidenceError, "manifest digest"):
                audit_replay(path, expected_manifest_sha256=ref["manifest_sha256"])
            with self.assertRaisesRegex(EvidenceError, "environment_mismatch"):
                audit_replay(path)

    def test_audit_recomputes_rewards_despite_rehashed_summary(self):
        with TemporaryDirectory() as directory:
            path, _ = self.bundle(directory)
            target = path/"decision.json"
            decision = json.loads(target.read_text())
            decision["results"][0]["reward"]["total"] = 999
            write_json(target, decision)
            manifest = json.loads((path/"manifest.json").read_text())
            manifest["artifacts"]["decision.json"] = {"sha256": sha256(target.read_bytes()).hexdigest(), "bytes": target.stat().st_size}
            write_json(path/"manifest.json", manifest)
            with self.assertRaisesRegex(EvidenceError, "replay_mismatch"):
                audit_replay(path)


class LegacyTests(unittest.TestCase):
    def test_policy_seeds_differ_and_unique_counts_match_library(self):
        with TemporaryDirectory() as directory:
            a, b = ASGSISystem(seed=7, audit_dir=directory+"/a"), ASGSISystem(seed=42, audit_dir=directory+"/b")
            self.assertNotEqual(a.agent.rng.getstate(), b.agent.rng.getstate())
            rows = a.train(iters=3, num_tasks=20)
            self.assertEqual(sum(r["verified_skills_added"] for r in rows), a.skill_graph.size())
            self.assertLessEqual(a.skill_graph.size(), 3)
            self.assertEqual(a.metadata["parameters_updated"], 0)

    def test_evaluation_preserves_memory_library_and_rng(self):
        with TemporaryDirectory() as directory:
            system = ASGSISystem(audit_dir=directory)
            system.train(iters=1, num_tasks=20)
            before = copy.deepcopy((system.agent.rng.getstate(), system.memory.notes, system.skill_graph.skills))
            env = VerifiableEnv(6)
            self.assertEqual(system.evaluate([env.sample_task() for _ in range(13)])["num_tasks"], 13)
            self.assertEqual(before, (system.agent.rng.getstate(), system.memory.notes, system.skill_graph.skills))

    def test_policy_never_receives_expected_answer(self):
        with TemporaryDirectory() as directory:
            system, test = ASGSISystem(audit_dir=directory), self
            class Spy:
                def act(self, obs, graph, verifier, memory):
                    test.assertFalse(hasattr(obs, "expected"))
                    call = ToolCall(obs.kind, obs.input_data)
                    answer = system.tools.call(call)
                    return [Step(0, "TOOL", {"tool_name": call.name, "tool_args": call.args}, answer)], answer
            system.agent = Spy()
            self.assertEqual(system.run_iteration(0, num_tasks=5)["success_rate"], 1)

    def test_one_passing_test_is_insufficient(self):
        with TemporaryDirectory() as directory:
            system = ASGSISystem(audit_dir=directory)
            task = Task("t", "concat", {"a": "x", "b": "y"}, "xy")
            traj = Trajectory(task, [Step(0, "TOOL", {"tool_name": "concat", "tool_args": task.input_data}, "xy")], "xy", True, 3, {})
            accepted, report = system.auditor.verify_skill(SkillCompiler().compile_from_trajectory(traj), [task])
            self.assertFalse(accepted)
            self.assertEqual(report["status"], INSUFFICIENT_EVIDENCE)

    def test_equal_values_do_not_confuse_argument_binding(self):
        task = Task("t", "concat", {"a": "x", "b": "x"}, "xx")
        traj = Trajectory(task, [Step(0, "TOOL", {"tool_name": "concat", "tool_args": task.input_data}, "xx")], "xx", True, 3, {})
        self.assertEqual(SkillCompiler().compile_from_trajectory(traj).program["argmap"], {"a": "a", "b": "b"})

    def test_model_failure_is_explicit_without_scripted_fallback(self):
        with TemporaryDirectory() as directory:
            system = ASGSISystem(audit_dir=directory)
            class Offline:
                base_url, model = "http://offline.invalid", "test"
                def generate(self, *args, **kwargs):
                    raise ConnectionError("offline")
            steps, answer = ModelBasedAgentPolicy(system.tools, Offline()).act(system.env.sample_task().observation(), system.skill_graph, system.auditor, system.memory)
            self.assertIsNone(answer)
            self.assertEqual(steps[0].action_type, "ERROR")
            self.assertFalse(steps[0].payload["fallback"])

    def test_analysis_uses_recorded_denominator_and_checkpoint(self):
        payload = {"schema_version": 2, "runs": {"7": [{"iteration": 4, "num_tasks": 13, "success_rate": 1, "avg_reward": 3, "actions": {"skill": 13, "direct": 0}}]}}
        result = analyze(payload)["checkpoints"]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["metrics"]["skill_pct"]["mean"], 100)
        self.assertIsNone(result[0]["metrics"]["skill_pct"]["ci95"])


class RunnerTests(unittest.TestCase):
    def test_complete_raw_result_report_and_audit_pipeline(self):
        config = {"seeds": [7], "methods": ["first_success", "self_verification"], "phases": ["static", "online"],
                  "online_batch_size": 3, "splits": {"promotion_size": 5, "evaluation_size": 5, "stress_size": 5}, "gate": {"min_cases": 5, "max_cases": 5}}
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)/"run"
            run(config, root)
            report = generate_report(root)
            self.assertEqual(len(report["per_run"]), 4)
            tasks = load_jsonl(root/"tasks.jsonl")
            self.assertEqual(len({r["reward"]["metric_hash"] for r in tasks}), 1)
            self.assertEqual(len(load_jsonl(root/"decisions.jsonl")), 40)
            audit = audit_run(root)
            self.assertEqual(audit["admission_bundles"], 40)
            self.assertEqual(audit["downstream_tasks"], len(tasks))
            for row in report["per_run"]:
                self.assertEqual((row["submissions"], row["faulty_candidates"], row["valid_candidates"]), (10, 7, 3))
            with self.assertRaisesRegex(ValueError, "empty"):
                run(config, root)

    def test_duplicate_seeds_do_not_count_as_independent_runs(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "distinct"):
                run({"seeds": [7, 7]}, Path(directory)/"run")

if __name__ == "__main__":
    unittest.main()
