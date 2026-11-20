"""Regression tests for the review-response experiments and mechanisms."""
import json
import random
import tempfile
import unittest
from pathlib import Path

from asg_si.evidence import EvidenceError, read_bundle, write_bundle
from asg_si.interfaces import INHERIT, check_chain, make_interface
from asg_si.llm import extract_json
from asg_si.splits import make_splits
from asg_si.benchmark import make_candidates
from asg_si.verifier import GateConfig, verify_in_process
from experiments import stochastic, retention, tamper, composition_depth, selection


class InterfaceTests(unittest.TestCase):
    def test_compatible_chain_with_inherited_tag(self):
        chain = [make_interface({}, {"unit": "mL", "validated": True}),
                 make_interface({"unit": "mL"}, {"unit": "L", "validated": INHERIT}),
                 make_interface({"unit": "L", "validated": True}, {})]
        self.assertTrue(check_chain(chain)["ok"])

    def test_refutes_unit_and_validation_mismatch(self):
        unit = check_chain([make_interface({}, {"unit": "mL"}), make_interface({"unit": "L"}, {})])
        self.assertEqual((unit["ok"], unit["tag"], unit["edge"]), (False, "unit", (0, 1)))
        raw = check_chain([make_interface({}, {"unit": "L", "validated": False}),
                           make_interface({"unit": "L"}, {"unit": "L", "validated": INHERIT}),
                           make_interface({"unit": "L", "validated": True}, {})])
        self.assertEqual((raw["ok"], raw["tag"]), (False, "validated"))


class SigningTests(unittest.TestCase):
    def setUp(self):
        splits = make_splits(7)
        candidates, _ = make_candidates(splits["discovery"])
        self.cand, self.cases = candidates[0], splits["promotion"]
        self.config = GateConfig(method="self_verification")
        self.decision = verify_in_process(self.cand, self.cases, self.config)

    def test_signature_detects_rehashed_forgery(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b"
            write_bundle(path, self.cand, self.cases, self.config, self.decision, signing_key=b"k")
            read_bundle(path, signing_key=b"k")
            forged = dict(self.decision, status="REJECT")
            tamper.rewrite(path, "decision.json", forged)
            tamper.refresh_manifest(path)
            read_bundle(path)  # per-artifact hashes alone are satisfied
            with self.assertRaises(EvidenceError):
                read_bundle(path, signing_key=b"k")

    def test_missing_signature_is_rejected_when_key_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b"
            write_bundle(path, self.cand, self.cases, self.config, self.decision)
            with self.assertRaises(EvidenceError):
                read_bundle(path, signing_key=b"k")

    def test_attack_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = tamper.evidence_attacks(Path(tmp))
        self.assertTrue(all(v == "not detected" for v in m["no attack"].values()))
        self.assertEqual(m["edit decision and recompute manifest hashes"]["hashes"], "not detected")
        self.assertTrue(m["edit decision and recompute manifest hashes"]["signature"].startswith("detected"))
        self.assertEqual(m["compromised verifier writes forged decision"]["signature"], "not detected")
        self.assertTrue(m["compromised verifier writes forged decision"]["replay"].startswith("detected"))

    def test_poisoned_destination_needs_coverage(self):
        rows = tamper.poisoning(trials=1)
        self.assertEqual((rows[0]["standard_gate"], rows[0]["coverage_gate"]), ("ACCEPT", "REJECT"))
        self.assertLessEqual(rows[0]["deployed_with_revocation_monitor"]["violations"], 1)


class StochasticTests(unittest.TestCase):
    def test_clopper_pearson_bounds(self):
        self.assertAlmostEqual(stochastic.cp_lower(60, 60), 0.05 ** (1 / 60), places=6)
        self.assertAlmostEqual(stochastic.cp_upper(0, 60), 1 - 0.05 ** (1 / 60), places=6)
        self.assertLess(stochastic.cp_lower(50, 60), 50 / 60)

    def test_statistical_gate_separates_adequate_and_inadequate(self):
        cases = stochastic.make_cases(5000, 15)
        single, three = stochastic.CANDIDATES[0], stochastic.CANDIDATES[2]
        self.assertEqual(stochastic.gate(single, cases, random.Random(1), "statistical_cp")[0], "REJECT")
        self.assertEqual(stochastic.gate(three, cases, random.Random(1), "statistical_cp")[0], "ACCEPT")


class RetentionTests(unittest.TestCase):
    def test_lifecycle_rejects_stale_and_admits_adapted(self):
        v2 = retention.RegistryV2()
        promo = make_splits(1004)["promotion"]
        from asg_si.benchmark import workflow
        from asg_si.types import Candidate
        stale = retention.repin(workflow("s"), v2)
        adapted = retention.repin(retention.adapted_program(), v2)
        d_stale = verify_in_process(Candidate("a", stale, stale.skill_id, (), "t"), promo, GateConfig(method="self_verification"), registry=v2)
        d_new = verify_in_process(Candidate("b", adapted, adapted.skill_id, (), "t"), promo, GateConfig(method="self_verification"), registry=v2)
        self.assertEqual((d_stale["status"], d_new["status"]), ("REJECT", "ACCEPT"))


class CompositionDepthTests(unittest.TestCase):
    def test_declared_contract_check_matches_primitive_semantics(self):
        prims = composition_depth.primitives()
        ok = ("prep_mL", "rescale_mL_cL", "rescale_cL_L", "render_liters")
        bad = ("rawprep_L", "render_liters")
        self.assertTrue(check_chain([prims[n][1] for n in ok])["ok"])
        self.assertFalse(check_chain([prims[n][1] for n in bad])["ok"])


class SelectionTests(unittest.TestCase):
    def test_library_has_requested_valid_share(self):
        splits = make_splits(7)
        lib = selection.make_library(50, random.Random(1), splits["discovery"][0])
        self.assertEqual(sum(f == "none" for _, f in lib), 20)


class LLMParsingTests(unittest.TestCase):
    def test_extract_json_from_fenced_and_thinking_output(self):
        self.assertEqual(extract_json('<think>x</think>```json\n{"a": {"b": 1}}\n```'), {"a": {"b": 1}})
        self.assertEqual(extract_json('text {"verdict": "ACCEPT"} tail'), {"verdict": "ACCEPT"})
        self.assertIsNone(extract_json("no json here"))


if __name__ == "__main__":
    unittest.main()
