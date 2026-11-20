import json
import tempfile
import unittest
from pathlib import Path

from experiments import alfworld_skills as a


class ExtractCodeTest(unittest.TestCase):
    def test_last_python_block_with_solve(self):
        text = "<think>x</think>```python\ndef helper():\n    pass\n```\n```python\ndef solve(env, goal):\n    env.step('look')\n```"
        self.assertEqual(a.extract_code(text), "def solve(env, goal):\n    env.step('look')\n")

    def test_bare_code(self):
        self.assertEqual(a.extract_code("def solve(env, goal):\n    return None"), "def solve(env, goal):\n    return None\n")

    def test_missing(self):
        self.assertIsNone(a.extract_code("I cannot write this."))


class VerdictTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(a._verdict('{"verdict": "accept", "reason": "ok"}'), ("ACCEPT", "ok"))
        self.assertEqual(a._verdict("no json")[0], "UNPARSEABLE")


class AnalyzeTest(unittest.TestCase):
    def test_methods_and_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pools = {tt: {"discovery": [], "evidence": [], "verify": [], "eval": ["g1", "g2", "g3", "g4", "g5"]}
                     for tt in a.TASK_TYPES}
            a.write_json(out / "prep.json", {"pools": pools})
            tt = a.TASK_TYPES[0]
            cands = [{"id": "good", "task_type": tt, "docs": "documented", "code": "x", "finish_reason": "stop"},
                     {"id": "bad", "task_type": tt, "docs": "sparse", "code": "y", "finish_reason": "stop"}]
            win = lambda n: [{"game": f"g{i}", "won": i < n, "steps": 3, "end": "won" if i < n else "step_budget"} for i in range(5)]  # noqa: E731
            runs = {"good": {"discovery": {"won": True}, "verify": [{"won": True}] * 10, "eval": win(5)},
                    "bad": {"discovery": {"won": True}, "verify": [{"won": True}] * 3 + [{"won": False}] * 7, "eval": win(1)}}
            verdicts = {"good": {"self_verification": "REJECT", "evidence_verification": "ACCEPT"},
                        "bad": {"self_verification": "ACCEPT", "evidence_verification": "ACCEPT"}}
            a.write_json(out / "candidates_m.json", cands)
            a.write_json(out / "exec_m.json", runs)
            a.write_json(out / "judge_m.json", verdicts)
            a.analyze(str(out), "m")
            s = json.loads((out / "summary_m.json").read_text())
            self.assertEqual(s["methods"]["first_success"]["admitted_inadequate"], 1)
            self.assertEqual(s["methods"]["independent_replay"]["admitted"], 1)
            self.assertEqual(s["methods"]["independent_replay"]["admitted_inadequate"], 0)
            self.assertEqual(s["methods"]["llm_self_verification"]["rejected_adequate"], 1)
            self.assertAlmostEqual(s["methods"]["independent_replay"]["library_success_mean"], 5 / 30)


if __name__ == "__main__":
    unittest.main()
