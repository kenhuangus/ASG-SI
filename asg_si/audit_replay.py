"""Reconstruct complete admission decisions, execution traces, and per-test rewards."""
import argparse
import json
from .evidence import EvidenceError, environment_snapshot, read_bundle
from .types import Candidate, EvaluationCase, digest
from .verifier import GateConfig, verify_in_process

def audit_replay(path, *, expected_manifest_sha256=None, signing_key=None):
    bundle = read_bundle(path, expected_manifest_sha256=expected_manifest_sha256, signing_key=signing_key)
    if bundle["manifest"]["environment"] != environment_snapshot():
        raise EvidenceError("environment_mismatch: restore recorded code, Python, and tool versions")
    files = bundle["artifacts"]
    original = dict(files["decision.json"])
    if "worker_failure" in original.get("reasons", []):
        raise EvidenceError("incomplete_evidence: worker failure has no replayable execution")
    original.pop("verification_ms", None)
    original.pop("worker", None)
    actual = verify_in_process(Candidate.from_dict(files["candidate.json"]),
                               [EvaluationCase.from_dict(c) for c in files["tests.json"]], GateConfig.from_dict(files["config.json"]))
    if digest(original) != digest(actual):
        raise EvidenceError("replay_mismatch: decision, execution, or rewards differ")
    return {"status": "REPRODUCED", "decision": actual["status"], "replayed_tests": actual["num_tests"],
            "reconstructed_rewards": len(actual["results"]), "externally_anchored": bundle["externally_anchored"]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    try:
        print(json.dumps(audit_replay(args.bundle, expected_manifest_sha256=args.manifest_sha256), indent=2))
    except (EvidenceError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")

if __name__ == "__main__":
    main()
