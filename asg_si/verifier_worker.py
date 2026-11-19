"""JSON-only replay worker; no policy objects or final-evaluation cases are sent."""
import json
import sys
from .types import Candidate, EvaluationCase
from .verifier import GateConfig, verify_in_process

def main():
    payload = json.load(sys.stdin)
    result = verify_in_process(Candidate.from_dict(payload["candidate"]),
                               [EvaluationCase.from_dict(c) for c in payload["cases"]],
                               GateConfig.from_dict(payload["config"]))
    json.dump(result, sys.stdout, allow_nan=False)

if __name__ == "__main__":
    main()
