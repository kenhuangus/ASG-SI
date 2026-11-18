"""Retrievable artifacts and full SHA-256 manifests; unsigned, with optional external roots."""
from hashlib import sha256
import json
from pathlib import Path
import platform

from .tools import ToolRegistry

class EvidenceError(ValueError):
    pass

def environment_snapshot():
    root = Path(__file__).parent
    project = root.parent
    sources = [*root.glob("*.py"), *(project/"experiments").glob("*.py"),
               *(project/name for name in ("asg_si_demo.py", "run_multi_seed.py", "analyze_results.py"))]
    return {"python": platform.python_version(), "tool_registry": ToolRegistry().snapshot(),
            "source_sha256": {str(p.relative_to(project)): sha256(p.read_bytes()).hexdigest() for p in sorted(sources) if p.is_file()}}

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+"\n")

def write_bundle(path, candidate, cases, config, decision):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    files = {"candidate.json": candidate.to_dict(), "tests.json": [c.to_dict() for c in cases],
             "config.json": config.to_dict(), "decision.json": decision}
    for name, value in files.items():
        write_json(path/name, value)
    manifest = {"format": "asg-si-evidence-1", "environment": environment_snapshot(),
                "artifacts": {name: {"sha256": sha256((path/name).read_bytes()).hexdigest(), "bytes": (path/name).stat().st_size} for name in files}}
    write_json(path/"manifest.json", manifest)
    return {"bundle": str(path), "manifest_sha256": sha256((path/"manifest.json").read_bytes()).hexdigest(),
            "artifact_count": len(files), "evidence_bytes": sum(p.stat().st_size for p in path.iterdir())}

def read_bundle(path, *, expected_manifest_sha256=None):
    path = Path(path)
    manifest_path = path/"manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise EvidenceError("incomplete_evidence: missing manifest")
    if expected_manifest_sha256 and sha256(manifest_path.read_bytes()).hexdigest() != expected_manifest_sha256:
        raise EvidenceError("integrity_failure: manifest digest mismatch")
    try:
        manifest = json.loads(manifest_path.read_text())
    except ValueError as error:
        raise EvidenceError("integrity_failure: malformed manifest") from error
    required = {"candidate.json", "tests.json", "config.json", "decision.json"}
    if manifest.get("format") != "asg-si-evidence-1" or set(manifest.get("artifacts", {})) != required:
        raise EvidenceError("incomplete_evidence: unexpected artifact inventory")
    artifacts = {}
    for name in sorted(required):
        item = path/name
        if not item.is_file() or item.is_symlink():
            raise EvidenceError(f"incomplete_evidence: {name}")
        content, metadata = item.read_bytes(), manifest["artifacts"][name]
        if len(content) != metadata["bytes"] or sha256(content).hexdigest() != metadata["sha256"]:
            raise EvidenceError(f"integrity_failure: {name}")
        artifacts[name] = json.loads(content)
    return {"manifest": manifest, "artifacts": artifacts, "externally_anchored": bool(expected_manifest_sha256)}
