"""Create (or update) the revised paper as a Google Doc with gws, then export a PDF for visual QA."""
import json
import subprocess
import sys
from pathlib import Path

GWS = ["node", r"C:\Users\kenhu\AppData\Roaming\npm\node_modules\@googleworkspace\cli\run.js"]
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def gws(*args, cwd=None):
    out = subprocess.run(GWS + list(args), capture_output=True, text=True, encoding="utf-8", cwd=cwd)
    if out.returncode != 0:
        raise SystemExit(out.stdout + out.stderr)
    return out.stdout


def main():
    docx, name = Path(sys.argv[1]).resolve(), sys.argv[2]
    file_id = sys.argv[3] if len(sys.argv) > 3 else None
    if file_id:
        gws("drive", "files", "update", "--params", json.dumps({"fileId": file_id}), "--upload", str(docx), "--upload-content-type", DOCX_MIME, cwd=docx.parent)
    else:
        res = json.loads(gws("drive", "files", "create", "--json", json.dumps({"name": name, "mimeType": "application/vnd.google-apps.document"}),
                             "--upload", str(docx), "--upload-content-type", DOCX_MIME, cwd=docx.parent))
        file_id = res["id"]
    gws("drive", "files", "export", "--params", json.dumps({"fileId": file_id, "mimeType": "application/pdf"}), "-o", "live_export.pdf", cwd=docx.parent)
    print(json.dumps({"id": file_id, "url": f"https://docs.google.com/document/d/{file_id}/edit", "pdf": str(docx.parent / "live_export.pdf")}))


if __name__ == "__main__":
    main()
