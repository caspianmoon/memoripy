from __future__ import annotations

import base64
import json
import shutil
import zlib
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parts_dir = root / ".github" / "v4_payload_parts"
encoded = "".join(path.read_text(encoding="ascii") for path in sorted(parts_dir.glob("*.part")))
data = json.loads(zlib.decompress(base64.b64decode(encoded.encode("ascii"))).decode("utf-8"))

applied = 0
deleted = 0
for relative, content in data["files"].items():
    if relative.startswith(".github/workflows/"):
        continue
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    applied += 1
for relative in data["delete"]:
    if relative.startswith(".github/workflows/"):
        continue
    path = root / relative
    if path.exists():
        path.unlink()
        deleted += 1

shutil.rmtree(parts_dir)
(root / ".github" / "v4_payload_ready").unlink(missing_ok=True)
Path(__file__).unlink()
print(f"Applied {applied} non-workflow files and deleted {deleted} non-workflow files")
