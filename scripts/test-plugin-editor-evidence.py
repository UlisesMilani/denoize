#!/usr/bin/env python3
"""Exercise the closed plug-in editor evidence generator and schema."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "scripts/generate-plugin-editor-evidence.py"
SCHEMA = ROOT / "schemas/denoize-plugin-editor-evidence-v1.schema.json"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def report(*, second: bool = True) -> str:
    neural = (
        "DENOIZE_EDITOR_HOST descriptor=denoize Neural rendered_colors=9 "
        "automation_events=3 bypass_value=1.0 lifecycle=true resize_contract=true\n"
        if second
        else ""
    )
    return (
        "denoize CLAP editor real-host smoke report\n"
        "host: clack-host 0.1.1 + baseview X11 parent\n"
        "display: Xvfb/X11\n"
        "descriptors: 2\n"
        "DENOIZE_EDITOR_HOST descriptor=denoize rendered_colors=8 "
        "automation_events=3 bypass_value=1.0 lifecycle=true resize_contract=true\n"
        f"{neural}"
        "Result: CLAP editor real-host smoke passed\n"
    )


def run(source: Path, output: Path, *, tag: str = "v9.8.7") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--tag",
            tag,
            "--commit",
            COMMIT,
            "--repository",
            "penguin425/denoize",
            "--operating-system",
            "ubuntu-24.04",
            "--architecture",
            "x86_64",
            "--report",
            str(source),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    with tempfile.TemporaryDirectory(prefix="denoize-editor-evidence-") as name:
        root = Path(name)
        source = root / "denoize-clap-editor-host-v9.8.7-x86_64-unknown-linux-gnu.txt"
        output = root / "denoize-plugin-editor-evidence-v1.json"
        source.write_text(report(), encoding="utf-8")
        result = run(source, output)
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        document = json.loads(output.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(document)
        assert document["source"]["commit"] == COMMIT
        assert document["descriptors"][0]["rendered_colors"] == 8
        assert document["descriptors"][1]["rendered_colors"] == 9
        assert document["claims"]["custom_editor"] is True
        assert document["run"]["report"]["name"] == source.name

        duplicate = run(source, output)
        assert duplicate.returncode != 0 and "refusing to replace" in duplicate.stderr

        incomplete = root / "incomplete.txt"
        incomplete.write_text(report(second=False), encoding="utf-8")
        incomplete_result = run(incomplete, root / "incomplete.json")
        assert incomplete_result.returncode != 0 and "two complete" in incomplete_result.stderr

        bad_tag = run(source, root / "bad-tag.json", tag="9.8.7")
        assert bad_tag.returncode != 0 and "invalid release tag" in bad_tag.stderr

        linked = root / "linked-report.txt"
        linked.symlink_to(source)
        linked_result = run(linked, root / "linked.json")
        assert linked_result.returncode != 0 and "not a regular file" in linked_result.stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
