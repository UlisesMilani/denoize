#!/usr/bin/env python3
"""Exercise the AUv3 evidence generator and its closed schema."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "scripts/generate-auv3-host-evidence.py"
SCHEMA = ROOT / "schemas/denoize-auv3-host-evidence-v1.schema.json"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def auval_report(*, failed: bool = False) -> str:
    second = "false" if failed else "true"
    result = "" if failed else "Result: AUv3 auval passed 2 components\n"
    return (
        "denoize AUv3 official validator report\n"
        "host: auval\n"
        "host_version: 15.6.1\n"
        "operating_system: macos\n"
        "architecture: arm64\n"
        "DENOIZE_AUV3_AUVAL_RESULT subtype=Dn01 passed=true\n"
        f"DENOIZE_AUV3_AUVAL_RESULT subtype=Dn02 passed={second}\n"
        f"{result}"
    )


def host_report(*, complete: bool = True) -> str:
    second = (
        "DENOIZE_AUV3_SMOKE component=Dn02 instantiated=true allocated=true "
        "state_round_trip=true teardown=true\n"
        if complete
        else ""
    )
    result = "Result: AUv3 AVFoundation host smoke passed\n" if complete else ""
    return (
        "denoize AUv3 AVFoundation real-host report\n"
        "host: AVFoundation\n"
        "host_version: 15.6.1\n"
        "operating_system: macos\n"
        "architecture: arm64\n"
        "DENOIZE_AUV3_SMOKE component=Dn01 instantiated=true allocated=true "
        "state_round_trip=true teardown=true\n"
        f"{second}{result}"
    )


def run(auval: Path, host: Path, output: Path, *, tag: str = "v9.8.7") -> subprocess.CompletedProcess[str]:
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
            "--auval-report",
            str(auval),
            "--host-report",
            str(host),
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
    with tempfile.TemporaryDirectory(prefix="denoize-auv3-evidence-") as name:
        root = Path(name)
        auval = root / "denoize-auv3-auval-v9.8.7-aarch64-apple-darwin.txt"
        host = root / "denoize-auv3-host-v9.8.7-aarch64-apple-darwin.txt"
        output = root / "denoize-auv3-host-evidence-v1-aarch64-apple-darwin.json"
        auval.write_text(auval_report(), encoding="utf-8")
        host.write_text(host_report(), encoding="utf-8")
        result = run(auval, host, output)
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        document = json.loads(output.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(document)
        assert document["format"] == "auv3"
        assert [item["subtype"] for item in document["components"]] == ["Dn01", "Dn02"]
        assert document["claims"]["self_contained_model"] is True
        assert document["runs"][1]["state_round_trip"] is True

        duplicate = run(auval, host, output)
        assert duplicate.returncode != 0 and "refusing to replace" in duplicate.stderr

        bad_auval = root / "bad-auval.txt"
        bad_auval.write_text(auval_report(failed=True), encoding="utf-8")
        failed = run(bad_auval, host, root / "failed.json")
        assert failed.returncode != 0 and "Dn02" in failed.stderr

        incomplete_host = root / "incomplete-host.txt"
        incomplete_host.write_text(host_report(complete=False), encoding="utf-8")
        incomplete = run(auval, incomplete_host, root / "incomplete.json")
        assert incomplete.returncode != 0 and "Dn02" in incomplete.stderr

        bad_tag = run(auval, host, root / "bad-tag.json", tag="9.8.7")
        assert bad_tag.returncode != 0 and "invalid release tag" in bad_tag.stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
