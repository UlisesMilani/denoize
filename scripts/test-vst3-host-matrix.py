#!/usr/bin/env python3
"""Exercise the closed VST3 host-matrix generator and schema."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "scripts/generate-vst3-host-matrix.py"
SCHEMA = ROOT / "schemas/denoize-plugin-host-matrix-v1.schema.json"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def run(
    report: Path,
    real_host_report: Path,
    output: Path,
    *,
    tag: str = "v9.8.7",
) -> subprocess.CompletedProcess[str]:
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
            "--validator-report",
            str(report),
            "--real-host-report",
            str(real_host_report),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def validator_report(passed: int = 94, failed: int = 0, boundaries: int = 2) -> str:
    successes = "\n".join("[Succeeded]" for _ in range(passed))
    high_rates = "\n".join(
        "Info:    1234567.8 Hz - processed successfully!" for _ in range(boundaries)
    )
    return f"{successes}\n{high_rates}\nResult: {passed} tests passed, {failed} tests failed\n"


def ardour_report(*, complete: bool = True) -> str:
    summary = (
        "DENOIZE_ARDOUR_SMOKE first_pass_frames=4096 restored_pass_frames=32768 "
        "sample_rate_hz=48000 descriptors=2 state_reload=true teardown=true\n"
        if complete
        else ""
    )
    return (
        "denoize VST3 real-host smoke report\n"
        "host: Ardour\n"
        "host_version: 8.4.0~ds1\n"
        "package_version: 1:8.4.0+ds1-2ubuntu8\n"
        "operating_system: ubuntu-24.04\n"
        "architecture: x86_64\n"
        "[Info]: Found Plugin: denoize\n"
        "[Info]: Found Plugin: denoize Neural\n"
        "DENOIZE_ARDOUR_TEARDOWN phase=create passed=true\n"
        "DENOIZE_ARDOUR_TEARDOWN phase=restore passed=true\n"
        f"{summary}"
        "Result: Ardour real-host smoke passed\n"
    )


def main() -> int:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    with tempfile.TemporaryDirectory(prefix="denoize-vst3-matrix-") as name:
        root = Path(name)
        report = root / "denoize-vst3-validator-v9.8.7-x86_64-unknown-linux-gnu.txt"
        host_report = root / "denoize-vst3-ardour-v9.8.7-x86_64-unknown-linux-gnu.txt"
        output = root / "denoize-vst3-host-matrix-v1.json"
        report.write_text(validator_report(), encoding="utf-8")
        host_report.write_text(ardour_report(), encoding="utf-8")
        result = run(report, host_report, output)
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        document = json.loads(output.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(document)
        assert document["claims"] == {
            "official_validator": True,
            "real_host_smoke": True,
            "single_precision_audio": True,
            "double_precision_audio": False,
            "custom_editor": False,
        }
        assert document["runs"][0]["tests_passed"] == 94
        assert document["runs"][0]["report"]["name"] == report.name
        assert document["runs"][1]["host"] == "Ardour"
        assert document["runs"][1]["first_pass_frames"] == 4096
        assert document["runs"][1]["restored_pass_frames"] == 32768
        assert document["runs"][1]["report"]["name"] == host_report.name

        duplicate = run(report, host_report, output)
        assert duplicate.returncode != 0 and "refusing to replace" in duplicate.stderr

        bad_report = root / "failed.txt"
        bad_report.write_text(validator_report(93, 1, 1), encoding="utf-8")
        bad_result = run(bad_report, host_report, root / "bad.json")
        assert bad_result.returncode != 0 and "94 passed" in bad_result.stderr

        bad_host_report = root / "incomplete-host.txt"
        bad_host_report.write_text(ardour_report(complete=False), encoding="utf-8")
        bad_host = run(report, bad_host_report, root / "bad-host.json")
        assert bad_host.returncode != 0 and "complete smoke summary" in bad_host.stderr

        bad_tag = run(report, host_report, root / "bad-tag.json", tag="9.8.7")
        assert bad_tag.returncode != 0 and "invalid release tag" in bad_tag.stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
