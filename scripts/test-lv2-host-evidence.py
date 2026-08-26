#!/usr/bin/env python3
"""Exercise the closed LV2 host-evidence generator and schema."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "scripts/generate-lv2-host-evidence.py"
SCHEMA = ROOT / "schemas/denoize-lv2-host-evidence-v1.schema.json"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def validation_report() -> str:
    return (
        "denoize LV2 validation report\n"
        "lv2_specification: 1.18.10\n"
        "operating_system: ubuntu-24.04\n"
        "architecture: x86_64\n"
        "Name:              denoize\n"
        "Name:              denoize Neural\n"
        "descriptor_count: 2\n"
        "metadata_validation: passed\n"
        "dsp_in_place_offline_host_processing: passed\n"
        "neural_worker_host: delegated-to-jalv\n"
        "binary_hardening: passed\n"
        "Result: LV2 validation passed\n"
    )


def jalv_report(*, complete: bool = True) -> str:
    summary = (
        "DENOIZE_LV2_JALV_SMOKE sample_rate_hz=48000 block_frames=480 "
        "descriptors=2 stereo_connected=true worker_host=true teardown=true\n"
        if complete
        else ""
    )
    return (
        "denoize LV2 Jalv real-host report\n"
        "host: Jalv\n"
        "host_package_version: 1.6.8-1build3\n"
        "jack_package_version: 1.9.21~dfsg-3ubuntu3\n"
        "operating_system: ubuntu-24.04\n"
        "architecture: x86_64\n"
        "sample_rate_hz: 48000\n"
        "block_frames: 480\n"
        "descriptors: 2\n"
        "audio_connections: stereo-in-stereo-out\n"
        "dsp_minimum_active_seconds: 5\n"
        "neural_minimum_active_seconds: 15\n"
        f"{summary}"
        "Result: Jalv real-host smoke passed\n"
    )


def ardour_report(*, state_record: bool = True) -> str:
    state = (
        "DENOIZE_LV2_ARDOUR_STATE properties=2 portable=true interface_errors=0\n"
        if state_record
        else ""
    )
    return (
        "denoize LV2 Ardour real-host smoke report\n"
        "host: Ardour\n"
        "host_version: 8.4.0~ds1\n"
        "package_version: 1:8.4.0+ds1-2ubuntu8\n"
        "operating_system: ubuntu-24.04\n"
        "architecture: x86_64\n"
        "DENOIZE_LV2_ARDOUR_TEARDOWN phase=create passed=true\n"
        "DENOIZE_LV2_ARDOUR_TEARDOWN phase=restore passed=true\n"
        f"{state}"
        "DENOIZE_LV2_ARDOUR_SMOKE first_pass_frames=4096 "
        "restored_pass_frames=32768 sample_rate_hz=48000 descriptors=2 "
        "state_reload=true teardown=true\n"
        "Result: Ardour LV2 real-host smoke passed\n"
    )


def run(
    validation: Path,
    jalv: Path,
    ardour: Path,
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
            "--validation-report",
            str(validation),
            "--jalv-report",
            str(jalv),
            "--ardour-report",
            str(ardour),
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
    with tempfile.TemporaryDirectory(prefix="denoize-lv2-evidence-") as name:
        root = Path(name)
        validation = root / "denoize-lv2-validation-v9.8.7-x86_64-unknown-linux-gnu.txt"
        jalv = root / "denoize-lv2-jalv-v9.8.7-x86_64-unknown-linux-gnu.txt"
        ardour = root / "denoize-lv2-ardour-v9.8.7-x86_64-unknown-linux-gnu.txt"
        output = root / "denoize-lv2-host-evidence-v1.json"
        validation.write_text(validation_report(), encoding="utf-8")
        jalv.write_text(jalv_report(), encoding="utf-8")
        ardour.write_text(ardour_report(), encoding="utf-8")

        result = run(validation, jalv, ardour, output)
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        document = json.loads(output.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(document)
        assert document["tag"] == "v9.8.7"
        assert document["source"]["commit"] == COMMIT
        assert document["descriptors"][1]["worker_required"] is True
        assert document["runs"][1]["worker_host"] is True
        assert document["runs"][2]["first_pass_frames"] == 4096
        assert document["runs"][2]["restored_pass_frames"] == 32768
        assert document["runs"][2]["state_interface_errors"] == 0

        duplicate = run(validation, jalv, ardour, output)
        assert duplicate.returncode != 0 and "refusing to replace" in duplicate.stderr

        incomplete_jalv = root / "incomplete-jalv.txt"
        incomplete_jalv.write_text(jalv_report(complete=False), encoding="utf-8")
        bad_worker = run(validation, incomplete_jalv, ardour, root / "bad-worker.json")
        assert bad_worker.returncode != 0 and "worker-host summary" in bad_worker.stderr

        incomplete_ardour = root / "incomplete-ardour.txt"
        incomplete_ardour.write_text(ardour_report(state_record=False), encoding="utf-8")
        bad_state = run(validation, jalv, incomplete_ardour, root / "bad-state.json")
        assert bad_state.returncode != 0 and "ARDOUR_STATE" in bad_state.stderr

        bad_tag = run(validation, jalv, ardour, root / "bad-tag.json", tag="9.8.7")
        assert bad_tag.returncode != 0 and "invalid release tag" in bad_tag.stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
