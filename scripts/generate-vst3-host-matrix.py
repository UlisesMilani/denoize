#!/usr/bin/env python3
"""Bind the pinned VST3 validator report into a closed release matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys


TAG_RE = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
PORTABLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,191}$")
RESULT_RE = re.compile(r"Result: ([0-9]+) tests passed, ([0-9]+) tests failed")
ARDOUR_SMOKE_RE = re.compile(
    r"^DENOIZE_ARDOUR_SMOKE first_pass_frames=([1-9][0-9]*) "
    r"restored_pass_frames=([1-9][0-9]*) sample_rate_hz=48000 "
    r"descriptors=2 state_reload=true teardown=true$",
    re.MULTILINE,
)
MAX_REPORT_BYTES = 2 * 1024 * 1024


class MatrixError(RuntimeError):
    pass


def regular_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise MatrixError(f"{label} is not a regular file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > MAX_REPORT_BYTES:
        raise MatrixError(
            f"{label} size must be in 1..={MAX_REPORT_BYTES} bytes"
        )
    return path.read_bytes()


def write_exclusive(path: Path, document: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise MatrixError(f"refusing to replace existing matrix: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        payload = (
            json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        with os.fdopen(descriptor, "wb", closefd=True) as destination:
            descriptor = -1
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def generate(args: argparse.Namespace) -> None:
    if not TAG_RE.fullmatch(args.tag):
        raise MatrixError(f"invalid release tag: {args.tag}")
    if not COMMIT_RE.fullmatch(args.commit):
        raise MatrixError("source commit must be a lowercase 40-character SHA-1")
    if not REPOSITORY_RE.fullmatch(args.repository):
        raise MatrixError("repository must be an owner/name pair")
    for label, value in (("operating system", args.operating_system), ("architecture", args.architecture)):
        if not PORTABLE_RE.fullmatch(value):
            raise MatrixError(f"invalid {label}: {value}")

    report_path = args.validator_report.resolve()
    payload = regular_bytes(report_path, "validator report")
    try:
        report = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MatrixError("validator report is not UTF-8") from error
    matches = RESULT_RE.findall(report)
    if matches != [("94", "0")]:
        raise MatrixError("validator report must contain one exact 94 passed / 0 failed result")
    if report.count("1234567.8 Hz - processed successfully!") != 2:
        raise MatrixError("both descriptors must pass the 1,234,567.8 Hz boundary")
    if report.count("[Succeeded]") != 94:
        raise MatrixError("validator report must contain exactly 94 successful test records")

    real_host_report_path = args.real_host_report.resolve()
    real_host_payload = regular_bytes(real_host_report_path, "real-host report")
    try:
        real_host_report = real_host_payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MatrixError("real-host report is not UTF-8") from error
    required_host_records = (
        "host: Ardour",
        "host_version: 8.4.0~ds1",
        "package_version: 1:8.4.0+ds1-2ubuntu8",
        "operating_system: ubuntu-24.04",
        "architecture: x86_64",
        "DENOIZE_ARDOUR_TEARDOWN phase=create passed=true",
        "DENOIZE_ARDOUR_TEARDOWN phase=restore passed=true",
        "Result: Ardour real-host smoke passed",
    )
    for record in required_host_records:
        if real_host_report.count(record) != 1:
            raise MatrixError(f"real-host report must contain one exact record: {record}")
    if real_host_report.count("[Info]: Found Plugin: denoize\n") != 1:
        raise MatrixError("real-host report must discover the standard descriptor once")
    if real_host_report.count("[Info]: Found Plugin: denoize Neural\n") != 1:
        raise MatrixError("real-host report must discover the neural descriptor once")
    smoke_matches = ARDOUR_SMOKE_RE.findall(real_host_report)
    if len(smoke_matches) != 1:
        raise MatrixError("real-host report must contain one complete smoke summary")
    first_pass_frames, restored_pass_frames = (int(value) for value in smoke_matches[0])

    document: dict[str, object] = {
        "schema": "denoize-plugin-host-matrix-v1",
        "schema_version": 1,
        "tag": args.tag,
        "source": {"repository": args.repository, "commit": args.commit},
        "format": "vst3",
        "adapter": {
            "strategy": "statically-linked-clap-wrapper",
            "clap_wrapper": {
                "version": "0.16.0",
                "commit": "1cca996e96f29ab2be7ae9f8cfe532bbc92e1dd6",
            },
            "clap_sdk": {
                "version": "1.2.6",
                "commit": "69a69252fdd6ac1d06e246d9a04c0a89d9607a17",
            },
            "vst3_sdk": {
                "version": "3.8.1",
                "commit": "3cdf9ca5d1f5b1b21e0a86832aa4abe55607bd96",
            },
        },
        "descriptors": [
            {
                "id": "org.penguin425.denoize",
                "name": "denoize",
                "audio_inputs": 1,
                "audio_outputs": 1,
                "parameters": 7,
            },
            {
                "id": "org.penguin425.denoize.neural",
                "name": "denoize Neural",
                "audio_inputs": 2,
                "audio_outputs": 1,
                "parameters": 4,
            },
        ],
        "runs": [
            {
                "host": "Steinberg VST3 Plug-in Validator",
                "host_version": "3.8.1",
                "operating_system": args.operating_system,
                "architecture": args.architecture,
                "evidence_kind": "official-validator",
                "status": "passed",
                "tests_passed": 94,
                "tests_failed": 0,
                "maximum_exercised_sample_rate_hz": 1_234_567.8,
                "report": {
                    "name": report_path.name,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
            },
            {
                "host": "Ardour",
                "host_version": "8.4.0~ds1",
                "operating_system": "ubuntu-24.04",
                "architecture": "x86_64",
                "evidence_kind": "real-host-smoke",
                "status": "passed",
                "tests_passed": 2,
                "tests_failed": 0,
                "maximum_exercised_sample_rate_hz": 48_000,
                "descriptors_exercised": 2,
                "first_pass_frames": first_pass_frames,
                "restored_pass_frames": restored_pass_frames,
                "state_reload": True,
                "teardown": True,
                "report": {
                    "name": real_host_report_path.name,
                    "size_bytes": len(real_host_payload),
                    "sha256": hashlib.sha256(real_host_payload).hexdigest(),
                },
            },
        ],
        "claims": {
            "official_validator": True,
            "real_host_smoke": True,
            "single_precision_audio": True,
            "double_precision_audio": False,
            "custom_editor": False,
        },
        "limitations": [
            "custom-editor-not-present",
            "double-precision-audio-not-supported-by-wrapper",
            "proprietary-hosts-not-exercised",
            "reference-input-reserved",
        ],
    }
    write_exclusive(args.output, document)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--tag", required=True)
    result.add_argument("--commit", required=True)
    result.add_argument("--repository", required=True)
    result.add_argument("--operating-system", required=True)
    result.add_argument("--architecture", required=True)
    result.add_argument("--validator-report", type=Path, required=True)
    result.add_argument("--real-host-report", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    try:
        generate(parser().parse_args())
    except (MatrixError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
