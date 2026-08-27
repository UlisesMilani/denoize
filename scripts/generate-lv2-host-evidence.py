#!/usr/bin/env python3
"""Bind the pinned LV2 validator, Jalv, and Ardour reports to a release."""

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
JALV_SMOKE_RE = re.compile(
    r"^DENOIZE_LV2_JALV_SMOKE sample_rate_hz=48000 block_frames=480 "
    r"descriptors=2 stereo_connected=true worker_host=true teardown=true$",
    re.MULTILINE,
)
ARDOUR_SMOKE_RE = re.compile(
    r"^DENOIZE_LV2_ARDOUR_SMOKE first_pass_frames=([1-9][0-9]*) "
    r"restored_pass_frames=([1-9][0-9]*) sample_rate_hz=48000 "
    r"descriptors=2 state_reload=true teardown=true$",
    re.MULTILINE,
)
MAX_REPORT_BYTES = 4 * 1024 * 1024


class EvidenceError(RuntimeError):
    pass


def regular_report(path: Path, label: str) -> tuple[bytes, str]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"{label} is not a regular file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > MAX_REPORT_BYTES:
        raise EvidenceError(f"{label} size must be in 1..={MAX_REPORT_BYTES} bytes")
    payload = path.read_bytes()
    try:
        return payload, payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{label} is not UTF-8") from error


def require_once(report: str, record: str, label: str) -> None:
    if report.count(record) != 1:
        raise EvidenceError(f"{label} must contain one exact record: {record}")


def file_record(path: Path, payload: bytes) -> dict[str, object]:
    return {
        "name": path.name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_exclusive(path: Path, document: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise EvidenceError(f"refusing to replace existing LV2 evidence: {path}")
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
        raise EvidenceError(f"invalid release tag: {args.tag}")
    if not COMMIT_RE.fullmatch(args.commit):
        raise EvidenceError("source commit must be a lowercase 40-character SHA-1")
    if not REPOSITORY_RE.fullmatch(args.repository):
        raise EvidenceError("repository must be an owner/name pair")

    validation_path = args.validation_report.resolve()
    validation_payload, validation = regular_report(validation_path, "validation report")
    for record in (
        "denoize LV2 validation report",
        "lv2_specification: 1.18.10",
        "operating_system: ubuntu-24.04",
        "architecture: x86_64",
        "descriptor_count: 2",
        "metadata_validation: passed",
        "dsp_in_place_offline_host_processing: passed",
        "neural_worker_host: delegated-to-jalv",
        "binary_hardening: passed",
        "Result: LV2 validation passed",
    ):
        require_once(validation, record, "validation report")
    require_once(validation, "Name:              denoize\n", "validation report")
    require_once(validation, "Name:              denoize Neural\n", "validation report")

    jalv_path = args.jalv_report.resolve()
    jalv_payload, jalv = regular_report(jalv_path, "Jalv report")
    for record in (
        "denoize LV2 Jalv real-host report",
        "host: Jalv",
        "host_package_version: 1.6.8-1build3",
        "jack_package_version: 1.9.21~dfsg-3ubuntu3",
        "operating_system: ubuntu-24.04",
        "architecture: x86_64",
        "sample_rate_hz: 48000",
        "block_frames: 480",
        "descriptors: 2",
        "audio_connections: stereo-in-stereo-out",
        "dsp_minimum_active_seconds: 5",
        "neural_minimum_active_seconds: 15",
        "Result: Jalv real-host smoke passed",
    ):
        require_once(jalv, record, "Jalv report")
    if len(JALV_SMOKE_RE.findall(jalv)) != 1:
        raise EvidenceError("Jalv report must contain one complete worker-host summary")

    ardour_path = args.ardour_report.resolve()
    ardour_payload, ardour = regular_report(ardour_path, "Ardour report")
    for record in (
        "denoize LV2 Ardour real-host smoke report",
        "host: Ardour",
        "host_version: 8.4.0~ds1",
        "package_version: 1:8.4.0+ds1-2ubuntu8",
        "operating_system: ubuntu-24.04",
        "architecture: x86_64",
        "DENOIZE_LV2_ARDOUR_TEARDOWN phase=create passed=true",
        "DENOIZE_LV2_ARDOUR_TEARDOWN phase=restore passed=true",
        "DENOIZE_LV2_ARDOUR_STATE properties=2 portable=true interface_errors=0",
        "Result: Ardour LV2 real-host smoke passed",
    ):
        require_once(ardour, record, "Ardour report")
    ardour_matches = ARDOUR_SMOKE_RE.findall(ardour)
    if len(ardour_matches) != 1:
        raise EvidenceError("Ardour report must contain one complete state-reload summary")
    first_pass_frames, restored_pass_frames = (int(value) for value in ardour_matches[0])

    document: dict[str, object] = {
        "schema": "denoize-lv2-host-evidence-v1",
        "schema_version": 1,
        "tag": args.tag,
        "source": {"repository": args.repository, "commit": args.commit},
        "format": "lv2",
        "adapter": {
            "strategy": "direct-rust-lv2",
            "lv2_specification": "1.18.10",
            "rust_lv2_version": "0.6.0",
            "lv2_dev_package": "1.18.10-2build1",
            "lilv_utils_package": "0.24.22-1build1",
            "sordi_package": "0.16.16-2build1",
            "jalv_package": "1.6.8-1build3",
            "jackd2_package": "1.9.21~dfsg-3ubuntu3",
            "ardour_package": "1:8.4.0+ds1-2ubuntu8",
        },
        "descriptors": [
            {
                "uri": "https://github.com/penguin425/denoize#lv2-dsp",
                "name": "denoize",
                "ports": 13,
                "audio_inputs": 2,
                "audio_outputs": 2,
                "latency_frames_48khz": 480,
                "state_property": "https://github.com/penguin425/denoize#dsp-state",
                "worker_required": False,
            },
            {
                "uri": "https://github.com/penguin425/denoize#lv2-neural",
                "name": "denoize Neural",
                "ports": 16,
                "audio_inputs": 2,
                "audio_outputs": 2,
                "latency_frames_48khz": 11_520,
                "state_property": "https://github.com/penguin425/denoize#neural-state",
                "worker_required": True,
            },
        ],
        "runs": [
            {
                "host": "LV2 reference tools and Lilv",
                "host_version": "1.18.10",
                "evidence_kind": "official-validation",
                "operating_system": "ubuntu-24.04",
                "architecture": "x86_64",
                "status": "passed",
                "descriptors_exercised": 2,
                "report": file_record(validation_path, validation_payload),
            },
            {
                "host": "Jalv",
                "host_version": "1.6.8-1build3",
                "evidence_kind": "real-host-worker-smoke",
                "operating_system": "ubuntu-24.04",
                "architecture": "x86_64",
                "status": "passed",
                "descriptors_exercised": 2,
                "sample_rate_hz": 48_000,
                "block_frames": 480,
                "worker_host": True,
                "teardown": True,
                "report": file_record(jalv_path, jalv_payload),
            },
            {
                "host": "Ardour",
                "host_version": "8.4.0~ds1",
                "evidence_kind": "real-host-state-smoke",
                "operating_system": "ubuntu-24.04",
                "architecture": "x86_64",
                "status": "passed",
                "descriptors_exercised": 2,
                "sample_rate_hz": 48_000,
                "first_pass_frames": first_pass_frames,
                "restored_pass_frames": restored_pass_frames,
                "state_properties": 2,
                "state_reload": True,
                "state_interface_errors": 0,
                "teardown": True,
                "report": file_record(ardour_path, ardour_payload),
            },
        ],
        "claims": {
            "direct_adapter": True,
            "official_metadata_validation": True,
            "lilv_discovery": True,
            "jalv_real_host": True,
            "ardour_real_host": True,
            "state_roundtrip": True,
            "worker_host": True,
            "sample_accurate_automation": True,
            "single_precision_audio": True,
            "double_precision_audio": False,
        },
        "limitations": [
            "custom-editor-not-present",
            "double-precision-audio-not-supported",
            "linux-x86_64-only",
            "lv2bench-neural-worker-not-supported",
            "proprietary-hosts-not-exercised",
        ],
    }
    write_exclusive(args.output, document)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--tag", required=True)
    result.add_argument("--commit", required=True)
    result.add_argument("--repository", required=True)
    result.add_argument("--validation-report", type=Path, required=True)
    result.add_argument("--jalv-report", type=Path, required=True)
    result.add_argument("--ardour-report", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    try:
        generate(parser().parse_args())
    except (EvidenceError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
