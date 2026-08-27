#!/usr/bin/env python3
"""Bind AUv3 auval and AVFoundation reports into one target-qualified record."""

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
VERSION_RE = re.compile(r"^host_version: ([0-9]+\.[0-9]+(?:\.[0-9]+)?)$", re.MULTILINE)
ARCHITECTURE_RE = re.compile(r"^architecture: (arm64|x86_64)$", re.MULTILINE)
MAX_REPORT_BYTES = 4 * 1024 * 1024


class EvidenceError(RuntimeError):
    pass


def regular_report(path: Path, label: str) -> tuple[bytes, str]:
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


def report_file(path: Path, payload: bytes) -> dict[str, object]:
    return {
        "name": path.name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def report_platform(report: str, label: str) -> tuple[str, str]:
    versions = VERSION_RE.findall(report)
    architectures = ARCHITECTURE_RE.findall(report)
    if len(versions) != 1 or len(architectures) != 1:
        raise EvidenceError(f"{label} must contain one host version and architecture")
    if report.count("operating_system: macos") != 1:
        raise EvidenceError(f"{label} must identify macOS exactly once")
    return versions[0], architectures[0]


def write_exclusive(path: Path, document: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise EvidenceError(f"refusing to replace existing evidence: {path}")
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
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            descriptor = -1
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
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

    auval_path = args.auval_report.resolve()
    auval_payload, auval = regular_report(auval_path, "auval report")
    for subtype in ("Dn01", "Dn02"):
        marker = f"DENOIZE_AUV3_AUVAL_RESULT subtype={subtype} passed=true"
        if auval.count(marker) != 1:
            raise EvidenceError(f"auval report must pass {subtype} exactly once")
    if auval.count("Result: AUv3 auval passed 2 components") != 1:
        raise EvidenceError("auval report has no complete success result")
    if "passed=false" in auval:
        raise EvidenceError("auval report contains a failed component")
    auval_version, auval_arch = report_platform(auval, "auval report")

    host_path = args.host_report.resolve()
    host_payload, host = regular_report(host_path, "AVFoundation host report")
    for subtype in ("Dn01", "Dn02"):
        marker = (
            f"DENOIZE_AUV3_SMOKE component={subtype} instantiated=true allocated=true "
            "state_round_trip=true teardown=true"
        )
        if host.count(marker) != 1:
            raise EvidenceError(f"AVFoundation host report must exercise {subtype} exactly once")
    if host.count("Result: AUv3 AVFoundation host smoke passed") != 1:
        raise EvidenceError("AVFoundation host report has no complete success result")
    host_version, host_arch = report_platform(host, "AVFoundation host report")
    if host_arch != auval_arch:
        raise EvidenceError("AUv3 reports disagree on architecture")

    document: dict[str, object] = {
        "schema": "denoize-auv3-host-evidence-v1",
        "schema_version": 1,
        "tag": args.tag,
        "source": {"repository": args.repository, "commit": args.commit},
        "format": "auv3",
        "adapter": {
            "strategy": "signed-embedded-clap-wrapper",
            "clap_wrapper": {
                "version": "0.16.0",
                "commit": "1cca996e96f29ab2be7ae9f8cfe532bbc92e1dd6",
            },
            "clap_sdk": {
                "version": "1.2.6",
                "commit": "69a69252fdd6ac1d06e246d9a04c0a89d9607a17",
            },
        },
        "components": [
            {
                "descriptor_id": "org.penguin425.denoize",
                "name": "denoize",
                "type": "aufx",
                "subtype": "Dn01",
                "manufacturer": "Dnze",
                "parameters": 7,
            },
            {
                "descriptor_id": "org.penguin425.denoize.neural",
                "name": "denoize Neural",
                "type": "aufx",
                "subtype": "Dn02",
                "manufacturer": "Dnze",
                "parameters": 4,
            },
        ],
        "bundled_model": {
            "name": "gtcrn-dns3",
            "filename": "gtcrn_simple.onnx",
            "size_bytes": 535_190,
            "sha256": "b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87",
            "authenticated_provenance": True,
        },
        "runs": [
            {
                "host": "auval",
                "host_version": auval_version,
                "operating_system": "macos",
                "architecture": auval_arch,
                "evidence_kind": "official-validator",
                "status": "passed",
                "components_exercised": 2,
                "state_round_trip": False,
                "teardown": False,
                "report": report_file(auval_path, auval_payload),
            },
            {
                "host": "AVFoundation",
                "host_version": host_version,
                "operating_system": "macos",
                "architecture": host_arch,
                "evidence_kind": "real-host-smoke",
                "status": "passed",
                "components_exercised": 2,
                "state_round_trip": True,
                "teardown": True,
                "report": report_file(host_path, host_payload),
            },
        ],
        "claims": {
            "official_auval": True,
            "avfoundation_real_host": True,
            "app_extension_sandbox": True,
            "self_contained_model": True,
            "embedded_editor": False,
        },
        "limitations": [
            "custom-view-not-exercised",
            "ios-not-shipped",
            "macos-only",
            "proprietary-third-party-hosts-not-exercised",
            "standalone-opens-standard-component",
        ],
    }
    write_exclusive(args.output, document)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--tag", required=True)
    result.add_argument("--commit", required=True)
    result.add_argument("--repository", required=True)
    result.add_argument("--auval-report", type=Path, required=True)
    result.add_argument("--host-report", type=Path, required=True)
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
