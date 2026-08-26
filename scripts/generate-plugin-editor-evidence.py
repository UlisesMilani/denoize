#!/usr/bin/env python3
"""Bind the real CLAP editor host smoke report into closed release evidence."""

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
PORTABLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
DESCRIPTOR_RE = re.compile(
    r"^DENOIZE_EDITOR_HOST descriptor=(denoize(?: Neural)?) "
    r"rendered_colors=([4-9]|[1-9][0-9]+) automation_events=3 "
    r"bypass_value=1\.0 lifecycle=true resize_contract=true$",
    re.MULTILINE,
)
MAX_REPORT_BYTES = 2 * 1024 * 1024


class EvidenceError(RuntimeError):
    pass


def regular_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"editor host report is not a regular file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > MAX_REPORT_BYTES:
        raise EvidenceError(
            f"editor host report size must be in 1..={MAX_REPORT_BYTES} bytes"
        )
    return path.read_bytes()


def write_exclusive(path: Path, document: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise EvidenceError(f"refusing to replace existing editor evidence: {path}")
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
    for label, value in (
        ("operating system", args.operating_system),
        ("architecture", args.architecture),
    ):
        if not PORTABLE_RE.fullmatch(value):
            raise EvidenceError(f"invalid {label}: {value}")

    # Validate the path exactly as supplied before resolving it. Resolving
    # first would erase the evidence that the caller supplied a symlink.
    payload = regular_bytes(args.report)
    report_path = args.report.resolve()
    try:
        report = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceError("editor host report is not UTF-8") from error
    required = (
        "denoize CLAP editor real-host smoke report",
        "host: clack-host 0.1.1 + baseview X11 parent",
        "display: Xvfb/X11",
        "descriptors: 2",
        "Result: CLAP editor real-host smoke passed",
    )
    for record in required:
        if report.count(record) != 1:
            raise EvidenceError(f"editor host report must contain one exact record: {record}")
    matches = DESCRIPTOR_RE.findall(report)
    if len(matches) != 2:
        raise EvidenceError("editor host report must contain two complete descriptor records")
    by_name = {name: int(colors) for name, colors in matches}
    if set(by_name) != {"denoize", "denoize Neural"} or len(matches) != len(by_name):
        raise EvidenceError("editor host report descriptor identities are incomplete or duplicated")

    document: dict[str, object] = {
        "schema": "denoize-plugin-editor-evidence-v1",
        "schema_version": 1,
        "tag": args.tag,
        "source": {"repository": args.repository, "commit": args.commit},
        "editor": {
            "format": "clap",
            "embedding": "native-child-window",
            "window_api": "x11",
            "baseview_commit": "aba6ad070828ba31174ae1e60c9b20d90d699e87",
            "renderer": "softbuffer-0.4.8",
            "accessibility_core": "accesskit-0.24.1",
        },
        "descriptors": [
            {
                "id": "org.penguin425.denoize",
                "name": "denoize",
                "rendered_colors": by_name["denoize"],
                "automation_events": 3,
                "bypass_value": 1.0,
                "lifecycle": True,
                "resize_contract": True,
            },
            {
                "id": "org.penguin425.denoize.neural",
                "name": "denoize Neural",
                "rendered_colors": by_name["denoize Neural"],
                "automation_events": 3,
                "bypass_value": 1.0,
                "lifecycle": True,
                "resize_contract": True,
            },
        ],
        "run": {
            "host": "clack-host",
            "host_version": "0.1.1",
            "operating_system": args.operating_system,
            "architecture": args.architecture,
            "display": "Xvfb/X11",
            "descriptors_exercised": 2,
            "status": "passed",
            "report": {
                "name": report_path.name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        },
        "claims": {
            "custom_editor": True,
            "native_embedded": True,
            "host_parameter_automation": True,
            "generic_parameter_fallback": True,
            "lifecycle_contract": True,
            "resize_contract": True,
        },
        "limitations": [
            "floating-window-not-supported",
            "linux-x11-evidence-only",
            "proprietary-hosts-not-exercised",
            "wayland-custom-editor-not-supported",
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
    result.add_argument("--report", type=Path, required=True)
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
