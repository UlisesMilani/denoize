#!/usr/bin/env python3
"""Fetch and validate issue #221 reporter evidence from a GitHub comment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request
from typing import Any


COMMENT_RE = re.compile(
    r"^https://github\.com/penguin425/denoize/issues/221#issuecomment-([1-9][0-9]*)$"
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PARAMETERS = ["Bypass", "Mix", "Output Gain", "Overload Fallback"]
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class ReporterError(RuntimeError):
    pass


def api(path: str) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "denoize-dpdfnet-promotion-evidence-v1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"https://api.github.com{path}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.URLError as error:
        raise ReporterError(f"GitHub API request failed: {error}") from error
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ReporterError("GitHub API response exceeds the size limit")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReporterError(f"GitHub API returned invalid JSON: {error}") from error
    if not isinstance(document, dict):
        raise ReporterError("GitHub API response must be an object")
    return document


def exact(document: dict, keys: set[str], label: str) -> None:
    if set(document) != keys:
        raise ReporterError(f"{label} has missing or unknown fields")


def bounded(value: Any, label: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum or "\x00" in value:
        raise ReporterError(f"{label} must be a bounded non-empty string")
    return value


def validate_payload(payload: dict[str, Any]) -> None:
    exact(
        payload,
        {
            "schema", "schema_version", "source_commit", "artifact_sha256", "environment",
            "runs", "accessibility", "quality_observation", "consent_to_publish",
        },
        "reporter payload",
    )
    if payload["schema"] != "denoize-dpdfnet-reporter-submission-v1" or payload["schema_version"] != 1:
        raise ReporterError("unsupported reporter-submission schema")
    if not isinstance(payload["source_commit"], str) or not COMMIT_RE.fullmatch(payload["source_commit"]):
        raise ReporterError("reporter submission has an invalid source commit")
    if not isinstance(payload["artifact_sha256"], str) or not SHA256_RE.fullmatch(payload["artifact_sha256"]):
        raise ReporterError("reporter submission has an invalid artifact digest")
    if payload["consent_to_publish"] is not True:
        raise ReporterError("reporter did not consent to publish the evidence")
    if payload["quality_observation"] not in {"dpdfnet-better", "equivalent", "gtcrn-better"}:
        raise ReporterError("quality_observation is invalid")
    environment = payload["environment"]
    if not isinstance(environment, dict):
        raise ReporterError("environment must be an object")
    exact(environment, {"windows_version", "cpu_model", "audio_device", "audio_driver", "reaper_version", "nvda_version", "osara_version"}, "reporter environment")
    for name, value in environment.items():
        bounded(value, name)
    version_match = re.fullmatch(r"([0-9]+)\.([0-9]+)", environment["reaper_version"])
    if version_match is None or tuple(map(int, version_match.groups())) < (7, 79):
        raise ReporterError("REAPER must be version 7.79 or newer")

    runs = payload["runs"]
    if not isinstance(runs, list) or not 3 <= len(runs) <= 16:
        raise ReporterError("reporter submission must contain 3..=16 buffer runs")
    buffers: set[int] = set()
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ReporterError(f"run {index} must be an object")
        exact(run, {"buffer_frames", "sample_rate_hz", "duration_seconds", "overload_events", "late_events", "audible_xruns", "continuous_audio"}, f"run {index}")
        for name in ("buffer_frames", "sample_rate_hz", "duration_seconds", "overload_events", "late_events", "audible_xruns"):
            if isinstance(run[name], bool) or not isinstance(run[name], int):
                raise ReporterError(f"run {index} {name} must be an integer")
        if run["sample_rate_hz"] != 48_000 or run["duration_seconds"] < 300:
            raise ReporterError(f"run {index} must cover at least 300 seconds at 48 kHz")
        if run["overload_events"] != 0 or run["late_events"] != 0 or run["audible_xruns"] != 0 or run["continuous_audio"] is not True:
            raise ReporterError(f"run {index} did not pass the realtime gate")
        if not 16 <= run["buffer_frames"] <= 8192:
            raise ReporterError(f"run {index} buffer size is outside 16..=8192")
        if run["buffer_frames"] in buffers:
            raise ReporterError("buffer runs must be unique")
        buffers.add(run["buffer_frames"])
    if min(buffers) > 128 or 480 not in buffers or max(buffers) < 1024:
        raise ReporterError("buffer coverage must include <=128, exactly 480, and >=1024 frames")

    accessibility = payload["accessibility"]
    if not isinstance(accessibility, dict):
        raise ReporterError("accessibility must be an object")
    exact(accessibility, {"nvda_active", "osara_active", "parameters_announced", "values_announced", "all_adjustable", "focus_stable", "host_or_plugin_crashes"}, "accessibility")
    if accessibility["parameters_announced"] != PARAMETERS:
        raise ReporterError("NVDA/OSARA must announce the four closed parameter names in order")
    if any(accessibility[name] is not True for name in ("nvda_active", "osara_active", "values_announced", "all_adjustable", "focus_stable")):
        raise ReporterError("one or more NVDA/OSARA checks failed")
    if accessibility["host_or_plugin_crashes"] != 0:
        raise ReporterError("the human test recorded a host or plug-in crash")


def generate(args: argparse.Namespace) -> None:
    match = COMMENT_RE.fullmatch(args.comment_url)
    if not match:
        raise ReporterError("comment URL must refer to issue #221 in penguin425/denoize")
    comment_id = int(match.group(1))
    issue = api("/repos/penguin425/denoize/issues/221")
    comment = api(f"/repos/penguin425/denoize/issues/comments/{comment_id}")
    issue_login = issue.get("user", {}).get("login")
    comment_login = comment.get("user", {}).get("login")
    if issue_login != "UlisesMilani" or comment_login != issue_login:
        raise ReporterError("the evidence comment was not authored by the issue #221 reporter")
    if comment.get("html_url") != args.comment_url:
        raise ReporterError("GitHub API comment URL differs from the requested URL")
    body = comment.get("body")
    if not isinstance(body, str) or len(body.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ReporterError("comment body is missing or too large")
    matches = re.findall(r"```json\s*\n(\{.*?\})\s*\n```", body, flags=re.DOTALL)
    if len(matches) != 1:
        raise ReporterError("comment must contain exactly one fenced JSON evidence object")
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError as error:
        raise ReporterError(f"comment evidence is invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ReporterError("comment evidence must be an object")
    validate_payload(payload)
    document = {
        "schema": "denoize-dpdfnet-reporter-evidence-v1",
        "schema_version": 1,
        "github": {
            "repository": "penguin425/denoize",
            "issue": 221,
            "comment_id": comment_id,
            "comment_url": args.comment_url,
            "author": comment_login,
            "created_at": comment.get("created_at"),
            "updated_at": comment.get("updated_at"),
            "comment_body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        },
        "payload": payload,
        "accepted": True,
    }
    output = args.output
    if output.exists() or output.is_symlink():
        raise ReporterError(f"refusing to replace existing reporter evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as destination:
            descriptor = -1
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--comment-url", required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    try:
        generate(parser().parse_args())
    except (ReporterError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
