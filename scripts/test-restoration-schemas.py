#!/usr/bin/env python3
"""Validate the closed deterministic-restoration report and mask schemas."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib

import jsonschema


ROOT = pathlib.Path(__file__).resolve().parents[1]
OPERATION_ORDER = ("declip", "declick", "dehum", "dereverb", "wind-plosive")


def valid_mask() -> dict:
    return {
        "schema": "denoize-restoration-mask-v1",
        "schema_version": 1,
        "channels": 1,
        "frames": 16,
        "runs": [
            {
                "channel": 0,
                "start_frame": 0,
                "frame_count": 7,
                "state": "untouched",
                "operations": [],
                "confidence": 0.0,
            },
            {
                "channel": 0,
                "start_frame": 7,
                "frame_count": 2,
                "state": "replaced",
                "operations": ["declick"],
                "confidence": 0.91,
            },
            {
                "channel": 0,
                "start_frame": 9,
                "frame_count": 7,
                "state": "untouched",
                "operations": [],
                "confidence": 0.0,
            },
        ],
    }


def valid_report(mask: dict) -> dict:
    mask_digest = hashlib.sha256(
        json.dumps(mask, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "denoize-restoration-report-v1",
        "schema_version": 1,
        "mode": "apply",
        "sample_rate": 48_000,
        "channels": 1,
        "frames": 16,
        "input_pcm_sha256": "1" * 64,
        "mask_sha256": mask_digest,
        "deterministic": True,
        "bypassed": False,
        "detected_samples": 2,
        "changed_samples": 2,
        "confidence": 0.91,
        "energy_delta_db": -0.02,
        "operations": [
            {
                "operation": "declick",
                "status": "applied",
                "detected_samples": 2,
                "changed_samples": 2,
                "confidence": 0.91,
                "energy_delta_db": -0.02,
                "warnings": [],
                "details": {
                    "kind": "declick",
                    "regions": 1,
                    "rejected_regions": 0,
                    "prediction_order": 12,
                    "maximum_gap_samples": 144,
                },
            }
        ],
        "warnings": [],
    }


def exact_coverage(mask: dict) -> bool:
    cursors = [0] * mask["channels"]
    previous = None
    for run in mask["runs"]:
        channel = run["channel"]
        position = (channel, run["start_frame"])
        if (
            channel >= len(cursors)
            or run["start_frame"] != cursors[channel]
            or (previous is not None and previous >= position)
        ):
            return False
        previous = position
        cursors[channel] += run["frame_count"]
    return all(cursor == mask["frames"] for cursor in cursors)


def report_semantics(report: dict) -> bool:
    operations = [entry["operation"] for entry in report["operations"]]
    ranks = [OPERATION_ORDER.index(operation) for operation in operations]
    maximum_samples = report["channels"] * report["frames"]
    return (
        len(operations) == len(set(operations))
        and ranks == sorted(ranks)
        and report["detected_samples"] <= maximum_samples
        and report["changed_samples"] <= maximum_samples
        and all(entry["operation"] == entry["details"]["kind"] for entry in report["operations"])
    )


def strict_json_bytes(value: dict) -> bytes:
    """Mirror serde_json's rejection of JSON-nonfinite IEEE-754 values."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def main() -> None:
    mask_schema = json.loads(
        (ROOT / "schemas/denoize-restoration-mask-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    report_schema = json.loads(
        (ROOT / "schemas/denoize-restoration-report-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator.check_schema(mask_schema)
    jsonschema.Draft202012Validator.check_schema(report_schema)
    mask_validator = jsonschema.Draft202012Validator(mask_schema)
    report_validator = jsonschema.Draft202012Validator(report_schema)
    mask = valid_mask()
    report = valid_report(mask)
    mask_validator.validate(mask)
    report_validator.validate(report)
    assert exact_coverage(mask)
    assert report_semantics(report)
    strict_json_bytes(mask)
    strict_json_bytes(report)

    unknown = copy.deepcopy(report)
    unknown["input_path"] = "/private/audio.wav"
    assert not report_validator.is_valid(unknown)
    nonfinite = copy.deepcopy(report)
    nonfinite["confidence"] = float("nan")
    try:
        strict_json_bytes(nonfinite)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite restoration JSON was serialized")
    wrong_details = copy.deepcopy(report)
    wrong_details["operations"][0]["details"] = {
        "kind": "dehum",
        "fundamental_hz": 50.0,
        "tracked_blocks": 1,
        "harmonic_count": 3,
        "attenuation_db": 20.0,
    }
    assert not report_validator.is_valid(wrong_details)
    detect_applied = copy.deepcopy(report)
    detect_applied["mode"] = "detect-only"
    assert not report_validator.is_valid(detect_applied)
    duplicate_report_operation = copy.deepcopy(report)
    duplicate_report_operation["operations"].append(
        copy.deepcopy(duplicate_report_operation["operations"][0])
    )
    duplicate_report_operation["operations"][1]["confidence"] = 0.8
    assert report_validator.is_valid(duplicate_report_operation)
    assert not report_semantics(duplicate_report_operation)
    unknown_mask = copy.deepcopy(mask)
    unknown_mask["runs"][1]["samples"] = [0.1, 0.2]
    assert not mask_validator.is_valid(unknown_mask)
    duplicate_operation = copy.deepcopy(mask)
    duplicate_operation["runs"][1]["operations"] = ["declick", "declick"]
    assert not mask_validator.is_valid(duplicate_operation)
    untouched_operation = copy.deepcopy(mask)
    untouched_operation["runs"][0]["operations"] = ["declick"]
    assert not mask_validator.is_valid(untouched_operation)
    detected_without_operation = copy.deepcopy(mask)
    detected_without_operation["runs"][1]["operations"] = []
    assert not mask_validator.is_valid(detected_without_operation)
    gap = copy.deepcopy(mask)
    gap["runs"][1]["start_frame"] = 8
    mask_validator.validate(gap)
    assert not exact_coverage(gap)
    out_of_order = copy.deepcopy(mask)
    out_of_order["channels"] = 2
    channel_one = copy.deepcopy(out_of_order["runs"])
    for run in channel_one:
        run["channel"] = 1
    out_of_order["runs"] = channel_one + out_of_order["runs"]
    mask_validator.validate(out_of_order)
    assert not exact_coverage(out_of_order)


if __name__ == "__main__":
    main()
