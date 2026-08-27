#!/usr/bin/env python3
"""Validate microphone-array report and signed promotion-evidence contracts."""

from __future__ import annotations

import copy
import json
import math
import pathlib

import jsonschema


ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIRED_STRATA = [
    "bad-channel",
    "channel-permutation",
    "clock-skew",
    "diffuse-noise",
    "directional-noise",
    "gain-phase-mismatch",
    "moving-source",
    "program-stereo",
    "real-meeting",
    "simulated-rir",
    "two-microphone",
    "unseen-geometry",
]


def schema(name: str) -> dict:
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


def valid_evidence() -> dict:
    return {
        "schema": "denoize-microphone-array-promotion-evidence-v1",
        "schema_version": 1,
        "payload": {
            "completed_at_unix_seconds": 1,
            "implementation": "native-wpe-mask-mvdr-v1",
            "implementation_source_revision": "0123456789abcdef",
            "implementation_source_sha256": "1" * 64,
            "configuration_sha256": "2" * 64,
            "corpus_manifest_sha256": "3" * 64,
            "evaluation_result_sha256": "4" * 64,
            "listening_result_sha256": "5" * 64,
            "strata": [
                {
                    "id": identifier,
                    "cases": 100,
                    "si_sdr_improvement_db": 0.0,
                    "wer_regression": 0.02,
                    "doa_error_degrees": 20.0,
                    "reference_coloration_db": 1.5,
                    "target_leakage_db": -3.0,
                    "non_finite_samples": 0,
                    "passed": True,
                }
                for identifier in REQUIRED_STRATA
            ],
            "real_meeting_cases": 100,
            "unseen_geometry_cases": 100,
            "permutation_cases": 100,
            "paced_realtime_blocks": 10_000,
            "worst_case_realtime_factor": 0.5,
            "callback_allocations": 0,
            "callback_locks": 0,
            "callback_waits": 0,
            "deadline_misses": 0,
            "listener_count": 20,
            "listener_preference": 0.5,
            "accepted": True,
        },
        "signature": {
            "algorithm": "ed25519",
            "key_id": "6" * 64,
            "value_base64": "A" * 86 + "==",
        },
    }


def valid_report() -> dict:
    return {
        "schema": "denoize-microphone-array-report-v1",
        "schema_version": 1,
        "implementation": "native-wpe-mask-mvdr-v1",
        "configuration_sha256": "0" * 64,
        "evidence_signing_key_id": "1" * 64,
        "evidence_evaluation_result_sha256": "2" * 64,
        "input_pcm_sha256": "3" * 64,
        "output_pcm_sha256": "4" * 64,
        "sample_rate": 48_000,
        "input_channels": 2,
        "input_frames": 3_211,
        "output_channels": 1,
        "output_frames": 3_211,
        "reference_microphone_id": "mic-0",
        "canonical_microphone_ids": ["mic-0", "mic-1"],
        "active_microphones": 2,
        "inactive_microphone_ids": [],
        "frame_size": 512,
        "hop_size": 128,
        "algorithmic_latency_milliseconds": 8.0,
        "solved_frequency_bins": 257,
        "fallback_frequency_bins": 0,
        "maximum_observed_condition_number": 10.0,
        "clipped_samples": 0,
        "non_finite_samples": 0,
        "exact_output_duration": True,
        "paths_recorded": 0,
        "limitations": [
            "explicit array semantics only",
            "deterministic WPE and mask-MVDR baseline",
            "reference fallback for ill-conditioned bins",
            "no bundled neural checkpoint",
            "moving-source streaming is not promoted",
        ],
    }


def expect_schema_failure(validator: jsonschema.Draft202012Validator, document: dict) -> None:
    if not list(validator.iter_errors(document)):
        raise AssertionError(
            "invalid microphone-array document unexpectedly passed JSON Schema"
        )


def validate_evidence_semantics(document: dict) -> None:
    payload = document["payload"]
    if [stratum["id"] for stratum in payload["strata"]] != REQUIRED_STRATA:
        raise AssertionError("microphone-array strata must be exact and sorted")
    all_passed = True
    for stratum in payload["strata"]:
        passed = (
            stratum["si_sdr_improvement_db"] >= 0.0
            and stratum["wer_regression"] <= 0.02
            and stratum["doa_error_degrees"] <= 20.0
            and abs(stratum["reference_coloration_db"]) <= 1.5
            and stratum["target_leakage_db"] <= -3.0
            and stratum["non_finite_samples"] == 0
        )
        if stratum["passed"] != passed:
            raise AssertionError("microphone-array stratum pass flag is inconsistent")
        all_passed &= passed
    global_gates = (
        payload["real_meeting_cases"] >= 100
        and payload["unseen_geometry_cases"] >= 100
        and payload["permutation_cases"] >= 100
        and payload["paced_realtime_blocks"] >= 10_000
        and payload["worst_case_realtime_factor"] <= 0.5
        and payload["callback_allocations"] == 0
        and payload["callback_locks"] == 0
        and payload["callback_waits"] == 0
        and payload["deadline_misses"] == 0
        and payload["listener_count"] >= 20
        and payload["listener_preference"] >= 0.5
    )
    if payload["accepted"] != (all_passed and global_gates):
        raise AssertionError("microphone-array accepted flag is inconsistent")


def validate_report_semantics(document: dict) -> None:
    if document["output_frames"] != document["input_frames"]:
        raise AssertionError("microphone-array report must preserve exact duration")
    identifiers = document["canonical_microphone_ids"]
    if identifiers != sorted(identifiers) or len(identifiers) != document["input_channels"]:
        raise AssertionError("canonical microphone IDs do not bind every input channel")
    if document["reference_microphone_id"] not in identifiers:
        raise AssertionError("reference microphone is missing from canonical IDs")
    inactive = document["inactive_microphone_ids"]
    if not set(inactive).issubset(identifiers):
        raise AssertionError("inactive microphone IDs are not declared channels")
    if document["reference_microphone_id"] in inactive:
        raise AssertionError("inactive reference microphone must fail before reporting")
    if document["active_microphones"] + len(inactive) != document["input_channels"]:
        raise AssertionError("active and inactive microphone counts are incomplete")
    frame_size = document["frame_size"]
    hop_size = document["hop_size"]
    if frame_size & (frame_size - 1) or hop_size > frame_size // 2:
        raise AssertionError("microphone-array STFT geometry is invalid")
    expected_latency = (frame_size - hop_size) * 1000 / document["sample_rate"]
    if not math.isclose(
        document["algorithmic_latency_milliseconds"], expected_latency, abs_tol=1e-9
    ):
        raise AssertionError("microphone-array latency does not match STFT geometry")
    if (
        document["solved_frequency_bins"] + document["fallback_frequency_bins"]
        != frame_size // 2 + 1
    ):
        raise AssertionError("microphone-array bin decisions do not cover the spectrum")


def main() -> None:
    evidence_validator = jsonschema.Draft202012Validator(
        schema("denoize-microphone-array-promotion-evidence-v1.schema.json")
    )
    report_validator = jsonschema.Draft202012Validator(
        schema("denoize-microphone-array-report-v1.schema.json")
    )
    evidence = valid_evidence()
    report = valid_report()
    evidence_validator.validate(evidence)
    report_validator.validate(report)
    validate_evidence_semantics(evidence)
    validate_report_semantics(report)

    invalid = copy.deepcopy(evidence)
    invalid["payload"]["callback_allocations"] = 1
    expect_schema_failure(evidence_validator, invalid)

    invalid = copy.deepcopy(evidence)
    invalid["payload"]["strata"][0]["id"] = "unexpected"
    evidence_validator.validate(invalid)
    try:
        validate_evidence_semantics(invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("incomplete array evaluation matrix unexpectedly passed")

    invalid = copy.deepcopy(evidence)
    invalid["payload"]["strata"][0]["wer_regression"] = 0.03
    evidence_validator.validate(invalid)
    try:
        validate_evidence_semantics(invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("inconsistent array promotion outcome unexpectedly passed")

    invalid = copy.deepcopy(report)
    invalid["output_frames"] -= 1
    report_validator.validate(invalid)
    try:
        validate_report_semantics(invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("array duration mismatch unexpectedly passed")

    invalid = copy.deepcopy(report)
    invalid["paths_recorded"] = 1
    expect_schema_failure(report_validator, invalid)
    print("microphone-array JSON schema tests passed")


if __name__ == "__main__":
    main()
