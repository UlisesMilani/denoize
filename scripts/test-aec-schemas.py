#!/usr/bin/env python3
"""Validate AEC report and signed promotion-evidence contracts."""

from __future__ import annotations

import copy
import json
import math
import pathlib

import jsonschema


ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMON_METRICS = {
    "latency.algorithmic-plus-buffering-ms": ("less-or-equal", 20.0),
    "output.duration-error-frames": ("less-or-equal", 0.0),
    "output.non-finite-samples": ("less-or-equal", 0.0),
}
KIND_METRICS = {
    "far-end-only": {
        "echo.erle-db": ("greater-or-equal", 10.0),
        "perceptual.aecmos-far-end": ("greater-or-equal", 3.5),
    },
    "near-end-only": {
        "content.word-accuracy-regression": ("less-or-equal", 0.02),
        "near-end.attenuation-db": ("less-or-equal", 1.0),
    },
    "double-talk": {
        "content.word-accuracy-regression": ("less-or-equal", 0.02),
        "near-end.attenuation-db": ("less-or-equal", 1.5),
        "perceptual.aecmos-double-talk": ("greater-or-equal", 3.2),
    },
    "transition": {
        "near-end.attenuation-db": ("less-or-equal", 1.5),
        "reset.stale-output-frames": ("less-or-equal", 0.0),
        "transition.reconvergence-ms": ("less-or-equal", 500.0),
    },
    "impairment": {
        "content.word-accuracy-regression": ("less-or-equal", 0.02),
        "echo.erle-db": ("greater-or-equal", 6.0),
        "near-end.attenuation-db": ("less-or-equal", 1.5),
        "perceptual.aecmos": ("greater-or-equal", 3.0),
    },
}
REQUIRED_STRATA = {
    "background-noise": "impairment",
    "clipping": "impairment",
    "clock-drift-negative": "transition",
    "clock-drift-positive": "transition",
    "delay-jump": "transition",
    "delay-negative": "transition",
    "delay-positive": "transition",
    "double-talk": "double-talk",
    "far-end-clean": "far-end-only",
    "linear-path": "far-end-only",
    "music-playback": "impairment",
    "near-end-clean": "near-end-only",
    "nonlinear-speaker": "impairment",
    "real-device": "impairment",
    "reference-loss": "transition",
    "room-change": "transition",
    "route-change": "transition",
}


def schema(name: str) -> dict:
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


def metric_documents(kind: str) -> list[dict]:
    policies = COMMON_METRICS | KIND_METRICS[kind]
    return [
        {
            "metric": name,
            "value": limit,
            "operator": operator,
            "limit": limit,
            "passed": True,
        }
        for name, (operator, limit) in sorted(policies.items())
    ]


def valid_evidence() -> dict:
    return {
        "schema": "denoize-aec-promotion-evidence-v1",
        "schema_version": 1,
        "payload": {
            "completed_at_unix_seconds": 1,
            "implementation": "native-pfdnlms-v1",
            "implementation_source_revision": "0123456789abcdef",
            "implementation_source_sha256": "1" * 64,
            "configuration_sha256": "2" * 64,
            "corpus_manifest_sha256": "3" * 64,
            "evaluation_result_sha256": "4" * 64,
            "listening_result_sha256": "5" * 64,
            "sample_rate": 48_000,
            "block_size_samples": 256,
            "tail_samples": 24_000,
            "maximum_delay_samples": 48_000,
            "strata": [
                {
                    "id": identifier,
                    "kind": kind,
                    "cases": 100,
                    "metrics": metric_documents(kind),
                }
                for identifier, kind in sorted(REQUIRED_STRATA.items())
            ],
            "real_device_cases": 100,
            "nonlinear_device_cases": 100,
            "delay_transition_cases": 100,
            "paced_realtime_blocks": 10_000,
            "worst_case_realtime_factor": 0.5,
            "callback_allocations": 0,
            "callback_locks": 0,
            "callback_waits": 0,
            "callback_io_operations": 0,
            "callback_log_operations": 0,
            "deadline_misses": 0,
            "stale_frames_after_reset": 0,
            "minimum_listeners": 20,
            "listener_count": 20,
            "listener_preference": 0.5,
            "listener_preference_limit": 0.5,
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
        "schema": "denoize-aec-report-v1",
        "schema_version": 1,
        "implementation": "native-pfdnlms-v1",
        "configuration_sha256": "0" * 64,
        "evidence_signing_key_id": "1" * 64,
        "evidence_evaluation_result_sha256": "2" * 64,
        "microphone_pcm_sha256": "3" * 64,
        "reference_pcm_sha256": "4" * 64,
        "output_pcm_sha256": "5" * 64,
        "microphone_frames": 1_024,
        "reference_frames": 1_024,
        "output_frames": 1_024,
        "microphone_sample_rate": 48_000,
        "reference_sample_rate": 48_000,
        "reference_clock_ppm": 0.0,
        "route_generation": 1,
        "delay": {
            "signed_delay_samples": -32,
            "confidence": 0.9,
            "polarity_inverted": False,
            "analyzed_samples": 1_024,
        },
        "block_size_samples": 256,
        "tail_samples": 24_000,
        "maximum_delay_samples": 48_000,
        "algorithmic_plus_buffering_milliseconds": 256 * 1000 / 48_000,
        "silence_blocks": 0,
        "far_end_only_blocks": 2,
        "near_end_only_blocks": 1,
        "double_talk_blocks": 1,
        "reference_uncertain_blocks": 0,
        "adaptation_blocks": 2,
        "reset_count": 1,
        "reset_reasons": {
            "initial": 1,
            "route_change": 0,
            "reference_discontinuity": 0,
            "clock_jump": 0,
            "delay_jump": 0,
            "non_finite_state": 0,
        },
        "clipped_samples": 0,
        "non_finite_output_samples": 0,
        "far_end_erle_db": 12.0,
        "exact_output_duration": True,
        "paths_recorded": 0,
        "limitations": [
            "mono microphone and reference only",
            "constant drift mapping only",
            "uncertain reference preserves microphone",
            "ERLE is far-end-only",
            "native linear baseline",
        ],
    }


def expect_schema_failure(validator: jsonschema.Draft202012Validator, document: dict) -> None:
    if not list(validator.iter_errors(document)):
        raise AssertionError("invalid AEC document unexpectedly passed JSON Schema")


def validate_evidence_semantics(document: dict) -> None:
    payload = document["payload"]
    strata = payload["strata"]
    identifiers = [item["id"] for item in strata]
    if identifiers != sorted(REQUIRED_STRATA):
        raise AssertionError("AEC strata must exactly match the sorted required matrix")
    all_metrics_passed = True
    for stratum in strata:
        expected_kind = REQUIRED_STRATA[stratum["id"]]
        if stratum["kind"] != expected_kind:
            raise AssertionError("AEC stratum kind mismatch")
        policies = COMMON_METRICS | KIND_METRICS[expected_kind]
        metric_names = [metric["metric"] for metric in stratum["metrics"]]
        if metric_names != sorted(policies):
            raise AssertionError("AEC metrics must exactly match the sorted policy")
        for metric in stratum["metrics"]:
            operator, hard_limit = policies[metric["metric"]]
            if metric["operator"] != operator:
                raise AssertionError("AEC metric operator mismatch")
            if operator == "less-or-equal":
                strict_enough = metric["limit"] <= hard_limit
                passed = metric["value"] <= metric["limit"]
            else:
                strict_enough = metric["limit"] >= hard_limit
                passed = metric["value"] >= metric["limit"]
            if not strict_enough or metric["passed"] != passed:
                raise AssertionError("AEC metric limit/pass semantics mismatch")
            all_metrics_passed &= metric["passed"]
    block = payload["block_size_samples"]
    if block & (block - 1):
        raise AssertionError("AEC block size must be a power of two")
    latency = block * 1000 / payload["sample_rate"]
    if latency > 20:
        raise AssertionError("AEC promoted geometry exceeds 20 ms")
    if payload["tail_samples"] < block or payload["tail_samples"] > 2 * payload["sample_rate"]:
        raise AssertionError("AEC tail geometry is invalid")
    if payload["maximum_delay_samples"] > 2 * payload["sample_rate"]:
        raise AssertionError("AEC delay geometry is invalid")
    accepted = all_metrics_passed and (
        payload["listener_preference"] >= payload["listener_preference_limit"]
    )
    if payload["accepted"] != accepted:
        raise AssertionError("AEC accepted flag is inconsistent")


def validate_report_semantics(document: dict) -> None:
    if document["output_frames"] != document["microphone_frames"]:
        raise AssertionError("AEC report must preserve exact microphone geometry")
    if abs(document["delay"]["signed_delay_samples"]) > document["maximum_delay_samples"]:
        raise AssertionError("AEC delay exceeds its configured signed range")
    expected_latency = (
        document["block_size_samples"] * 1000 / document["microphone_sample_rate"]
    )
    if not math.isclose(
        document["algorithmic_plus_buffering_milliseconds"], expected_latency, abs_tol=1e-9
    ):
        raise AssertionError("AEC latency does not match the promoted block geometry")
    classified = sum(
        document[name]
        for name in (
            "silence_blocks",
            "far_end_only_blocks",
            "near_end_only_blocks",
            "double_talk_blocks",
            "reference_uncertain_blocks",
        )
    )
    expected_blocks = math.ceil(document["microphone_frames"] / document["block_size_samples"])
    if classified != expected_blocks:
        raise AssertionError("AEC report block classifications do not cover the render")
    if document["adaptation_blocks"] > document["far_end_only_blocks"]:
        raise AssertionError("AEC adaptation may occur only during far-end-only blocks")
    if sum(document["reset_reasons"].values()) != document["reset_count"]:
        raise AssertionError("AEC reset reasons do not sum to reset_count")


def main() -> None:
    evidence_validator = jsonschema.Draft202012Validator(
        schema("denoize-aec-promotion-evidence-v1.schema.json")
    )
    report_validator = jsonschema.Draft202012Validator(
        schema("denoize-aec-report-v1.schema.json")
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
    invalid["payload"]["block_size_samples"] = 192
    evidence_validator.validate(invalid)
    try:
        validate_evidence_semantics(invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("non-power-of-two AEC geometry unexpectedly passed semantics")
    invalid = copy.deepcopy(evidence)
    invalid["payload"]["strata"][0]["id"] = "unexpected"
    evidence_validator.validate(invalid)
    try:
        validate_evidence_semantics(invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("incomplete AEC matrix unexpectedly passed semantics")
    invalid = copy.deepcopy(report)
    invalid["output_frames"] -= 1
    report_validator.validate(invalid)
    try:
        validate_report_semantics(invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("AEC duration mismatch unexpectedly passed semantics")
    invalid = copy.deepcopy(report)
    invalid["paths_recorded"] = 1
    expect_schema_failure(report_validator, invalid)
    print("AEC JSON schema tests passed")


if __name__ == "__main__":
    main()
