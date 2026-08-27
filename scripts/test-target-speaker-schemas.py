#!/usr/bin/env python3
"""Validate target-speaker report and signed promotion-evidence contracts."""

from __future__ import annotations

import copy
import json
import pathlib

import jsonschema


ROOT = pathlib.Path(__file__).resolve().parents[1]
PRESENT_METRICS = {
    "content.target-word-error-rate": ("less-or-equal", 0.35),
    "extraction.si-sdr-improvement-db": ("greater-or-equal", 3.0),
    "interferer.speaker-similarity": ("less-or-equal", 0.30),
    "interferer.word-leakage-rate": ("less-or-equal", 0.02),
    "output.duration-error-frames": ("less-or-equal", 0.0),
    "output.non-finite-samples": ("less-or-equal", 0.0),
    "perceptual.dnsmos-p808": ("greater-or-equal", 3.0),
    "presence.recall": ("greater-or-equal", 0.95),
    "speaker.target-similarity": ("greater-or-equal", 0.70),
}
ABSENT_METRICS = {
    "interferer.speaker-similarity": ("less-or-equal", 0.30),
    "interferer.word-leakage-rate": ("less-or-equal", 0.01),
    "output.duration-error-frames": ("less-or-equal", 0.0),
    "output.non-finite-samples": ("less-or-equal", 0.0),
    "output.rms-dbfs": ("less-or-equal", -60.0),
    "presence.false-positive-rate": ("less-or-equal", 0.01),
}
CAUSAL_MAX_REGRESSION = {
    "content.target-word-error-rate": 0.02,
    "extraction.si-sdr-improvement-db": 0.5,
    "interferer.speaker-similarity": 0.02,
    "interferer.word-leakage-rate": 0.005,
    "output.duration-error-frames": 0.0,
    "output.non-finite-samples": 0.0,
    "output.rms-dbfs": 3.0,
    "perceptual.dnsmos-p808": 0.1,
    "presence.false-positive-rate": 0.005,
    "presence.recall": 0.02,
    "speaker.target-similarity": 0.02,
}
REQUIRED_STRATA = {
    "channel-mismatch": "target-present",
    "child-speaker": "target-present",
    "code-switching": "target-present",
    "codec-enrollment": "target-present",
    "different-sex": "target-present",
    "many-interferers": "target-present",
    "noisy-enrollment": "target-present",
    "one-interferer": "target-present",
    "real-t-conversation": "target-present",
    "reverberant-enrollment": "target-present",
    "same-sex": "target-present",
    "same-words": "target-present",
    "similar-voices": "target-present",
    "singing": "target-present",
    "speech-absent": "target-absent",
    "target-absent": "target-absent",
    "target-absent-same-words": "target-absent",
    "target-absent-similar-interferer": "target-absent",
    "target-present-clean": "target-present",
    "ts-superb": "target-present",
    "unseen-domain": "target-present",
    "whisper": "target-present",
}
GATES = {
    "geometry",
    "finite-normalized-samples",
    "energy-gain",
    "peak-gain",
    "new-clipping",
    "target-presence",
    "promotion-evidence",
}


def schema(name: str) -> dict:
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


def metric_documents(kind: str) -> list[dict]:
    policies = PRESENT_METRICS if kind == "target-present" else ABSENT_METRICS
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
        "schema": "denoize-target-speaker-promotion-evidence-v1",
        "schema_version": 1,
        "payload": {
            "completed_at_unix_seconds": 1,
            "model_package_sha256": "0" * 64,
            "source_revision": "0123456789abcdef",
            "source_sha256": "1" * 64,
            "checkpoint_sha256": "2" * 64,
            "corpus_manifest_sha256": "3" * 64,
            "evaluation_result_sha256": "4" * 64,
            "real_t_result_sha256": "5" * 64,
            "ts_superb_result_sha256": "6" * 64,
            "strata": [
                {
                    "id": identifier,
                    "kind": kind,
                    "cases": 10,
                    "metrics": metric_documents(kind),
                }
                for identifier, kind in sorted(REQUIRED_STRATA.items())
            ],
            "target_speaker_count": 100,
            "interferer_speaker_count": 100,
            "language_count": 2,
            "presence_expected_calibration_error": 0.05,
            "presence_expected_calibration_error_limit": 0.05,
            "minimum_listeners": 20,
            "listener_count": 20,
            "listener_preference": 0.5,
            "listener_preference_limit": 0.5,
            "accepted": True,
        },
        "signature": {
            "algorithm": "ed25519",
            "key_id": "7" * 64,
            "value_base64": "A" * 86 + "==",
        },
    }


def causal_metric_documents(kind: str) -> list[dict]:
    policies = PRESENT_METRICS if kind == "target-present" else ABSENT_METRICS
    return [
        {
            "metric": name,
            "operator": operator,
            "offline_value": limit,
            "causal_value": limit,
            "hard_limit": limit,
            "maximum_regression": CAUSAL_MAX_REGRESSION[name],
            "passed": True,
        }
        for name, (operator, limit) in sorted(policies.items())
    ]


def valid_causal_evidence() -> dict:
    return {
        "schema": "denoize-causal-target-speaker-promotion-evidence-v1",
        "schema_version": 1,
        "payload": {
            "completed_at_unix_seconds": 1,
            "model_package_sha256": "0" * 64,
            "source_revision": "0123456789abcdef",
            "source_sha256": "1" * 64,
            "checkpoint_sha256": "2" * 64,
            "offline_evaluation_result_sha256": "3" * 64,
            "causal_evaluation_result_sha256": "4" * 64,
            "state_reset_flush_result_sha256": "5" * 64,
            "latency_result_sha256": "6" * 64,
            "realtime_callback_result_sha256": "7" * 64,
            "transition_result_sha256": "8" * 64,
            "strata": [
                {
                    "id": identifier,
                    "kind": kind,
                    "offline_cases": 10,
                    "causal_cases": 10,
                    "metrics": causal_metric_documents(kind),
                }
                for identifier, kind in sorted(REQUIRED_STRATA.items())
            ],
            "model_sample_rate_hz": 16_000,
            "frame_samples": 160,
            "algorithmic_latency_samples": 1_440,
            "flush_samples": 1_440,
            "perturbation_latency_cases": 100,
            "effective_latency_milliseconds": 90.0,
            "effective_latency_limit_milliseconds": 100.0,
            "realtime": {
                "paced_blocks": 10_000,
                "deadline_misses": 0,
                "overload_blocks": 0,
                "queue_capacity_blocks": 16,
                "maximum_queue_depth_blocks": 15,
                "callback_allocations": 0,
                "callback_locks": 0,
                "callback_waits": 0,
                "callback_file_io_operations": 0,
                "callback_network_operations": 0,
                "callback_log_operations": 0,
                "callback_inference_calls": 0,
            },
            "transitions": {
                "absent_to_present_cases": 100,
                "present_to_absent_cases": 100,
                "uncertain_transition_cases": 100,
                "enrollment_mismatch_cases": 100,
                "reference_loss_cases": 100,
                "late_results_injected": 100,
                "late_results_discarded": 100,
                "stale_generation_results_injected": 100,
                "stale_generation_results_discarded": 100,
                "false_attribution_publications": 0,
            },
            "accepted": True,
        },
        "signature": {
            "algorithm": "ed25519",
            "key_id": "9" * 64,
            "value_base64": "A" * 86 + "==",
        },
    }


def valid_report() -> dict:
    return {
        "schema": "denoize-target-speaker-report-v1",
        "schema_version": 1,
        "denoize_version": "0.77.0",
        "network_accessed": False,
        "deterministic": True,
        "model": {
            "package_sha256": "0" * 64,
            "public_key_sha256": "1" * 64,
            "package_id": "org.example.target-speaker",
            "package_revision": "1",
            "precision_profile": "fp32",
            "source_revision": "0123456789abcdef",
            "source_sha256": "2" * 64,
            "source_license_spdx": "Apache-2.0",
            "checkpoint_sha256": "3" * 64,
            "checkpoint_license_spdx": "LicenseRef-audit-required",
            "accelerator": "cpu",
        },
        "promotion_evidence": {
            "signing_key_id": "4" * 64,
            "corpus_manifest_sha256": "5" * 64,
            "evaluation_result_sha256": "6" * 64,
            "real_t_result_sha256": "7" * 64,
            "ts_superb_result_sha256": "8" * 64,
            "strata": 22,
            "target_speakers": 100,
            "interferer_speakers": 100,
            "languages": 2,
            "accepted": True,
        },
        "decision": "accepted-present",
        "model_invoked": True,
        "candidate_accepted": True,
        "output_published": True,
        "candidate_retained": True,
        "source_sample_rate": 48_000,
        "source_channels": 2,
        "source_frames": 48_000,
        "output_channels": 1,
        "output_frames": 48_000,
        "mixture_mixdown_policy": "arithmetic-mean-mono-v1",
        "mixture_pcm_sha256": "9" * 64,
        "candidate_pcm_sha256": "a" * 64,
        "output_pcm_sha256": "a" * 64,
        "enrollment": {
            "input_sample_rate": 48_000,
            "input_channels": 1,
            "input_frames": 144_000,
            "model_sample_rate": 16_000,
            "model_samples": 48_000,
            "mixdown_policy": "arithmetic-mean-mono-v1",
            "raw_audio_retained": False,
            "embedding_retained": False,
            "digest_recorded": False,
        },
        "presence": {
            "state": "present",
            "absent_probability": 0.01,
            "uncertain_probability": 0.01,
            "present_probability": 0.98,
            "minimum_absent_probability": 0.9,
            "minimum_present_probability": 0.9,
        },
        "measurements": {
            "mixture_rms_dbfs": -20.0,
            "candidate_rms_dbfs": -22.0,
            "mixture_peak_dbfs": -2.0,
            "candidate_peak_dbfs": -3.0,
            "energy_delta_db": -2.0,
            "mixture_clipping_ratio": 0.0,
            "candidate_clipping_ratio": 0.0,
        },
        "safety_gates": [
            {"kind": kind, "observed": 1.0, "limit": 1.0, "passed": True}
            for kind in sorted(GATES)
        ],
        "runtime_speaker_identity_verified": False,
        "interferer_leakage_measured_at_runtime": False,
        "limitations": [f"known limitation {index}" for index in range(6)],
        "warnings": [],
    }


def valid_causal_report() -> dict:
    return {
        "schema": "denoize-causal-target-speaker-report-v1",
        "schema_version": 1,
        "denoize_version": "0.82.0",
        "network_accessed": False,
        "deterministic": True,
        "model": {
            "package_sha256": "0" * 64,
            "public_key_sha256": "1" * 64,
            "package_id": "org.example.causal-target-speaker",
            "package_revision": "1",
            "precision_profile": "fp32",
            "source_revision": "0123456789abcdef",
            "source_sha256": "2" * 64,
            "source_license_spdx": "Apache-2.0",
            "checkpoint_sha256": "3" * 64,
            "checkpoint_license_spdx": "LicenseRef-audit-required",
            "accelerator": "cpu",
        },
        "promotion_evidence": {
            "offline_signing_key_id": "4" * 64,
            "causal_signing_key_id": "5" * 64,
            "offline_evaluation_result_sha256": "6" * 64,
            "causal_evaluation_result_sha256": "7" * 64,
            "state_reset_flush_result_sha256": "8" * 64,
            "latency_result_sha256": "9" * 64,
            "realtime_callback_result_sha256": "a" * 64,
            "transition_result_sha256": "b" * 64,
            "strata": 22,
            "accepted": True,
        },
        "source_sample_rate": 48_000,
        "source_channels": 2,
        "source_frames": 48_000,
        "output_channels": 1,
        "output_frames": 48_000,
        "model_sample_rate": 16_000,
        "frame_samples": 160,
        "algorithmic_latency_samples": 1_440,
        "flush_samples": 1_440,
        "input_blocks": 100,
        "flush_blocks": 9,
        "decision_counts": {
            "published_present_blocks": 90,
            "muted_absent_blocks": 1,
            "muted_uncertain_blocks": 1,
            "muted_present_warmup_blocks": 3,
            "muted_safety_gate_blocks": 0,
            "muted_flush_blocks": 0,
        },
        "presence_transitions": 2,
        "rendered_audio_published": True,
        "mixture_mixdown_policy": "arithmetic-mean-mono-v1",
        "mixture_pcm_sha256": "c" * 64,
        "output_pcm_sha256": "d" * 64,
        "enrollment": {
            "input_sample_rate": 48_000,
            "input_channels": 1,
            "input_frames": 144_000,
            "model_sample_rate": 16_000,
            "model_samples": 48_000,
            "mixdown_policy": "arithmetic-mean-mono-v1",
            "raw_audio_retained": False,
            "embedding_retained": False,
            "digest_recorded": False,
        },
        "runtime_speaker_identity_verified": False,
        "interferer_leakage_measured_at_runtime": False,
        "limitations": [f"known causal limitation {index}" for index in range(7)],
        "warnings": [],
    }


def evidence_semantics(document: dict) -> bool:
    payload = document["payload"]
    strata = payload["strata"]
    ids = [entry["id"] for entry in strata]
    if ids != sorted(set(ids)) or not set(REQUIRED_STRATA).issubset(ids):
        return False
    all_passed = True
    for stratum in strata:
        if REQUIRED_STRATA.get(stratum["id"], stratum["kind"]) != stratum["kind"]:
            return False
        policies = PRESENT_METRICS if stratum["kind"] == "target-present" else ABSENT_METRICS
        metrics = stratum["metrics"]
        names = [metric["metric"] for metric in metrics]
        if names != sorted(set(names)) or not set(policies).issubset(names):
            return False
        for metric in metrics:
            expected = (
                metric["value"] >= metric["limit"]
                if metric["operator"] == "greater-or-equal"
                else metric["value"] <= metric["limit"]
            )
            if metric["passed"] != expected:
                return False
            if metric["metric"] in policies:
                operator, hard_limit = policies[metric["metric"]]
                if metric["operator"] != operator:
                    return False
                if operator == "greater-or-equal" and metric["limit"] < hard_limit:
                    return False
                if operator == "less-or-equal" and metric["limit"] > hard_limit:
                    return False
            all_passed = all_passed and expected
    expected_accepted = (
        all_passed
        and payload["presence_expected_calibration_error"]
        <= payload["presence_expected_calibration_error_limit"]
        and payload["listener_count"] >= payload["minimum_listeners"]
        and payload["listener_preference"] >= payload["listener_preference_limit"]
    )
    return payload["accepted"] == expected_accepted


def causal_evidence_semantics(document: dict) -> bool:
    payload = document["payload"]
    strata = payload["strata"]
    ids = [entry["id"] for entry in strata]
    if ids != sorted(set(ids)) or set(ids) != set(REQUIRED_STRATA):
        return False
    all_passed = True
    for stratum in strata:
        if REQUIRED_STRATA[stratum["id"]] != stratum["kind"]:
            return False
        policies = PRESENT_METRICS if stratum["kind"] == "target-present" else ABSENT_METRICS
        metrics = stratum["metrics"]
        names = [metric["metric"] for metric in metrics]
        if names != sorted(set(names)) or set(names) != set(policies):
            return False
        for metric in metrics:
            operator, hard_limit = policies[metric["metric"]]
            hard_limit_is_strong_enough = (
                metric["hard_limit"] >= hard_limit
                if operator == "greater-or-equal"
                else metric["hard_limit"] <= hard_limit
            )
            if (
                metric["operator"] != operator
                or not hard_limit_is_strong_enough
                or metric["maximum_regression"] < 0
                or metric["maximum_regression"]
                > CAUSAL_MAX_REGRESSION[metric["metric"]]
            ):
                return False
            if operator == "greater-or-equal":
                passed = (
                    metric["offline_value"] >= metric["hard_limit"]
                    and metric["causal_value"] >= metric["hard_limit"]
                    and metric["causal_value"]
                    >= metric["offline_value"] - metric["maximum_regression"]
                )
            else:
                passed = (
                    metric["offline_value"] <= metric["hard_limit"]
                    and metric["causal_value"] <= metric["hard_limit"]
                    and metric["causal_value"]
                    <= metric["offline_value"] + metric["maximum_regression"]
                )
            if metric["passed"] != passed:
                return False
            all_passed = all_passed and passed
    realtime = payload["realtime"]
    realtime_passed = (
        realtime["paced_blocks"] >= 10_000
        and realtime["deadline_misses"] == 0
        and realtime["overload_blocks"] == 0
        and 16 <= realtime["queue_capacity_blocks"] <= 256
        and realtime["maximum_queue_depth_blocks"] < realtime["queue_capacity_blocks"]
        and all(
            realtime[name] == 0
            for name in (
                "callback_allocations",
                "callback_locks",
                "callback_waits",
                "callback_file_io_operations",
                "callback_network_operations",
                "callback_log_operations",
                "callback_inference_calls",
            )
        )
    )
    transitions = payload["transitions"]
    transitions_passed = (
        all(
            transitions[name] >= 100
            for name in (
                "absent_to_present_cases",
                "present_to_absent_cases",
                "uncertain_transition_cases",
                "enrollment_mismatch_cases",
                "reference_loss_cases",
                "late_results_injected",
                "stale_generation_results_injected",
            )
        )
        and transitions["late_results_discarded"] == transitions["late_results_injected"]
        and transitions["stale_generation_results_discarded"]
        == transitions["stale_generation_results_injected"]
        and transitions["false_attribution_publications"] == 0
    )
    expected = (
        all_passed
        and payload["algorithmic_latency_samples"] * 1000
        <= 100 * payload["model_sample_rate_hz"]
        and payload["flush_samples"] >= payload["algorithmic_latency_samples"]
        and payload["perturbation_latency_cases"] >= 100
        and payload["effective_latency_milliseconds"]
        <= payload["effective_latency_limit_milliseconds"]
        <= 100.0
        and realtime_passed
        and transitions_passed
    )
    return payload["accepted"] == expected


def main() -> None:
    report_schema = schema("denoize-target-speaker-report-v1.schema.json")
    causal_report_schema = schema("denoize-causal-target-speaker-report-v1.schema.json")
    evidence_schema = schema("denoize-target-speaker-promotion-evidence-v1.schema.json")
    causal_schema = schema(
        "denoize-causal-target-speaker-promotion-evidence-v1.schema.json"
    )
    for document_schema in (
        report_schema,
        causal_report_schema,
        evidence_schema,
        causal_schema,
    ):
        jsonschema.Draft202012Validator.check_schema(document_schema)
    report_validator = jsonschema.Draft202012Validator(report_schema)
    causal_report_validator = jsonschema.Draft202012Validator(causal_report_schema)
    evidence_validator = jsonschema.Draft202012Validator(evidence_schema)
    causal_validator = jsonschema.Draft202012Validator(causal_schema)
    report = valid_report()
    causal_report = valid_causal_report()
    evidence = valid_evidence()
    causal = valid_causal_evidence()
    report_validator.validate(report)
    causal_report_validator.validate(causal_report)
    evidence_validator.validate(evidence)
    causal_validator.validate(causal)
    assert evidence_semantics(evidence)
    assert causal_evidence_semantics(causal)
    assert {gate["kind"] for gate in report["safety_gates"]} == GATES

    private_path = copy.deepcopy(report)
    private_path["enrollment_path"] = "/private/enrollment.wav"
    assert not report_validator.is_valid(private_path)
    retained = copy.deepcopy(report)
    retained["enrollment"]["raw_audio_retained"] = True
    assert not report_validator.is_valid(retained)
    wrong_presence = copy.deepcopy(report)
    wrong_presence["presence"]["state"] = "uncertain"
    assert not report_validator.is_valid(wrong_presence)
    withheld = copy.deepcopy(report)
    withheld.update(
        {
            "decision": "withheld-uncertain",
            "candidate_accepted": False,
            "output_published": False,
            "candidate_retained": False,
            "output_frames": None,
            "candidate_pcm_sha256": None,
            "output_pcm_sha256": None,
        }
    )
    withheld["presence"]["state"] = "uncertain"
    report_validator.validate(withheld)
    leaked_hash = copy.deepcopy(withheld)
    leaked_hash["candidate_pcm_sha256"] = "b" * 64
    assert not report_validator.is_valid(leaked_hash)

    retained_causal = copy.deepcopy(causal_report)
    retained_causal["enrollment"]["raw_audio_retained"] = True
    assert not causal_report_validator.is_valid(retained_causal)
    attributed_causal = copy.deepcopy(causal_report)
    attributed_causal["runtime_speaker_identity_verified"] = True
    assert not causal_report_validator.is_valid(attributed_causal)
    path_causal = copy.deepcopy(causal_report)
    path_causal["enrollment_path"] = "/private/enrollment.wav"
    assert not causal_report_validator.is_valid(path_causal)

    weak = copy.deepcopy(evidence)
    present = next(entry for entry in weak["payload"]["strata"] if entry["kind"] == "target-present")
    similarity = next(metric for metric in present["metrics"] if metric["metric"] == "speaker.target-similarity")
    similarity.update({"value": 0.1, "limit": 0.1})
    assert evidence_validator.is_valid(weak)
    assert not evidence_semantics(weak)

    slow = copy.deepcopy(causal)
    slow["payload"]["effective_latency_limit_milliseconds"] = 100.01
    assert not causal_validator.is_valid(slow)
    callback_inference = copy.deepcopy(causal)
    callback_inference["payload"]["realtime"]["callback_inference_calls"] = 1
    assert causal_validator.is_valid(callback_inference)
    assert not causal_evidence_semantics(callback_inference)
    causal_regression = copy.deepcopy(causal)
    present = next(
        entry
        for entry in causal_regression["payload"]["strata"]
        if entry["kind"] == "target-present"
    )
    similarity = next(
        metric
        for metric in present["metrics"]
        if metric["metric"] == "speaker.target-similarity"
    )
    similarity["causal_value"] = 0.679
    assert causal_validator.is_valid(causal_regression)
    assert not causal_evidence_semantics(causal_regression)

    print("target-speaker JSON schema tests passed")


if __name__ == "__main__":
    main()
