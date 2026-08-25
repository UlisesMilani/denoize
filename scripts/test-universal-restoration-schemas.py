#!/usr/bin/env python3
"""Validate universal-restoration report, mask, and promotion evidence schemas."""

from __future__ import annotations

import copy
import json
import pathlib

import jsonschema


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEGRADATIONS = {
    "additive-noise",
    "reverberation",
    "clipping",
    "bandwidth-limitation",
    "codec-distortion",
    "packet-loss",
    "wind",
}
GATES = {
    "geometry",
    "finite-samples",
    "energy-gain",
    "peak-gain",
    "new-clipping",
    "silence-injection",
    "native-quality-regression",
}
REQUIRED_STRATA = (
    "accent",
    "age",
    "clean-bypass",
    "degradation-additive-noise",
    "degradation-bandwidth-limitation",
    "degradation-clipping",
    "degradation-codec-distortion",
    "degradation-packet-loss",
    "degradation-reverberation",
    "degradation-wind",
    "emotion",
    "language",
    "near-clean-bypass",
    "non-speech",
    "seen-corpus",
    "sex",
    "singing",
    "speech",
    "unseen-corpus",
    "whisper",
)
REQUIRED_METRICS = (
    "content.phoneme-similarity-delta",
    "content.word-error-rate-delta",
    "hallucination.new-word-rate",
    "objective.si-sdr-improvement-db",
    "output.duration-error-frames",
    "output.non-finite-samples",
    "perceptual.quality-delta",
    "performance.realtime-factor",
    "speaker.similarity-delta",
)


def schema(name: str) -> dict:
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


def valid_mask() -> dict:
    return {
        "schema": "denoize-universal-restoration-mask-v1",
        "schema_version": 1,
        "channels": 1,
        "frames": 8,
        "runs": [
            {"channel": 0, "start_frame": 0, "frame_count": 2, "state": "untouched"},
            {"channel": 0, "start_frame": 2, "frame_count": 4, "state": "replaced"},
            {"channel": 0, "start_frame": 6, "frame_count": 2, "state": "untouched"},
        ],
    }


def exact_coverage(mask: dict) -> bool:
    cursor = [0] * mask["channels"]
    previous = None
    for run in mask["runs"]:
        channel = run["channel"]
        position = (channel, run["start_frame"])
        if (
            channel >= len(cursor)
            or cursor[channel] != run["start_frame"]
            or (previous is not None and previous >= position)
        ):
            return False
        cursor[channel] += run["frame_count"]
        previous = position
    return all(value == mask["frames"] for value in cursor)


def valid_report() -> dict:
    return {
        "schema": "denoize-universal-restoration-report-v1",
        "schema_version": 1,
        "denoize_version": "0.75.0",
        "network_accessed": False,
        "model_family": "discriminative",
        "render_role": "primary",
        "model": {
            "package_sha256": "0" * 64,
            "public_key_sha256": "1" * 64,
            "package_id": "org.example.urgent-bsrnn",
            "package_revision": "1",
            "precision_profile": "fp32",
            "source_revision": "b1dc3ad1e86419ff0bd666f455bda7936bff0e9a",
            "source_sha256": "2" * 64,
            "source_license_spdx": "Apache-2.0",
            "checkpoint_sha256": "3" * 64,
            "checkpoint_license_spdx": "LicenseRef-audit-required",
            "accelerator": "cpu",
        },
        "decision": "accepted",
        "model_invoked": True,
        "candidate_accepted": True,
        "deterministic": True,
        "sample_rate": 48_000,
        "channels": 1,
        "frames": 8,
        "input_pcm_sha256": "4" * 64,
        "candidate_pcm_sha256": "5" * 64,
        "output_pcm_sha256": "5" * 64,
        "mask_sha256": "6" * 64,
        "changed_samples": 4,
        "degradations": [
            {
                "degradation": degradation,
                "detected": degradation == "additive-noise",
                "confidence": 0.8,
                "severity": 0.6,
                "score": 0.48,
            }
            for degradation in sorted(DEGRADATIONS)
        ],
        "measurements": {
            "input_rms_dbfs": -24.0,
            "candidate_rms_dbfs": -25.0,
            "input_peak_dbfs": -3.0,
            "candidate_peak_dbfs": -4.0,
            "energy_delta_db": -1.0,
            "input_clipping_ratio": 0.0,
            "candidate_clipping_ratio": 0.0,
            "native_quality_score_delta": 2.0,
        },
        "safety_gates": [
            {"kind": gate, "observed": 0.0, "limit": 6.0, "passed": True}
            for gate in sorted(GATES)
        ],
        "semantic_fidelity_assessed": False,
        "speaker_identity_assessed": False,
        "promotion_evidence_verified": False,
        "limitations": [
            "signal gates do not assess words",
            "the adapter has a fixed spectral contract",
            "promotion needs signed external evidence",
        ],
        "warnings": [],
    }


def valid_evidence() -> dict:
    metrics = [
        {
            "metric": metric,
            "value": 0.0,
            "operator": "greater-or-equal",
            "limit": 0.0,
            "passed": True,
        }
        for metric in REQUIRED_METRICS
    ]
    return {
        "schema": "denoize-universal-promotion-evidence-v1",
        "schema_version": 1,
        "payload": {
            "completed_at_unix_seconds": 1,
            "model_package_sha256": "0" * 64,
            "model_family": "discriminative",
            "source_revision": "b1dc3ad1e86419ff0bd666f455bda7936bff0e9a",
            "source_sha256": "1" * 64,
            "checkpoint_sha256": "2" * 64,
            "corpus_manifest_sha256": "3" * 64,
            "evaluation_result_sha256": "4" * 64,
            "strata": [
                {"id": stratum, "cases": 1, "metrics": copy.deepcopy(metrics)}
                for stratum in REQUIRED_STRATA
            ],
            "minimum_listeners": 1,
            "listener_count": 1,
            "listener_preference": 0.5,
            "listener_preference_limit": 0.5,
            "accepted": True,
        },
        "signature": {
            "algorithm": "ed25519",
            "key_id": "5" * 64,
            "value_base64": "A" * 86 + "==",
        },
    }


def evidence_semantics(document: dict) -> bool:
    payload = document["payload"]
    strata = payload["strata"]
    ids = [entry["id"] for entry in strata]
    if ids != sorted(set(ids)) or not set(REQUIRED_STRATA).issubset(ids):
        return False
    passed = True
    for stratum in strata:
        metrics = stratum["metrics"]
        names = [entry["metric"] for entry in metrics]
        if names != sorted(set(names)) or not set(REQUIRED_METRICS).issubset(names):
            return False
        for metric in metrics:
            expected = (
                metric["value"] >= metric["limit"]
                if metric["operator"] == "greater-or-equal"
                else metric["value"] <= metric["limit"]
            )
            if metric["passed"] != expected:
                return False
            passed = passed and expected
    expected_accepted = (
        passed
        and payload["listener_count"] >= payload["minimum_listeners"]
        and payload["listener_preference"] >= payload["listener_preference_limit"]
    )
    return payload["accepted"] == expected_accepted


def main() -> None:
    mask_schema = schema("denoize-universal-restoration-mask-v1.schema.json")
    report_schema = schema("denoize-universal-restoration-report-v1.schema.json")
    evidence_schema = schema("denoize-universal-promotion-evidence-v1.schema.json")
    for document_schema in (mask_schema, report_schema, evidence_schema):
        jsonschema.Draft202012Validator.check_schema(document_schema)
    mask_validator = jsonschema.Draft202012Validator(mask_schema)
    report_validator = jsonschema.Draft202012Validator(report_schema)
    evidence_validator = jsonschema.Draft202012Validator(evidence_schema)

    mask = valid_mask()
    report = valid_report()
    evidence = valid_evidence()
    mask_validator.validate(mask)
    report_validator.validate(report)
    evidence_validator.validate(evidence)
    assert exact_coverage(mask)
    assert {item["degradation"] for item in report["degradations"]} == DEGRADATIONS
    assert {item["kind"] for item in report["safety_gates"]} == GATES
    assert evidence_semantics(evidence)

    private_path = copy.deepcopy(report)
    private_path["input_path"] = "/private/speech.wav"
    assert not report_validator.is_valid(private_path)
    generative_primary = copy.deepcopy(report)
    generative_primary["model_family"] = "generative"
    assert not report_validator.is_valid(generative_primary)
    bypass_with_candidate = copy.deepcopy(report)
    bypass_with_candidate.update(
        {
            "decision": "bypassed-clean",
            "model_invoked": False,
            "candidate_accepted": False,
            "changed_samples": 0,
            "safety_gates": [],
        }
    )
    assert not report_validator.is_valid(bypass_with_candidate)
    gap = copy.deepcopy(mask)
    gap["runs"][1]["start_frame"] = 3
    mask_validator.validate(gap)
    assert not exact_coverage(gap)
    missing_stratum = copy.deepcopy(evidence)
    missing_stratum["payload"]["strata"].pop()
    evidence_validator.validate(missing_stratum)
    assert not evidence_semantics(missing_stratum)
    inconsistent_metric = copy.deepcopy(evidence)
    inconsistent_metric["payload"]["strata"][0]["metrics"][0]["passed"] = False
    evidence_validator.validate(inconsistent_metric)
    assert not evidence_semantics(inconsistent_metric)
    unknown = copy.deepcopy(evidence)
    unknown["payload"]["training_command"] = "python train.py"
    assert not evidence_validator.is_valid(unknown)


if __name__ == "__main__":
    main()
