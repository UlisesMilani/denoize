#!/usr/bin/env python3
"""Validate causal target-sound evidence, report, and snapshot contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
SHA = "ab" * 32
STRATA = [
    ("binaural-spatial", "binaural-spatial"),
    ("class-confusable", "target-present"),
    ("clean-bypass", "target-absent"),
    ("low-snr", "target-present"),
    ("multi-instance", "target-present"),
    ("music-foreground", "protected-foreground"),
    ("query-alias", "target-present"),
    ("speech-foreground", "protected-foreground"),
    ("target-absent", "target-absent"),
    ("target-present", "target-present"),
    ("tonal-target", "target-present"),
    ("transient-target", "target-present"),
    ("unseen-domain", "target-present"),
    ("unseen-interferer", "target-present"),
]
PRESENT = [
    ("extraction.target-si-sdr-improvement-db", "greater-or-equal", 3.0),
    ("output.clipped-samples", "less-or-equal", 0.0),
    ("output.duration-mismatch-samples", "less-or-equal", 0.0),
    ("output.non-finite-samples", "less-or-equal", 0.0),
    ("output.protected-foreground-sdr-db", "greater-or-equal", 20.0),
    ("presence.expected-calibration-error", "less-or-equal", 0.05),
    ("presence.false-negative-rate", "less-or-equal", 0.05),
    ("recombination.maximum-absolute-error", "less-or-equal", 1.0e-5),
    ("residual.target-leakage-db", "less-or-equal", -20.0),
]
ABSENT = [
    ("output.clipped-samples", "less-or-equal", 0.0),
    ("output.duration-mismatch-samples", "less-or-equal", 0.0),
    ("output.non-finite-samples", "less-or-equal", 0.0),
    ("presence.expected-calibration-error", "less-or-equal", 0.05),
    ("presence.false-positive-rate", "less-or-equal", 0.01),
    ("recombination.maximum-absolute-error", "less-or-equal", 1.0e-5),
    ("target.output-rms-dbfs", "less-or-equal", -60.0),
]
BINAURAL = [
    ("extraction.target-si-sdr-improvement-db", "greater-or-equal", 3.0),
    ("output.clipped-samples", "less-or-equal", 0.0),
    ("output.duration-mismatch-samples", "less-or-equal", 0.0),
    ("output.non-finite-samples", "less-or-equal", 0.0),
    ("presence.expected-calibration-error", "less-or-equal", 0.05),
    ("presence.false-negative-rate", "less-or-equal", 0.05),
    ("recombination.maximum-absolute-error", "less-or-equal", 1.0e-5),
    ("residual.target-leakage-db", "less-or-equal", -20.0),
    ("spatial.ild-error-db", "less-or-equal", 1.0),
    ("spatial.itd-error-microseconds", "less-or-equal", 100.0),
]


def load(name: str) -> dict:
    return json.loads((SCHEMAS / name).read_text(encoding="utf-8"))


def policies(kind: str) -> list[tuple[str, str, float]]:
    if kind == "target-absent":
        return ABSENT
    if kind == "binaural-spatial":
        return BINAURAL
    return PRESENT


def metric_documents(kind: str) -> list[dict]:
    return [
        {
            "metric": name,
            "operator": operator,
            "offline_value": limit,
            "causal_value": limit,
            "hard_limit": limit,
            "maximum_regression": 0.0,
            "passed": True,
        }
        for name, operator, limit in policies(kind)
    ]


def valid_device(index: int) -> dict:
    return {
        "device_id": f"device-{index}",
        "device_class": "desktop",
        "operating_system": "Example OS",
        "audio_stack": "Example Audio",
        "sample_rate_hz": 48000,
        "channels": 2,
        "capture_milliseconds": 1.0,
        "chunk_milliseconds": 1.0,
        "lookahead_milliseconds": 1.0,
        "resampling_milliseconds": 1.0,
        "inference_milliseconds": 1.0,
        "buffering_milliseconds": 1.0,
        "host_milliseconds": 1.0,
        "output_milliseconds": 1.0,
        "total_milliseconds": 8.0,
    }


def valid_evidence() -> dict:
    return {
        "schema": "denoize-causal-target-sound-promotion-evidence-v1",
        "schema_version": 1,
        "payload": {
            "completed_at_unix_seconds": 1,
            "offline_model_package_sha256": SHA,
            "causal_model_package_sha256": SHA,
            "causal_source_revision": "revision-1",
            "causal_source_sha256": SHA,
            "causal_checkpoint_sha256": SHA,
            "offline_configuration_sha256": SHA,
            "causal_configuration_sha256": SHA,
            "query_catalog_sha256": SHA,
            "query_catalog_revision": "catalog-1",
            "query_class_ids_sha256": SHA,
            "query_class_count": 2,
            "offline_evaluation_result_sha256": SHA,
            "causal_evaluation_result_sha256": SHA,
            "state_reset_flush_result_sha256": SHA,
            "snapshot_roundtrip_result_sha256": SHA,
            "recombination_result_sha256": SHA,
            "latency_result_sha256": SHA,
            "realtime_callback_result_sha256": SHA,
            "transition_result_sha256": SHA,
            "strata": [
                {
                    "id": identifier,
                    "kind": kind,
                    "offline_cases": 50,
                    "causal_cases": 50,
                    "metrics": metric_documents(kind),
                }
                for identifier, kind in STRATA
            ],
            "model_sample_rate_hz": 48000,
            "model_channels": 2,
            "frame_samples": 480,
            "algorithmic_latency_samples": 960,
            "flush_samples": 960,
            "perturbation_latency_cases": 100,
            "effective_latency_limit_milliseconds": 100.0,
            "worst_effective_latency_milliseconds": 8.0,
            "device_measurements": [valid_device(index) for index in range(3)],
            "realtime": {
                "paced_blocks": 10000,
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
                "reset_cases": 100,
                "discontinuity_cases": 100,
                "dropout_cases": 100,
                "overload_fallback_cases": 100,
                "snapshot_roundtrip_cases": 100,
                "resampler_boundary_cases": 100,
                "query_mutation_rejections": 100,
                "late_results_injected": 100,
                "late_results_discarded": 100,
                "stale_generation_results_injected": 100,
                "stale_generation_results_discarded": 100,
                "partial_semantic_removal_publications": 0,
                "recombination_violations": 0,
            },
            "accepted": True,
        },
        "signature": {
            "algorithm": "ed25519",
            "key_id": SHA,
            "value_base64": "A" * 86 + "==",
        },
    }


def valid_report() -> dict:
    return {
        "schema": "denoize-causal-target-sound-report-v1",
        "schema_version": 1,
        "denoize_version": "0.90.0",
        "network_accessed": False,
        "deterministic": True,
        "mode": "preserve",
        "configuration_sha256": SHA,
        "query": {
            "query_sha256": SHA,
            "catalog_sha256": SHA,
            "catalog_revision": "catalog-1",
            "class_ids_sha256": SHA,
            "class_count": 2,
            "class_id": "rain",
            "class_index": 1,
            "canonical_label": "Rain",
            "encoding": "one-hot-v1",
            "open_text_accepted": False,
        },
        "model": {
            "package_sha256": SHA,
            "public_key_sha256": SHA,
            "package_id": "example.model",
            "package_revision": "revision-1",
            "precision_profile": "fp32",
            "package_license_spdx": "MIT",
            "source_revision": "revision-1",
            "source_sha256": SHA,
            "source_license_spdx": "MIT",
            "checkpoint_sha256": SHA,
            "checkpoint_license_spdx": "MIT",
            "accelerator": "cpu",
        },
        "promotion_evidence": {
            "offline_signing_key_id": SHA,
            "causal_signing_key_id": SHA,
            "offline_model_package_sha256": SHA,
            "offline_evaluation_result_sha256": SHA,
            "causal_evaluation_result_sha256": SHA,
            "state_reset_flush_result_sha256": SHA,
            "snapshot_roundtrip_result_sha256": SHA,
            "recombination_result_sha256": SHA,
            "latency_result_sha256": SHA,
            "realtime_callback_result_sha256": SHA,
            "transition_result_sha256": SHA,
            "device_measurements": 3,
            "strata": 14,
            "accepted": True,
        },
        "source_sample_rate": 48000,
        "source_channels": 2,
        "source_frames": 48000,
        "model_sample_rate": 48000,
        "model_channels": 2,
        "frame_samples": 480,
        "algorithmic_latency_samples": 960,
        "flush_samples": 960,
        "input_blocks": 100,
        "flush_blocks": 2,
        "decision_counts": {
            "published_present_blocks": 50,
            "fallback_absent_blocks": 10,
            "fallback_uncertain_blocks": 10,
            "fallback_present_warmup_blocks": 10,
            "fallback_safety_gate_blocks": 10,
            "fallback_flush_blocks": 12,
            "fallback_overload_blocks": 0,
        },
        "presence_transitions": 5,
        "source_clock_withheld_frames": 100,
        "source_clock_conservative_fallback": False,
        "target_published": True,
        "residual_published": True,
        "output_published": True,
        "input_pcm_sha256": SHA,
        "target_pcm_sha256": SHA,
        "residual_pcm_sha256": SHA,
        "output_pcm_sha256": SHA,
        "maximum_model_recombination_error": 0.001,
        "maximum_publication_recombination_error": 0.0,
        "partial_semantic_removal_fallbacks": 0,
        "path_fields_recorded": 0,
        "limitations": [f"limitation {index}" for index in range(8)],
        "warnings": [],
    }


def valid_snapshot() -> dict:
    return {
        "schema": "denoize-causal-target-sound-snapshot-v1",
        "schema_version": 1,
        "model_package_sha256": SHA,
        "configuration_sha256": SHA,
        "query_sha256": SHA,
        "query_catalog_sha256": SHA,
        "selected_class_id": "rain",
        "snapshot_generation": 1,
        "next_frame": 480,
        "present_streak": 1,
        "states": [
            {"element_type": "float32", "shape": [1, 2], "values": [0.0, 1.0]},
            {"element_type": "int64", "shape": [1], "values": [2]},
        ],
    }


def release_wiring_includes_all_schemas() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    upload_blocks = [
        fragment.split("--clobber", 1)[0]
        for fragment in workflow.split('gh release upload "$GITHUB_REF_NAME"')[1:]
    ]
    required = (
        '"$causal_target_sound_promotion_evidence_schema"',
        '"$causal_target_sound_report_schema"',
        '"$causal_target_sound_snapshot_schema"',
    )
    if not any(all(asset in block for asset in required) for block in upload_blocks):
        raise AssertionError("all causal target-sound schemas must be uploaded together")
    verifier = (ROOT / "scripts/verify-release-assets.sh").read_text(encoding="utf-8")
    publisher = (ROOT / "scripts/publish-crates-io.sh").read_text(encoding="utf-8")
    for name in (
        "denoize-causal-target-sound-promotion-evidence-v1.schema.json",
        "denoize-causal-target-sound-report-v1.schema.json",
        "denoize-causal-target-sound-snapshot-v1.schema.json",
    ):
        if name not in verifier or name not in publisher:
            raise AssertionError(f"causal target-sound release wiring omits {name}")


def must_reject(validator: jsonschema.Draft202012Validator, document: dict) -> None:
    try:
        validator.validate(document)
    except jsonschema.ValidationError:
        return
    raise AssertionError("schema unexpectedly accepted invalid document")


def main() -> None:
    evidence_schema = load("denoize-causal-target-sound-promotion-evidence-v1.schema.json")
    report_schema = load("denoize-causal-target-sound-report-v1.schema.json")
    snapshot_schema = load("denoize-causal-target-sound-snapshot-v1.schema.json")
    for document in (evidence_schema, report_schema, snapshot_schema):
        jsonschema.Draft202012Validator.check_schema(document)
    evidence = jsonschema.Draft202012Validator(evidence_schema)
    report = jsonschema.Draft202012Validator(report_schema)
    snapshot = jsonschema.Draft202012Validator(snapshot_schema)
    evidence.validate(valid_evidence())
    report.validate(valid_report())
    snapshot.validate(valid_snapshot())

    invalid = copy.deepcopy(valid_evidence())
    invalid["payload"]["device_measurements"].pop()
    must_reject(evidence, invalid)
    invalid = copy.deepcopy(valid_evidence())
    invalid["payload"]["unknown"] = True
    must_reject(evidence, invalid)
    invalid = copy.deepcopy(valid_report())
    invalid["partial_semantic_removal_fallbacks"] = 1
    must_reject(report, invalid)
    invalid = copy.deepcopy(valid_snapshot())
    invalid["states"][0]["element_type"] = "float64"
    must_reject(snapshot, invalid)
    release_wiring_includes_all_schemas()
    print("causal target-sound schemas: valid and fail-closed examples passed")


if __name__ == "__main__":
    main()
