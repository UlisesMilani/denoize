#!/usr/bin/env python3
"""Validate bounded music-restoration evidence and report contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
SHA = "ab" * 32
STRATA = [
    "aac-64k",
    "clean-bypass",
    "genre-unseen",
    "long-form",
    "mono",
    "mp3-64k",
    "neural-codec",
    "percussion-transients",
    "phase-critical",
    "stereo-image",
    "unseen-codec",
    "wideband-reference",
]


def schema(name: str) -> dict:
    return json.loads((SCHEMAS / name).read_text(encoding="utf-8"))


def valid_evidence() -> dict:
    return {
        "schema": "denoize-music-restoration-promotion-evidence-v1",
        "schema_version": 1,
        "payload": {
            "completed_at_unix_seconds": 1,
            "task": "codec-repair",
            "model_package_sha256": SHA,
            "source_revision": "revision-1",
            "source_sha256": SHA,
            "checkpoint_sha256": SHA,
            "configuration_sha256": SHA,
            "artifact_bom_sha256": SHA,
            "training_dataset_license_manifest_sha256": SHA,
            "evaluation_corpus_manifest_sha256": SHA,
            "evaluation_corpus_license_manifest_sha256": SHA,
            "evaluation_result_sha256": SHA,
            "listening_result_sha256": SHA,
            "strata": [
                {
                    "id": identifier,
                    "cases": 100,
                    "multi_mel_snr_improvement_db": 1.0,
                    "zimtohrli_regression": 0.0,
                    "fad_clap_regression": 0.0,
                    "low_band_snr_db": 60.0,
                    "transient_loss_rate": 0.01,
                    "stereo_correlation_error": 0.01,
                    "phase_error_radians": 0.1,
                    "duration_mismatch_samples": 0,
                    "clipped_samples": 0,
                    "non_finite_samples": 0,
                    "passed": True,
                }
                for identifier in STRATA
            ],
            "paired_clips": 1000,
            "full_length_tracks": 50,
            "instrument_classes": 8,
            "genres": 8,
            "clean_bypass_cases": 100,
            "mono_cases": 100,
            "stereo_cases": 100,
            "listener_count": 20,
            "listener_preference": 0.5,
            "redistributed_restricted_artifacts": 0,
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
        "schema": "denoize-music-restoration-report-v1",
        "schema_version": 1,
        "denoize_version": "0.88.0",
        "task": "codec-repair",
        "configuration_sha256": SHA,
        "network_accessed": False,
        "deterministic": True,
        "candidate_render": True,
        "recovered_ground_truth_claimed": False,
        "dry_stems_produced": False,
        "creative_mastering_applied": False,
        "model": {
            "package_sha256": SHA,
            "public_key_sha256": SHA,
            "package_id": "org.example.music-restoration",
            "package_revision": "1",
            "precision_profile": "float32-cpu",
            "package_license_spdx": "LicenseRef-Operator-Supplied",
            "source_revision": "revision-1",
            "source_sha256": SHA,
            "source_license_spdx": "Apache-2.0",
            "checkpoint_sha256": SHA,
            "checkpoint_license_spdx": "LicenseRef-Operator-Supplied",
            "training_datasets": [
                {
                    "id": "operator-corpus",
                    "revision": "2026-08-27",
                    "sha256": SHA,
                    "license_spdx": "LicenseRef-Operator-Verified",
                }
            ],
            "accelerator": "cpu",
        },
        "promotion_evidence": {
            "signing_key_id": SHA,
            "artifact_bom_sha256": SHA,
            "training_dataset_license_manifest_sha256": SHA,
            "evaluation_corpus_manifest_sha256": SHA,
            "evaluation_corpus_license_manifest_sha256": SHA,
            "evaluation_result_sha256": SHA,
            "listening_result_sha256": SHA,
            "paired_clips": 1000,
            "full_length_tracks": 50,
            "listener_count": 20,
            "accepted": True,
        },
        "source_sample_rate": 48000,
        "source_channels": 1,
        "source_frames": 96000,
        "output_sample_rate": 48000,
        "output_channels": 1,
        "output_frames": 96000,
        "model_sample_rate": 48000,
        "model_channels": 1,
        "model_window_samples": 48000,
        "model_hop_samples": 24000,
        "model_state_frames_per_window": 100,
        "model_windows": 3,
        "decision_frames": 200,
        "applied_decision_frames": 50,
        "bypassed_decision_frames": 100,
        "uncertain_decision_frames": 50,
        "applied_source_samples": 24000,
        "changed_samples": 24000,
        "regions": [
            {
                "start_sample": 0,
                "end_sample": 24000,
                "decision": "apply",
                "confidence": 0.9,
            },
            {
                "start_sample": 72000,
                "end_sample": 96000,
                "decision": "uncertain",
                "confidence": 0.7,
            },
        ],
        "input_pcm_sha256": SHA,
        "output_pcm_sha256": SHA,
        "correction_pcm_sha256": SHA,
        "correction_recombination_maximum_absolute_error": 0.0,
        "maximum_absolute_correction": 0.1,
        "maximum_output_peak": 0.9,
        "input_stereo_correlation": None,
        "output_stereo_correlation": None,
        "stereo_correlation_delta": None,
        "input_mid_side_energy_ratio_db": None,
        "output_mid_side_energy_ratio_db": None,
        "mid_side_energy_ratio_delta_db": None,
        "exact_output_geometry": True,
        "path_fields_recorded": 0,
        "limitations": [
            "candidate render only",
            "no recovered-ground-truth claim",
            "no dry stems",
            "no creative mastering",
            "uncertain regions bypassed",
            "correction residual required",
            "operator-supplied checkpoint",
        ],
    }


def evidence_semantics(document: dict) -> None:
    payload = document["payload"]
    if [item["id"] for item in payload["strata"]] != STRATA:
        raise AssertionError("music-restoration strata must be exact and sorted")
    all_passed = True
    for item in payload["strata"]:
        expected = (
            0 <= item["multi_mel_snr_improvement_db"] <= 240
            and -1 <= item["zimtohrli_regression"] <= 0.01
            and -1000 <= item["fad_clap_regression"] <= 0.02
            and 40 <= item["low_band_snr_db"] <= 240
            and 0 <= item["transient_loss_rate"] <= 0.02
            and 0 <= item["stereo_correlation_error"] <= 0.02
            and 0 <= item["phase_error_radians"] <= 0.2
            and item["duration_mismatch_samples"] == 0
            and item["clipped_samples"] == 0
            and item["non_finite_samples"] == 0
        )
        if item["passed"] is not expected:
            raise AssertionError("music-restoration stratum pass flag is inconsistent")
        all_passed &= expected
    expected_accepted = (
        all_passed
        and payload["listener_preference"] >= 0.5
        and payload["redistributed_restricted_artifacts"] == 0
    )
    if payload["accepted"] is not expected_accepted:
        raise AssertionError("music-restoration accepted flag is inconsistent")


def report_semantics(document: dict) -> None:
    if not (
        document["output_sample_rate"] == document["source_sample_rate"]
        and document["output_channels"] == document["source_channels"]
        and document["output_frames"] == document["source_frames"]
        and document["model_channels"] == document["source_channels"]
    ):
        raise AssertionError("music-restoration output/model geometry is inconsistent")
    if document["model_window_samples"] % document["model_state_frames_per_window"]:
        raise AssertionError("state clock must divide the model window")
    state_hop = (
        document["model_window_samples"]
        // document["model_state_frames_per_window"]
    )
    if document["model_hop_samples"] % state_hop:
        raise AssertionError("model hop must align to the state clock")
    model_frames = (
        document["source_frames"] * document["model_sample_rate"]
        + document["source_sample_rate"] // 2
    ) // document["source_sample_rate"]
    expected_decisions = (model_frames + state_hop - 1) // state_hop
    if document["decision_frames"] != expected_decisions:
        raise AssertionError("decision frame count is inconsistent")
    expected_windows = (
        1
        if model_frames <= document["model_window_samples"]
        else (
            model_frames
            - document["model_window_samples"]
            + document["model_hop_samples"]
            - 1
        )
        // document["model_hop_samples"]
        + 1
    )
    if document["model_windows"] != expected_windows:
        raise AssertionError("model window count is inconsistent")
    if (
        document["applied_decision_frames"]
        + document["bypassed_decision_frames"]
        + document["uncertain_decision_frames"]
        != document["decision_frames"]
    ):
        raise AssertionError("decision summary is inconsistent")
    if document["changed_samples"] > (
        document["source_frames"] * document["source_channels"]
    ):
        raise AssertionError("changed sample count is impossible")
    previous_end = 0
    applied_samples = 0
    for region in document["regions"]:
        if not (
            previous_end
            <= region["start_sample"]
            < region["end_sample"]
            <= document["source_frames"]
        ):
            raise AssertionError("regions must be ordered and bounded")
        previous_end = region["end_sample"]
        if region["decision"] == "apply":
            applied_samples += region["end_sample"] - region["start_sample"]
    if applied_samples != document["applied_source_samples"]:
        raise AssertionError("applied source duration is inconsistent")
    dataset_ids = [item["id"] for item in document["model"]["training_datasets"]]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise AssertionError("training dataset IDs must be unique")
    stereo_names = [
        "input_stereo_correlation",
        "output_stereo_correlation",
        "stereo_correlation_delta",
        "input_mid_side_energy_ratio_db",
        "output_mid_side_energy_ratio_db",
        "mid_side_energy_ratio_delta_db",
    ]
    stereo_values = [document[name] for name in stereo_names]
    if document["source_channels"] == 1:
        if any(value is not None for value in stereo_values):
            raise AssertionError("mono reports must omit all stereo metrics")
    elif any(value is None for value in stereo_values):
        raise AssertionError("stereo reports must include every stereo metric")


def reject(validator: jsonschema.Draft202012Validator, document: dict) -> None:
    if not list(validator.iter_errors(document)):
        raise AssertionError(
            "invalid music-restoration document unexpectedly passed JSON Schema"
        )


def reject_semantics(document: dict, validator) -> None:
    try:
        validator(document)
    except AssertionError:
        return
    raise AssertionError("invalid music-restoration semantics unexpectedly passed")


def release_workflow_uploads_schemas() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    upload_blocks = [
        fragment.split("--clobber", 1)[0]
        for fragment in workflow.split('gh release upload "$GITHUB_REF_NAME"')[1:]
    ]
    required = (
        '"$music_restoration_promotion_evidence_schema"',
        '"$music_restoration_report_schema"',
    )
    if not any(all(asset in block for asset in required) for block in upload_blocks):
        raise AssertionError(
            "music-restoration schemas must be uploaded to every tagged release"
        )


def main() -> None:
    evidence_validator = jsonschema.Draft202012Validator(
        schema("denoize-music-restoration-promotion-evidence-v1.schema.json")
    )
    report_validator = jsonschema.Draft202012Validator(
        schema("denoize-music-restoration-report-v1.schema.json")
    )

    evidence = valid_evidence()
    evidence_validator.validate(evidence)
    evidence_semantics(evidence)
    report = valid_report()
    report_validator.validate(report)
    report_semantics(report)

    invalid = copy.deepcopy(evidence)
    invalid["payload"]["redistributed_restricted_artifacts"] = 1
    reject(evidence_validator, invalid)
    invalid = copy.deepcopy(evidence)
    invalid["payload"]["strata"][0], invalid["payload"]["strata"][1] = (
        invalid["payload"]["strata"][1],
        invalid["payload"]["strata"][0],
    )
    reject_semantics(invalid, evidence_semantics)
    invalid = copy.deepcopy(evidence)
    invalid["payload"]["strata"][0]["transient_loss_rate"] = 0.03
    reject_semantics(invalid, evidence_semantics)
    invalid = copy.deepcopy(report)
    invalid["dry_stems_produced"] = True
    reject(report_validator, invalid)
    invalid = copy.deepcopy(report)
    invalid["decision_frames"] += 1
    reject_semantics(invalid, report_semantics)
    invalid = copy.deepcopy(report)
    invalid["input_stereo_correlation"] = 0.5
    reject_semantics(invalid, report_semantics)
    invalid = copy.deepcopy(report)
    invalid["model"]["training_datasets"].append(
        copy.deepcopy(invalid["model"]["training_datasets"][0])
    )
    reject_semantics(invalid, report_semantics)
    release_workflow_uploads_schemas()

    print("music-restoration JSON schema and semantic tests passed")


if __name__ == "__main__":
    main()
