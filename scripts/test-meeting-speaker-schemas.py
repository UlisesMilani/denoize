#!/usr/bin/env python3
"""Validate meeting speaker-track evidence, report, and consent contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
SHA = "ab" * 32
STRATA = [
    "array-available",
    "cross-talk",
    "far-field",
    "four-plus-speakers",
    "language-switch",
    "long-meeting",
    "overlap",
    "real-meeting",
    "single-channel",
    "speaker-count",
    "unknown-speech",
    "unseen-room",
]


def schema(name: str) -> dict:
    return json.loads((SCHEMAS / name).read_text(encoding="utf-8"))


def valid_evidence() -> dict:
    return {
        "schema": "denoize-meeting-speaker-promotion-evidence-v1",
        "schema_version": 1,
        "payload": {
            "completed_at_unix_seconds": 1,
            "model_package_sha256": SHA,
            "source_revision": "revision-1",
            "source_sha256": SHA,
            "checkpoint_sha256": SHA,
            "configuration_sha256": SHA,
            "corpus_manifest_sha256": SHA,
            "corpus_license_manifest_sha256": SHA,
            "evaluation_result_sha256": SHA,
            "listening_result_sha256": SHA,
            "strata": [
                {
                    "id": identifier,
                    "cases": 100,
                    "permutation_si_sdr_improvement_db": 1.0,
                    "diarization_error_rate": 0.2,
                    "jaccard_error_rate": 0.3,
                    "overlap_f1": 0.7,
                    "track_swap_rate": 0.01,
                    "tcp_wer_regression": 0.0,
                    "unknown_false_assignment_rate": 0.005,
                    "non_finite_samples": 0,
                    "passed": True,
                }
                for identifier in STRATA
            ],
            "real_meeting_cases": 100,
            "distinct_speakers": 100,
            "language_count": 2,
            "speaker_count_expected_calibration_error": 0.04,
            "listener_count": 20,
            "listener_preference": 0.5,
            "retained_enrollment_recordings": 0,
            "retained_speaker_embeddings": 0,
            "accepted": True,
        },
        "signature": {
            "algorithm": "ed25519",
            "key_id": SHA,
            "value_base64": "A" * 86 + "==",
        },
    }


def valid_labels() -> dict:
    return {
        "schema": "denoize-meeting-track-labels-v1",
        "schema_version": 1,
        "labels": [
            {
                "track_id": "speaker-001",
                "label": "facilitator",
                "consent_record_sha256": SHA,
                "target_speaker_report_sha256": SHA,
                "raw_enrollment_retained": False,
                "speaker_embedding_retained": False,
            }
        ],
    }


def valid_report() -> dict:
    return {
        "schema": "denoize-meeting-speaker-report-v1",
        "schema_version": 1,
        "denoize_version": "0.87.0",
        "configuration_sha256": SHA,
        "network_accessed": False,
        "deterministic": True,
        "model": {
            "package_sha256": SHA,
            "public_key_sha256": SHA,
            "package_id": "org.example.meeting-css",
            "package_revision": "1",
            "precision_profile": "float32-cpu",
            "source_revision": "revision-1",
            "source_sha256": SHA,
            "source_license_spdx": "Apache-2.0",
            "checkpoint_sha256": SHA,
            "checkpoint_license_spdx": "LicenseRef-Operator-Supplied",
            "accelerator": "cpu",
        },
        "promotion_evidence": {
            "signing_key_id": SHA,
            "corpus_manifest_sha256": SHA,
            "corpus_license_manifest_sha256": SHA,
            "evaluation_result_sha256": SHA,
            "listening_result_sha256": SHA,
            "real_meeting_cases": 100,
            "distinct_speakers": 100,
            "languages": 2,
            "accepted": True,
        },
        "source_sample_rate": 48000,
        "source_channels": 1,
        "source_frames": 96000,
        "model_sample_rate": 16000,
        "model_input_channels": 1,
        "model_window_samples": 32000,
        "model_hop_samples": 16000,
        "model_activity_frames": 200,
        "model_windows": 1,
        "maximum_tracks": 2,
        "published_tracks": 1,
        "track_summaries": [
            {
                "id": "speaker-001",
                "label": None,
                "pcm_sha256": SHA,
                "active_samples": 48000,
                "uncertain_samples": 0,
                "segments": [
                    {
                        "start_sample": 0,
                        "end_sample": 48000,
                        "state": "active",
                        "confidence": 0.9,
                        "overlap": False,
                    }
                ],
                "consent_record_sha256": None,
                "target_speaker_report_sha256": None,
            }
        ],
        "unknown_regions": [
            {"start_sample": 48000, "end_sample": 96000, "confidence": 0.9}
        ],
        "overlap_regions": [],
        "permutation_ambiguous_windows": 0,
        "mixture_pcm_sha256": SHA,
        "unassigned_pcm_sha256": SHA,
        "recombination_maximum_absolute_error": 0.0,
        "exact_output_duration": True,
        "raw_enrollment_retained": False,
        "speaker_embeddings_retained": False,
        "path_fields_recorded": 0,
        "limitations": [
            "anonymous by default",
            "unknown is explicit",
            "eight-track cap",
            "bounded permutation",
            "residual required",
            "no transcription",
            "operator-supplied checkpoint",
        ],
    }


def evidence_semantics(document: dict) -> None:
    payload = document["payload"]
    if [item["id"] for item in payload["strata"]] != STRATA:
        raise AssertionError("meeting-speaker strata must be exact and sorted")
    all_passed = True
    for item in payload["strata"]:
        expected = (
            0 <= item["permutation_si_sdr_improvement_db"] <= 240
            and 0 <= item["diarization_error_rate"] <= 0.30
            and 0 <= item["jaccard_error_rate"] <= 0.40
            and 0.60 <= item["overlap_f1"] <= 1
            and 0 <= item["track_swap_rate"] <= 0.02
            and -1 <= item["tcp_wer_regression"] <= 0.02
            and 0 <= item["unknown_false_assignment_rate"] <= 0.01
            and item["non_finite_samples"] == 0
        )
        if item["passed"] is not expected:
            raise AssertionError("meeting-speaker stratum pass flag is inconsistent")
        all_passed &= expected
    if payload["accepted"] is not (all_passed and payload["listener_preference"] >= 0.5):
        raise AssertionError("meeting-speaker accepted flag is inconsistent")


def report_semantics(document: dict) -> None:
    if document["model_window_samples"] % document["model_activity_frames"]:
        raise AssertionError("activity clock must divide the model window")
    activity_hop = document["model_window_samples"] // document["model_activity_frames"]
    if document["model_hop_samples"] % activity_hop:
        raise AssertionError("model hop must align to the activity clock")
    model_frames = (
        document["source_frames"] * document["model_sample_rate"]
        + document["source_sample_rate"] // 2
    ) // document["source_sample_rate"]
    expected_windows = (
        1
        if model_frames <= document["model_window_samples"]
        else (
            model_frames - document["model_window_samples"]
            + document["model_hop_samples"]
            - 1
        )
        // document["model_hop_samples"]
        + 1
    )
    if document["model_windows"] != expected_windows:
        raise AssertionError("model window count is inconsistent")
    if document["permutation_ambiguous_windows"] > document["model_windows"] - 1:
        raise AssertionError("ambiguous window count is inconsistent")
    tracks = document["track_summaries"]
    if len(tracks) != document["published_tracks"]:
        raise AssertionError("published track count must match track summaries")
    if [item["id"] for item in tracks] != sorted({item["id"] for item in tracks}):
        raise AssertionError("track IDs must be unique and sorted")
    for track in tracks:
        label_fields = [
            track["label"],
            track["consent_record_sha256"],
            track["target_speaker_report_sha256"],
        ]
        if not (all(value is None for value in label_fields) or all(value is not None for value in label_fields)):
            raise AssertionError("label and consent fields must be all-present or all-absent")
        previous = 0
        active = 0
        uncertain = 0
        for segment in track["segments"]:
            if not previous <= segment["start_sample"] < segment["end_sample"] <= document["source_frames"]:
                raise AssertionError("track segments must be ordered and bounded")
            previous = segment["end_sample"]
            duration = segment["end_sample"] - segment["start_sample"]
            if segment["state"] == "active":
                active += duration
            else:
                uncertain += duration
        if not active or active != track["active_samples"] or uncertain != track["uncertain_samples"]:
            raise AssertionError("track duration summary is inconsistent")
    for name in ("unknown_regions", "overlap_regions"):
        previous = 0
        for region in document[name]:
            if not previous <= region["start_sample"] < region["end_sample"] <= document["source_frames"]:
                raise AssertionError(f"{name} must be ordered and bounded")
            previous = region["end_sample"]


def label_semantics(document: dict) -> None:
    ids = [item["track_id"] for item in document["labels"]]
    names = [item["label"] for item in document["labels"]]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise AssertionError("meeting track IDs and labels must be unique")


def reject(validator: jsonschema.Draft202012Validator, document: dict) -> None:
    if not list(validator.iter_errors(document)):
        raise AssertionError("invalid meeting-speaker document unexpectedly passed JSON Schema")


def main() -> None:
    evidence_validator = jsonschema.Draft202012Validator(
        schema("denoize-meeting-speaker-promotion-evidence-v1.schema.json")
    )
    report_validator = jsonschema.Draft202012Validator(
        schema("denoize-meeting-speaker-report-v1.schema.json")
    )
    label_validator = jsonschema.Draft202012Validator(
        schema("denoize-meeting-track-labels-v1.schema.json")
    )

    evidence = valid_evidence()
    evidence_validator.validate(evidence)
    evidence_semantics(evidence)
    report = valid_report()
    report_validator.validate(report)
    report_semantics(report)
    labels = valid_labels()
    label_validator.validate(labels)
    label_semantics(labels)

    invalid = copy.deepcopy(evidence)
    invalid["payload"]["retained_speaker_embeddings"] = 1
    reject(evidence_validator, invalid)
    invalid = copy.deepcopy(evidence)
    invalid["payload"]["strata"][0], invalid["payload"]["strata"][1] = (
        invalid["payload"]["strata"][1],
        invalid["payload"]["strata"][0],
    )
    try:
        evidence_semantics(invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("permuted strata unexpectedly passed semantic validation")
    invalid = copy.deepcopy(evidence)
    invalid["payload"]["strata"][0]["tcp_wer_regression"] = -2
    reject(evidence_validator, invalid)
    invalid = copy.deepcopy(report)
    invalid["track_summaries"][0]["label"] = "facilitator"
    try:
        report_semantics(invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("partial label metadata unexpectedly passed semantic validation")
    invalid = copy.deepcopy(labels)
    invalid["labels"][0]["speaker_embedding_retained"] = True
    reject(label_validator, invalid)
    invalid = copy.deepcopy(labels)
    invalid["labels"][0]["label"] = "facilitator\u0000"
    reject(label_validator, invalid)

    print("meeting-speaker JSON schema and semantic tests passed")


if __name__ == "__main__":
    main()
