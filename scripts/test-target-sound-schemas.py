#!/usr/bin/env python3
"""Validate closed-catalog target-sound query, evidence, and report contracts."""

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
BASE_GATES = [
    "query-catalog",
    "geometry",
    "finite-normalized-samples",
    "model-recombination",
    "published-recombination",
    "target-peak",
    "residual-peak",
    "target-energy-gain",
    "residual-energy-gain",
    "target-presence",
    "promotion-evidence",
]
STEREO_GATES = [
    "target-stereo-correlation",
    "target-mid-side-energy",
    "residual-stereo-correlation",
    "residual-mid-side-energy",
]
GATE_LIMITS = {
    "model-recombination": (0.0, 0.10),
    "published-recombination": (0.0, 1.0e-6),
    "target-peak": (0.5, 1.0),
    "residual-peak": (0.5, 1.0),
    "target-energy-gain": (0.0, 12.0),
    "residual-energy-gain": (0.0, 12.0),
    "target-stereo-correlation": (0.0, 0.25),
    "target-mid-side-energy": (0.0, 6.0),
    "residual-stereo-correlation": (0.0, 0.25),
    "residual-mid-side-energy": (0.0, 6.0),
}
DEFAULT_GATE_LIMITS = {
    "model-recombination": 0.01,
    "published-recombination": 1.0e-12,
    "target-peak": 1.0,
    "residual-peak": 1.0,
    "target-energy-gain": 3.0,
    "residual-energy-gain": 3.0,
    "target-stereo-correlation": 0.05,
    "target-mid-side-energy": 1.5,
    "residual-stereo-correlation": 0.05,
    "residual-mid-side-energy": 1.5,
}


def schema(name: str) -> dict:
    return json.loads((SCHEMAS / name).read_text(encoding="utf-8"))


def valid_query() -> dict:
    return {
        "schema": "denoize-target-sound-query-v1",
        "schema_version": 1,
        "catalog_revision": "catalog-1",
        "classes": [
            {"id": "alarm", "canonical_label": "Alarm"},
            {"id": "baby-cry", "canonical_label": "Baby cry"},
        ],
        "selected_class_id": "baby-cry",
    }


def policies(kind: str) -> list[tuple[str, str, float]]:
    if kind == "target-absent":
        return ABSENT
    if kind == "binaural-spatial":
        return BINAURAL
    return PRESENT


def outcomes(kind: str) -> list[dict]:
    return [
        {
            "metric": name,
            "value": limit,
            "operator": operator,
            "limit": limit,
            "passed": True,
        }
        for name, operator, limit in policies(kind)
    ]


def valid_evidence() -> dict:
    return {
        "schema": "denoize-target-sound-promotion-evidence-v1",
        "schema_version": 1,
        "payload": {
            "completed_at_unix_seconds": 1,
            "model_package_sha256": SHA,
            "source_revision": "revision-1",
            "source_sha256": SHA,
            "checkpoint_sha256": SHA,
            "configuration_sha256": SHA,
            "query_catalog_sha256": SHA,
            "query_catalog_revision": "catalog-1",
            "query_class_ids_sha256": SHA,
            "query_class_count": 2,
            "class_coverage_manifest_sha256": SHA,
            "evaluated_class_count": 2,
            "minimum_present_cases_per_class": 20,
            "minimum_absent_cases_per_class": 20,
            "worst_class_false_positive_rate": 0.01,
            "worst_class_false_negative_rate": 0.05,
            "artifact_bom_sha256": SHA,
            "training_dataset_license_manifest_sha256": SHA,
            "evaluation_corpus_manifest_sha256": SHA,
            "evaluation_corpus_license_manifest_sha256": SHA,
            "evaluation_result_sha256": SHA,
            "listening_result_sha256": SHA,
            "strata": [
                {
                    "id": identifier,
                    "kind": kind,
                    "cases": 50,
                    "metrics": outcomes(kind),
                }
                for identifier, kind in STRATA
            ],
            "paired_cases": 1000,
            "target_absent_cases": 200,
            "protected_foreground_cases": 200,
            "binaural_cases": 200,
            "listener_count": 20,
            "listener_preference": 0.5,
            "redistributed_restricted_artifacts": 0,
            "unresolved_artifact_licenses": 0,
            "unresolved_training_dataset_licenses": 0,
            "unresolved_evaluation_dataset_licenses": 0,
            "accepted": True,
        },
        "signature": {
            "algorithm": "ed25519",
            "key_id": SHA,
            "value_base64": "A" * 86 + "==",
        },
    }


def gate(kind: str) -> dict:
    if kind in {
        "query-catalog",
        "geometry",
        "finite-normalized-samples",
        "promotion-evidence",
    }:
        return {"kind": kind, "observed": 1.0, "limit": 1.0, "passed": True}
    if kind == "target-presence":
        return {"kind": kind, "observed": 0.98, "limit": 0.9, "passed": True}
    measurements = {
        "model-recombination": 0.001,
        "published-recombination": 0.0,
        "target-peak": 0.5,
        "residual-peak": 0.6,
        "target-energy-gain": -4.0,
        "residual-energy-gain": -2.0,
        "target-stereo-correlation": 0.0,
        "target-mid-side-energy": 0.0,
        "residual-stereo-correlation": 0.0,
        "residual-mid-side-energy": 0.0,
    }
    return {
        "kind": kind,
        "observed": measurements[kind],
        "limit": DEFAULT_GATE_LIMITS[kind],
        "passed": True,
    }


def valid_report(channels: int = 1) -> dict:
    spatial = None if channels == 1 else 0.0
    gates = [gate(kind) for kind in BASE_GATES]
    if channels == 2:
        gates[9:9] = [gate(kind) for kind in STEREO_GATES]
    return {
        "schema": "denoize-target-sound-report-v1",
        "schema_version": 1,
        "denoize_version": "0.89.0",
        "configuration_sha256": SHA,
        "mode": "preserve",
        "network_accessed": False,
        "deterministic": True,
        "closed_class_query": True,
        "model_invoked": True,
        "query": {
            "query_sha256": SHA,
            "catalog_sha256": SHA,
            "catalog_revision": "catalog-1",
            "class_ids_sha256": SHA,
            "class_count": 2,
            "class_id": "baby-cry",
            "class_index": 1,
            "canonical_label": "Baby cry",
            "encoding": "one-hot-v1",
            "open_text_accepted": False,
        },
        "model": {
            "package_sha256": SHA,
            "public_key_sha256": SHA,
            "package_id": "org.example.target-sound",
            "package_revision": "1",
            "precision_profile": "float32-cpu",
            "package_license_spdx": "LicenseRef-Operator-Supplied",
            "source_revision": "revision-1",
            "source_sha256": SHA,
            "source_license_spdx": "MIT",
            "checkpoint_sha256": SHA,
            "checkpoint_license_spdx": "LicenseRef-Operator-Supplied",
            "training_datasets": [
                {
                    "id": "operator-corpus",
                    "revision": "2026-08-30",
                    "sha256": SHA,
                    "license_spdx": "LicenseRef-Operator-Verified",
                }
            ],
            "accelerator": "cpu",
        },
        "promotion_evidence": {
            "signing_key_id": SHA,
            "class_coverage_manifest_sha256": SHA,
            "evaluated_class_count": 2,
            "minimum_present_cases_per_class": 20,
            "minimum_absent_cases_per_class": 20,
            "worst_class_false_positive_rate": 0.01,
            "worst_class_false_negative_rate": 0.05,
            "artifact_bom_sha256": SHA,
            "training_dataset_license_manifest_sha256": SHA,
            "evaluation_corpus_manifest_sha256": SHA,
            "evaluation_corpus_license_manifest_sha256": SHA,
            "evaluation_result_sha256": SHA,
            "listening_result_sha256": SHA,
            "strata": 14,
            "paired_cases": 1000,
            "target_absent_cases": 200,
            "protected_foreground_cases": 200,
            "binaural_cases": 200,
            "listener_count": 20,
            "accepted": True,
        },
        "decision": "accepted-present",
        "candidate_accepted": True,
        "target_published": True,
        "residual_published": True,
        "output_published": True,
        "candidates_retained": True,
        "source_sample_rate": 48000,
        "source_channels": channels,
        "source_frames": 96000,
        "model_sample_rate": 32000,
        "model_channels": channels,
        "model_window_samples": 32000,
        "model_hop_samples": 16000,
        "model_windows": 3,
        "input_pcm_sha256": SHA,
        "target_pcm_sha256": SHA,
        "residual_pcm_sha256": SHA,
        "output_pcm_sha256": SHA,
        "presence": {
            "state": "present",
            "absent_probability": 0.01,
            "uncertain_probability": 0.01,
            "present_probability": 0.98,
            "minimum_absent_probability": 0.9,
            "minimum_present_probability": 0.9,
        },
        "measurements": {
            "input_rms_dbfs": -20.0,
            "target_rms_dbfs": -24.0,
            "residual_rms_dbfs": -22.0,
            "input_peak": 0.8,
            "target_peak": 0.5,
            "residual_peak": 0.6,
            "target_energy_gain_db": -4.0,
            "residual_energy_gain_db": -2.0,
            "model_recombination_maximum_absolute_error": 0.001,
            "publication_recombination_maximum_absolute_error": 0.0,
            "target_stereo_correlation_delta": spatial,
            "target_mid_side_energy_ratio_delta_db": spatial,
            "residual_stereo_correlation_delta": spatial,
            "residual_mid_side_energy_ratio_delta_db": spatial,
        },
        "safety_gates": gates,
        "path_fields_recorded": 0,
        "limitations": [
            "finite catalog only",
            "presence head is not an independent detector",
            "leakage is evaluated at promotion time",
            "signed claims depend on evaluator truthfulness",
            "target and residual are estimates",
            "offline adapter only",
            "no checkpoint is bundled",
        ],
        "warnings": [],
    }


def query_semantics(document: dict) -> None:
    ids = [item["id"] for item in document["classes"]]
    if len(ids) != len(set(ids)):
        raise AssertionError("target-sound class IDs must be unique")
    if document["selected_class_id"] not in ids:
        raise AssertionError("target-sound selected class must exist")


def evidence_semantics(document: dict) -> None:
    payload = document["payload"]
    if payload["evaluated_class_count"] != payload["query_class_count"]:
        raise AssertionError("target-sound evidence omits catalog classes")
    class_case_floor = payload["evaluated_class_count"] * (
        payload["minimum_present_cases_per_class"]
        + payload["minimum_absent_cases_per_class"]
    )
    if payload["paired_cases"] < class_case_floor:
        raise AssertionError("target-sound evidence lacks per-class cases")
    class_coverage_valid = (
        payload["worst_class_false_positive_rate"] <= 0.01
        and payload["worst_class_false_negative_rate"] <= 0.05
    )
    observed = [(item["id"], item["kind"]) for item in payload["strata"]]
    if observed != STRATA:
        raise AssertionError("target-sound strata must be exact and sorted")
    all_passed = True
    for item in payload["strata"]:
        expected_policies = policies(item["kind"])
        observed_metrics = [
            (metric["metric"], metric["operator"]) for metric in item["metrics"]
        ]
        expected_metrics = [
            (name, operator) for name, operator, _ in expected_policies
        ]
        if observed_metrics != expected_metrics:
            raise AssertionError("target-sound metrics differ")
        for metric, (_, operator, hard_limit) in zip(
            item["metrics"], expected_policies, strict=True
        ):
            weaker = (
                metric["limit"] < hard_limit
                if operator == "greater-or-equal"
                else metric["limit"] > hard_limit
            )
            if weaker:
                raise AssertionError("target-sound metric declares a weaker hard limit")
            passed = (
                metric["value"] >= metric["limit"]
                if metric["operator"] == "greater-or-equal"
                else metric["value"] <= metric["limit"]
            )
            if metric["passed"] is not passed:
                raise AssertionError("target-sound metric passed flag is inconsistent")
            all_passed &= passed
    licenses_clear = all(
        payload[name] == 0
        for name in (
            "redistributed_restricted_artifacts",
            "unresolved_artifact_licenses",
            "unresolved_training_dataset_licenses",
            "unresolved_evaluation_dataset_licenses",
        )
    )
    expected_accepted = all_passed and class_coverage_valid and licenses_clear
    if payload["accepted"] is not expected_accepted:
        raise AssertionError("target-sound accepted flag is inconsistent")


def report_semantics(document: dict) -> None:
    if document["model_channels"] != document["source_channels"]:
        raise AssertionError("target-sound model/source channels differ")
    if document["model_hop_samples"] > document["model_window_samples"]:
        raise AssertionError("target-sound hop exceeds window")
    if document["query"]["class_index"] >= document["query"]["class_count"]:
        raise AssertionError("target-sound class index is outside catalog")
    probabilities = [
        document["presence"]["absent_probability"],
        document["presence"]["uncertain_probability"],
        document["presence"]["present_probability"],
    ]
    if abs(sum(probabilities) - 1.0) > 0.001:
        raise AssertionError("target-sound probabilities are not normalized")
    if (
        probabilities[2] >= document["presence"]["minimum_present_probability"]
        and probabilities[2] > probabilities[0]
        and probabilities[2] > probabilities[1]
    ):
        expected_presence = "present"
    elif (
        probabilities[0] >= document["presence"]["minimum_absent_probability"]
        and probabilities[0] > probabilities[1]
        and probabilities[0] > probabilities[2]
    ):
        expected_presence = "absent"
    else:
        expected_presence = "uncertain"
    if document["presence"]["state"] != expected_presence:
        raise AssertionError("target-sound presence state conflicts with probabilities")
    expected_gates = BASE_GATES + (
        STEREO_GATES if document["source_channels"] == 2 else []
    )
    observed_gates = [item["kind"] for item in document["safety_gates"]]
    if set(observed_gates) != set(expected_gates) or len(observed_gates) != len(
        expected_gates
    ):
        raise AssertionError("target-sound safety gates are not exact")
    for item in document["safety_gates"]:
        if item["kind"] in {
            "query-catalog",
            "geometry",
            "finite-normalized-samples",
            "promotion-evidence",
        }:
            passed = item["observed"] == 1.0 and item["limit"] == 1.0
        elif item["kind"] == "target-presence":
            passed = (
                item["observed"] == probabilities[2]
                and item["limit"]
                == document["presence"]["minimum_present_probability"]
                and expected_presence == "present"
            )
        else:
            minimum, maximum = GATE_LIMITS[item["kind"]]
            if not minimum <= item["limit"] <= maximum:
                raise AssertionError("target-sound gate limit weakens the runtime contract")
            passed = item["observed"] <= item["limit"]
        if item["passed"] is not passed:
            raise AssertionError("target-sound gate result is inconsistent")
    measurement_bindings = {
        "model-recombination": "model_recombination_maximum_absolute_error",
        "published-recombination": "publication_recombination_maximum_absolute_error",
        "target-peak": "target_peak",
        "residual-peak": "residual_peak",
        "target-energy-gain": "target_energy_gain_db",
        "residual-energy-gain": "residual_energy_gain_db",
        "target-stereo-correlation": "target_stereo_correlation_delta",
        "target-mid-side-energy": "target_mid_side_energy_ratio_delta_db",
        "residual-stereo-correlation": "residual_stereo_correlation_delta",
        "residual-mid-side-energy": "residual_mid_side_energy_ratio_delta_db",
    }
    gates_by_kind = {item["kind"]: item for item in document["safety_gates"]}
    for kind, measurement in measurement_bindings.items():
        if kind not in gates_by_kind:
            continue
        if gates_by_kind[kind]["observed"] != document["measurements"][measurement]:
            raise AssertionError("target-sound gate is not bound to its measurement")
    accepted = document["decision"] == "accepted-present"
    if accepted != all(item["passed"] for item in document["safety_gates"]):
        raise AssertionError("target-sound decision conflicts with gates")
    expected_flags = [
        document[name]
        for name in (
            "candidate_accepted",
            "target_published",
            "residual_published",
            "output_published",
            "candidates_retained",
        )
    ]
    if any(value is not accepted for value in expected_flags):
        raise AssertionError("target-sound publication flags are inconsistent")
    expected_decision_presence = {
        "accepted-present": "present",
        "withheld-safety-gate": "present",
        "withheld-absent": "absent",
        "withheld-uncertain": "uncertain",
    }
    if expected_decision_presence[document["decision"]] != expected_presence:
        raise AssertionError("target-sound decision conflicts with presence")
    evidence = document["promotion_evidence"]
    if evidence["evaluated_class_count"] != document["query"]["class_count"]:
        raise AssertionError("target-sound report class coverage is incomplete")
    class_case_floor = evidence["evaluated_class_count"] * (
        evidence["minimum_present_cases_per_class"]
        + evidence["minimum_absent_cases_per_class"]
    )
    if evidence["paired_cases"] < class_case_floor:
        raise AssertionError("target-sound report per-class coverage is incomplete")
    digest_names = (
        "target_pcm_sha256",
        "residual_pcm_sha256",
        "output_pcm_sha256",
    )
    if any((document[name] is not None) is not accepted for name in digest_names):
        raise AssertionError("target-sound candidate digests are inconsistent")
    spatial_names = (
        "target_stereo_correlation_delta",
        "target_mid_side_energy_ratio_delta_db",
        "residual_stereo_correlation_delta",
        "residual_mid_side_energy_ratio_delta_db",
    )
    spatial = [document["measurements"][name] for name in spatial_names]
    if document["source_channels"] == 1 and any(value is not None for value in spatial):
        raise AssertionError("mono target-sound reports must omit spatial measurements")
    if document["source_channels"] == 2 and any(value is None for value in spatial):
        raise AssertionError("stereo target-sound reports require spatial measurements")
    datasets = [item["id"] for item in document["model"]["training_datasets"]]
    if len(datasets) != len(set(datasets)):
        raise AssertionError("target-sound training dataset IDs must be unique")


def reject(validator: jsonschema.Draft202012Validator, document: dict) -> None:
    if not list(validator.iter_errors(document)):
        raise AssertionError("invalid target-sound document passed JSON Schema")


def reject_semantics(document: dict, validator) -> None:
    try:
        validator(document)
    except AssertionError:
        return
    raise AssertionError("invalid target-sound semantics unexpectedly passed")


def release_wiring_includes_all_schemas() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    upload_blocks = [
        fragment.split("--clobber", 1)[0]
        for fragment in workflow.split('gh release upload "$GITHUB_REF_NAME"')[1:]
    ]
    required = (
        '"$target_sound_query_schema"',
        '"$target_sound_promotion_evidence_schema"',
        '"$target_sound_report_schema"',
    )
    if not any(all(asset in block for asset in required) for block in upload_blocks):
        raise AssertionError("all target-sound schemas must be uploaded together")
    verifier = (ROOT / "scripts/verify-release-assets.sh").read_text(encoding="utf-8")
    publisher = (ROOT / "scripts/publish-crates-io.sh").read_text(encoding="utf-8")
    for name in (
        "denoize-target-sound-query-v1.schema.json",
        "denoize-target-sound-promotion-evidence-v1.schema.json",
        "denoize-target-sound-report-v1.schema.json",
    ):
        if name not in verifier or name not in publisher:
            raise AssertionError(f"target-sound release wiring omits {name}")


def main() -> None:
    query_validator = jsonschema.Draft202012Validator(
        schema("denoize-target-sound-query-v1.schema.json")
    )
    evidence_validator = jsonschema.Draft202012Validator(
        schema("denoize-target-sound-promotion-evidence-v1.schema.json")
    )
    report_validator = jsonschema.Draft202012Validator(
        schema("denoize-target-sound-report-v1.schema.json")
    )

    query = valid_query()
    query_validator.validate(query)
    query_semantics(query)
    evidence = valid_evidence()
    evidence_validator.validate(evidence)
    evidence_semantics(evidence)
    stronger_evidence = copy.deepcopy(evidence)
    stronger_evidence["payload"]["strata"][1]["metrics"][0]["limit"] = 4.0
    stronger_evidence["payload"]["strata"][1]["metrics"][0]["value"] = 4.0
    evidence_validator.validate(stronger_evidence)
    evidence_semantics(stronger_evidence)
    report = valid_report()
    report_validator.validate(report)
    report_semantics(report)
    stereo_report = valid_report(2)
    report_validator.validate(stereo_report)
    report_semantics(stereo_report)

    invalid = copy.deepcopy(query)
    invalid["selected_class_id"] = "free-text"
    reject_semantics(invalid, query_semantics)
    invalid = copy.deepcopy(query)
    invalid["classes"].append(copy.deepcopy(invalid["classes"][0]))
    reject(query_validator, invalid)
    invalid = copy.deepcopy(evidence)
    invalid["payload"]["unresolved_artifact_licenses"] = 1
    reject(evidence_validator, invalid)
    invalid = copy.deepcopy(evidence)
    invalid["payload"]["strata"][0], invalid["payload"]["strata"][1] = (
        invalid["payload"]["strata"][1],
        invalid["payload"]["strata"][0],
    )
    reject_semantics(invalid, evidence_semantics)
    invalid = copy.deepcopy(evidence)
    invalid["payload"]["strata"][1]["metrics"][0]["limit"] = -999.0
    reject_semantics(invalid, evidence_semantics)
    invalid = copy.deepcopy(evidence)
    invalid["payload"]["evaluated_class_count"] = 3
    reject_semantics(invalid, evidence_semantics)
    invalid = copy.deepcopy(report)
    invalid["query"]["open_text_accepted"] = True
    reject(report_validator, invalid)
    invalid = copy.deepcopy(report)
    invalid["safety_gates"][0]["passed"] = False
    reject_semantics(invalid, report_semantics)
    invalid = copy.deepcopy(report)
    next(
        item
        for item in invalid["safety_gates"]
        if item["kind"] == "published-recombination"
    )["limit"] = 1.0
    reject_semantics(invalid, report_semantics)
    invalid = copy.deepcopy(report)
    invalid["presence"]["state"] = "absent"
    reject_semantics(invalid, report_semantics)
    invalid = copy.deepcopy(report)
    invalid["measurements"]["target_peak"] = 0.4
    reject_semantics(invalid, report_semantics)
    invalid = copy.deepcopy(report)
    invalid["measurements"]["target_stereo_correlation_delta"] = 0.0
    reject_semantics(invalid, report_semantics)
    invalid = copy.deepcopy(report)
    invalid["model"]["training_datasets"].append(
        copy.deepcopy(invalid["model"]["training_datasets"][0])
    )
    reject_semantics(invalid, report_semantics)
    release_wiring_includes_all_schemas()

    print("target-sound JSON schema and semantic tests passed")


if __name__ == "__main__":
    main()
