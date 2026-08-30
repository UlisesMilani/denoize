#!/usr/bin/env python3
"""Validate the closed runtime-model package v2 and numerical-vector schemas."""

from __future__ import annotations

import copy
import json
import pathlib

import jsonschema


ROOT = pathlib.Path(__file__).resolve().parents[1]


def valid_vectors() -> dict:
    return {
        "schema": "denoize-runtime-model-numerical-vectors-v1",
        "profile_id": "fp32",
        "cases": [
            {
                "id": "identity",
                "inputs": [
                    {
                        "name": "input",
                        "element_type": "float32",
                        "shape": [1, 4],
                        "values": [-0.5, 0.0, 0.25, 0.75],
                    }
                ],
                "outputs": [
                    {
                        "name": "output",
                        "element_type": "float32",
                        "shape": [1, 4],
                        "values": [-0.5, 0.0, 0.25, 0.75],
                    }
                ],
                "tolerance": {"absolute": 0.000001, "relative": 0.000001},
            }
        ],
    }


def valid_manifest() -> dict:
    digest = "0" * 64
    resources = {
        "max_session_memory_bytes": 67108879,
        "max_worker_memory_bytes": 4096,
        "max_gpu_session_memory_bytes": 0,
        "max_gpu_worker_memory_bytes": 0,
        "accelerators": ["cpu"],
    }
    tensor = lambda name: {
        "name": name,
        "role": "audio",
        "element_type": "float32",
        "axes": [
            {"name": "batch", "kind": "batch", "fixed": 1},
            {"name": "samples", "kind": "sample", "fixed": None},
        ],
        "optional": False,
        "state_id": None,
    }
    return {
        "schema": "denoize-runtime-model-package-v2",
        "format_version": 2,
        "package_id": "example.identity-v2",
        "package_revision": "1",
        "signing_key_id": "0123456789ABCDEF",
        "runtime": {
            "kind": "onnx-audio-graph-v2",
            "sample_rate_hz": 16000,
            "mode": "finite-and-streaming",
        },
        "frontend": {
            "normalization": "pcm-f32-minus-one-to-one-v1",
            "resampling": "bandlimited-waveform-v1",
            "duration": "preserve-input-frames-v1",
            "channels": {
                "policy": "independent-mono",
                "roles": [],
                "geometry": None,
            },
        },
        "tensors": {"inputs": [tensor("input")], "outputs": [tensor("output")]},
        "state_pairs": [],
        "latency": {
            "frame_samples": 4,
            "hop_samples": 4,
            "left_context_samples": 0,
            "right_context_samples": 0,
            "lookahead_samples": 0,
            "algorithmic_latency_samples": 0,
            "flush_samples": 0,
        },
        "components": [
            {
                "id": "model-fp32",
                "kind": "onnx-model",
                "file": {"filename": "model.onnx", "size_bytes": 5, "sha256": digest},
            },
            {
                "id": "license",
                "kind": "license-notice",
                "file": {"filename": "LICENSE.txt", "size_bytes": 7, "sha256": digest},
            },
            {
                "id": "provenance",
                "kind": "provenance-json",
                "file": {"filename": "provenance.json", "size_bytes": 2, "sha256": digest},
            },
            {
                "id": "vectors-fp32",
                "kind": "numerical-vectors-json",
                "file": {"filename": "vectors.json", "size_bytes": 32, "sha256": digest},
            },
        ],
        "precision_profiles": [
            {
                "id": "fp32",
                "element_type": "float32",
                "model_component": "model-fp32",
                "numerical_vectors_component": "vectors-fp32",
                "resources": resources,
            }
        ],
        "default_precision_profile": "fp32",
        "license": {"spdx": "MIT", "notice_component": "license"},
        "provenance": {
            "component": "provenance",
            "source_repository": "https://example.invalid/source",
            "source_revision": "0123456789abcdef",
            "source_sha256": digest,
            "source_license_spdx": "MIT",
            "checkpoint_source": "https://example.invalid/checkpoint",
            "checkpoint_sha256": "1" * 64,
            "checkpoint_license_spdx": "MIT",
            "conversion_tool": "example-converter",
            "conversion_revision": "1",
            "training_datasets": [
                {
                    "id": "synthetic",
                    "source": "urn:denoize:test:synthetic",
                    "revision": "1",
                    "sha256": "2" * 64,
                    "license_spdx": "CC0-1.0",
                }
            ],
        },
    }


def main() -> None:
    manifest_schema = json.loads(
        (ROOT / "schemas/denoize-runtime-model-package-v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    vectors_schema = json.loads(
        (
            ROOT
            / "schemas/denoize-runtime-model-numerical-vectors-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(manifest_schema)
    jsonschema.Draft202012Validator.check_schema(vectors_schema)
    manifest_validator = jsonschema.Draft202012Validator(manifest_schema)
    vectors_validator = jsonschema.Draft202012Validator(vectors_schema)
    manifest = valid_manifest()
    vectors = valid_vectors()
    manifest_validator.validate(manifest)
    vectors_validator.validate(vectors)

    gpu_resources = copy.deepcopy(manifest)
    gpu_resources["precision_profiles"][0]["resources"]["accelerators"] = ["cuda"]
    manifest_validator.validate(gpu_resources)

    semantic = copy.deepcopy(manifest)
    semantic["tensors"]["inputs"].append(
        {
            "name": "query",
            "role": "query",
            "element_type": "float32",
            "axes": [
                {"name": "batch", "kind": "batch", "fixed": 1},
                {"name": "classes", "kind": "feature", "fixed": 2},
            ],
            "optional": False,
            "state_id": None,
        }
    )
    residual = copy.deepcopy(semantic["tensors"]["outputs"][0])
    residual["name"] = "residual"
    residual["role"] = "residual"
    semantic["tensors"]["outputs"].append(residual)
    manifest_validator.validate(semantic)
    open_text = copy.deepcopy(semantic)
    open_text["tensors"]["inputs"][1]["role"] = "natural-language"
    assert not manifest_validator.is_valid(open_text)

    unknown = copy.deepcopy(manifest)
    unknown["command"] = "python converter.py"
    assert not manifest_validator.is_valid(unknown)
    scripted = copy.deepcopy(manifest)
    scripted["components"][0]["kind"] = "script"
    assert not manifest_validator.is_valid(scripted)
    unsafe_name = copy.deepcopy(manifest)
    unsafe_name["components"][0]["file"]["filename"] = "../model.onnx"
    assert not manifest_validator.is_valid(unsafe_name)
    private_source = copy.deepcopy(manifest)
    private_source["provenance"]["source_repository"] = "/home/user/source"
    assert not manifest_validator.is_valid(private_source)
    secret_source = copy.deepcopy(manifest)
    secret_source["provenance"]["checkpoint_source"] = (
        "https://user:secret@example.invalid/checkpoint?token=secret"
    )
    assert not manifest_validator.is_valid(secret_source)
    fragment_source = copy.deepcopy(manifest)
    fragment_source["provenance"]["training_datasets"][0]["source"] = (
        "urn:denoize:dataset#private-fragment"
    )
    assert not manifest_validator.is_valid(fragment_source)
    oversized_license = copy.deepcopy(manifest)
    oversized_license["components"][1]["file"]["size_bytes"] = 16777217
    assert not manifest_validator.is_valid(oversized_license)
    optional_output = copy.deepcopy(manifest)
    optional_output["tensors"]["outputs"][0]["optional"] = True
    assert not manifest_validator.is_valid(optional_output)
    unknown_vector = copy.deepcopy(vectors)
    unknown_vector["cases"][0]["outputs"][0]["path"] = "/tmp/output"
    assert not vectors_validator.is_valid(unknown_vector)
    unbounded_tolerance = copy.deepcopy(vectors)
    unbounded_tolerance["cases"][0]["tolerance"]["absolute"] = 0.010001
    assert not vectors_validator.is_valid(unbounded_tolerance)
    float_overflow = copy.deepcopy(vectors)
    float_overflow["cases"][0]["inputs"][0]["values"][0] = 3.5e38
    assert not vectors_validator.is_valid(float_overflow)
    fractional_integer = copy.deepcopy(vectors)
    fractional_integer["cases"][0]["inputs"][0]["element_type"] = "int64"
    fractional_integer["cases"][0]["inputs"][0]["values"][0] = 0.5
    assert not vectors_validator.is_valid(fractional_integer)


if __name__ == "__main__":
    main()
