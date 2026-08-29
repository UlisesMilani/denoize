#!/usr/bin/env python3
"""Validate the Stage 23 and Stage 32 project contracts."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

from jsonschema import Draft202012Validator, RefResolver


PROJECT_SCHEMAS = {
    "denoize-project-v1": "denoize-project-v1.schema.json",
    "denoize-project-verification-v1": "denoize-project-verification-v1.schema.json",
    "denoize-project-render-v1": "denoize-project-render-v1.schema.json",
    "denoize-project-execution-plan-v1": "denoize-project-execution-plan-v1.schema.json",
    "denoize-project-execution-receipt-v1": "denoize-project-execution-receipt-v1.schema.json",
    "denoize-project-receipt-verification-v1": "denoize-project-receipt-verification-v1.schema.json",
    "denoize-project-bundle-v1": "denoize-project-bundle-v1.schema.json",
    "denoize-project-bundle-import-v1": "denoize-project-bundle-import-v1.schema.json",
    "denoize-project-batch-v1": "denoize-project-batch-v1.schema.json",
    "denoize-project-watch-cycle-v1": "denoize-project-watch-cycle-v1.schema.json",
    "denoize-project-v2": "denoize-project-v2.schema.json",
    "denoize-project-v2-verification-v1": "denoize-project-v2-verification-v1.schema.json",
    "denoize-project-v2-render-v1": "denoize-project-v2-render-v1.schema.json",
    "denoize-project-v2-journal-entry-v1": "denoize-project-v2-journal-entry-v1.schema.json",
    "denoize-project-v2-journal-inspection-v1": "denoize-project-v2-journal-inspection-v1.schema.json",
    "denoize-project-v2-checkpoint-v1": "denoize-project-v2-checkpoint-v1.schema.json",
    "denoize-project-v2-cache-request-v1": "denoize-project-v2-cache-request-v1.schema.json",
    "denoize-project-v2-cache-key-v1": "denoize-project-v2-cache-key-v1.schema.json",
    "denoize-project-v2-cache-record-v1": "denoize-project-v2-cache-record-v1.schema.json",
    "denoize-project-v2-cache-verification-v1": "denoize-project-v2-cache-verification-v1.schema.json",
    "denoize-project-v2-interchange-v1": "denoize-project-v2-interchange-v1.schema.json",
    "denoize-project-v2-external-inspection-v1": "denoize-project-v2-external-inspection-v1.schema.json",
    "denoize-project-v2-provenance-v1": "denoize-project-v2-provenance-v1.schema.json",
}

RUNTIME_PROJECT_SCHEMAS = {
    "denoize-project-v1",
    "denoize-project-verification-v1",
    "denoize-project-render-v1",
    "denoize-project-execution-plan-v1",
    "denoize-project-execution-receipt-v1",
    "denoize-project-receipt-verification-v1",
    "denoize-project-bundle-v1",
    "denoize-project-bundle-import-v1",
    "denoize-project-batch-v1",
    "denoize-project-watch-cycle-v1",
    "denoize-project-v2",
    "denoize-project-v2-verification-v1",
    "denoize-project-v2-render-v1",
    "denoize-project-v2-journal-inspection-v1",
    "denoize-project-v2-cache-key-v1",
    "denoize-project-v2-interchange-v1",
    "denoize-project-v2-external-inspection-v1",
    "denoize-project-v2-provenance-v1",
}


def run(binary: Path, *arguments: object) -> subprocess.CompletedProcess[str]:
    command = [str(binary), *(str(argument) for argument in arguments)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if completed.stderr:
        raise RuntimeError(f"command wrote unexpected stderr: {' '.join(command)}\n{completed.stderr}")
    return completed


def run_json(binary: Path, *arguments: object) -> dict[str, object]:
    completed = run(binary, *arguments)
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"command did not produce one JSON document: {' '.join(map(str, arguments))}: {error}"
        ) from error
    if not isinstance(document, dict):
        raise RuntimeError("project command output is not a JSON object")
    return document


def read_json(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return document


def write_source(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        samples = [1_000, -1_000, 2_000, -2_000, 3_000, -3_000, 4_000, -4_000]
        output.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def load_validators(repository: Path) -> dict[str, Draft202012Validator]:
    schemas: dict[str, dict[str, object]] = {}
    for schema_name, filename in PROJECT_SCHEMAS.items():
        path = repository / "schemas" / filename
        schema = read_json(path)
        Draft202012Validator.check_schema(schema)
        schemas[schema_name] = schema
    store = {str(schema["$id"]): schema for schema in schemas.values()}
    validators: dict[str, Draft202012Validator] = {}
    for schema_name, schema in schemas.items():
        validators[schema_name] = Draft202012Validator(
            schema, resolver=RefResolver.from_schema(schema, store=store)
        )
    return validators


def validate_documents(
    validators: dict[str, Draft202012Validator], documents: list[dict[str, object]]
) -> None:
    observed: set[str] = set()
    for document in documents:
        schema_name = document.get("schema")
        if not isinstance(schema_name, str) or schema_name not in validators:
            raise RuntimeError(f"unexpected project schema identity: {schema_name!r}")
        validators[schema_name].validate(document)
        future = dict(document)
        future["unsupported_future_record"] = True
        if validators[schema_name].is_valid(future):
            raise RuntimeError(f"{schema_name} does not reject an unknown future field")
        observed.add(schema_name)
    missing = sorted(RUNTIME_PROJECT_SCHEMAS - observed)
    if missing:
        raise RuntimeError(f"no runtime document was validated for: {', '.join(missing)}")

    manifest = next(
        document for document in documents if document.get("schema") == "denoize-project-v1"
    )
    unsupported_geometry = json.loads(json.dumps(manifest))
    unsupported_geometry["timelines"][0]["channels"] = 65
    if validators["denoize-project-v1"].is_valid(unsupported_geometry):
        raise RuntimeError("project schema accepts channels above the runtime limit")
    unsafe_locator = json.loads(json.dumps(manifest))
    unsafe_locator["sources"][0]["locator"] = "nested/.."
    if validators["denoize-project-v1"].is_valid(unsafe_locator):
        raise RuntimeError("project schema accepts a traversal locator")

    manifest_v2 = next(
        document for document in documents if document.get("schema") == "denoize-project-v2"
    )
    unsafe_v2 = json.loads(json.dumps(manifest_v2))
    unsafe_v2["sources"][0]["storage"]["locator"] = "nested/../source.wav"
    if validators["denoize-project-v2"].is_valid(unsafe_v2):
        raise RuntimeError("project v2 schema accepts a traversal locator")
    future_node = json.loads(json.dumps(manifest_v2))
    future_node["graphs"][0]["clips"][0]["future_node"] = True
    if validators["denoize-project-v2"].is_valid(future_node):
        raise RuntimeError("project v2 schema accepts an unknown executable node field")
    later_without_parent = json.loads(json.dumps(manifest_v2))
    later_without_parent["root_revision"] = 2
    if validators["denoize-project-v2"].is_valid(later_without_parent):
        raise RuntimeError("project v2 schema accepts a later root without its parent")
    initial_with_parent = json.loads(json.dumps(manifest_v2))
    initial_with_parent["parent_digest"] = "00" * 32
    if validators["denoize-project-v2"].is_valid(initial_with_parent):
        raise RuntimeError("project v2 schema accepts an initial root with a parent")

    provenance_v2 = next(
        document
        for document in documents
        if document.get("schema") == "denoize-project-v2-provenance-v1"
    )
    action_without_owner = json.loads(json.dumps(provenance_v2))
    del action_without_owner["payload"]["actions"][0]["graph_id"]
    if validators["denoize-project-v2-provenance-v1"].is_valid(action_without_owner):
        raise RuntimeError("project v2 provenance accepts an action without its owner graph")


def exercise(binary: Path, repository: Path) -> None:
    validators = load_validators(repository)
    with tempfile.TemporaryDirectory(prefix="denoize-project-schema-") as raw_directory:
        root = Path(raw_directory)
        source = root / "source.wav"
        setting = root / "settings.toml"
        manifest = root / "project.json"
        plan = root / "plan.json"
        output = root / "assembled.wav"
        receipt = root / "receipt.json"
        secret = root / "receipt-secret.json"
        public = root / "receipt-public.json"
        bundle = root / "project.dpb"
        imported = root / "imported"
        batch_output = root / "batch-output"
        inbox = root / "inbox"
        watched_output = root / "watched-output"
        watched_manifest = inbox / "watched.json"
        manifest_v2 = root / "project-v2.json"
        muted_manifest_v2 = root / "project-v2-muted.json"
        output_v2 = root / "rendered-v2.wav"
        muted_output_v2 = root / "rendered-v2-muted.wav"
        provenance_v2 = root / "rendered-v2.provenance.json"
        otio = root / "project.otio"
        journal_v2 = root / "project-v2.ndjson"
        write_source(source)
        setting.write_text("strength = 0.5\n", encoding="utf-8")

        documents: list[dict[str, object]] = []
        documents.append(
            run_json(
                binary,
                "project",
                "create",
                manifest,
                "--root",
                root,
                "--project-id",
                "_schema-smoke",
                "--source",
                "source=source.wav",
                "--selection",
                "selection=source,0,0.001",
                "--setting",
                "settings=settings.toml",
            )
        )
        documents.append(
            run_json(binary, "project", "validate", manifest, "--root", root)
        )
        documents.append(
            run_json(
                binary,
                "project",
                "plan",
                "create",
                manifest,
                output,
                "--root",
                root,
                "--output",
                plan,
            )
        )
        run(binary, "receipts", "keygen", secret, public)
        documents.append(
            run_json(
                binary,
                "project",
                "assemble",
                manifest,
                output,
                "--root",
                root,
                "--plan",
                plan,
                "--receipt",
                receipt,
                "--receipt-key",
                secret,
            )
        )
        documents.append(read_json(plan))
        documents.append(read_json(receipt))
        documents.append(
            run_json(
                binary,
                "project",
                "receipt",
                "verify",
                receipt,
                "--root",
                root,
                "--public-key",
                public,
                "--plan",
                plan,
            )
        )
        documents.append(
            run_json(
                binary,
                "project",
                "bundle",
                "create",
                manifest,
                bundle,
                "--root",
                root,
            )
        )
        inspected = run_json(binary, "project", "bundle", "inspect", bundle)
        if inspected != documents[-1]:
            raise RuntimeError("project bundle create and inspect reports differ")
        documents.append(
            run_json(binary, "project", "bundle", "import", bundle, imported)
        )

        batch_output.mkdir()
        documents.append(
            run_json(
                binary,
                "project",
                "batch",
                manifest,
                "--root",
                root,
                "--output-dir",
                batch_output,
            )
        )

        documents.append(
            run_json(
                binary,
                "project",
                "v2",
                "migrate",
                manifest,
                manifest_v2,
                "--root",
                root,
            )
        )
        documents.append(run_json(binary, "project", "v2", "inspect", manifest_v2))
        documents.append(
            run_json(
                binary,
                "project",
                "v2",
                "validate",
                manifest_v2,
                "--root",
                root,
            )
        )
        documents.append(
            run_json(
                binary,
                "project",
                "v2",
                "render",
                manifest_v2,
                output_v2,
                "--root",
                root,
                "--max-memory-mib",
                "64",
            )
        )
        muted_document = read_json(manifest_v2)
        for graph in muted_document["graphs"]:
            for track in graph["tracks"]:
                track["muted"] = True
        muted_manifest_v2.write_text(
            json.dumps(muted_document, separators=(",", ":")), encoding="utf-8"
        )
        documents.append(
            run_json(
                binary,
                "project",
                "v2",
                "render",
                muted_manifest_v2,
                muted_output_v2,
                "--root",
                root,
                "--max-memory-mib",
                "64",
            )
        )
        documents.append(
            run_json(binary, "project", "v2", "cache", "key", manifest_v2)
        )
        documents.append(
            run_json(
                binary,
                "project",
                "v2",
                "interchange",
                "assess",
                manifest_v2,
                "--format",
                "otio",
            )
        )
        documents.append(
            run_json(
                binary,
                "project",
                "v2",
                "otio",
                "export",
                manifest_v2,
                otio,
                "--root",
                root,
                "--accept-losses",
            )
        )
        documents.append(run_json(binary, "project", "v2", "otio", "inspect", otio))
        signed_provenance = run_json(
            binary,
            "project",
            "v2",
            "provenance",
            "sign",
            manifest_v2,
            output_v2,
            provenance_v2,
            "--root",
            root,
            "--secret-key",
            secret,
            "--format",
            "wav-f32",
        )
        documents.append(signed_provenance)
        verified_provenance = run_json(
            binary,
            "project",
            "v2",
            "provenance",
            "verify",
            provenance_v2,
            output_v2,
            "--public-key",
            public,
        )
        if verified_provenance != signed_provenance:
            raise RuntimeError("project v2 provenance sign and verify reports differ")
        journal_v2.write_bytes(b"")
        documents.append(
            run_json(binary, "project", "v2", "journal", "inspect", journal_v2)
        )

        inbox.mkdir()
        run_json(
            binary,
            "project",
            "create",
            watched_manifest,
            "--root",
            root,
            "--project-id",
            ".watched-schema-smoke",
            "--source",
            "source=source.wav",
            "--selection",
            "selection=source,0,0.001",
        )
        documents.append(
            run_json(
                binary,
                "project",
                "watch",
                inbox,
                watched_output,
                "--root",
                root,
                "--receipt-key",
                secret,
                "--once",
                "--settle-ms",
                "0",
            )
        )

        validate_documents(validators, documents)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--denoize", type=Path, required=True)
    arguments = parser.parse_args()
    binary = arguments.denoize.resolve()
    repository = Path(__file__).resolve().parent.parent
    if not binary.is_file():
        parser.error(f"denoize binary does not exist: {binary}")
    exercise(binary, repository)
    print(f"validated {len(PROJECT_SCHEMAS)} project JSON contracts through Stage 32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
