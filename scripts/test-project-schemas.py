#!/usr/bin/env python3
"""Validate every Stage 23 project document against its checked-in schema."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

from jsonschema import Draft202012Validator


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
    validators: dict[str, Draft202012Validator] = {}
    for schema_name, filename in PROJECT_SCHEMAS.items():
        path = repository / "schemas" / filename
        schema = read_json(path)
        Draft202012Validator.check_schema(schema)
        validators[schema_name] = Draft202012Validator(schema)
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
    missing = sorted(set(validators) - observed)
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
    print(f"validated {len(PROJECT_SCHEMAS)} Stage 23 project JSON contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
