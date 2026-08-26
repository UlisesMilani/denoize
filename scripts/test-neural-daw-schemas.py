#!/usr/bin/env python3
"""Validate the neural DAW state and CLI automation contracts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


def run(binary: Path, model_dir: Path, *args: str, success: bool = True) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["DENOIZE_MODEL_DIR"] = str(model_dir)
    result = subprocess.run(
        [str(binary), *args],
        cwd=ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if success and result.returncode != 0:
        raise AssertionError(result.stderr.decode("utf-8", errors="replace"))
    if not success and result.returncode == 0:
        raise AssertionError(f"command unexpectedly succeeded: {args!r}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--denoize", type=Path, default=ROOT / "target" / "debug" / "denoize")
    arguments = parser.parse_args()
    binary = arguments.denoize.resolve()

    state_schema = load("denoize-neural-daw-session-v1.schema.json")
    cli_schema = load("denoize-cli-output-v1.schema.json")
    Draft202012Validator.check_schema(state_schema)
    Draft202012Validator.check_schema(cli_schema)
    state_validator = Draft202012Validator(state_schema)
    cli_validator = Draft202012Validator(cli_schema)

    with tempfile.TemporaryDirectory(prefix="denoize-neural-schema-") as temporary:
        directory = Path(temporary)
        model_dir = directory / "models"
        state_path = directory / "session.json"

        info = json.loads(
            run(
                binary,
                model_dir,
                "plugin",
                "neural",
                "info",
                "--sample-rate",
                "48000",
                "--json",
            ).stdout
        )
        cli_validator.validate(info)
        assert info["model_installed"] is False
        assert info["latency_frames"] == 11520

        latency = json.loads(
            run(
                binary,
                model_dir,
                "plugin",
                "neural",
                "latency",
                "--sample-rate",
                "44100",
                "--json",
            ).stdout
        )
        cli_validator.validate(latency)
        assert latency["chunk_frames"] == 441
        assert latency["latency_frames"] == latency["measured_latency_frames"] == 10584

        state = json.loads(
            run(
                binary,
                model_dir,
                "plugin",
                "neural",
                "session",
                "create",
                str(state_path),
                "--mono",
                "--mix",
                "0.75",
                "--fallback",
                "last-safe-gain",
                "--json",
            ).stdout
        )
        state_validator.validate(state)
        assert json.loads(state_path.read_text(encoding="utf-8")) == state

        validation = json.loads(
            run(
                binary,
                model_dir,
                "plugin",
                "neural",
                "session",
                "validate",
                str(state_path),
                "--json",
            ).stdout
        )
        cli_validator.validate(validation)

        duplicate = run(
            binary,
            model_dir,
            "plugin",
            "neural",
            "session",
            "create",
            str(state_path),
            success=False,
        )
        assert duplicate.stdout == b""
        state_validator.validate(json.loads(state_path.read_text(encoding="utf-8")))

        fractional = run(
            binary,
            model_dir,
            "plugin",
            "neural",
            "latency",
            "--sample-rate",
            "44100.5",
            "--json",
        )
        fractional_document = json.loads(fractional.stdout)
        cli_validator.validate(fractional_document)
        assert fractional_document["latency_frames"] == 10608
        assert fractional_document["measured_latency_frames"] == 10608
        assert fractional_document["matches_reported"] is True

        for field, value in (
            ("future", True),
            ("model_sha256", "0" * 64),
            ("latency_policy", "future"),
        ):
            invalid = dict(state)
            invalid[field] = value
            assert list(state_validator.iter_errors(invalid))
            invalid_path = directory / f"invalid-{field}.json"
            invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
            rejected = run(
                binary,
                model_dir,
                "plugin",
                "neural",
                "session",
                "validate",
                str(invalid_path),
                "--json",
                success=False,
            )
            assert rejected.stdout == b""


if __name__ == "__main__":
    main()
