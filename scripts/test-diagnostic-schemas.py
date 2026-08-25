#!/usr/bin/env python3
"""Exercise and validate the stable native diagnostic JSON contracts."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import struct
import subprocess
import tempfile
import wave

import jsonschema


ROOT = pathlib.Path(__file__).resolve().parents[1]


def write_fixture(path: pathlib.Path, damaged: bool) -> None:
    rate = 48_000
    rng = random.Random(425)
    frames = bytearray()
    for index in range(rate * 2):
        time = index / rate
        envelope = 1.0 if index % 12_000 < 9_000 else 0.08
        clean = envelope * (
            0.28 * math.sin(math.tau * 180.0 * time)
            + 0.14 * math.sin(math.tau * 510.0 * time)
            + 0.07 * math.sin(math.tau * 2_300.0 * time)
        )
        value = clean
        if damaged:
            value = (
                clean * 2.4
                + rng.uniform(-0.16, 0.16)
                + 0.18 * math.sin(math.tau * 60.0 * time)
                + 0.09 * math.sin(math.tau * 120.0 * time)
                + 0.05 * math.sin(math.tau * 180.0 * time)
            )
        sample = round(max(-1.0, min(1.0, value)) * 32_767)
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(frames)


def run(binary: pathlib.Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(binary), *arguments],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
    )


def validator(schema: dict, diagnostic_schema: dict) -> jsonschema.Draft202012Validator:
    store = {
        diagnostic_schema["$id"]: diagnostic_schema,
        "denoize-diagnostic-v1.schema.json": diagnostic_schema,
    }
    resolver = jsonschema.RefResolver.from_schema(schema, store=store)
    return jsonschema.Draft202012Validator(schema, resolver=resolver)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--denoize", type=pathlib.Path, required=True)
    args = parser.parse_args()
    binary = args.denoize.resolve()
    diagnostic_schema = json.loads(
        (ROOT / "schemas/denoize-diagnostic-v1.schema.json").read_text(encoding="utf-8")
    )
    assessment_schema = json.loads(
        (ROOT / "schemas/denoize-assessment-v1.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(diagnostic_schema)
    jsonschema.Draft202012Validator.check_schema(assessment_schema)

    with tempfile.TemporaryDirectory(prefix="denoize-diagnostic-contract-") as raw:
        directory = pathlib.Path(raw)
        before = directory / "damaged.wav"
        after = directory / "clean.wav"
        write_fixture(before, True)
        write_fixture(after, False)

        first = run(binary, "diagnose", str(before), "--analysis-seconds", "1", "--json")
        assert first.returncode == 0, first.stderr
        assert not first.stderr
        assert str(directory) not in first.stdout
        assert len(first.stdout.encode("utf-8")) < 64 * 1024
        diagnosis = json.loads(first.stdout)
        validator(diagnostic_schema, diagnostic_schema).validate(diagnosis)
        assert diagnosis["network_accessed"] is False
        assert diagnosis["input"]["source_analyzed_frames"] == 48_000
        assert any(
            finding["kind"] == "clipping" and finding["detected"]
            for finding in diagnosis["findings"]
        )

        repeated = run(binary, "diagnose", str(before), "--analysis-seconds", "1", "--json")
        assert repeated.returncode == 0, repeated.stderr
        assert repeated.stdout == first.stdout

        compared = run(
            binary,
            "assess",
            str(before),
            str(after),
            "--analysis-seconds",
            "2",
            "--json",
        )
        assert compared.returncode == 0, compared.stderr
        assessment = json.loads(compared.stdout)
        validator(assessment_schema, diagnostic_schema).validate(assessment)
        assert str(directory) not in compared.stdout
        assert assessment["verdict"] == "improved"
        assert assessment["comparison"]["presentation_preserved"] is True
        assert assessment["comparison"]["semantic_fidelity_assessed"] is False
        assert assessment["comparison"]["quality_score_delta"] > 3

        single = run(binary, "assess", str(after), "--json")
        assert single.returncode == 0, single.stderr
        single_assessment = json.loads(single.stdout)
        validator(assessment_schema, diagnostic_schema).validate(single_assessment)
        assert single_assessment["verdict"] == "single-input"
        assert single_assessment["baseline"] is None
        assert single_assessment["comparison"] is None
        assessment_validator = validator(assessment_schema, diagnostic_schema)
        assert not assessment_validator.is_valid(
            {**single_assessment, "verdict": "improved"}
        )
        assert not assessment_validator.is_valid(
            {**assessment, "verdict": "single-input"}
        )

        invalid = run(
            binary,
            "diagnose",
            str(directory / "missing.wav"),
            "--analysis-seconds",
            "0",
        )
        assert invalid.returncode != 0
        assert not invalid.stdout
        assert "between 1 and 60" in invalid.stderr
        assert "missing.wav" not in invalid.stderr


if __name__ == "__main__":
    main()
