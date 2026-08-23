#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tmp_dir=$(mktemp -d)
cleanup() {
  find "$tmp_dir" -depth -delete
}
trap cleanup EXIT

denoize_bin=${DENOIZE_BIN:-$repo_dir/target/debug/denoize}
if [[ ! -x "$denoize_bin" ]]; then
  if [[ -n "${DENOIZE_BIN:-}" ]]; then
    echo "DENOIZE_BIN is not executable: $denoize_bin" >&2
    exit 2
  fi
  (
    cd "$repo_dir"
    cargo build --locked --no-default-features --bin denoize
  )
fi

python3 - "$tmp_dir" <<'PY'
import hashlib
import json
import pathlib
import struct
import sys
import wave

root = pathlib.Path(sys.argv[1])
audio_dir = root / "corpus" / "audio"
audio_dir.mkdir(parents=True)
sample_rate = 16_000
frames = sample_rate


def triangle(index: int, period: int) -> int:
    phase = index % period
    half = period // 2
    return phase if phase < half else period - phase


clean = []
noisy = []
for index in range(frames):
    sample = (triangle(index, 80) - 20) * 500
    sample += (triangle(index, 40) - 10) * 250
    noise = (((index * 7_919) % 997) - 498) * 2
    clean.append(max(-32_768, min(32_767, sample)))
    noisy.append(max(-32_768, min(32_767, sample + noise)))


def write_wav(path: pathlib.Path, samples: list[int]) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(struct.pack(f"<{len(samples)}h", *samples))


clean_path = audio_dir / "clean.wav"
noisy_path = audio_dir / "noisy.wav"
write_wav(clean_path, clean)
write_wav(noisy_path, noisy)

parameters = {
    "clean_signal": "integer dual-triangle periods=80,40 amplitudes=500,250",
    "frames": frames,
    "noise": "integer modular sequence multiplier=7919 modulus=997 amplitude=2",
    "sample_format": "signed-16-bit-pcm-mono",
    "sample_rate": sample_rate,
}
parameters_digest = hashlib.sha256(
    json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def fingerprint(path: pathlib.Path) -> dict[str, object]:
    content = path.read_bytes()
    return {"len": len(content), "digest": hashlib.sha256(content).hexdigest()}


license_metadata = {
    "spdx_id": "CC0-1.0",
    "name": "Creative Commons Zero v1.0 Universal",
    "url": "https://creativecommons.org/publicdomain/zero/1.0/",
}
source = {
    "uri": "urn:denoize:evaluation-fixture:cc0-synthetic-v1",
    "revision": f"sha256-{parameters_digest}",
}
preparation = {
    "description": "Deterministic integer synthetic signal with additive modular noise",
    "tool": "scripts/test-evaluation-evidence.sh",
    "tool_version": "1.0.0",
    "parameters_digest": parameters_digest,
}


def artifact(path: str, absolute_path: pathlib.Path) -> dict[str, object]:
    return {
        "path": path,
        "fingerprint": fingerprint(absolute_path),
        "license": license_metadata,
        "source": source,
        "preparation": preparation,
    }


manifest = {
    "schema": "denoize-evaluation-corpus-v1",
    "schema_version": 1,
    "corpus_id": "cc0-synthetic-contract",
    "corpus_version": "1.0.0",
    "title": "CC0 deterministic release-evaluation contract fixture",
    "cases": [
        {
            "id": "synthetic-noise-001",
            "clean": artifact("audio/clean.wav", clean_path),
            "noisy": artifact("audio/noisy.wav", noisy_path),
            "tags": ["synthetic"],
        }
    ],
    "recipe": {
        "backend": "classical",
        "preset": "speech",
        "accelerator": "cpu",
        "deterministic": True,
        "seed": None,
        "channel_mode": "independent",
        "sgmse_profile": "balanced",
        "model": None,
        "model_sample_rate": None,
    },
    "policy": {
        "warmup_runs": 0,
        "measured_runs": 1,
        "silence_threshold_dbfs": -90.0,
        "dropout_window_ms": 20,
        "thresholds": [
            {
                "metric": "objective.si-sdr-improvement-db",
                "aggregation": "minimum",
                "operator": "greater-or-equal",
                "value": -200.0,
            },
            {
                "metric": "perceptual.musical-noise",
                "aggregation": "maximum",
                "operator": "less-or-equal",
                "value": 1.0,
            },
            {
                "metric": "output.decode-integrity",
                "aggregation": "minimum",
                "operator": "greater-or-equal",
                "value": 1.0,
            },
            {
                "metric": "performance.realtime-factor",
                "aggregation": "maximum",
                "operator": "less-or-equal",
                "value": 10_000.0,
            },
        ],
        "regression_tolerances": [
            {
                "metric": "objective.si-sdr-improvement-db",
                "aggregation": "minimum",
                "direction": "higher-is-better",
                "max_regression": 0.0,
            }
        ],
        "listening": {
            "required": False,
            "rationale": "Synthetic contract fixture; no human preference claim is made",
            "protocol": None,
        },
    },
}
(root / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

manifest="$tmp_dir/manifest.json"
corpus_root="$tmp_dir/corpus"
secret_key="$tmp_dir/evaluation-secret.json"
public_key="$tmp_dir/evaluation-public.json"
result="$tmp_dir/evaluation-result.json"

"$denoize_bin" receipts keygen "$secret_key" "$public_key" >/dev/null
"$denoize_bin" evaluate validate "$manifest" \
  --corpus-root "$corpus_root" --json > "$tmp_dir/corpus-verification.json"
"$denoize_bin" evaluate run "$manifest" \
  --corpus-root "$corpus_root" \
  --key "$secret_key" \
  --output "$result" \
  --json > "$tmp_dir/run-output.json"
"$denoize_bin" evaluate verify "$result" \
  --key "$public_key" \
  --manifest "$manifest" \
  --json > "$tmp_dir/evaluation-verification.json"
"$denoize_bin" evaluate compare "$result" "$result" \
  --key "$public_key" \
  --json > "$tmp_dir/evaluation-comparison.json"

python3 - "$repo_dir" "$tmp_dir" <<'PY'
import copy
import json
import pathlib
import stat
import sys

try:
    import jsonschema
except ImportError as error:
    raise SystemExit(
        "python3-jsonschema is required to test evaluation evidence schemas"
    ) from error

repo = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
schema_dir = repo / "schemas"
schema_names = [
    "denoize-evaluation-comparison-v1.schema.json",
    "denoize-evaluation-corpus-v1.schema.json",
    "denoize-evaluation-corpus-verification-v1.schema.json",
    "denoize-evaluation-result-v1.schema.json",
    "denoize-evaluation-verification-v1.schema.json",
    "denoize-listening-result-v1.schema.json",
]
schemas = {}
for name in schema_names:
    schema = json.loads((schema_dir / name).read_text(encoding="utf-8"))
    validator_type = jsonschema.validators.validator_for(schema)
    validator_type.check_schema(schema)
    schemas[name] = validator_type(schema)

instances = [
    ("manifest.json", "denoize-evaluation-corpus-v1.schema.json"),
    (
        "corpus-verification.json",
        "denoize-evaluation-corpus-verification-v1.schema.json",
    ),
    ("evaluation-result.json", "denoize-evaluation-result-v1.schema.json"),
    (
        "evaluation-verification.json",
        "denoize-evaluation-verification-v1.schema.json",
    ),
    (
        "evaluation-comparison.json",
        "denoize-evaluation-comparison-v1.schema.json",
    ),
]
loaded = {}
for instance_name, schema_name in instances:
    instance = json.loads((root / instance_name).read_text(encoding="utf-8"))
    schemas[schema_name].validate(instance)
    loaded[instance_name] = instance

run_output = json.loads((root / "run-output.json").read_text(encoding="utf-8"))
result = loaded["evaluation-result.json"]
if run_output != result:
    raise SystemExit("evaluate run stdout differs from the atomically published result")
if not result["payload"]["accepted"]:
    raise SystemExit("contract-fixture evaluation was unexpectedly rejected")
if not loaded["evaluation-verification.json"]["accepted"]:
    raise SystemExit("signed contract-fixture evaluation did not verify as accepted")
if not loaded["evaluation-comparison.json"]["passed"]:
    raise SystemExit("self regression comparison did not pass")

encoded = json.dumps(result, sort_keys=True)
for private_locator in ("clean.wav", "noisy.wav", str(root)):
    if private_locator in encoded:
        raise SystemExit(f"signed result leaked corpus locator: {private_locator}")

secret_mode = stat.S_IMODE((root / "evaluation-secret.json").stat().st_mode)
if secret_mode & 0o077:
    raise SystemExit(f"evaluation secret key permissions are too broad: {secret_mode:o}")

tampered = copy.deepcopy(result)
tampered["payload"]["cases"][0]["objective"]["snr_db"] += 0.25
(root / "tampered-result.json").write_text(
    json.dumps(tampered, indent=2) + "\n", encoding="utf-8"
)
PY

if "$denoize_bin" evaluate verify "$tmp_dir/tampered-result.json" \
  --key "$public_key" >/dev/null 2>&1; then
  echo "tampered evaluation result unexpectedly verified" >&2
  exit 1
fi

if "$denoize_bin" evaluate run "$manifest" \
  --corpus-root "$corpus_root" \
  --key "$secret_key" \
  --output "$result" >/dev/null 2>&1; then
  echo "evaluation result unexpectedly overwrote an existing file" >&2
  exit 1
fi

echo "licensed-corpus evaluation evidence contract passed"
