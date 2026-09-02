#!/usr/bin/env python3
"""Create a model-anonymous DPDFNet/GTCRN paired-listening bundle."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import html
import json
import os
from pathlib import Path
import shutil
import tempfile
import wave


EXPECTED_STRATA = {
    "recorded-noise": 4,
    "babble": 3,
    "source-preservation": 3,
    "synthetic-noise": 2,
}
MODEL_DPDFNET = "dpdfnet2-48khz-hr"
MODEL_GTCRN = "gtcrn-dns3"
MAX_AUDIO_BYTES = 64 * 1024 * 1024


class BundleError(RuntimeError):
    pass


def load_json(path: Path) -> tuple[dict, bytes]:
    if path.is_symlink() or not path.is_file():
        raise BundleError(f"JSON input is not a regular file: {path}")
    payload = path.read_bytes()
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"invalid JSON input {path}: {error}") from error
    if not isinstance(document, dict):
        raise BundleError(f"JSON input must be an object: {path}")
    return document, payload


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def keyed(key: bytes, label: str) -> bytes:
    return hmac.new(key, label.encode("utf-8"), hashlib.sha256).digest()


def classify(case: dict) -> str:
    if case.get("kind") == "clean-preservation":
        return "source-preservation"
    if case.get("kind") != "noise-matrix":
        raise BundleError(f"unsupported listening case kind for {case.get('id')}")
    noise = case.get("noise")
    if noise == "three-talker-babble":
        return "babble"
    if isinstance(noise, str) and noise.startswith("freesound-"):
        return "recorded-noise"
    return "synthetic-noise"


def audio_record(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise BundleError(f"audio input is not a regular file: {path}")
    size = path.stat().st_size
    if not 44 <= size <= MAX_AUDIO_BYTES:
        raise BundleError(f"audio size outside 44..={MAX_AUDIO_BYTES}: {path}")
    try:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            frames = source.getnframes()
            sample_width = source.getsampwidth()
            compression = source.getcomptype()
    except (EOFError, wave.Error) as error:
        raise BundleError(f"invalid WAV input {path}: {error}") from error
    if channels != 1 or sample_rate != 48_000 or frames <= 0:
        raise BundleError(f"listening WAV must be non-empty mono 48 kHz: {path}")
    if sample_width not in {2, 3, 4} or compression != "NONE":
        raise BundleError(f"listening WAV must be uncompressed PCM: {path}")
    payload = path.read_bytes()
    return {
        "size_bytes": len(payload),
        "sha256": digest(payload),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "frames": frames,
    }


def canonical(document: dict) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_exclusive(path: Path, payload: bytes, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise BundleError(f"refusing to replace existing {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as destination:
            descriptor = -1
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def html_document(protocol: dict, protocol_sha256: str) -> str:
    rows: list[str] = []
    for index, trial in enumerate(protocol["trials"], start=1):
        trial_id = html.escape(trial["trial_id"])
        prompt = (
            "Which output sounds more natural and preserves the source with fewer artifacts?"
            if trial["question"] == "source-preservation"
            else "Which output best preserves the target speech while reducing unwanted noise?"
        )
        reference = trial["audio"]["reference"]["path"]
        input_audio = trial["audio"]["input"]["path"]
        a_audio = trial["audio"]["a"]["path"]
        b_audio = trial["audio"]["b"]["path"]
        rows.append(
            f'<fieldset data-trial="{trial_id}"><legend>Trial {index}</legend>'
            f"<p>{html.escape(prompt)}</p>"
            f'<label>Reference <audio controls preload="none" src="{html.escape(reference)}"></audio></label>'
            f'<label>Input <audio controls preload="none" src="{html.escape(input_audio)}"></audio></label>'
            f'<label>A <audio controls preload="none" src="{html.escape(a_audio)}"></audio></label>'
            f'<label>B <audio controls preload="none" src="{html.escape(b_audio)}"></audio></label>'
            f'<p><label><input required type="radio" name="{trial_id}" value="a"> A</label> '
            f'<label><input required type="radio" name="{trial_id}" value="b"> B</label> '
            f'<label><input required type="radio" name="{trial_id}" value="tie"> Tie / no preference</label></p>'
            "</fieldset>"
        )
    trial_ids = json.dumps([trial["trial_id"] for trial in protocol["trials"]])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>denoize blinded listening test</title>
<style>body{{font:16px system-ui;max-width:900px;margin:2rem auto;padding:0 1rem}}fieldset{{margin:1.2rem 0;padding:1rem}}fieldset>label{{display:block;margin:.5rem 0}}audio{{display:block;width:min(100%,32rem)}}button{{font-size:1rem;padding:.7rem 1rem}}</style></head>
<body><h1>Blinded paired listening test</h1>
<p>Use the same headphones and volume throughout. Do not inspect file metadata. Listen to every pair before choosing A, B, or Tie.</p>
<form id="response"><p><label>Pseudonymous listener ID <input id="listener" required pattern="[A-Za-z0-9._-]{{3,64}}" maxlength="64"></label></p>
<p><label><input id="consent" type="checkbox" required> I consent to this anonymous response being included in aggregate results.</label></p>
{''.join(rows)}
<button type="submit">Download response JSON</button></form>
<script>
const trialIds={trial_ids};
document.getElementById('response').addEventListener('submit', event=>{{
  event.preventDefault();
  const form=event.currentTarget;
  if(!form.reportValidity()) return;
  const listener=document.getElementById('listener').value;
  const trials=trialIds.map(trial_id=>({{trial_id,preference:new FormData(form).get(trial_id)}}));
  const result={{schema:'denoize-dpdfnet-blind-listener-response-v1',schema_version:1,protocol_sha256:'{protocol_sha256}',listener_id:listener,consent:true,trials}};
  const blob=new Blob([JSON.stringify(result,null,2)+'\\n'],{{type:'application/json'}});
  const link=document.createElement('a'); link.href=URL.createObjectURL(blob);
  link.download='listener-'+listener+'.json'; link.click(); URL.revokeObjectURL(link.href);
}});
</script></body></html>
"""


def prepare(args: argparse.Namespace) -> None:
    matrix, matrix_payload = load_json(args.matrix_result)
    if args.audio_dir.is_symlink() or not args.audio_dir.is_dir():
        raise BundleError(f"audio directory is unavailable: {args.audio_dir}")
    audio_root = args.audio_dir.resolve()
    if args.randomization_key.is_symlink() or not args.randomization_key.is_file():
        raise BundleError(
            f"randomization key is not a regular file: {args.randomization_key}"
        )
    key_path = args.randomization_key.resolve()
    key = key_path.read_bytes()
    if len(key) != 32:
        raise BundleError("randomization key must contain exactly 32 opaque bytes")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise BundleError(
            f"refusing to replace existing public bundle: {args.output_dir}"
        )
    if args.answer_key.exists() or args.answer_key.is_symlink():
        raise BundleError(f"refusing to replace existing answer key: {args.answer_key}")
    output = args.output_dir.resolve()
    answer_key_path = args.answer_key.resolve()

    cases = {case.get("id"): case for case in matrix.get("cases", []) if isinstance(case, dict)}
    selected: list[tuple[dict, Path, str]] = []
    for directory in sorted(audio_root.iterdir(), key=lambda value: value.name):
        if not directory.is_dir() or directory.is_symlink():
            continue
        case = cases.get(directory.name)
        if case is None:
            raise BundleError(f"audio case is absent from matrix: {directory.name}")
        selected.append((case, directory, classify(case)))
    counts = {name: sum(stratum == name for _, _, stratum in selected) for name in EXPECTED_STRATA}
    if counts != EXPECTED_STRATA:
        raise BundleError(f"listening strata must be {EXPECTED_STRATA}, observed {counts}")

    duplicate_sources: set[str] = set()
    for stratum in EXPECTED_STRATA:
        choices = [case for case, _, value in selected if value == stratum]
        choice = min(choices, key=lambda case: keyed(key, f"duplicate:{case['id']}") )
        duplicate_sources.add(choice["id"])

    trial_specs: list[dict] = []
    core_ids: dict[str, str] = {}
    for case, directory, stratum in selected:
        trial_id = keyed(key, f"trial:core:{case['id']}").hex()[:24]
        core_ids[case["id"]] = trial_id
        trial_specs.append({"trial_id": trial_id, "case": case, "directory": directory, "stratum": stratum, "role": "core"})
        if case["id"] in duplicate_sources:
            trial_specs.append({
                "trial_id": keyed(key, f"trial:repeat:{case['id']}").hex()[:24],
                "case": case,
                "directory": directory,
                "stratum": stratum,
                "role": "repeat",
            })
    trial_specs.sort(key=lambda trial: keyed(key, f"order:{trial['trial_id']}"))

    temp_parent = output.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=temp_parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        public_trials: list[dict] = []
        private_trials: list[dict] = []
        for trial in trial_specs:
            trial_id = trial["trial_id"]
            trial_dir = staging / "audio" / trial_id
            trial_dir.mkdir(parents=True)
            swap = bool(keyed(key, f"side:{trial_id}")[0] & 1)
            sides = (
                {"a": ("gtcrn.wav", MODEL_GTCRN), "b": ("dpdfnet2.wav", MODEL_DPDFNET)}
                if swap
                else {"a": ("dpdfnet2.wav", MODEL_DPDFNET), "b": ("gtcrn.wav", MODEL_GTCRN)}
            )
            sources = {
                "reference": trial["directory"] / "clean.wav",
                "input": trial["directory"] / "noisy.wav",
                "a": trial["directory"] / sides["a"][0],
                "b": trial["directory"] / sides["b"][0],
            }
            audio: dict[str, dict] = {}
            for label, source in sources.items():
                record = audio_record(source)
                destination = trial_dir / f"{label}.wav"
                shutil.copyfile(source, destination)
                record["path"] = destination.relative_to(staging).as_posix()
                audio[label] = record
            question = "source-preservation" if trial["stratum"] == "source-preservation" else "noise-reduction"
            public_trials.append({
                "trial_id": trial_id,
                "stratum": trial["stratum"],
                "question": question,
                "audio": audio,
            })
            private_trials.append({
                "trial_id": trial_id,
                "source_case_id": trial["case"]["id"],
                "stratum": trial["stratum"],
                "role": trial["role"],
                "duplicate_of": core_ids[trial["case"]["id"]] if trial["role"] == "repeat" else None,
                "a_model": sides["a"][1],
                "b_model": sides["b"][1],
            })

        bundle_id = keyed(key, f"bundle:{digest(matrix_payload)}").hex()[:24]
        protocol = {
            "schema": "denoize-dpdfnet-blind-protocol-v1",
            "schema_version": 1,
            "bundle_id": bundle_id,
            "source_matrix_sha256": digest(matrix_payload),
            "policy": {
                "core_trials": 12,
                "repeat_trials": 4,
                "minimum_retained_listeners": 20,
                "maximum_listener_duplicate_inconsistency": 0.5,
                "maximum_aggregate_duplicate_inconsistency": 0.25,
                "minimum_overall_dpdfnet_preference": 0.55,
                "minimum_overall_bootstrap_95ci_lower": 0.5,
                "minimum_stratum_dpdfnet_preference": {
                    "recorded-noise": 0.5,
                    "babble": 0.4,
                    "source-preservation": 0.45,
                    "synthetic-noise": 0.5,
                },
                "tie_score": 0.5,
                "listener_cluster_bootstrap_resamples": 20_000,
            },
            "trials": public_trials,
        }
        protocol_payload = canonical(protocol)
        protocol_sha256 = digest(protocol_payload)
        (staging / "protocol.json").write_bytes(protocol_payload)
        (staging / "index.html").write_text(
            html_document(protocol, protocol_sha256), encoding="utf-8", newline="\n"
        )
        answer_key = {
            "schema": "denoize-dpdfnet-blind-answer-key-v1",
            "schema_version": 1,
            "bundle_id": bundle_id,
            "protocol_sha256": protocol_sha256,
            "source_matrix_sha256": digest(matrix_payload),
            "randomization_key_sha256": digest(key),
            "trials": private_trials,
        }
        write_exclusive(answer_key_path, canonical(answer_key), "answer key")
        os.replace(staging, output)
    print(f"public bundle: {output}")
    print(f"private answer key: {answer_key_path}")
    print(f"protocol SHA-256: {protocol_sha256}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--matrix-result", type=Path, required=True)
    result.add_argument("--audio-dir", type=Path, required=True)
    result.add_argument("--randomization-key", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--answer-key", type=Path, required=True)
    return result


def main() -> int:
    try:
        prepare(parser().parse_args())
    except (BundleError, OSError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
