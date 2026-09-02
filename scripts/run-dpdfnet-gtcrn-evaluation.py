#!/usr/bin/env python3
"""Run the reproducible corpus, deadline, and robustness evaluation for issue #221."""

from __future__ import annotations

import argparse
import array
import hashlib
import html
import json
import math
import os
import pathlib
import random
import statistics
import subprocess
import sys
import urllib.request
import wave
from collections import defaultdict
from typing import Any, Callable, Iterable


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
VCTK_REVISION = "410276703e63dc4ff4564ff75722611871dfe893"
DPDFNET_REVISION = "dd6818d00f50c836fed43a6243ebe49116de5964"
DEEPFILTERNET_REVISION = "d375b2d8309e0935d165700c91da9de862a99c31"
GTCRN_REVISION = "3862c44808dca492ea5a8a145d2dc2a1028d08c8"
SAMPLE_RATE = 48_000
SNR_LEVELS = (-5.0, 0.0, 5.0, 15.0)
NOISE_NAMES = (
    "freesound-2530",
    "freesound-573577",
    "white",
    "pink",
    "hum-60hz",
    "impulsive",
    "three-talker-babble",
)
SPEAKERS = tuple(f"p{number}" for number in range(225, 235))


def vctk_asset(speaker: str, digest: str, size: int) -> tuple[str, str, int]:
    return (
        "https://huggingface.co/datasets/srinathnr/TTS_DATASET/resolve/"
        f"{VCTK_REVISION}/test/wav/{speaker}/{speaker}_001.wav",
        digest,
        size,
    )


ASSETS: dict[str, tuple[str, str, int]] = {
    "dpdfnet2_48khz_hr.onnx": (
        "https://huggingface.co/Ceva-IP/DPDFNet/resolve/"
        f"{DPDFNET_REVISION}/onnx/dpdfnet2_48khz_hr.onnx",
        "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b",
        10_493_337,
    ),
    "dpdfnet8_48khz_hr.onnx": (
        "https://huggingface.co/Ceva-IP/DPDFNet/resolve/"
        f"{DPDFNET_REVISION}/onnx/dpdfnet8_48khz_hr.onnx",
        "7b3afbb260a08fe9af3d16e3bda992971be1e7e951d1dee7c2d235f5c43f5631",
        14_857_107,
    ),
    "gtcrn_simple.onnx": (
        "https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/"
        f"{GTCRN_REVISION}/stream/onnx_models/gtcrn_simple.onnx",
        "b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87",
        535_190,
    ),
    "noise_freesound_2530.wav": (
        "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/"
        f"{DEEPFILTERNET_REVISION}/assets/noise_freesound_2530.wav",
        "cc774d58170fc5b8f143345ef8ff31583feef71f23f447336fdd56c8478d8a8a",
        6_564_092,
    ),
    "noise_freesound_573577.wav": (
        "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/"
        f"{DEEPFILTERNET_REVISION}/assets/noise_freesound_573577.wav",
        "cb367b36e4e9d72d112377dd57bf354e13f0b30f8402c9e841ac47639e773497",
        474_010,
    ),
    "p225_001.wav": vctk_asset(
        "p225", "d30590d385b63336cd4f398e8760c54e54270c06ea99d2848802f9c78143b2d9", 196_990
    ),
    "p226_001.wav": vctk_asset(
        "p226", "8b2a3dab620ab4d8bbc13f6e51a0df346ff062a265e394d77ad6c8dec44e0a28", 438_448
    ),
    "p227_001.wav": vctk_asset(
        "p227", "25e603850e536279e2468e278f1f85660dd2e7973faf3b753ffce548c4b0bcf3", 450_806
    ),
    "p228_001.wav": vctk_asset(
        "p228", "e1f3504c4550e569e7c7d82152196e41565e5fb76b71c9149dc9d1e157cfd457", 258_270
    ),
    "p229_001.wav": vctk_asset(
        "p229", "0f8e78b7d6688880f801e6bcdf0a089139491fd7bc3002c93ce82cf1f22e46f8", 205_054
    ),
    "p230_001.wav": vctk_asset(
        "p230", "0ee0d3913b574eae66527b39db830d60a21e7ba3a4d638809d957517eee9d364", 213_094
    ),
    "p231_001.wav": vctk_asset(
        "p231", "91c92b1f3a785d7fc2669b54e22314004f0b175a83d965ac8672b3c6ab6dda5c", 205_286
    ),
    "p232_001.wav": vctk_asset(
        "p232", "892892f8e8393fe0dd904368bb04bc77df0492472728d06aea502c311a76c32c", 245_928
    ),
    "p233_001.wav": vctk_asset(
        "p233", "1a6a6893b26dced6772002b5be341f1f7d1e28c046e1878afe4e5f290e3cebf9", 364_828
    ),
    "p234_001.wav": vctk_asset(
        "p234", "ec9cb480438974e18b2a2693702595e72f22a531013f23a20ce26ec621450aa8", 311_810
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/denoize-dpdfnet-gtcrn-evaluation"),
    )
    parser.add_argument("--stress-seconds", type=int, default=60)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--skip-visqol", action="store_true")
    parser.add_argument("--skip-listening-bundle", action="store_true")
    return parser.parse_args()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire(root: pathlib.Path, name: str) -> pathlib.Path:
    url, expected_digest, expected_size = ASSETS[name]
    destination = root / "assets" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.stat().st_size == expected_size and sha256(destination) == expected_digest:
            return destination
        raise SystemExit(f"existing asset does not match pinned identity: {destination}")
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        print(f"downloading {name}", flush=True)
        urllib.request.urlretrieve(url, partial)
        actual_size = partial.stat().st_size
        actual_digest = sha256(partial)
        if (actual_size, actual_digest) != (expected_size, expected_digest):
            raise SystemExit(
                f"asset verification failed for {name}: size={actual_size}, sha256={actual_digest}"
            )
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def read_pcm16(path: pathlib.Path) -> tuple[int, list[float]]:
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        if reader.getsampwidth() != 2 or reader.getcomptype() != "NONE":
            raise SystemExit(f"fixture must be uncompressed PCM16: {path}")
        values = array.array("h")
        values.frombytes(reader.readframes(reader.getnframes()))
    if sys.byteorder != "little":
        values.byteswap()
    if channels == 1:
        return sample_rate, [value / 32768.0 for value in values]
    mono = [
        sum(values[index + channel] for channel in range(channels)) / channels / 32768.0
        for index in range(0, len(values), channels)
    ]
    return sample_rate, mono


def quantize(samples: Iterable[float]) -> list[int]:
    return [
        max(-32768, min(32767, round(max(-1.0, min(32767 / 32768, value)) * 32768.0)))
        for value in samples
    ]


def write_pcm16(path: pathlib.Path, samples: Iterable[float]) -> list[float]:
    encoded = quantize(samples)
    values = array.array("h", encoded)
    if sys.byteorder != "little":
        values.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(values.tobytes())
    return [value / 32768.0 for value in encoded]


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


class XorShift64:
    def __init__(self, seed: int) -> None:
        self.state = seed or 0x9E3779B97F4A7C15

    def uniform(self) -> float:
        state = self.state
        state ^= state >> 12
        state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 27
        self.state = state & 0xFFFFFFFFFFFFFFFF
        value = (self.state * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF
        return ((value >> 11) / float(1 << 53)) * 2.0 - 1.0


def loop_noise(source: list[float], frames: int, seed: int) -> list[float]:
    if not source:
        raise SystemExit("noise source is empty")
    offset = seed % len(source)
    return [source[(offset + index) % len(source)] for index in range(frames)]


def white_noise(frames: int, seed: int) -> list[float]:
    generator = XorShift64(seed)
    return [generator.uniform() for _ in range(frames)]


def pink_noise(frames: int, seed: int) -> list[float]:
    generator = XorShift64(seed)
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
    output: list[float] = []
    for _ in range(frames):
        white = generator.uniform()
        b0 = 0.99886 * b0 + white * 0.0555179
        b1 = 0.99332 * b1 + white * 0.0750759
        b2 = 0.96900 * b2 + white * 0.1538520
        b3 = 0.86650 * b3 + white * 0.3104856
        b4 = 0.55000 * b4 + white * 0.5329522
        b5 = -0.7616 * b5 - white * 0.0168980
        pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362
        b6 = white * 0.115926
        output.append(pink)
    return output


def hum_noise(frames: int, seed: int) -> list[float]:
    phase = (seed % 10_000) / 10_000.0 * math.tau
    return [
        (0.75 + 0.25 * math.sin(math.tau * 0.7 * index / SAMPLE_RATE + phase))
        * (
            math.sin(math.tau * 60.0 * index / SAMPLE_RATE + phase)
            + 0.45 * math.sin(math.tau * 120.0 * index / SAMPLE_RATE + phase * 0.3)
            + 0.2 * math.sin(math.tau * 180.0 * index / SAMPLE_RATE + phase * 0.7)
        )
        for index in range(frames)
    ]


def impulsive_noise(frames: int, seed: int) -> list[float]:
    generator = XorShift64(seed)
    output = [0.0] * frames
    position = seed % max(1, SAMPLE_RATE // 20)
    while position < frames:
        polarity = -1.0 if generator.uniform() < 0.0 else 1.0
        width = SAMPLE_RATE // 125
        for index in range(width):
            if position + index >= frames:
                break
            output[position + index] += polarity * math.exp(-index / (SAMPLE_RATE * 0.0007))
        position += SAMPLE_RATE // 10 + int((generator.uniform() + 1.0) * SAMPLE_RATE // 40)
    return output


def babble_noise(
    speaker: str, frames: int, seed: int, clean_audio: dict[str, list[float]]
) -> list[float]:
    candidates = [candidate for candidate in SPEAKERS if candidate != speaker]
    start = seed % len(candidates)
    selected = [candidates[(start + offset * 3) % len(candidates)] for offset in range(3)]
    tracks = [
        loop_noise(clean_audio[candidate], frames, seed ^ stable_seed(candidate))
        for candidate in selected
    ]
    return [sum(samples) / len(tracks) for samples in zip(*tracks)]


def rms(samples: Iterable[float]) -> float:
    values = list(samples)
    return math.sqrt(sum(value * value for value in values) / max(1, len(values)))


def make_pair(clean: list[float], noise: list[float], snr_db: float) -> tuple[list[float], list[float]]:
    clean_rms = rms(clean)
    noise_mean = sum(noise) / len(noise)
    noise = [value - noise_mean for value in noise]
    noise_rms = rms(noise)
    if clean_rms <= 1.0e-12 or noise_rms <= 1.0e-12:
        raise SystemExit("fixture has insufficient clean or noise energy")
    noise_gain = clean_rms / noise_rms * 10.0 ** (-snr_db / 20.0)
    mixed = [speech + noise_gain * interference for speech, interference in zip(clean, noise)]
    peak = max(abs(value) for value in mixed)
    scale = min(1.0, 0.98 / peak) if peak > 0.0 else 1.0
    return [value * scale for value in clean], [value * scale for value in mixed]


def measured_snr(clean: list[float], noisy: list[float]) -> float:
    noise = [mixture - speech for speech, mixture in zip(clean, noisy)]
    return 20.0 * math.log10(rms(clean) / max(rms(noise), 1.0e-20))


def relative(path: pathlib.Path, root: pathlib.Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def build_manifest(root: pathlib.Path, paths: dict[str, pathlib.Path]) -> pathlib.Path:
    clean_audio: dict[str, list[float]] = {}
    for speaker in SPEAKERS:
        sample_rate, samples = read_pcm16(paths[f"{speaker}_001.wav"])
        if sample_rate != SAMPLE_RATE:
            raise SystemExit(f"VCTK fixture {speaker} is {sample_rate} Hz, expected {SAMPLE_RATE}")
        clean_audio[speaker] = samples
    real_noise: dict[str, list[float]] = {}
    for name, asset in (
        ("freesound-2530", "noise_freesound_2530.wav"),
        ("freesound-573577", "noise_freesound_573577.wav"),
    ):
        sample_rate, samples = read_pcm16(paths[asset])
        if sample_rate != SAMPLE_RATE:
            raise SystemExit(f"noise fixture {asset} is {sample_rate} Hz, expected {SAMPLE_RATE}")
        real_noise[name] = samples

    cases: list[dict[str, Any]] = []
    for speaker in SPEAKERS:
        clean = clean_audio[speaker]
        cases.append(
            {
                "id": f"{speaker}__clean",
                "kind": "clean-preservation",
                "speaker": speaker,
                "noise": None,
                "requested_snr_db": None,
                "actual_snr_db": None,
                "clean": relative(paths[f"{speaker}_001.wav"], root),
                "noisy": relative(paths[f"{speaker}_001.wav"], root),
                "sample_rate": SAMPLE_RATE,
                "write_audio": False,
            }
        )
        for noise_name in NOISE_NAMES:
            seed = stable_seed(f"denoize-issue-221:{speaker}:{noise_name}")
            if noise_name in real_noise:
                noise = loop_noise(real_noise[noise_name], len(clean), seed)
            elif noise_name == "white":
                noise = white_noise(len(clean), seed)
            elif noise_name == "pink":
                noise = pink_noise(len(clean), seed)
            elif noise_name == "hum-60hz":
                noise = hum_noise(len(clean), seed)
            elif noise_name == "impulsive":
                noise = impulsive_noise(len(clean), seed)
            elif noise_name == "three-talker-babble":
                noise = babble_noise(speaker, len(clean), seed, clean_audio)
            else:
                raise AssertionError(noise_name)
            for requested_snr in SNR_LEVELS:
                snr_slug = f"m{abs(int(requested_snr)):02d}" if requested_snr < 0 else f"p{int(requested_snr):02d}"
                case_id = f"{speaker}__{noise_name}__snr-{snr_slug}"
                clean_path = root / "fixtures" / case_id / "clean.wav"
                noisy_path = root / "fixtures" / case_id / "noisy.wav"
                scaled_clean, mixed = make_pair(clean, noise, requested_snr)
                decoded_clean = write_pcm16(clean_path, scaled_clean)
                decoded_noisy = write_pcm16(noisy_path, mixed)
                cases.append(
                    {
                        "id": case_id,
                        "kind": "noise-matrix",
                        "speaker": speaker,
                        "noise": noise_name,
                        "requested_snr_db": requested_snr,
                        "actual_snr_db": measured_snr(decoded_clean, decoded_noisy),
                        "clean": relative(clean_path, root),
                        "noisy": relative(noisy_path, root),
                        "sample_rate": SAMPLE_RATE,
                        "write_audio": False,
                    }
                )

    probe = next(
        case
        for case in cases
        if case["id"] == "p226__pink__snr-p00"
    )
    for sample_rate in (8_000, 16_000, 44_100, 48_000, 96_000):
        case = dict(probe)
        case["id"] = f"p226__pink__snr-p00__rate-{sample_rate}"
        case["kind"] = "sample-rate"
        case["sample_rate"] = sample_rate
        cases.append(case)

    identity = {
        "asset_sha256": {name: identity[1] for name, identity in sorted(ASSETS.items())},
        "cases": cases,
        "generator": "denoize-dpdfnet-evaluation-v1",
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema": "denoize-dpdfnet-evaluation-manifest-v1",
        "fixture_fingerprint": fingerprint,
        "cases": cases,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"fixture manifest: {path} ({len(cases)} cases, fingerprint {fingerprint})")
    return path


def run_matrix(
    manifest: pathlib.Path,
    result: pathlib.Path,
    paths: dict[str, pathlib.Path],
    *,
    visqol: bool = False,
    audio_dir: pathlib.Path | None = None,
) -> None:
    features = "dpdfnet,gtcrn,visqol" if visqol else "dpdfnet,gtcrn"
    command = [
        "cargo",
        "run",
        "--locked",
        "--release",
        "--features",
        features,
        "--example",
        "dpdfnet_gtcrn_matrix",
        "--",
        "--manifest",
        str(manifest),
        "--dpdfnet2-model",
        str(paths["dpdfnet2_48khz_hr.onnx"]),
        "--dpdfnet8-model",
        str(paths["dpdfnet8_48khz_hr.onnx"]),
        "--gtcrn-model",
        str(paths["gtcrn_simple.onnx"]),
        "--json",
        str(result),
    ]
    if audio_dir is not None:
        command.extend(["--audio-dir", str(audio_dir)])
    subprocess.run(command, check=True, cwd=REPOSITORY_ROOT)


def subset_manifest(
    root: pathlib.Path,
    manifest: dict[str, Any],
    name: str,
    selected: Callable[[dict[str, Any]], bool],
    *,
    write_audio: bool = False,
) -> pathlib.Path:
    cases = []
    for original in manifest["cases"]:
        if selected(original):
            case = dict(original)
            case["write_audio"] = write_audio
            cases.append(case)
    payload = {
        "schema": manifest["schema"],
        "fixture_fingerprint": manifest["fixture_fingerprint"],
        "cases": cases,
    }
    path = root / f"{name}-manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_stress(root: pathlib.Path, paths: dict[str, pathlib.Path], seconds: int) -> list[pathlib.Path]:
    if not 1 <= seconds <= 3600:
        raise SystemExit("--stress-seconds must be between 1 and 3600")
    output = root / "stress"
    output.mkdir(exist_ok=True)
    model_paths = {
        "dpdfnet2": paths["dpdfnet2_48khz_hr.onnx"],
        "dpdfnet8": paths["dpdfnet8_48khz_hr.onnx"],
        "gtcrn": paths["gtcrn_simple.onnx"],
        "gtcrn-daw": paths["gtcrn_simple.onnx"],
    }
    results = []
    for model, model_path in model_paths.items():
        for parallel in (1, 2, 4):
            result = output / f"{model}-parallel-{parallel}.json"
            subprocess.run(
                [
                    "cargo",
                    "run",
                    "--locked",
                    "--release",
                    "--features",
                    "dpdfnet,gtcrn",
                    "--example",
                    "dpdfnet_gtcrn_stress",
                    "--",
                    "--model",
                    model,
                    "--model-path",
                    str(model_path),
                    "--seconds",
                    str(seconds),
                    "--parallel",
                    str(parallel),
                    "--json",
                    str(result),
                ],
                check=True,
                cwd=REPOSITORY_ROOT,
            )
            results.append(result)
    return results


def nested(item: dict[str, Any], path: str) -> float | None:
    value: Any = item
    for component in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(component)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cluster_bootstrap(
    cases: list[dict[str, Any]], values: list[float], seed: int
) -> tuple[float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for case, value in zip(cases, values):
        grouped[case["speaker"]].append(value)
    clusters = [statistics.fmean(group) for group in grouped.values()]
    generator = random.Random(seed)
    samples = [
        statistics.fmean(generator.choice(clusters) for _ in clusters)
        for _ in range(20_000)
    ]
    return quantile(samples, 0.025), quantile(samples, 0.975)


def paired_summary(
    cases: list[dict[str, Any]],
    left: str,
    right: str,
    metric: str,
    *,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    paired_cases: list[dict[str, Any]] = []
    differences: list[float] = []
    left_values: list[float] = []
    right_values: list[float] = []
    for case in cases:
        left_value = nested(case, f"{left}.{metric}")
        right_value = nested(case, f"{right}.{metric}")
        if left_value is None or right_value is None:
            continue
        difference = right_value - left_value if lower_is_better else left_value - right_value
        paired_cases.append(case)
        differences.append(difference)
        left_values.append(left_value)
        right_values.append(right_value)
    if not differences:
        return {"count": 0}
    low, high = cluster_bootstrap(
        paired_cases,
        differences,
        stable_seed(f"bootstrap:{left}:{right}:{metric}"),
    )
    return {
        "count": len(differences),
        "left_mean": statistics.fmean(left_values),
        "right_mean": statistics.fmean(right_values),
        "paired_difference_mean": statistics.fmean(differences),
        "paired_difference_median": statistics.median(differences),
        "speaker_cluster_bootstrap_95ci": [low, high],
        "left_win_fraction": sum(value > 0.0 for value in differences) / len(differences),
        "ties": sum(value == 0.0 for value in differences),
        "positive_means_left_is_better": True,
    }


PAIR_METRICS = {
    "enhanced_si_sdr_db": ("quality.enhanced.si_sdr_db", False),
    "si_sdr_improvement_db": ("quality.improvement.si_sdr_db", False),
    "enhanced_segmental_snr_db": ("quality.enhanced.segmental_snr_db", False),
    "stoi": ("quality.enhanced.stoi", False),
    "stoi_improvement": ("quality.improvement.stoi", False),
    "visqol": ("quality.enhanced.visqol", False),
    "visqol_improvement": ("quality.improvement.visqol", False),
    "musical_noise": ("quality.enhanced.artifact_scores.musical_noise_score", True),
    "pumping": ("quality.enhanced.artifact_scores.pumping_score", True),
    "transient_loss": ("quality.enhanced.artifact_scores.transient_loss_score", True),
}


def summarize_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    primary = [case for case in matrix["cases"] if case["kind"] == "noise-matrix"]
    clean = [case for case in matrix["cases"] if case["kind"] == "clean-preservation"]
    rates = [case for case in matrix["cases"] if case["kind"] == "sample-rate"]
    comparisons: dict[str, Any] = {}
    for name, (metric, lower_is_better) in PAIR_METRICS.items():
        comparisons[name] = paired_summary(
            primary,
            "dpdfnet2_48khz_hr",
            "gtcrn",
            metric,
            lower_is_better=lower_is_better,
        )
    dpdfnet2_vs_8 = {
        name: paired_summary(
            primary,
            "dpdfnet2_48khz_hr",
            "dpdfnet8_48khz_hr",
            metric,
            lower_is_better=lower_is_better,
        )
        for name, (metric, lower_is_better) in PAIR_METRICS.items()
    }
    strata: dict[str, Any] = {"noise": {}, "snr_db": {}}
    for noise in NOISE_NAMES:
        subset = [case for case in primary if case["noise"] == noise]
        strata["noise"][noise] = paired_summary(
            subset,
            "dpdfnet2_48khz_hr",
            "gtcrn",
            "quality.improvement.si_sdr_db",
        )
    for snr_db in SNR_LEVELS:
        subset = [case for case in primary if case["requested_snr_db"] == snr_db]
        strata["snr_db"][str(snr_db)] = paired_summary(
            subset,
            "dpdfnet2_48khz_hr",
            "gtcrn",
            "quality.improvement.si_sdr_db",
        )
    clean_summary = {
        metric: paired_summary(
            clean,
            "dpdfnet2_48khz_hr",
            "gtcrn",
            path,
            lower_is_better=lower,
        )
        for metric, (path, lower) in PAIR_METRICS.items()
        if metric
        in {
            "enhanced_si_sdr_db",
            "stoi",
            "visqol",
            "musical_noise",
            "pumping",
            "transient_loss",
        }
    }
    return {
        "case_counts": {
            "total": len(matrix["cases"]),
            "noise_matrix": len(primary),
            "clean_preservation": len(clean),
            "sample_rate": len(rates),
            "speakers": len({case["speaker"] for case in primary}),
            "noises": len({case["noise"] for case in primary}),
            "snr_levels": len({case["requested_snr_db"] for case in primary}),
        },
        "dpdfnet2_vs_gtcrn": comparisons,
        "dpdfnet2_vs_dpdfnet8": dpdfnet2_vs_8,
        "strata": strata,
        "clean_preservation": clean_summary,
        "sample_rate_cases": rates,
    }


def summarize_stress(paths: list[pathlib.Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def choose_listening_cases(matrix: dict[str, Any]) -> list[str]:
    primary = [case for case in matrix["cases"] if case["kind"] == "noise-matrix"]
    clean = [case for case in matrix["cases"] if case["kind"] == "clean-preservation"]

    def delta(
        case: dict[str, Any],
        left: str = "dpdfnet2_48khz_hr",
        right: str = "gtcrn",
        metric: str = "quality.improvement.si_sdr_db",
    ) -> float:
        left_value = nested(case, f"{left}.{metric}")
        right_value = nested(case, f"{right}.{metric}")
        return (left_value or 0.0) - (right_value or 0.0)

    def extrema(cases: list[dict[str, Any]], count: int, metric: str) -> list[str]:
        ranked = sorted(cases, key=lambda case: (delta(case, metric=metric), case["id"]))
        candidates = [
            ranked[0],
            min(ranked, key=lambda case: (abs(delta(case, metric=metric)), case["id"])),
            ranked[-1],
            ranked[len(ranked) // 2],
        ]
        selected: list[str] = []
        for case in candidates + ranked:
            if case["id"] not in selected:
                selected.append(case["id"])
            if len(selected) == count:
                return selected
        raise RuntimeError(f"only {len(selected)} unique listening cases for {count} slots")

    recorded = [case for case in primary if str(case["noise"]).startswith("freesound-")]
    babble = [case for case in primary if case["noise"] == "three-talker-babble"]
    synthetic = [
        case
        for case in primary
        if case["noise"] not in {"freesound-2530", "freesound-573577", "three-talker-babble"}
    ]
    selected = [
        *extrema(recorded, 4, "quality.improvement.si_sdr_db"),
        *extrema(babble, 3, "quality.improvement.si_sdr_db"),
        *extrema(clean, 3, "quality.enhanced.si_sdr_db"),
        *extrema(synthetic, 2, "quality.improvement.si_sdr_db"),
    ]
    if len(selected) != 12 or len(set(selected)) != 12:
        raise RuntimeError("formal listening selection must contain 12 unique cases")
    return selected


def write_listening_index(
    root: pathlib.Path, matrix: dict[str, Any], selected_ids: list[str]
) -> None:
    by_id = {case["id"]: case for case in matrix["cases"]}
    rows = []
    for case_id in selected_ids:
        case = by_id[case_id]
        links = " ".join(
            f'<a href="{html.escape(case_id)}/{name}.wav">{name}</a>'
            for name in ("clean", "noisy", "dpdfnet2", "dpdfnet8", "gtcrn")
        )
        dp2 = nested(case, "dpdfnet2_48khz_hr.quality.improvement.si_sdr_db")
        dp8 = nested(case, "dpdfnet8_48khz_hr.quality.improvement.si_sdr_db")
        gt = nested(case, "gtcrn.quality.improvement.si_sdr_db")
        rows.append(
            "<tr>"
            f"<td>{html.escape(case_id)}</td><td>{dp2:.3f}</td><td>{dp8:.3f}</td>"
            f"<td>{gt:.3f}</td><td>{links}</td></tr>"
        )
    document = (
        "<!doctype html><meta charset=utf-8><title>Issue 221 listening bundle</title>"
        "<style>body{font-family:system-ui;max-width:1200px;margin:3rem auto}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.45rem}"
        "td:nth-child(n+2):nth-child(-n+4){text-align:right}</style>"
        "<h1>DPDFNet / GTCRN blinded-listening candidates</h1>"
        "<p>SI-SDR improvements are navigation aids, not listening votes. "
        "Use prepare-dpdfnet-blind-listening.py to randomize and relabel these "
        "files before the formal paired-preference test.</p>"
        "<table><thead><tr><th>case</th><th>DPDFNet-2 ΔSI-SDR</th>"
        "<th>DPDFNet-8 ΔSI-SDR</th><th>GTCRN ΔSI-SDR</th><th>audio</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    (root / "index.html").write_text(document, encoding="utf-8")


def markdown_summary(summary: dict[str, Any]) -> str:
    quality = summary["quality"]["dpdfnet2_vs_gtcrn"]
    lines = [
        "# DPDFNet issue #221 exhaustive evaluation summary",
        "",
        "Positive paired differences mean DPDFNet-2 is better. Confidence intervals use a "
        "speaker-cluster bootstrap (20,000 resamples).",
        "",
        "| Metric | DPDFNet-2 mean | GTCRN mean | Paired delta | 95% CI | DPDFNet-2 wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "si_sdr_improvement_db",
        "stoi_improvement",
        "visqol_improvement",
        "musical_noise",
        "pumping",
        "transient_loss",
    ):
        item = quality[name]
        if item.get("count", 0) == 0:
            continue
        low, high = item["speaker_cluster_bootstrap_95ci"]
        lines.append(
            f"| {name} | {item['left_mean']:.6f} | {item['right_mean']:.6f} | "
            f"{item['paired_difference_mean']:+.6f} | [{low:+.6f}, {high:+.6f}] | "
            f"{item['left_win_fraction']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Sustained timing",
            "",
            "| Model/path | Streams | Mean | p99 | p99.9 | Max | Over budget | Throughput | Peak RSS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary.get("stress", []):
        timing = item["timing"]
        memory = item["memory"]
        lines.append(
            f"| {item['model']} | {item['parallel_streams']} | {timing['mean_ms']:.3f} ms | "
            f"{timing['p99_ms']:.3f} ms | {timing['p99_9_ms']:.3f} ms | "
            f"{timing['maximum_ms']:.3f} ms | {timing['calls_over_budget_fraction']:.3%} | "
            f"{timing['aggregate_realtime_throughput_x']:.2f}x | "
            f"{(memory['peak_rss_bytes'] or 0) / 2**20:.1f} MiB |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {name: acquire(root, name) for name in ASSETS}
    manifest_path = build_manifest(root, paths)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.prepare_only:
        return
    matrix_result_path = root / "matrix-result.json"
    if not args.skip_quality:
        run_matrix(manifest_path, matrix_result_path, paths)
    if not matrix_result_path.is_file():
        raise SystemExit(f"quality result is absent: {matrix_result_path}")
    matrix = json.loads(matrix_result_path.read_text(encoding="utf-8"))

    stress_paths: list[pathlib.Path]
    if args.skip_stress:
        stress_paths = sorted((root / "stress").glob("*.json"))
    else:
        stress_paths = run_stress(root, paths, args.stress_seconds)

    visqol_result_path = root / "visqol-result.json"
    if not args.skip_visqol:
        visqol_manifest = subset_manifest(
            root,
            manifest,
            "visqol",
            lambda case: case["kind"] == "clean-preservation"
            or (
                case["kind"] == "noise-matrix"
                and case["noise"]
                in {"freesound-2530", "freesound-573577", "three-talker-babble"}
                and case["requested_snr_db"] in {-5.0, 5.0}
            ),
        )
        run_matrix(visqol_manifest, visqol_result_path, paths, visqol=True)

    if not args.skip_listening_bundle:
        selected_ids = choose_listening_cases(matrix)
        listening_manifest = subset_manifest(
            root,
            manifest,
            "listening",
            lambda case: case["id"] in set(selected_ids),
            write_audio=True,
        )
        listening_dir = root / "listening"
        listening_dir.mkdir(parents=True, exist_ok=True)
        run_matrix(
            listening_manifest,
            root / "listening-result.json",
            paths,
            audio_dir=listening_dir,
        )
        write_listening_index(listening_dir, matrix, selected_ids)

    summary: dict[str, Any] = {
        "schema": "denoize-dpdfnet-gtcrn-evaluation-summary-v1",
        "source_commit": os.environ.get("DENOIZE_EVIDENCE_SOURCE_COMMIT"),
        "fixture_fingerprint": manifest["fixture_fingerprint"],
        "matrix_result_sha256": sha256(matrix_result_path),
        "models": {
            "dpdfnet2-48khz-hr": ASSETS["dpdfnet2_48khz_hr.onnx"][1],
            "dpdfnet8-48khz-hr": ASSETS["dpdfnet8_48khz_hr.onnx"][1],
            "gtcrn-dns3": ASSETS["gtcrn_simple.onnx"][1],
        },
        "quality": summarize_matrix(matrix),
        "stress": summarize_stress(stress_paths),
        "visqol": None,
    }
    if visqol_result_path.is_file():
        summary["visqol"] = summarize_matrix(
            json.loads(visqol_result_path.read_text(encoding="utf-8"))
        )
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = root / "summary.md"
    markdown_path.write_text(markdown_summary(summary), encoding="utf-8")
    print(f"summary JSON: {summary_path}")
    print(f"summary Markdown: {markdown_path}")
    if not args.skip_listening_bundle:
        print(f"listening bundle: {root / 'listening' / 'index.html'}")


if __name__ == "__main__":
    main()
