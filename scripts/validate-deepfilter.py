#!/usr/bin/env python3
"""Run DeepFilterNet's pinned end-to-end real-speech quality regression."""

from __future__ import annotations

import argparse
import hashlib
import math
import pathlib
import random
import struct
import subprocess
import tempfile
import urllib.request
import wave


FIXTURE_URL = (
    "https://raw.githubusercontent.com/espnet/espnet/"
    "443028662106472c60fe8bd892cb277e5b488651/test_utils/st_test.wav"
)
FIXTURE_SHA256 = "55441b4929df3806be67cb9dfca28a8554c2f7fc111b742baff3fe90a490ae1c"
NOISE_SEED = 425
NOISE_AMPLITUDE = 0.05
MINIMUM_SI_SNR_IMPROVEMENT_DB = 0.5
# v0.33.0 on this fixture after compensating its known 1,440-sample model delay.
REFERENCE_BASELINE_SI_SNR_DB = 3.327
MAXIMUM_BASELINE_REGRESSION_DB = 0.1
BASELINE_MODEL_RATE = 48_000
BASELINE_DELAY_SAMPLES = 1_440


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--denoize", required=True, type=pathlib.Path)
    parser.add_argument("--baseline", type=pathlib.Path)
    return parser.parse_args()


def read_wav(path: pathlib.Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as reader:
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
            raise SystemExit("DeepFilterNet fixture must be 16-bit mono PCM")
        sample_rate = reader.getframerate()
        count = reader.getnframes()
        samples = struct.unpack(f"<{count}h", reader.readframes(count))
    return [sample / 32768.0 for sample in samples], sample_rate


def write_wav(path: pathlib.Path, samples: list[float], sample_rate: int) -> None:
    quantized = [
        max(-32768, min(32767, round(sample * 32768.0))) for sample in samples
    ]
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(struct.pack(f"<{len(quantized)}h", *quantized))


def si_snr(reference: list[float], estimate: list[float]) -> float:
    if len(reference) != len(estimate):
        raise SystemExit("enhanced fixture duration changed")
    reference_mean = sum(reference) / len(reference)
    estimate_mean = sum(estimate) / len(estimate)
    reference = [sample - reference_mean for sample in reference]
    estimate = [sample - estimate_mean for sample in estimate]
    energy = sum(sample * sample for sample in reference)
    scale = sum(a * b for a, b in zip(estimate, reference)) / energy
    target = [scale * sample for sample in reference]
    residual = [sample - projected for sample, projected in zip(estimate, target)]
    return 10.0 * math.log10(
        (sum(sample * sample for sample in target) + 1e-12)
        / (sum(sample * sample for sample in residual) + 1e-12)
    )


def run_backend(
    denoize: pathlib.Path,
    noisy_path: pathlib.Path,
    output_path: pathlib.Path,
    expected_rate: int,
) -> list[float]:
    subprocess.run(
        [
            str(denoize),
            str(noisy_path),
            str(output_path),
            "--backend",
            "deepfilter",
        ],
        check=True,
    )
    enhanced, sample_rate = read_wav(output_path)
    if sample_rate != expected_rate:
        raise SystemExit("enhanced fixture sample rate changed")
    if not all(math.isfinite(sample) for sample in enhanced):
        raise SystemExit("enhanced fixture contains a non-finite sample")
    return enhanced


def main() -> None:
    args = parse_args()
    denoize = args.denoize.resolve()
    baseline = args.baseline.resolve() if args.baseline else None
    for label, executable in (("denoize", denoize), ("baseline", baseline)):
        if executable is not None and not executable.is_file():
            raise SystemExit(f"{label} executable not found: {executable}")

    with tempfile.TemporaryDirectory(prefix="denoize-deepfilter-") as directory:
        root = pathlib.Path(directory)
        clean_path = root / "clean.wav"
        noisy_path = root / "noisy.wav"
        enhanced_path = root / "enhanced.wav"
        with urllib.request.urlopen(FIXTURE_URL, timeout=30) as response:
            clean_path.write_bytes(response.read())
        digest = hashlib.sha256(clean_path.read_bytes()).hexdigest()
        if digest != FIXTURE_SHA256:
            raise SystemExit(f"speech fixture sha256 mismatch: {digest}")

        clean, sample_rate = read_wav(clean_path)
        generator = random.Random(NOISE_SEED)
        noisy = [
            max(
                -1.0,
                min(
                    32767.0 / 32768.0,
                    sample + generator.gauss(0.0, NOISE_AMPLITUDE),
                ),
            )
            for sample in clean
        ]
        write_wav(noisy_path, noisy, sample_rate)
        actual_noisy, _ = read_wav(noisy_path)
        noisy_score = si_snr(clean, actual_noisy)

        enhanced = run_backend(denoize, noisy_path, enhanced_path, sample_rate)
        enhanced_score = si_snr(clean, enhanced)
        improvement = enhanced_score - noisy_score
        print(f"noisy SI-SNR: {noisy_score:.3f} dB")
        print(f"enhanced SI-SNR: {enhanced_score:.3f} dB")
        print(f"improvement: {improvement:.3f} dB")
        if improvement < MINIMUM_SI_SNR_IMPROVEMENT_DB:
            raise SystemExit(
                "DeepFilterNet speech quality regression: "
                f"expected >= {MINIMUM_SI_SNR_IMPROVEMENT_DB:.1f} dB improvement"
            )
        delay = round(BASELINE_DELAY_SAMPLES * sample_rate / BASELINE_MODEL_RATE)
        if delay <= 0 or delay >= len(clean):
            raise SystemExit("invalid DeepFilterNet baseline delay")
        reference_score = si_snr(clean[:-delay], enhanced[:-delay])
        reference_regression = REFERENCE_BASELINE_SI_SNR_DB - reference_score
        print(f"reference comparison SI-SNR: {reference_score:.3f} dB")
        print(f"reference baseline regression: {reference_regression:.3f} dB")
        if reference_regression > MAXIMUM_BASELINE_REGRESSION_DB:
            raise SystemExit(
                "DeepFilterNet reference baseline regression: "
                f"expected <= {MAXIMUM_BASELINE_REGRESSION_DB:.1f} dB"
            )

        if baseline is not None:
            baseline_path = root / "baseline.wav"
            baseline_output = run_backend(
                baseline, noisy_path, baseline_path, sample_rate
            )
            comparison_reference = clean[:-delay]
            comparison_enhanced = enhanced[:-delay]
            comparison_baseline = baseline_output[delay:]
            current_score = si_snr(comparison_reference, comparison_enhanced)
            baseline_score = si_snr(comparison_reference, comparison_baseline)
            regression = baseline_score - current_score
            print(f"baseline delay compensation: {delay} samples")
            print(f"comparison SI-SNR: {current_score:.3f} dB")
            print(f"baseline SI-SNR: {baseline_score:.3f} dB")
            print(f"baseline regression: {regression:.3f} dB")
            if regression > MAXIMUM_BASELINE_REGRESSION_DB:
                raise SystemExit(
                    "DeepFilterNet baseline regression: "
                    f"expected <= {MAXIMUM_BASELINE_REGRESSION_DB:.1f} dB"
                )


if __name__ == "__main__":
    main()
