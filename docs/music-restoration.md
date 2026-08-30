# Bounded music and general-audio restoration

Stage 35 adds one deliberately narrow operation: render a conservative repair
candidate for a complete mono or stereo music mixture affected by codec loss or
missing bandwidth. It does not estimate dry stems, reverse artistic mastering,
or claim that missing phase or spectrum was recovered as ground truth.

## Why this is separate from speech restoration

Speech metrics and safety gates do not adequately protect instrumental timbre,
stereo image, percussion transients, genre balance, or mastering intent. The
[Music Source Restoration task](https://arxiv.org/abs/2505.21827) formalizes a
broader degradation family, while its
[2026 challenge summary](https://arxiv.org/abs/2601.04343) reports strong
instrument-dependent differences under Multi-Mel-SNR, Zimtohrli, FAD-CLAP,
and professional listening. The
[MSRBench analysis](https://arxiv.org/abs/2510.10995) also shows why ordinary
scale-invariant waveform scores can mis-rank phase-affected music.

denoize therefore starts with a smaller, mixture-preserving product boundary:

- `codec-repair` may propose a correction for audible codec damage;
- `bandwidth-extension` may propose missing-band content but labels the result
  as a candidate, never recovered truth;
- clean and uncertain regions receive exactly zero in-memory correction;
- dry-stem separation, prompt-controlled mastering, EQ, compression, widening,
  and text-directed generation are different future operations.

## Closed graph contract

The operator supplies a signed runtime package v2 and a trusted Minisign key.
The package must be finite and stateless, declare one or two independent
program channels, and expose exactly:

- one `float32` audio input `[batch=1, channel=C, sample=W]`;
- one same-shape candidate-audio output `[1,C,W]`;
- one `repair_state` output `[1,F,3]`, ordered `bypass`, `uncertain`, `apply`.

Every name, rank, fixed dimension, channel role, window, hop, state clock,
sample rate, graph byte, numerical vector, precision profile, resource ceiling,
accelerator allowlist, source/checkpoint identity, license, and training dataset
is authenticated before the program audio is opened. The state clock must
divide the window and align exactly with the package hop.

No model or checkpoint ships with denoize. Source-code licensing alone is not
enough: the exact checkpoint, every inherited model, and each training dataset
must be represented in the package provenance and the separately signed
promotion evidence. The detailed paper and artifact audit is in
[the restoration research review](restoration-research.md#stage-35--music-and-general-audio-restoration).

## Conservative rendering

Overlapping model windows are averaged on the authenticated clock. A frame is
applied only when `apply` is the unique winning class above the configured
threshold and the run lasts for the configured minimum number of consecutive
frames. A confident `bypass` frame and every ambiguous frame receive zero
correction. Short apply runs become `uncertain` rather than being expanded.

The correction is limited both before and after resampling. It is added to the
original source-rate mixture, so sample rate, channel count, and frame count are
unchanged. Publication fails before any candidate appears when a sample is
non-finite, the candidate or correction exceeds its peak bound, or stereo
correlation or Mid/Side energy ratio moves farther than configured.

The Rust result contains the exact `f64` correction used to form its in-memory
candidate and verifies `input + correction = output` to `1e-12`. The CLI writes
that residual as uncompressed float32 WAV for broad tool compatibility; the
report's PCM identities describe the pre-encoding in-memory values, not a hash
of the WAV container bytes. Container-level identity should be recorded by an
execution receipt or ordinary SHA-256 when it is needed.

## CLI

```bash
denoize music-restore program.wav candidate.wav \
  --correction correction.wav \
  --report music-restoration-report.json \
  --task codec-repair \
  --model-package music-restoration.dmp \
  --model-package-key operator-model.pub \
  --promotion-evidence music-restoration-evidence.json \
  --promotion-evidence-key evaluator.pub.json \
  --max-memory 2048 --pretty
```

Candidate, correction, and report paths are mandatory and must be distinct
from every input, package, key, and evidence file. Both audio artifacts must be
WAV. Every encode and the report are staged before publication; the candidate
is committed last, so an interrupted multi-file commit cannot leave an
unaudited candidate without a previously published residual and report.
No-clobber is the default; `--replace` is explicit.

The first release is deliberately full-buffered. Its estimator charges decoded
source PCM, both resampler plans, model-rate windows and accumulators, candidate,
correction, decisions, and the authenticated model working set. Processing has
an internal 2-GiB ceiling even without `--max-memory`; the CLI option can set a
lower ceiling. The six-hour duration bound is therefore only an input bound,
not a promise that every high-rate geometry fits in memory.

Evidence can be verified without opening program audio:

```bash
denoize music-restore evidence verify \
  music-restoration-evidence.json evaluator.pub.json --pretty
```

## Promotion evidence

`denoize-music-restoration-promotion-evidence-v1` binds the task and exact
package, source, checkpoint, runtime configuration, artifact BOM, training-data
license manifest, evaluation-corpus and license manifests, objective result,
listening result, and Ed25519 evaluator. Its 12 required sorted strata are:

1. `aac-64k`
2. `clean-bypass`
3. `genre-unseen`
4. `long-form`
5. `mono`
6. `mp3-64k`
7. `neural-codec`
8. `percussion-transients`
9. `phase-critical`
10. `stereo-image`
11. `unseen-codec`
12. `wideband-reference`

Every stratum requires at least ten paired cases and passes only with:

- Multi-Mel-SNR improvement in `0..=240 dB`;
- Zimtohrli regression at most `0.01` and FAD-CLAP regression at most `0.02`;
- low-band SNR at least `40 dB`;
- transient-loss rate and stereo-correlation error at most `0.02`;
- phase error at most `0.20` radians;
- zero duration mismatch, clipped samples, and non-finite samples.

Global gates require at least 1,000 paired clips, 50 full-length tracks, eight
instrument classes, eight genres, 100 clean-bypass examples, 100 mono and 100
stereo examples, and 20 listeners with preference at least `0.50`. Restricted
artifacts redistributed by the evaluated package must be exactly zero. These
are minimum hard gates, not a universal quality claim.

## Report boundary

`denoize-music-restoration-report-v1` records no paths. It contains the exact
task/configuration, full package/source/checkpoint/training-dataset BOM and
licenses, evidence identities and coverage, source/model/output geometry and
clocks, apply/uncertain regions, decision counts, input/output/correction PCM
digests, correction and peak maxima, stereo metrics, and the fixed declarations
that the render is deterministic, network-free, candidate-only, and produced
neither dry stems nor creative mastering.

## Files and API

- [promotion evidence schema](../schemas/denoize-music-restoration-promotion-evidence-v1.schema.json)
- [report schema](../schemas/denoize-music-restoration-report-v1.schema.json)
- Rust API: `MusicRestorationSession`, `MusicRestorationConfig`,
  `MusicRestorationResult`, `MusicRestorationReport`, and
  `SignedMusicRestorationPromotionEvidence`

Dry-stem estimation remains opt-in research work. It needs a separately named
operation, residual/conservation semantics, instrument-specific and phase-aware
evaluation, and a complete redistributable model/data chain; it will not be
silently introduced behind either Stage 35 task.
