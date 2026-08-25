# Deterministic audio restoration

`denoize restore` is a separate, inspectable repair pipeline for damaged audio.
It does not select a neural model, access the network, change sample rate or
channel geometry, or conceal an uncertain result. The five operations run in a
fixed order: de-clip, de-click, de-hum, WPE de-reverberation, then wind/plosive
repair.

This stage establishes conservative signal-processing baselines before any
universal or generative restorer is eligible for the default path. The design
and acceptance rationale are in the
[restoration research review](restoration-research.md#stage-26--deterministic-restoration).

## Quick start

Apply every operation and atomically publish an audio output, report, and mask:

```sh
denoize restore damaged.wav restored.wav \
  --report restoration-report.json \
  --mask restoration-mask.json \
  --max-memory 2048 \
  --pretty
```

Inspect possible damage without changing or writing audio:

```sh
denoize restore damaged.wav --detect-only \
  --operations declip,declick,dehum \
  --report restoration-report.json \
  --mask restoration-mask.json
```

Run finite multichannel WPE explicitly:

```sh
denoize restore room.wav dry.wav \
  --operations dereverb \
  --wpe-channel-mode multichannel \
  --wpe-delay 3 --wpe-taps 8 --wpe-iterations 3
```

Destinations are no-clobber by default. `--replace` is required to replace a
regular file or symlink. Input, audio output, report, and mask must all resolve
to distinct destinations, and every existing destination is checked before
decoding starts. Each published file uses a private same-directory stage and an
atomic destination commit. The files are independently atomic rather than a
single filesystem-wide transaction.

Restoration currently requires regular-file input and whole-file decode; stdin,
stdout audio, and resumable streaming are not supported. Use `--max-memory` to
make decode and restoration admission fail before the heavy working set is
allocated.

## Operations and safety gates

### De-hum

The de-hum baseline searches 49–51 Hz and 59–61 Hz in 0.1 Hz increments. It
scores the first six harmonics so a missing fundamental can still be tracked,
then fits each present harmonic independently per channel with a robust
reweighted sinusoidal regression. Half-overlapped raised-cosine blocks keep
corrections continuous. The configured attenuation is a ceiling, not a promise
to subtract every stable low-frequency tone.

Blocks below the harmonic-support confidence gate are untouched. This is
important for bass notes and other sustained musical partials. The report
records the mean tracked fundamental, block count, maximum fitted harmonic
count, confidence, attenuation ceiling, changed samples, and energy delta.
The approach follows the drifting harmonic-complex model and time/frequency
selectivity trade-off reviewed in
[Brandt and Bitzer](https://uol.de/f/6/dept/mediphysik/ag/sigproc/download/papers/phd/Brandt_PhD_Thesis.pdf).

### De-click

De-clicking warps the signal, fits a bounded autoregressive predictor, and uses
a median/MAD prediction residual to find isolated impulses. Nearby candidates
are merged only within the configured short interval. Regions that are too
long or lack enough context are reported as rejected and left untouched.
Accepted regions are reconstructed from regularized forward and backward AR
predictions, blended across the gap.

This is intended for clicks, crackle, and very short losses—not long missing
passages. Prediction context is marked as `padded`; only the actual damaged
interval contributes to `detected_samples`. The detector is based on the
warped-linear-prediction family described by
[Esquef, Karjalainen, and Välimäki](https://research.spa.aalto.fi/publications/papers/dsp2002-declick/).

### De-clip

De-clipping first estimates separate positive and negative flat-top levels.
Ambiguous, overlong, or pervasive plateaus are rejected so intentionally hard-
limited or square-like material is not automatically rewritten. Each accepted
short interval is reconstructed by a finite analysis-sparse FFT projection:
reliable context samples are reset on every iteration, while clipped samples
retain the appropriate upper or lower clipping inequality. The FFT and
iteration count are capped.

The method is an inspectable analysis-sparse baseline motivated by the
[A-SPADE analysis and correction](https://arxiv.org/abs/1809.09847). It does
not claim to recover the original waveform uniquely. A warning records regions
that reached the iteration cap; hard reliable-sample and clipping constraints
remain enforced.

### WPE de-reverberation

Finite weighted prediction error (WPE) estimates delayed late reverberation in
the STFT domain, using bounded complex weighted regression, explicit diagonal
regularization, a fixed iteration cap, and a maximum attenuation ceiling. The
output is overlap-added and cropped to the exact input frame count.

`independent` mode solves each channel separately. `multichannel` mode uses all
input channels as predictors for each target and is intentionally capped at
four channels; larger inputs must use independent mode. Ill-conditioned
frequency bins are bypassed and counted rather than emitting unstable samples.
The report exposes frame/hop sizes, delay, taps, effective context, iterations,
solved and rejected bins, and convergence. The algorithm follows
[Nakatani et al.](https://doi.org/10.1109/TASL.2010.2052251).

WPE is global late-tail suppression, so a confident application can mark most
of a file. It is not a room-impulse-response estimator and should be auditioned
carefully on music and strong early reflections.

### Wind and plosives

The deterministic wind/plosive operation addresses only short low-frequency
bursts. It combines low/high-band energy ratio, excess over the recording's
median low-band baseline, temporal modulation, and cross-channel coherence.
Persistent bass is rejected by the burst gate. Accepted intervals receive a
confidence-scaled local high-pass blend with short raised-cosine fades and an
attenuation ceiling.

This is deliberately not general wind-source separation. Long wind, broadband
turbulence, or ambiguous voiced fundamentals are left untouched or reported at
low confidence. Strong recent single-channel work can require an auxiliary
ultrasonic sensor, as in
[DeWinder](https://www.isca-archive.org/interspeech_2024/yuan24_interspeech.html),
so an audio-only baseline must retain this limitation.

## Detect-only and mask semantics

`--detect-only` executes the same bounded detectors but the returned `Audio`
must remain sample-for-sample and bit-for-bit identical to decoded input. It
requires at least one observable result: `--report`, `--mask`, `--json`, or
`--pretty`. An optional audio destination writes that unchanged decoded audio.

The closed
[`denoize-restoration-mask-v1`](../schemas/denoize-restoration-mask-v1.schema.json)
document is channel-ordered run-length encoding. Runs cover every frame exactly
once per channel, without gaps or overlap:

- `untouched`: no selected operation marked the interval;
- `padded`: analysis or interpolation context, not counted as detected damage;
- `detected`: accepted damage that was not rewritten in detect-only mode or by
  a conservative bypass;
- `replaced`: at least one operation changed the decoded sample value.

Each run carries all contributing operations and their maximum confidence.
When masks overlap, `replaced` has highest priority, followed by `detected`,
`padded`, and `untouched`; later context padding cannot hide a real detection.
`detected_samples` counts the unique accepted-damage frames tracked before
state priority is applied; a replaced fade/context frame is not mislabeled as
detected damage. `changed_samples` compares final output PCM with input PCM.
The mask is capped at 4,000,000 runs and fails rather than allocating an
unbounded adversarial RLE document.

## Report contract

The closed
[`denoize-restoration-report-v1`](../schemas/denoize-restoration-report-v1.schema.json)
document contains:

- exact sample rate, channel count, and frame count;
- a domain-separated SHA-256 of canonical decoded PCM and a SHA-256 of the
  compact serialized mask;
- mode, deterministic/bypass state, detected and changed sample counts,
  confidence, energy delta, and deduplicated warnings;
- exactly one typed detail object per selected operation, in canonical
  execution order.

No source or destination pathname is serialized. The PCM digest is still a
content fingerprint and may correlate private audio, so redact it before
sharing when that correlation is sensitive. Strict schemas reject unknown
fields and JSON-nonfinite numbers. The mask's exact-coverage invariant is
enforced by the Rust validator and parity tests because JSON Schema alone
cannot express the per-channel running cursor.

## Rust API

The public API is independent from `DenoiserConfig`:

```rust
use denoize::{restore_audio, RestorationConfig, RestorationMode};

let mut config = RestorationConfig::default();
config.mode = RestorationMode::DetectOnly;
let result = restore_audio(&audio, &config)?;
assert_eq!(result.audio.frames(), audio.frames());
result.mask.validate()?;
# Ok::<(), String>(())
```

All configuration structs reject unknown fields when deserialized and validate
finite numeric ranges, unique operation selection, FFT geometry, iteration
caps, and channel mode before processing. `estimate_restoration_memory_bytes`
provides the conservative admission estimate used by CLI and Desktop. It
accounts separately for the peak WPE planes and the worst allowed RLE mask,
rather than estimating from the usually small compressed mask.

## Determinism and limitations

The pipeline has no randomness, model selection, network access, wall-clock
input, or adaptive state outside the supplied audio and configuration. It uses
a canonical operation order even if operations are requested in another order.
Repeated runs of the same build and target are byte-tested. Cross-architecture
floating-point results should be compared with explicit numerical tolerances;
the report does not assert that different CPU implementations serialize
bit-identical floating-point values.

Every operation preserves decoded sample rate, channel count, channel order,
and frame count. Metadata is copied only when an audio output is requested and
can be disabled with `--no-metadata`. Output encoding can quantize repaired PCM,
so mask/report counts describe the in-memory restoration result before encoder
quantization.

These methods estimate plausible repairs; they do not prove original content,
semantic fidelity, speaker identity, or authorship. Low confidence and warnings
are expected outcomes. A future neural or generative stage must pass separate
undamaged-bypass, semantic, speaker, demographic, unseen-distortion, and human-
listening gates before it can supersede these defaults.
