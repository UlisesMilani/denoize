# Explicit-geometry microphone-array enhancement

Stage 31 exposes an offline, deterministic microphone-array baseline through
`denoize array` and `MicrophoneArraySession`. It accepts only a two-to-four
channel recording whose channel identities, coordinates, calibration, and
reference microphone are declared in a closed JSON configuration. It never
guesses that ordinary stereo or surround is an array.

No neural spatial checkpoint is bundled or promoted. SpatialNet,
OnlineSpatialNet, DFSNet, DeFTAN-AA, and coordinate-aware neural models remain
research comparisons until immutable redistributable weights, complete training
data terms, numerical vectors, and real-device evidence clear the package-v2
gate.

## Command line

```text
denoize array meeting-array.wav enhanced.wav \
  --array-config array-config.json \
  --promotion-evidence array-evidence.json \
  --promotion-evidence-key evaluator-public-key.json \
  --report array-report.json \
  --max-memory 1024 --pretty
```

`array-enhance` is an alias for `array`. Configuration and evidence files are
bounded regular files. The CLI authenticates the evidence and binds the exact
configuration before opening the audio source. Input, configuration, evidence,
key, output, and report must be distinct publication paths. Output and report
publication are atomic; existing destinations require `--replace`.

Verify evidence independently with:

```text
denoize array evidence verify \
  array-evidence.json evaluator-public-key.json --pretty
```

## Closed geometry

The coordinate system is fixed to meters and a right-handed
x-forward/y-left/z-up frame. Every microphone has a bounded ASCII ID, a unique
position separated from every other position by at least 0.1 mm, signed sample
skew, gain mismatch, and phase mismatch. The reference ID must name exactly one
microphone. The decoded file must have the same channel count, sample rate,
frame count on every channel, finite normalized PCM, and at least one frame.

```json
{
  "sample_rate": 48000,
  "geometry": {
    "input_semantics": "microphone-array",
    "coordinate_unit": "meters",
    "handedness": "right-handed-x-forward-y-left-z-up",
    "reference_microphone_id": "mic-0",
    "microphones": [
      {
        "id": "mic-0",
        "x": -0.04,
        "y": 0.0,
        "z": 0.0,
        "sample_skew": 0,
        "gain_mismatch_db": 0.0,
        "phase_mismatch_degrees": 0.0
      },
      {
        "id": "mic-1",
        "x": 0.04,
        "y": 0.0,
        "z": 0.0,
        "sample_skew": 0,
        "gain_mismatch_db": 0.0,
        "phase_mismatch_degrees": 0.0
      }
    ]
  },
  "frame_size": 512,
  "hop_size": 128,
  "wpe_prediction_delay_frames": 3,
  "wpe_prediction_taps": 8,
  "wpe_iterations": 3,
  "diagonal_loading": 0.001,
  "maximum_condition_number": 1000000.0,
  "covariance_smoothing": 0.05,
  "inactive_channel_rms": 0.0000001,
  "maximum_peak": 1.0
}
```

The configuration digest canonicalizes microphone entries by ID. Permuting a
channel and its matching geometry entry therefore preserves the digest and must
produce numerically equivalent output. Permuting audio without the matching
geometry changes semantics and is not treated as equivalent.

## Processing and fallback

The inspectable baseline applies the declared sample, gain, and phase
calibration, then multichannel weighted prediction error dereverberation. A
bounded speech/noise mask estimates spatial covariance for each STFT bin.
Noise covariance receives diagonal loading; a principal speech vector defines
the steering direction; a pivot and condition-number bound gate the complex
MVDR solve.

Each singular or ill-conditioned frequency bin uses the declared reference
channel. An array with fewer than two active microphones also uses that
reference path. An inactive reference microphone fails the complete operation
before publication. Output is mono, has exactly the input frame count, contains
no non-finite samples, and is bounded by `maximum_peak`.

This fallback is intentionally local and deterministic. It does not invoke a
neural model, choose another semantic target, or reinterpret program stereo.

## Promotion evidence

[`denoize-microphone-array-promotion-evidence-v1`](../schemas/denoize-microphone-array-promotion-evidence-v1.schema.json)
is an Ed25519-signed document. The payload binds the exact implementation
source, canonical configuration, corpus manifest, objective evaluation, and
listening result. Its sorted matrix contains exactly these strata:

- bad channel, channel permutation, and clock skew;
- diffuse and directional noise;
- gain/phase mismatch and moving source;
- ordinary program stereo;
- real meetings and simulated RIRs;
- two-microphone and unseen-geometry arrays.

Every stratum requires at least ten cases, non-negative SI-SDR improvement, no
more than 0.02 absolute WER regression, no more than 20 degrees DOA error,
reference coloration within 1.5 dB, target leakage at or below -3 dB, and zero
non-finite samples. Global gates require at least 100 real-meeting, unseen-
geometry, and permutation cases; at least 10,000 paced blocks; worst-case RTF
at or below 0.5; zero callback allocations, locks, waits, or deadline misses;
and at least 20 listeners with preference at or above 0.5.

The current native path is offline. Callback fields reserve and close the
future streaming promotion boundary; release evidence must not claim a
streaming neural implementation on the strength of offline tests.

## Report and privacy

[`denoize-microphone-array-report-v1`](../schemas/denoize-microphone-array-report-v1.schema.json)
records configuration/evidence/input/output digests, canonical IDs, active and
inactive channel counts, STFT geometry and latency, solved/fallback bin counts,
the maximum observed condition estimate, clipping, and exact duration. It
records no filesystem paths (`paths_recorded` is always zero).

The report validates that every frequency bin has one decision, every input
channel is classified active or inactive, the reference remains active, output
is finite mono PCM of exact duration, and the evidence/configuration binding is
cryptographically identifiable. The implementation performs no transcription
and retains no audio outside the caller-selected output.

## Rust API

Load `MicrophoneArrayConfig` and
`SignedMicrophoneArrayPromotionEvidence`, load a trusted `ReceiptPublicKey`,
then call `MicrophoneArraySession::prepare`. Preparation authenticates and
binds all control data. `MicrophoneArraySession::enhance` accepts an in-memory
`Audio` only after that boundary and returns `MicrophoneArrayResult`.

Use `estimate_microphone_array_memory_bytes` before accepting a render. The CLI
includes decoded PCM, WPE/STFT/covariance state, output, input-session state,
and retained metadata in `--max-memory` enforcement.

## Known limitations

- The released baseline is offline WPE plus mask-estimated MVDR, not a jointly
  optimized convolutional or neural beamformer.
- Only two through four synchronized channels are accepted.
- Calibration is fixed for the render; continuously varying independent device
  clocks are not tracked.
- Moving-source and low-latency streaming neural claims remain unpromoted.
- Ill-conditioned bins preserve the declared reference channel rather than
  inventing an alternative target.
