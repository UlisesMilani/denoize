# Neural DAW plug-in

`denoize Neural` is the model-backed CLAP effect introduced in v0.76.0. It is
published in the same `denoize.clap` binary as the existing fixed-memory DSP
effect, but uses the independent stable ID
`org.penguin425.denoize.neural`. A project therefore cannot silently replace
one processor with the other.

The first model is the pinned causal `gtcrn-dns3` graph. The plug-in does not
embed or download a model from a DAW process. Install and verify it before the
host activates the effect:

```sh
denoize models install gtcrn
denoize plugin neural info --sample-rate 48000 --pretty
denoize plugin neural latency --sample-rate 48000 --pretty
```

The expected graph SHA-256 is
`b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87`.
Activation fails closed if the managed artifact or its authenticated install
provenance is absent, redirected, replaced, or inconsistent with this build.
The model is verified and the tract graph is prepared on the permanent neural
worker before activation returns.

## Scheduling contract

At activation the plug-in fixes the finite host sample rate, mono/stereo layout,
10 ms scheduler block, 16-block input and result queues, 40 preallocated audio
blocks, model profile, and reported latency. The latency is:

```text
chunk_frames   = ceil(sample_rate * 0.010)
latency_frames = 24 * chunk_frames
```

CLAP represents the sample rate as a floating-point value. The scheduler uses
that exact host value for frame geometry, including unusual fractional rates;
the integer-rate audio backend receives the nearest rate only for its internal
resampling ratio. Ordinary DAW rates are therefore exact, while validator and
specialized-host rates remain ABI-compatible and keep a correct reported delay.

The host-facing plug-in contract accepts finite rates through 1,234,568 Hz so
the official VST3 3.8.1 validator's 1,234,567.8 Hz boundary is exercised
without a format-specific clamp. This does not widen denoize's file decoding,
encoding, or offline-restoration ceiling of 768 kHz.

That is 10,584 frames at 44.1 kHz, 11,520 at 48 kHz, and 23,040 at
96 kHz—240 ms at those rates. The policy name is
`fixed-24x10ms-worker-v1`. The fixed budget deliberately includes model and
resampler startup plus scheduling headroom; it is not a claim that every
machine can finish inference before every deadline. The CLAP latency extension
reports this value only after activation, as required by the stabilized CLAP
contract.

The release-profile reference gate measured RTF 0.567 over 100 consecutive
blocks. The first resampler/WOLA-aligned output requires eleven input blocks,
so the former 120 ms prototype left only one scheduler quantum of startup
headroom. The 240 ms contract is intentional; a machine that cannot sustain
RTF below one remains safe but will use the selected overload fallback.

The host audio thread performs only these bounded operations:

1. copy finite input samples into the current preallocated block;
2. push or pop blocks through bounded lock-free queues;
3. read the result for the exact delayed input-frame identity;
4. apply sample-accurate bypass, mix, output gain, and fallback selection.

It performs no allocation, lock acquisition, model inference, worker wait,
filesystem/network I/O, logging, or host thread-pool request. Model inference,
resampling, recurrent state, output validation, and any temporary allocation
stay on one named worker. The official CLAP thread-pool extension is not used:
its own specification warns that synchronization may violate hard real-time
rules and `request_exec` waits for completion.

Each block carries a generation and absolute input-frame start. A host reset or
dropped input block cold-resets recurrent and resampler state. Results from an
older generation, results that arrive after their exact deadline, non-finite
output, and output above the fixed safety peak are never played. This prevents
stale audio from crossing transport resets or plug-in sessions.

## Overload behavior

The user selects one closed fallback:

- `delayed-dry` (default) preserves the declared delay and original samples;
- `last-safe-gain` applies the last bounded, smoothed per-channel gain to the
  delayed dry signal;
- `silence` emits zero only when explicitly selected.

The callback never waits for a late result and never changes the reported
latency. Bypass is also delayed, so host plug-in-delay compensation and A/B
switching stay aligned. Neural output that arrives after its block deadline is
discarded rather than leaking into a later time position.

## Ports, automation, and state

Both mono and stereo configurations expose one main input/output pair and one
independent reference input with the same channel count. The reference port is
reserved and ignored in v0.76.0; it is the stable routing foundation for
target-speaker and acoustic-echo stages, not an implied AEC implementation.
Ordinary stereo uses one linked mid estimate and applies the correction equally
to left and right, preserving the side signal.

The stable parameter IDs are bypass, mix, output gain, and overload fallback.
CLAP events are consumed in sample-offset batches. Host snapshots and portable
files use the same closed
[`denoize-neural-daw-session-v1`](../schemas/denoize-neural-daw-session-v1.schema.json)
document. It binds plug-in ID, exact model ID and digest, latency policy, port
configuration, and every parameter; unknown fields and future versions fail.
Standalone creation is no-clobber by default:

```sh
denoize plugin neural session create neural.json \
  --stereo --mix 0.8 --fallback delayed-dry --pretty
denoize plugin neural session validate neural.json --json
```

Files are capped at 64 KiB, must be regular non-symlinks, and are published
atomically. Neither a model path nor voice/audio data is serialized.

## Evidence and limitations

GTCRN was selected because its official ICASSP 2024 implementation is causal,
publishes a streaming ONNX graph, and reports an ultra-light 48.2K-parameter,
33 MMAC/s configuration and 0.07 real-time factor on its reference CPU
([official repository](https://github.com/Xiaobin-Rong/gtcrn)). Those upstream
numbers justify testing the candidate; they are not a deadline guarantee for a
particular DAW, host block pattern, or machine. denoize separately tests the
pinned graph, fixed latency, worker stalls, queue saturation, allocation
failure, generation resets, mono/stereo layouts, f32/f64 buffers, automation,
portable state, and the official CLAP validator.

The design follows the central conclusion of
[RTNeural](https://arxiv.org/abs/2106.03037): compact recurrent inference can be
engineered for real-time systems, but the deployed graph and execution path
still require direct measurement. The audio callback boundary follows the
[CLAP specification](https://github.com/free-audio/clap), including its latency,
audio-port, state, parameter, and thread rules.

This release is speech enhancement, not general restoration, target-speaker
extraction, AEC, or spatial beamforming. v0.79.0 adds the same accessible native
CLAP editor used by the DSP descriptor; hosts retain generic rendering of all
four parameters when the custom API is unsupported or creation fails. See
[Accessible plug-in editor](plugin-editor.md). v0.78.2 added a statically bound
VST3 3.8 adapter with
official-validator, pinned Ardour 8.4 processing/state-reload smoke, packaging,
checksum, SBOM, and signed host-matrix evidence; see
[VST3 plug-in](vst3-plugin.md). Its matrix intentionally does not claim
double-precision VST3 audio, a VST3 custom view, or compatibility with untested
proprietary hosts. v0.80.0 adds the macOS AUv3 gate with a sandbox-visible
verified copy of the pinned graph, Apple `auval`, and independent AVFoundation
lifecycle/state evidence; see [AUv3 plug-in](auv3-plugin.md). v0.81.0 adds a
direct Linux LV2 adapter whose inference is delegated to the host Worker and
whose Atom/Patch automation and portable State are exercised in Jalv and
Ardour; see [LV2 plug-in](lv2-plugin.md). iOS and untested proprietary hosts
remain separate release gates.
