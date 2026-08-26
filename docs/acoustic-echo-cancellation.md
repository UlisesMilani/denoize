# Fail-closed acoustic echo cancellation

Stage 30 introduces a typed far-end-reference acoustic echo cancellation (AEC)
path. It is not ordinary denoising: the microphone and playback reference are
separate inputs with an explicit clock relationship, signed delay can be
positive or negative, and route changes cold-reset all adaptive state.

The first promoted implementation is `native-pfdnlms-v1`. It deliberately
keeps the safe path inspectable:

1. constant reference clock offset is mapped to the microphone clock;
2. normalized FFT cross-correlation estimates signed bulk delay;
3. a partitioned frequency-domain normalized-LMS filter estimates the linear
   acoustic echo path;
4. double-talk classification freezes adaptation so near-end speech cannot
   train the echo path;
5. a conservative residual suppressor attenuates only the linear error and
   uses a substantially higher gain floor during double talk;
6. missing or low-confidence reference preserves the microphone exactly.

No neural checkpoint is required or bundled. A future nonlinear neural
post-filter may receive the aligned microphone/reference, linear echo estimate,
and error only through a dedicated package-v2 adapter. It cannot replace
explicit alignment, route generation, or the native fallback.

## CLI

Build with the `aec` feature (included by `full`) and provide independently
distributed promotion evidence and its Ed25519 public key:

```sh
denoize aec microphone.wav playback-reference.wav cleaned.wav \
  --promotion-evidence aec-evidence.json \
  --promotion-evidence-key evaluator-public-key.json \
  --aec-config aec-config.json \
  --reference-clock-ppm -37.5 \
  --initial-delay-samples -240 \
  --route-generation 18 \
  --report aec-report.json \
  --pretty
```

`--aec-config` is optional and defaults to the promoted 48 kHz baseline. A
custom configuration is useful only when its exact canonical JSON digest and
filter geometry are bound by the signed evidence. Evidence, configuration, and
key authentication completes before either audio file is opened. Microphone,
reference, configuration, evidence, key, output, and report paths must all be
distinct. Output and report publication is atomic and no-clobber unless
`--replace` is explicit.

Verify evidence without opening audio:

```sh
denoize aec evidence verify \
  aec-evidence.json evaluator-public-key.json --pretty
```

The file path currently accepts exactly one microphone channel and one typed
far-end channel. It preserves the microphone sample rate and exact frame count.
Reference rate conversion occurs only when both nominal rates and
`--reference-clock-ppm` are recorded in the report; there is no silent clock
assumption.

## Configuration and latency

`AecConfig` is a closed JSON object. The default uses 256-sample blocks at
48 kHz (5.333 ms algorithmic-plus-buffering latency), a 500 ms linear tail, a
signed one-second delay range, and a three-second delay-analysis window. Block
size must be a power of two and every accepted configuration must remain at or
below 20 ms. Tail and signed delay range are each bounded to two seconds;
analysis is bounded to ten seconds and the complete native state to 512 MiB.

The configuration also binds adaptation rate and regularization, leakage,
reference activity and delay confidence, double-talk correlation, residual
suppression, far-end and double-talk gain floors, and a normalized absolute
peak ceiling. A change to any field changes `configuration_sha256`, so signed
evidence for a different tuning cannot be replayed.

## Real-time library boundary

`AecSession::prepare` authenticates the evidence and exact configuration.
`AecSession::stream` then constructs a fixed-size `AecStream` for one route
generation. Construction plans both FFTs and allocates all filter, history,
scratch, overlap, echo, and error buffers. `AecStream::process_block` accepts
exactly one configured block and performs no allocation, locks, waits, file or
network I/O, or logging. Native FFT/adaptation work does execute on the calling
audio thread; a host that cannot budget the signed worst-case real-time factor
must schedule the stream on its own permanent worker and use latency-aligned
microphone fallback.

`AecSession::realtime_adapter` adds the typed sidechain bridge for hosts whose
callback quantum differs from the promoted block. It preallocates four mono
blocks, accepts equal arbitrary-length microphone/reference/output slices, and
advertises exactly one AEC block of latency. It never changes latency when a
callback boundary moves. Route or reference-confidence changes discard the
partial input and latency block before any new-generation output is returned.

`set_route_generation`, reference discontinuity, clock jumps, delay jumps, and
non-finite adaptive state are cold-reset boundaries. Filter spectra, reference
history, overlap, echo, and error buffers are all zeroed. A route generation is
an absolute token; state from an older route is never reused for a new one.

## Promotion evidence

[`denoize-aec-promotion-evidence-v1`](../schemas/denoize-aec-promotion-evidence-v1.schema.json)
is a closed, bounded, domain-separated Ed25519 document. It binds the exact
implementation source, configuration, corpus manifest, evaluation result,
listening result, sample rate, block, tail, and signed delay range.

Exactly 17 sorted strata are required: background noise, clipping, positive and
negative clock drift, delay jump, positive and negative delay, double talk,
clean far end, linear path, music playback, clean near end, nonlinear speaker,
real device, reference loss, room change, and route change. Depending on the
stratum, hard limits cover valid-region ERLE, near-end attenuation, word-
accuracy regression, far-end/double-talk/full-band AECMOS, exact duration,
non-finite output, reconvergence, stale reset output, and no more than 20 ms
latency. Stricter signed limits are permitted; weaker limits are rejected.

Promotion additionally requires at least 100 real-device, nonlinear-device,
and transition cases; 10,000 paced blocks; worst-case single-thread RTF at or
below 0.5; zero callback allocations, locks, waits, I/O, logging, deadline
misses, and stale frames after reset; at least 20 listeners; and a mechanically
consistent preference decision. Learned MOS is corroborating evidence, not the
sole gate.

## Report and privacy

[`denoize-aec-report-v1`](../schemas/denoize-aec-report-v1.schema.json) records
the evidence/configuration identities, domain-separated microphone/reference/
output PCM digests, explicit clock mapping, signed delay and confidence,
latency/filter geometry, talk-state/adaptation/reset/clipping counts, and ERLE
only when far-end-only regions exist. It records zero filesystem paths. Block
counts must cover the complete render, adaptation cannot exceed far-end-only
blocks, output duration must equal microphone duration, and non-finite output
must remain zero.

## Evidence basis and limitations

The hybrid architecture follows the adaptive-filter-plus-RNN decomposition in
[Haubner et al. (2020)](https://arxiv.org/abs/2005.09237), while retaining the
linear path even before a learned residual filter is admitted. The
[ICASSP 2021 AEC Challenge](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/06/0000151.pdf)
motivates real-device/double-talk/listening gates rather than ERLE/PESQ alone.
The [ICASSP 2023 AEC Challenge](https://arxiv.org/abs/2309.12553) motivates the
20 ms class, full-band AECMOS, word-accuracy, and personalized/nonlinear test
matrix. The neural-Kalman review by
[Haubner et al. (2025)](https://arxiv.org/abs/2501.16367) supports keeping an
interpretable acoustic-path state and using learned components only for
reconvergence control or residual nonlinear echo.

The initial release is mono and models constant clock offset within one file;
an abrupt clock step requires a reset. Its native residual suppressor is not a
claim of nonlinear loudspeaker cancellation. A live host still owns capture/
playback clock measurement, reference continuity, route generation, worker
scheduling where needed, and real-device promotion evidence.
