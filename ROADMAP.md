# Neural backend roadmap

This document separates deployable implementations from architectural names.
A named model backend is complete only when denoize can load a documented
pretrained model, run it without Python, preserve the input channel count and
duration, and pass an end-to-end audio fixture test. The model-agnostic ONNX
foundation is complete when its public tensor contract is validated before
inference, a loaded graph can be reused without reopening its pathname, and
end-to-end fixtures cover the accepted layouts, resampling, channels, and
duration. Model quality remains a gate for each named adapter, not for an
arbitrary user-supplied graph.

## Managed model operations roadmap

Model distribution and local lifecycle work proceeds in this order. A stage is
complete only when its CLI and desktop surfaces, offline behavior, failure
atomicity, documentation, and release assets are covered by automated tests.

| Order | Stage | Status |
|---:|---|---|
| 1 | Signed, sequence-monotonic model catalog; exact artifact size/SHA-256; content-addressed installation provenance | Implemented |
| 2 | `models doctor`, `verify`, `repair`, and `prune` for corrupt, missing, stale, and orphaned cache state | Implemented |
| 3 | Signing-key rotation, explicit revocation, expiry policy, and emergency trust-root recovery | Implemented |
| 4 | Signed offline bundles containing catalog, signature, models, licenses, and provenance for closed networks | Implemented |
| 5 | Stable JSON output for catalog/model health, provenance, recipe identity, and automation | Implemented |
| 6 | Hardware capability discovery, explicit accelerator selection, and deterministic CPU fallback | Implemented |
| 7 | Process-level RAM/temporary/GPU admission, memory-weighted workers, and opt-in OS child isolation for third-party codec/model failures | Implemented |
| 8 | Bounded streaming for compressed inputs and restartable processing checkpoints for long-running jobs | Implemented |
| 9 | Input- and device-aware quality/model recommendation with reproducible calibration evidence | Implemented |
| 10 | Reproducible releases with per-artifact SBOMs, signed build provenance, and offline verification for binaries, crates, and converted models | Implemented |

Stages 2–5 extend the authenticated distribution system without weakening its
rollback or provenance guarantees. Stages 6–9 are runtime improvements and
must retain a portable CPU path and deterministic validation fixtures. Stage 10
extends authentication from downloaded model bytes to the release and model
conversion processes that produced every distributed artifact.

Stage 7 exposes a cloneable library `ResourceGovernor` and connects it to CLI,
desktop, batch, streaming, and live processing. Admission atomically reserves
denoize-owned RAM, staged-output bytes, CPU/GPU worker slots, and conservative
GPU/model allowances; retained metadata and configured decoder scratch budgets
participate in the worker weight. The CLI's optional `--isolate` child adds an
`RLIMIT_AS` boundary on Unix or a Job Object process-memory boundary on Windows.
Cooperative counters deliberately do not claim allocator-exact RSS, filesystem
quota, or driver-exact VRAM enforcement.

## Investigation status

| Model | Upstream artifact | Native integration gap | Status |
|---|---|---|---|
| BSRNN | [ESPnet VCTK+DEMAND xtiny checkpoint](https://huggingface.co/wyz/vctk_bsrnn_xtiny_causal) (CC-BY-4.0) | External conversion is required because upstream publishes PyTorch only | Implemented |
| MP-SENet | [Official MIT repository](https://github.com/yxlu-0102/MP-SENet) with PyTorch checkpoints | External conversion is required because upstream publishes PyTorch only | Implemented |
| MossFormer2 | [Apache-2.0 ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) and the official 48 kHz checkpoint | External conversion is required because upstream publishes PyTorch only | Implemented |
| SGMSE+ | [Official MIT repository](https://github.com/sp-uhh/sgmse) with PyTorch Lightning checkpoints | External conversion plus a native iterative predictor/corrector sampler | Implemented |

None of these upstream projects currently publishes a model artifact with a
documented ONNX contract that can be embedded directly in this Rust CLI. Their
PyTorch checkpoints are not treated as implemented support.

## Implemented foundation

The `onnx` feature provides a Pure-Rust tract backend for one-input,
one-output `float32` waveform models:

- input layout `[batch, samples]` or `[batch, channels, samples]`;
- batch and model channel dimension are fixed to one;
- file channels are processed independently;
- audio is resampled to and from the configured model rate;
- output duration and original channel count are preserved;
- missing files, unsupported ranks, short outputs, and non-finite samples are
  rejected with explicit errors.

`OnnxWaveformModel::load` establishes that contract once and exposes its layout
and any fixed input/output length to embedders. It retains the parsed graph,
caches the optimized graph for the most recent model-rate input length, and
therefore neither reparses the graph nor observes later pathname replacement
on repeated calls. The module-level `onnx::process` function is the compatible
single-call wrapper.

`BackendSession::prepare` is the common reusable layer for finite processing.
CLI batches share one prepared session for every equal backend/model option
set, and VAD regions use the same session instead of reopening a graph per
region. Fixed-shape adapters retain one optimized graph; dynamic BSRNN,
SGMSE+, and generic waveform adapters retain the most recently required tensor
shape. DeepFilterNet's non-`Send` runtime is cached once per worker thread. The
stateful `StreamingBackendSession` provides the continuous file-streaming API
for Classical, RNNoise, DeepFilterNet, MossFormer2, and GTCRN. The low-latency
live capture/playback path deliberately accepts only Classical, RNNoise, and
GTCRN.

The generated rank-2 and rank-3 ONNX fixtures exercise real tract inference,
sample-rate conversion, multichannel independence, exact duration restoration,
deterministic ordering, fixed-shape rejection, and cache reuse. The dedicated
model adapters demonstrate the same Pure-Rust deployment layer with real
pretrained graphs and their own numerical and speech-quality gates; the managed
official GTCRN graph additionally exercises stateful multi-input inference
without Python. These checks complete the external ONNX inference foundation.

This contract can host exported waveform models, but it does not make any of
the named roadmap models complete by itself.

## MP-SENet adapter

The `mpsenet` feature implements the official 16 kHz frontend in Rust: RMS
normalization, centered 400-point periodic-Hann STFT with 100-sample hop,
0.3-power magnitude compression, parallel magnitude/phase inference, inverse
STFT, 50%-overlapped reconstruction of the official 32,000-sample training
segments, and exact input-duration restoration. `scripts/export-mpsenet.py`
converts an official `g_best_vb` or `g_best_dns` checkpoint into the adapter's
two-input/two-output ONNX contract. The converted model is covered by a pinned
automated real-speech quality fixture.

The converter pins upstream revision
`89932cfe90d1dacb8e170e4a331d762462c21792` and verifies the official checkpoint
SHA-256 before export. On a fixed two-second 16 kHz fixture, the converted graph
matched upstream PyTorch through ONNX Runtime with magnitude correlation above
`0.9999999999` and phase correlation above `0.9999999999`; tract matched ONNX
Runtime at the same correlation threshold. End-to-end Rust/PyTorch waveform
correlation was `0.9900` (MSE `8.56e-6`), with the remaining difference dominated
by phase wrapping in low-energy FFT bins across the two FFT implementations.
On the pinned two-second Apache-2.0 ESPnet speech fixture, the Rust end-to-end
quality gate improved SI-SNR from `2.719 dB` to `10.282 dB` (`+7.563 dB`). The
converted graph is about 9 MiB. On the reference x86-64 Linux host, inference
for the fixture took 43.67 seconds and the complete process used 410,048 KiB
maximum RSS.

## BSRNN adapter

The `bsrnn` feature implements the causal ESPnet BSRNN frontend and inference
contract at 48 kHz: per-channel sample-standard-deviation normalization,
centered 960-point periodic-Hann STFT with a 480-sample hop, whole-utterance
recurrent inference, inverse STFT, de-normalization, and exact input channel
count/rate/duration restoration. `scripts/export-bsrnn.py` converts the pinned
`wyz/vctk_bsrnn_xtiny_causal` checkpoint into a dynamic-frame
`[1, frames, 481, 2]` ONNX graph and can verify it against PyTorch using ONNX
Runtime.

The model revision is `59e1f2263b7946b1970a222d1beef9adc5a67eaa`, the
checkpoint SHA-256 is
`e3cb771a452e0503144af74720b476e81b57f518b789b37ba2c253c6cc22d70b`,
and the reference architecture is pinned to Apache-2.0 ESPnet revision
`5208894ceaa534732164212357b63d83dd137eab`. The model is CC-BY-4.0 and the
adapted reference implementation is Apache-2.0; denoize does not bundle its
weights.

On the fixed 67-frame numerical fixture, PyTorch and ONNX Runtime correlation
was `0.999999999998` (MSE `1.88e-11`, maximum absolute error `2.34e-4`). On the
same fixture's PyTorch and Rust waveforms, after the CLI's documented PCM
clipping and quantization, correlated at `0.99999999958` (MSE `2.18e-10`,
maximum absolute error `1.85e-4`). On the pinned two-second Apache-2.0 ESPnet
speech fixture, the Rust end-to-end quality gate improved SI-SNR from
`2.719 dB` to `9.612 dB` (`+6.892 dB`). A release build on the reference x86-64
Linux host processed it in 1.58 seconds (1.3x realtime) with 44,628 KiB maximum
RSS. The model is about 2.4 MiB; memory and latency grow with utterance length
because upstream inference is recurrent and whole-utterance.

## MossFormer2 adapter

The `mossformer2` feature implements the ClearerVoice 48 kHz frontend and its
four-second deployment contract: 60-bin Kaldi fbank features with first- and
second-order deltas, a non-centred 1,920-point symmetric-Hamming STFT with a
384-sample hop, real spectral-mask application, three-second-stride segmented
inference, 0.5-second edge discard, resampling, and exact input-duration and
channel restoration. `scripts/export-mossformer2.py` pins and verifies the
official checkpoint and rewrites the fixed 496-frame graph to tract-supported
primitive ONNX operations.

The architecture revision is `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`,
the model revision is `eff8c97925c8bec812af707814b3e5d777fd4503`, and the
checkpoint SHA-256 is
`03692b9f773bbd6bb43b9c5a41f96b1e28affd66e13796b7bec66ad3d8b227c6`.
Both architecture and model are Apache-2.0; weights are external. On a fixed
496-frame numerical fixture, the compatibility rewrite matched its source
graph exactly, while tract and ONNX Runtime correlated at
`0.999999999997` (MSE `4.93e-12`, maximum absolute error `4.49e-5`). The graph
is about 217 MiB. A four-second release-build CLI run on the reference x86-64
Linux host took 7.74 seconds and used 483,400 KiB maximum RSS. On the pinned
four-second Apache-2.0 ESPnet speech fixture, the Rust end-to-end quality gate
improved SI-SNR from `2.683 dB` to `13.928 dB` (`+11.246 dB`).

## Completion gates

For each named backend:

1. Pin the upstream architecture and checkpoint revision and record its license.
2. Supply a reproducible conversion or a native safe-tensors loader.
3. Implement the exact normalization, STFT, chunking, and reconstruction used
   by upstream inference.
4. Verify numerical parity against upstream inference on a fixed fixture.
5. Add a denoising quality regression fixture, not only shape tests.
6. Document model download, checksum, sample rate, latency, and memory use.
7. Include the backend in `full` only when release binaries can actually run it.

SGMSE+ additionally requires deterministic sampler tests and an explicit
quality/speed choice because its iterative inference cost differs substantially
from one-pass enhancement networks.

## SGMSE+ adapter

The `sgmse` feature implements the official 16 kHz VoiceBank+DEMAND inference
path: noisy-peak normalization, centered 510-point periodic-Hann STFT with a
128-sample hop, magnitude-square-root complex transform scaled by 0.15,
multiple-of-64 spectral padding, inverse transform, and exact duration/channel
restoration. `scripts/export-sgmse.py` loads the official EMA parameters and
exports the dynamic-frame NCSN++ score network with explicit real/imaginary
channels for tract.

The architecture revision is `1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e`,
the model revision is `b6485214b3662a7f90309f397cacf1384046783c`, and the
checkpoint SHA-256 is
`e3875747b5646092d5c556bae68e5af639e2c1f45f009c669f379cd4d415cbd8`.
Both code and model are MIT licensed; weights are external. The explicit
quality/speed choice is the upstream quality configuration: 30 OUVE reverse
steps, one ALD corrector step per reverse step, `snr=0.5`, and therefore 60
score-network evaluations. Sampling uses a documented fixed SplitMix64 and
Box-Muller normal stream so repeated runs are deterministic.

On a fixed 64-frame score fixture, PyTorch and ONNX Runtime correlated above
`0.999999999999` (MSE `4.66e-12`, maximum absolute error `1.53e-5`). On the
pinned two-second Apache-2.0 ESPnet speech fixture, the Rust end-to-end output
correlated with the same deterministic Python/ONNX sampler at
`0.9999999972` (MSE `2.35e-11`, maximum PCM difference `3.05e-5`). The quality
gate improved SI-SNR from `2.719 dB` to `11.471 dB` (`+8.752 dB`). The graph is
about 252 MiB. A release build on the reference x86-64 Linux host took 737.92
seconds for the two-second fixture and used 1,204,648 KiB maximum RSS.

## Product delivery stages

This sequence tracks the operational work around the neural backends. A stage
is marked implemented only after its CLI and desktop surfaces, documentation,
focused and broad tests, release package, CI, tag, and published assets have
been verified. Stages are released in order rather than accumulated into one
unreviewable release.

| Stage | Deliverable | Status |
|---:|---|---|
| 1 | Signed, rollback-resistant managed-model catalog and install provenance | Released in v0.49.0 |
| 2 | Conservative model-cache doctor, repair, and prune workflows | Released in v0.50.0 |
| 3 | Signed trust-root rotation with rollback and expiry policy | Released in v0.51.0 |
| 4 | Signed offline multi-model transfer bundles | Released in v0.52.0 |
| 5 | Stable CLI, model, and hardware automation contracts | Released in v0.53.0 |
| 6 | Explicit CPU/Metal/CUDA discovery and accelerator selection | Released in v0.54.0 |
| 7 | Process-wide RAM, temporary-space, CPU, GPU, and isolation admission | Released in v0.55.0 |
| 8 | Bounded compressed-input streaming with durable restart checkpoints | Released in v0.56.0 |
| 9 | Network-free backend and preset recommendation with on-device benchmark calibration and an explainable decision report | Released in v0.57.0 |
| 10 | Release SBOMs, build provenance, and asset-to-source verification | Released in v0.58.0 |
| 11 | Read-only execution plans, signed receipts, and offline result verification | Released in v0.59.0 |
| 12 | Native gapless/granule/edit-aware checkpoints, encoded output, and bounded non-seekable streams | Released in v0.60.0 |
| 13 | Parser and resource-amplification fuzzing, deterministic fault injection, and crash/power-loss simulation | Released in v0.61.0 |
| 14 | Desktop isolation, recovery, bounded non-destructive preview and A/B comparison, redacted diagnostics, accessibility, and localization | Released in v0.62.0 |
| 15 | Streaming feature parity: bounded VAD, two-pass loudness, metadata, and additional AI backends | Released in v0.63.0 |
| 16 | Live-device resilience: asynchronous resampling, clock-drift correction, hotplug recovery, and latency diagnostics | Released in v0.64.0 |
| 17 | Signed, self-describing custom-model runtime packages with frontend, license, resource, and tensor contracts | Released in v0.65.0 |
| 18 | Local watch-folder automation with settle detection, retry, quarantine, and receipts | Planned |
| 19 | Local authenticated IPC and job-control API with bounded requests, capability-scoped authorization, durable status/cancel control, and stable automation contracts | Planned |
| 20 | Real-time-safe DAW plug-in integration with portable presets, deterministic session restoration, and measured latency | Planned |
| 21 | Reproducible licensed-corpus objective, perceptual, performance, output-quality, and regression evaluation with publishable evidence manifests | Planned |
| 22 | Signed updates with staged activation, health-checked rollback, and offline recovery | Planned |
| 23 | Portable projects and sample-accurate region/timeline workflows with bounded edit graphs, deterministic assembly, and signed provenance | Planned |

Stage 8 accepts regular-file WAV, FLAC, and Ogg Vorbis input, writes an atomic
WAV, and supports the stateful Classical, RNNoise, and GTCRN backends. Its
checkpoint binds the input, effective recipe, model, decoder geometry, and
block size; it synchronizes a bounded journal and PCM spool, replays backend
state deterministically, and records the staged output fingerprint before
publication. A restart therefore resumes an incomplete stream or reconciles a
completed commit whose data sidecars were not yet removed. Stage 12 extends the
same contract to gapless MP3, granule-aware Ogg Opus, frame-aware ADTS AAC, and
edit-aware M4A AAC/ALAC; it adds encoded output, complete pre-publication decode
verification, bounded stdin/stdout and library `Read`/`Write` spools, and v2
stream plans and signed receipts. Checkpoints persist presentation PCM only
after codec delay, granule, or edit-list mapping has been applied.

Stage 13 makes robustness failures reproducible instead of treating a long
random fuzz run as evidence by itself. Checked-in seed cases and a fixed,
versioned mutator exercise every supported audio container plus the signed and
durable JSON/bundle formats under explicit input, allocation, iteration, and
wall-clock ceilings. Nightly coverage-guided fuzzing uses the same entry points;
every minimized finding becomes a deterministic regression seed before it is
considered fixed. Resource-amplification assertions distinguish denoize-owned
capacity, declared codec scratch, and third-party private allocation rather
than presenting an RSS sample as an exact portable bound.

Fault injection is compiled only into debug/test builds and uses an exact
point, occurrence, and action (`error` or immediate process exit), so a test
cannot silently fire at a nearby operation after code is reordered. Injection
points bracket staged-file synchronization and publication, journal
prepare/completion, checkpoint spool/state updates, signed-receipt publication,
and managed-model state changes. Child-process tests enumerate those durable
prefixes, restart from each abrupt exit, and require an old or fully committed
output, coherent restart state, reusable locks, no partial automation output,
and no deletion of an unverified artifact. Power-loss tests simulate loss at
acknowledged synchronization boundaries on a local filesystem; they do not
claim to model faulty hardware, remote filesystems, drive write caches, or a
kernel that violates its documented durability semantics.

Stage 9 keeps recommendation read-only and network-free. It analyzes a bounded
signal prefix, considers compiled backends, verified local managed models,
one read-only hardware/runtime snapshot, and CPU/GPU resource limits, then emits
stable reason codes and explicit settings. Optional calibration runs a fixed
hash-identified Classical Hi-Fi workload locally and preserves the raw timing
evidence; backend headroom is an explainable cost-class estimate, not a claim
that every neural candidate was executed or that wall-clock time is
deterministic.

Stage 17 introduces a separate trust boundary for operator-selected custom
models. A `.dmp` is a signed, length-delimited container rather than an archive:
its manifest binds exact model and license bytes, SPDX identity, runtime and
sample rate, waveform frontend, float32 tensor geometry, permitted
accelerators, and conservative session/worker CPU/GPU reservations. The trust
key is supplied separately, so replacing a key embedded beside hostile bytes
cannot make those bytes authoritative. CLI, desktop, and library entry points
all use the same verifier and show the authenticated identity before execution.

The generic ONNX adapter reads the verified model range without extraction or
path-based sidecars, compares the signed tensor declaration with the parsed
graph, and re-hashes the package when preparing a session. Resource admission
uses signed values only after enforcing denoize's own conservative model and
GPU baselines. Resume and execution evidence bind the whole package
fingerprint; changing model, notice, signature, or contract changes that
identity. V1 intentionally supports the reproducible mono waveform frontend
only—spectral, recurrent multi-input, or code-bearing packages require a future
versioned adapter rather than weakening the v1 contract.

Every stage from Stage 12 onward also carries an upgrade-compatibility gate:
persisted presets, journals, checkpoints, receipts, and automation schemas must
migrate from at least the two preceding releases or reject an unknown future
format without modifying it. Stage 12 preserves the v0.58/v0.59 v1 checkpoint,
v3 batch-journal, and v1 execution-document contracts while adding separate v2
stream schemas. Anonymous `Read`/`Write` spools are finite but ephemeral:
atomic publication applies only to a filesystem transaction, and restart
requires durable regular-file input and output.

Stage 14 adds a bounded, non-destructive audition path rather than a second
processing engine. Users can select a short region, render candidate recipes,
switch between loudness-matched original and processed audio, run a blind A/B
comparison, and persist the chosen recipe. Preview work uses the same backend,
resource admission, and source fingerprint as the final job. Its bounded region
locator is versioned so Stage 23 can reuse it without changing presentation
time; cancelling a preview leaves no output or durable state. Preview,
final-file, and batch decoding/inference run in supervised child processes.
Final workers publish bounded authenticated progress, and a shared
commit/cancel fence prevents output publication after cancellation or protocol
failure. File and batch jobs keep owner-private recovery records for their exact
live stages.
Recovery can retry a request or remove only verified denoize-owned stages; it
does not delete an existing output or restart journal. Diagnostics contain only
bounded schema-defined event codes and capability/count fields. Rust failures
use structured codes with localized Japanese/English summaries and preserved
technical details. Keyboard/ARIA navigation, visible focus, reduced motion,
and forced-colors checks are joined by a real-WebView interaction test in the
desktop release gate.

Stage 15 closes the bounded file-streaming feature gap without converting the
pipeline back into a whole-file allocation. A fixed-history VAD aligns original
and processed presentation samples across backend latency. Loudness uses a
fixed-memory EBU R128 analysis pass over an anonymous bounded PCM spool and a
constant-gain encoding pass; restart checkpoints reuse their existing PCM
spool. Metadata is applied before verification for regular-file and stdout
destinations. DeepFilterNet processes reusable 48 kHz hops, while MossFormer2
retains only its official four-second window and three-second stride; both
reuse a prepared model and preserve the exact input presentation length.

Stage 16 separates the capture and playback clocks. A bounded asynchronous
sinc converter accepts different nominal device rates, while a bounded PI
controller makes small ratio adjustments to hold a configured playback queue.
The target queue, correction ceiling, and recovery window are explicit CLI,
configuration-file, library, and desktop settings. DSP remains on the worker
thread; capture uses a non-waiting bounded handoff, while playback consumes a
bounded queue without waiting for the worker to release it. Overload discards
stale complete chunks or the oldest complete playback frames, emits silence on
playback contention, and cold-resets causal state at a sequence gap. Stage 20 remains
responsible for the stronger allocation-free/lock-free plug-in callback
contract.

Device/configuration and stream callback failures enter a finite exponential
backoff loop. Named devices are reacquired by an unambiguous exact name, while
duplicate names fail closed and default-device sessions follow the new system
default. Each successful generation rebuilds device-bound state, primes the
playback queue before starting output, and
retains the validated processing configuration and runtime selection. CLI
NDJSON and the desktop report connection phase, device generation, independent
sample rates, queue depth, underrun/overflow/drop counts, bounded drift
correction, and a capture-to-playback latency estimate. The estimate includes
measured callback timing, chunking, converter/backend delay, processing time,
and queue time; it is not a hardware loopback guarantee.

Stage 19 exposes only local, authenticated control surfaces and reuses the
bounded JSON contracts, resource admission, signed receipts, and regular-file
publication rules rather than creating a second execution model. Capability
grants are explicit and revocable; discovery, transport, request size, timeout,
and concurrency limits are part of the public contract. Its durable queue
supports explicit priority plus pause and resume at a verified checkpoint
boundary; work without a resumable boundary remains cancel-and-retry rather
than pretending to pause safely. Before admission, clients can inspect a
bounded dry-run report of requested RAM, temporary space, destination changes,
and overwrite policy. Bounded job history links that report to the resulting
execution plan and signed receipt without retaining input paths indefinitely.

Stage 20 brings the existing causal/live backends into DAW hosts without
allocating, blocking, performing filesystem or network I/O, or changing model
state on the real-time audio thread. The first portable plug-in format must
ship with deterministic preset/session round trips, latency reporting, channel
layout negotiation, bypass/tail behavior, and host-level regression fixtures;
additional platform formats follow only when they preserve the same contract.

Stage 21 turns model-quality, output-integrity, and speed claims into
reproducible release evidence. Corpus manifests pin licenses, source revisions,
checksums, signal preparation, objective and perceptual metrics, listening-test
protocol where automation cannot substitute for human judgment,
hardware/runtime context, and accepted thresholds while keeping restricted
audio out of the repository and release artifacts. Output-quality reports add
duration and channel-layout agreement, clipping and true peak, DC offset,
unexpected silence or dropouts, loudness, decode integrity, and the output
fingerprint. Local and CI runners consume the same manifest, emit
machine-readable signed results, and fail closed on missing provenance or
incomparable configurations.

Stage 22 makes application updates an explicit recoverable transaction rather
than an in-place replacement. Signed manifests bind the release channel,
platform artifact, SBOM, provenance, compatibility range, and rollback policy;
the updater downloads or imports an offline bundle into a bounded staging area,
verifies it before activation, and retains the last known-good installation.
Startup health checks either confirm the new version or atomically restore that
known-good version without weakening anti-rollback policy. CLI and desktop
surfaces expose read-only check and dry-run reports, explicit `apply`, `status`,
and `recover` operations, and durable redacted diagnostics; they never silently
downgrade, delete the only recoverable installation, or require a network for
recovery.

Stage 23 makes partial-file processing an explicit timeline contract instead
of an ad hoc trim. Region locators bind the source fingerprint, presentation
timebase, channel map, ordered selections, padding, and crossfades; bounded edit
graphs compose those regions without whole-file PCM retention. A versioned,
portable project manifest records source locators and fingerprints, timelines,
models, presets, plans, and receipts without embedding source audio by default;
relocation and missing-source recovery require an exact fingerprint match. CLI,
desktop, plans, receipts, and batch/watch automation share the same
deterministic assembly semantics, and unsupported overlapping or future edit
records fail closed without changing source, project, or existing output.
An offline export/import bundle carries the project manifest, settings,
presets, trusted model-package references, and verification evidence; source
audio and model payloads are included only by an explicit bounded option and
retain their existing signature, license, and fingerprint checks.
