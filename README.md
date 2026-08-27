# denoize

**The pursuit of the world's highest-fidelity audio denoising — in pure Rust.**

`denoize` removes background noise from WAV recordings with **maximum transparency**:
preserving timbre, transients, dynamics, stereo imaging, and natural "air".

## Implemented technology stack

### Classical DSP (always available)
- STFT/ISTFT + Perfect Reconstruction OLA + high overlap
- IMCRA/MCRA noise estimation + SPP + spectral-flatness profiling
- Ephraim-Malah Decision-Directed SNR
- **8 gain estimators**: OMLSA, LogMMSE, MMSE-STSA, Wiener, SpecSub, SpecSub-NL, SpecSub-Geo
- Transient protection, cepstral smoothing, pre-emphasis
- **Advanced windows**: Kaiser, Flat-top, principal DPSS/Slepian (+ Hann/Hamming/Sine/Blackman)
- **Multiband spectral subtraction** (Bark bands)
- **Perceptual weighting** (Bark-scale gain shaping)
- **Musical-noise post-filter**

### Optional AI backends (feature-gated)
| Backend | Feature | Description |
|---------|---------|-------------|
| `rnnoise` | `--features rnnoise` | RNNoise via nnnoiseless (pure-Rust) |
| `deepfilter` | `--features deepfilter` | DeepFilterNet v3 (tract ONNX, embedded model) |
| `onnx` | `--features onnx` | External waveform-to-waveform ONNX model (tract, Pure Rust) |
| `mpsenet` | `--features mpsenet` | MP-SENet magnitude/phase enhancement adapter (external converted model) |
| `bsrnn` | `--features bsrnn` | ESPnet BSRNN spectral enhancement adapter (external converted model) |
| `mossformer2` | `--features mossformer2` | ClearerVoice MossFormer2 48 kHz mask adapter (external converted model) |
| `sgmse` | `--features sgmse` | SGMSE+ iterative diffusion adapter (external converted model) |
| `gtcrn` | `--features gtcrn` | Official 48K-parameter causal GTCRN; reusable offline, `--stream`, library, and realtime sessions |

Build everything: `cargo build --release --features full`

The generic ONNX backend is the deployment foundation for future neural
models. Raw `.onnx` files and signed package v1 intentionally accept only
single-input/single-output waveform models. Signed package v2 additionally
authenticates named multi-input/output tensors, recurrent state, channel roles
and microphone geometry, latency/context, per-accelerator precision profiles,
resource budgets, provenance, and numerical conformance vectors. The generic
adapter executes only a finite-capable, independent-mono graph with one required
waveform input and one waveform output; expressive v2 graphs fail closed until
their dedicated restoration, target-speaker, AEC, or spatial adapter is
selected.
Library embedders can load [`OnnxWaveformModel`](https://docs.rs/denoize/latest/denoize/struct.OnnxWaveformModel.html)
once, inspect its validated `float32` waveform contract, and reuse the most
recently compiled input length instead of parsing and optimizing the graph on
every call. The CLI convenience path keeps the same `--onnx-model` contract.

> The prebuilt GitHub binaries include every backend. Because the DeepFilterNet
> Rust crate is not available from crates.io, the crates.io package's `full`
> feature currently includes RNNoise, generic ONNX, MP-SENet, BSRNN,
> MossFormer2, SGMSE+, GTCRN, and live-device support, but not DeepFilterNet.
> Building live support from source on Linux requires the ALSA development
> package (for example, `libasound2-dev` on Debian/Ubuntu); prebuilt archives do
> not require compiler headers.

### Hardware acceleration

CPU inference remains the compatibility default. Full builds register Apple
Metal on Apple targets and NVIDIA CUDA on Linux/Windows targets for the generic
ONNX, MP-SENet, BSRNN, MossFormer2, SGMSE+, and GTCRN adapters. Inspect the
current binary and host without opening a model or using the network:

```sh
denoize hardware
denoize hardware --json
denoize noisy.wav clean.wav -b gtcrn --accelerator auto
```

`--accelerator auto` uses the stable Metal-then-CUDA preference and reports an
explicit CPU fallback when the backend is CPU-only, deterministic mode is
active, or no compiled runtime passes its dependency probe. `gpu`, `metal`, and
`cuda` are strict requests. With an explicit backend, an unsupported or
unavailable request fails before input decoding; automatic backend selection
must inspect the decoded input before it can validate backend compatibility.
`--deterministic` always executes on CPU; combine it with `cpu` or `auto`.
File/stream JSON results expose `requested`, `effective`, and `fallback`, and
the effective runtime is included in batch recipe identity. The capability
contract is published as
[`denoize-hardware-v1`](schemas/denoize-hardware-v1.schema.json).
For an available GPU the report also includes the device name, CUDA compute
capability when applicable, and the runtime-reported device-memory limit
(total global memory for CUDA; recommended maximum working set for Metal).

CUDA availability requires a compatible NVIDIA driver plus the CUDA runtime,
NVRTC, cuBLAS, cuDNN, CUDA development headers, CCCL headers, and a writable
tract kernel-cache directory. `denoize hardware` reports the first missing
prerequisite. The first CUDA model preparation can compile and cache kernels;
the host probe does not claim that every user-supplied ONNX graph is supported
by a GPU transform, so model preparation errors remain explicit.

### Offline input and device recommendation

`denoize recommend` analyzes the signal, compiled backends, locally installed
models authenticated by the embedded signed catalog, resource limit, and
current CPU/GPU availability without updating the catalog or model cache,
downloading a model, or contacting a service:

```sh
denoize recommend noisy.wav
denoize recommend noisy.wav --goal quality --calibrate
denoize recommend noisy.wav --goal speed --max-memory 256 \
  --max-gpu-memory 2048 --json
```

The command analyzes at most 12 seconds by default (configurable from 1 to 60).
WAV, FLAC, and Ogg Vorbis are analyzed through the bounded block decoder; other
supported formats use their existing whole-file decoder under the same
`--max-memory` ceiling and then analyze only the bounded prefix. Filesystem
inputs retain the regular-file and same-opened-object guarantees. Recommendation
still requires a filesystem input; `-` is reserved for bounded `--stream`
processing rather than recommendation.

`--calibrate` adds a fixed, hash-identified Classical Hi-Fi workload with one
warmup and three measured runs after its fixed scratch allowance fits the same
memory ceiling. The report records raw timings and median
baseline realtime headroom, then combines that evidence with documented
backend cost classes; it does not pretend to be a full-reference audio-quality
benchmark or an exact runtime prediction for a neural model. Every candidate
includes stable reason codes, eligibility, local model/runtime state, estimated
denoize-owned CPU and GPU memory, and the suggested explicit arguments. GPU
eligibility respects both `--max-gpu-memory` and a runtime-reported device limit
when one is available. Recommendation captures one read-only hardware snapshot
and does not create or test a CUDA kernel cache; actual processing revalidates
cache writability before preparing a model. Compact and pretty output use the
versioned
[`denoize-recommendation-v1`](schemas/denoize-recommendation-v1.schema.json)
contract. Backends that require a caller-supplied model path remain visible as
excluded candidates; the report does not disclose paths, so only configurations
that can be reproduced without inventing a model argument are auto-selected.
The analyzed-sample SHA-256 is a content fingerprint; remove it before sharing
a report when correlating the source audio would be sensitive.

### Native degradation diagnosis and no-reference assessment

**denoize diagnose** performs bounded, deterministic, network-free triage when
a clean reference is unavailable:

~~~sh
denoize diagnose damaged.wav
denoize diagnose damaged.wav --analysis-seconds 20 --pretty
denoize assess before.wav after.wav --json
~~~

The report separates additive noise, clipping, 50/60 Hz hum, clicks,
reverberation, bandwidth limitation, short dropouts, wind/plosive energy, and
codec risk. Each finding carries direct evidence, continuous severity,
confidence, and a recommended restoration action. **assess** can evaluate one
input or compare before/after quality while separately checking sample rate,
channel count, and presentation duration.

The native score and MOS proxy are triage estimates, not human MOS. Schema v1
always reports semantic fidelity as unassessed; it cannot prove preservation of
words, phonemes, speaker identity, language, or prosody and cannot authorize a
generative result by score alone. See [Native diagnostics](docs/diagnostics.md)
and the closed
[denoize-diagnostic-v1](schemas/denoize-diagnostic-v1.schema.json) and
[denoize-assessment-v1](schemas/denoize-assessment-v1.schema.json) contracts.

### Deterministic audio restoration

**denoize restore** provides conservative, model-free de-clipping,
prediction-residual de-clicking, harmonic de-hum, finite WPE de-reverberation,
and short wind/plosive repair:

~~~sh
denoize restore damaged.wav restored.wav \
  --report restoration-report.json --mask restoration-mask.json
denoize restore damaged.wav --detect-only \
  --operations declip,declick,dehum --pretty
~~~

The pipeline never changes decoded sample rate, channels, or frame count.
Detect-only mode is bit-exact and can export a complete channel/frame RLE mask
without writing audio. Apply mode uses conservative confidence gates, explicit
iteration and attenuation ceilings, bounded memory admission, atomic
no-clobber outputs, and a canonical operation order. Reports contain PCM and
mask digests, detected/changed counts, confidence, energy delta, warnings, and
closed per-operation evidence, but no filesystem path. Context padding is
separate from accepted damage and from samples actually replaced.

See [Deterministic restoration](docs/restoration.md), the closed
[report](schemas/denoize-restoration-report-v1.schema.json) and
[mask](schemas/denoize-restoration-mask-v1.schema.json) contracts, and the
[research and acceptance review](docs/restoration-research.md#stage-26--deterministic-restoration).

### Fail-closed universal speech restoration

**denoize universal** runs a signed runtime package v2 through the dedicated
48 kHz BSRNN spectral adapter. The discriminative `primary` path is the safe
default; hybrid and generative packages require both an `alternate` role and an
explicit experimental opt-in:

~~~sh
denoize universal degraded.wav restored.wav \
  --model-package urgent-bsrnn.dmp \
  --model-package-key publisher.pub \
  --report universal-report.json \
  --mask universal-mask.json \
  --max-memory 4096 --pretty
~~~

The package signature, provenance, exact graph interface, selected-runtime
resource profile, and numerical vectors pass before source inference. Clean
input bypasses the model. A private candidate is published only when geometry,
finite-sample, energy, peak, new-clipping, silence-injection, and native-quality
gates all pass; otherwise the decoded input is written unchanged. Reports bind
package/key/source/checkpoint and PCM/mask SHA-256 values without paths.

Signal gates do not prove preservation of words, phonemes, prosody, or speaker
identity. Model promotion therefore uses separately signed evidence covering 20
required demographic/material/degradation strata, nine metrics per stratum,
and human-listening thresholds. Upstream URGENT and UniPASE weights are not
bundled because their complete artifact-level training-data redistribution
chain has not been established. See [Universal restoration](docs/universal-restoration.md),
its [research audit](docs/restoration-research.md#stage-27--universal-speech-restoration),
and the closed [report](schemas/denoize-universal-restoration-report-v1.schema.json),
[mask](schemas/denoize-universal-restoration-mask-v1.schema.json), and
[promotion evidence](schemas/denoize-universal-promotion-evidence-v1.schema.json)
contracts.

### Fail-closed target-speaker extraction

**denoize target-speaker** extracts one enrolled speaker from a mixture through
a dedicated signed package v2 graph. It is offline and mono in Stage 29:

~~~sh
denoize target-speaker meeting.wav enrollment.wav target.wav \
  --model-package target-speaker.dmp \
  --model-package-key publisher.pub \
  --promotion-evidence promotion.json \
  --promotion-evidence-key evaluator-public-key.json \
  --report target-speaker-report.json \
  --max-memory 4096 --pretty
~~~

The graph must expose exactly one mixture input, one enrollment input, one
same-length audio output, and calibrated `absent`/`uncertain`/`present`
probabilities. Package components, provenance, graph names/shapes, numerical
vectors, and separately signed REAL-T/TS-SUPERB/absence promotion evidence are
verified before either audio file is decoded.

Only a confidently present target whose candidate passes geometry, finite,
energy, peak, clipping, presence, and evidence gates creates an audio file.
Absent, uncertain, and unsafe candidates create no audio and never fall back to
the mixture, silence, or an unverified voice. Enrollment working buffers are
zeroized immediately after inference; reports contain no enrollment samples,
embedding, digest, or path.

No checkpoint is bundled because the audited WeSep/REAL-TSE/MeanFlow candidates
do not yet provide a complete artifact-level redistribution and protected-
stratum evidence chain. See [Target-speaker extraction](docs/target-speaker.md),
the [paper and artifact audit](docs/restoration-research.md#stage-29--target-speaker-extraction),
and the closed [report](schemas/denoize-target-speaker-report-v1.schema.json)
and [promotion evidence](schemas/denoize-target-speaker-promotion-evidence-v1.schema.json)
contracts.

## Supported input formats

| Format | Decoder | Notes |
|--------|---------|-------|
| WAV/BWF | `hound` | 8–32 bit int / float; BWF metadata chunks are preserved for supported tags |
| RF64 | native RF64 reader | 64-bit-size PCM/WAVE, bounded chunk reads |
| AIFF/AIFC | `symphonia` | PCM and supported AIFC codecs |
| CAF | `symphonia` | PCM and ALAC/other supported CAF codecs |
| MP3 | `symphonia` + bounded `nanomp3` fallback (Pure Rust) | Xing/Info + LAME gapless trim, ID3v2, no resampling |
| M4A/AAC/ALAC | `oxideav-aac` + `symphonia` fallback | AAC-LC/ALAC decode with MP4 v0/v1 unity-rate edit-list timing; MP4 AAC-LC uses the 24-bit MPEG-4 buffer/sample safety ceiling and charges payload-proportional decoder work before access-unit allocation when `--max-memory` is set (ALAC is unaffected) |
| FLAC | `claxon` | Lossless FLAC |
| Ogg Opus/Vorbis | `opus` + `ogg` / `symphonia` | Mono/stereo; native sample rate decode |

### Output formats

| Format | Encoder | Notes |
|--------|---------|-------|
| WAV | `hound` | Lossless; preserves bit depth |
| MP3 | `shine-rs` (Pure Rust) | `--mp3-bitrate` (default 192 kbps) |
| M4A | `oxideav-aac` + MP4 mux | GitHub/source builds; positive `--m4a-bitrate` (default 192 kbps) |
| FLAC | `flacenc` | Lossless, pure Rust |
| Ogg Opus | `opus` + `ogg` | 128 kbps, mono/stereo |

MP3 inputs with a Xing/Info header and a compatible LAME, Lavf, or Lavc
extension are decoded over their exact signalled presentation span: encoder
delay and end padding are not exposed to the denoising pipeline. Untagged raw
MP3 has only MPEG-frame timing, so its decoded duration remains frame-rounded.
Symphonia remains the primary decoder; a fixed-size streaming compatibility
fallback is used only for its exact invalid-bit-reservoir error on untagged,
contiguous Layer III frames, and never for files carrying gapless timing.
The built-in Shine encoder completes its final bit cache and emits at least two
MPEG frames for short-clip interoperability; clips shorter than that minimum
therefore contain trailing encoded silence.

M4A AAC-LC and ALAC inputs are rendered on the container presentation
timeline. Unity-rate v0/v1 edit lists may trim or splice media and insert
leading or interior silence; unsupported rates and malformed timing are
rejected instead of returning mis-timed PCM.

Channel order is kept planar and unchanged through WAV/FLAC and denoising. The
standard layouts mono, stereo, 2.1, quad, 5.0, 5.1, 6.1, and 7.1 are reported
when their channel count is recognized. MP3, M4A, and ADTS AAC encoders in the
current release accept only mono/stereo; surround input is rejected instead of
being mixed implicitly. Use `--downmix stereo` when a documented, explicit
surround-to-stereo render is intended (LFE is not copied into the full-range
stereo pair). WAVE_FORMAT_EXTENSIBLE speaker masks are read, preserved, and
written for multichannel WAV files; `--report` also shows each channel's
azimuth/elevation pan coordinate. A non-standard but valid mask is used for
position-aware downmixing instead of being guessed from the channel count.

Stereo processing can be selected with `--channels mid-side`. This uses a
reversible, energy-preserving Mid/Side transform (`M=(L+R)/sqrt(2)`,
`S=(L-R)/sqrt(2)`) and reconstructs the original channel order and speaker
metadata after denoising.

```sh
# MP3 / M4A input and output — no manual ffmpeg conversion
denoize noisy.mp3 clean.mp3 -p hifi
denoize noisy.m4a clean.m4a -b deepfilter
denoize noisy.wav clean.wav --mp3-bitrate 320

# User-supplied waveform model: [1, samples] or [1, 1, samples]
denoize noisy.wav clean.wav -b onnx \
  --onnx-model model.onnx --onnx-rate 16000

# Signed model + license + frontend/tensor/resource contracts
denoize noisy.wav clean.wav -b onnx \
  --model-package voice-cleaner.dmp \
  --model-package-key vendor-model.pub

# Inspect either signed package version; v2 also reports named I/O, state,
# latency, precision profiles, provenance, and numerical-vector coverage
denoize models package inspect voice-cleaner.dmp vendor-model.pub

# Official MP-SENet checkpoint converted with scripts/export-mpsenet.py
denoize noisy.wav clean.wav -b mpsenet \
  --onnx-model mp-senet-vb.onnx --onnx-rate 16000

# ESPnet BSRNN xtiny checkpoint converted with scripts/export-bsrnn.py
denoize noisy.wav clean.wav -b bsrnn \
  --onnx-model bsrnn-xtiny.onnx --onnx-rate 48000

# ClearerVoice MossFormer2 48 kHz model
denoize noisy.wav clean.wav -b mossformer2 \
  --onnx-model mossformer2-se-48k.onnx --onnx-rate 48000

# Official SGMSE+ VoiceBank model (30-step quality sampler)
denoize noisy.wav clean.wav -b sgmse \
  --onnx-model sgmse-vb.onnx --onnx-rate 16000 --sgmse-profile quality

# Verified official GTCRN model (manual model path is unnecessary afterwards)
denoize models install gtcrn
denoize models verify all
denoize models update gtcrn
denoize models remove gtcrn
denoize noisy.wav clean.wav -b gtcrn

# Stereo coupling, pipes, metrics, and directory batches
denoize stereo.wav clean.flac --channels linked
denoize surround-5.1.wav stereo.mp3 --downmix stereo
cat noisy.wav | denoize - - > clean.wav
denoize metrics reference.wav clean.wav --json
denoize recordings/ cleaned/ --batch
```

To prepare the pinned official MP-SENet VoiceBank model:

```sh
git clone https://github.com/yxlu-0102/MP-SENet.git
git -C MP-SENet checkout 89932cfe90d1dacb8e170e4a331d762462c21792
python3 -m pip install torch onnx onnxscript pesq joblib matplotlib
python3 scripts/export-mpsenet.py \
  --repo MP-SENet \
  --checkpoint MP-SENet/best_ckpt/g_best_vb \
  --output mp-senet-vb.onnx
```

The VoiceBank graph is about 9 MiB and expects 16 kHz audio. On the reference
x86-64 Linux host, a two-second mono speech fixture took 43.67 seconds after
model loading and the complete process used 410,048 KiB maximum RSS. Run the
pinned real-speech quality gate after conversion:

```sh
python3 scripts/validate-mpsenet.py \
  --denoize target/release/denoize \
  --model mp-senet-vb.onnx
```

To prepare the pinned ESPnet BSRNN xtiny model (CC-BY-4.0):

```sh
curl -L \
  'https://huggingface.co/wyz/vctk_bsrnn_xtiny_causal/resolve/59e1f2263b7946b1970a222d1beef9adc5a67eaa/exp_vctk/enh_train_enh_bsrnn_xtiny_raw/58epoch.pth' \
  -o 58epoch.pth
echo 'e3cb771a452e0503144af74720b476e81b57f518b789b37ba2c253c6cc22d70b  58epoch.pth' \
  | sha256sum -c -
python3 -m pip install torch onnx onnxruntime
python3 scripts/export-bsrnn.py \
  --checkpoint 58epoch.pth \
  --output bsrnn-xtiny.onnx \
  --verify
```

The adapter resamples to 48 kHz and reproduces the published model's
variance normalization, centered 960-point Hann STFT with a 480-sample hop,
whole-utterance recurrent inference, and inverse STFT. The converted model is
about 2.4 MiB. On a release build on the project reference x86-64 Linux host,
the fixed two-second regression fixture took 1.58 seconds (1.3x realtime) and
used 44,628 KiB maximum RSS. Runtime and memory grow with utterance length.

Run the reproducible real-speech quality gate after conversion:

```sh
python3 scripts/validate-bsrnn.py \
  --denoize target/release/denoize \
  --model bsrnn-xtiny.onnx
```

To prepare the pinned Apache-2.0 MossFormer2 SE 48 kHz model:

```sh
git clone https://github.com/modelscope/ClearerVoice-Studio.git
git -C ClearerVoice-Studio checkout 6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61
curl -L \
  'https://huggingface.co/alibabasglab/MossFormer2_SE_48K/resolve/eff8c97925c8bec812af707814b3e5d777fd4503/last_best_checkpoint.pt' \
  -o last_best_checkpoint.pt
echo '03692b9f773bbd6bb43b9c5a41f96b1e28affd66e13796b7bec66ad3d8b227c6  last_best_checkpoint.pt' \
  | sha256sum -c -
python3 -m pip install torch onnx onnxruntime numpy einops rotary-embedding-torch
python3 scripts/export-mossformer2.py \
  --repo ClearerVoice-Studio \
  --checkpoint last_best_checkpoint.pt \
  --output mossformer2-se-48k.onnx \
  --verify
```

The adapter uses 48 kHz audio, 40 ms Kaldi fbank frames at an 8 ms shift,
first- and second-order deltas, a non-centred 1,920-point symmetric-Hamming
STFT, and the official four-second/three-second-stride edge-discard
reconstruction. The converted graph is about 217 MiB. On the reference
x86-64 Linux host, a four-second mono fixture took 7.74 seconds and used
483,400 KiB maximum RSS in a release build. Model weights are not bundled.

Run the pinned real-speech quality gate after conversion:

```sh
python3 scripts/validate-mossformer2.py \
  --denoize target/release/denoize \
  --model mossformer2-se-48k.onnx
```

To prepare the pinned MIT-licensed SGMSE+ VoiceBank+DEMAND model:

```sh
git clone https://github.com/sp-uhh/sgmse.git
git -C sgmse checkout 1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e
curl -L \
  'https://huggingface.co/sp-uhh/speech-enhancement-sgmse/resolve/b6485214b3662a7f90309f397cacf1384046783c/train_vb_29nqe0uh_epoch%3D115.ckpt?download=true' \
  -o 'train_vb_29nqe0uh_epoch=115.ckpt'
echo 'e3875747b5646092d5c556bae68e5af639e2c1f45f009c669f379cd4d415cbd8  train_vb_29nqe0uh_epoch=115.ckpt' \
  | sha256sum -c -
python3 -m pip install torch onnx onnxruntime numpy
python3 scripts/export-sgmse.py \
  --source sgmse \
  --checkpoint 'train_vb_29nqe0uh_epoch=115.ckpt' \
  --output sgmse-vb.onnx \
  --verify
```

The adapter reproduces the official noisy-peak normalization, centered
510-point periodic-Hann STFT with a 128-sample hop, complex square-root
spectrum transform, and OUVE predictor/corrector sampler. The explicit
quality/speed choice is the upstream 30 reverse steps with one ALD corrector
step (`snr=0.5`), or 60 score-network evaluations. The graph is about 252 MiB
and weights are not bundled. On the reference x86-64 Linux host, the pinned
two-second mono fixture took 737.92 seconds and used 1,204,648 KiB maximum RSS
in a release build. This backend prioritizes generative quality rather than
interactive speed.

Run the pinned quality gate after conversion (expect a long CPU run):

```sh
python3 scripts/validate-sgmse.py \
  --denoize target/release/denoize \
  --model sgmse-vb.onnx
```

## Quick start

```sh
cargo build --release --features full

# Best classical quality
./target/release/denoize noisy.wav clean.wav -p hifi

# RNNoise AI backend
./target/release/denoize noisy.wav clean.wav -b rnnoise

# DeepFilterNet v3 AI backend
./target/release/denoize noisy.wav clean.wav -b deepfilter

# Advanced DSP options
./target/release/denoize noisy.wav clean.wav \
  --window kaiser --kaiser-beta 10 \
  --multiband --perceptual --postfilter \
  -a specsub-nl -s 0.5

# Principal periodic DPSS/Slepian taper (NW must be in (0, 8])
./target/release/denoize noisy.wav clean.wav \
  --window dpss --dpss-nw 3.0
```

For DPSS, `NW` is the time-bandwidth product and defaults to `3.0`. Increasing
it widens the main lobe and concentrated frequency band while strengthening
the taper toward the frame edges. DPSS is a classical-backend STFT option; the
equivalent TOML is:

```toml
backend = "classical"
window = "dpss"
dpss_nw = 3.0
```

### Managed model downloads

Every build contains a versioned model catalog and trust-root policy. Remote
catalog updates are accepted only after detached-minisign verification against
the active root, strict schema and expiry validation, and monotonic
sequence/rollback checks. Trust-root rotations advance exactly one version and
must satisfy distinct-signature thresholds from both the current and candidate
root; catalog signing keys can be given closed sequence windows or explicit
revocation cutoffs, while rotations cannot weaken the active expiration policy.
Model
installs and updates then use the active catalog's exact size and SHA-256 while
supporting explicit network policy, authenticated mirrors, resumable transfers,
and air-gapped local files. Run `denoize models --help` for the dedicated
command reference.

```sh
# Inspect, update, or air-gap import the signed catalog.
denoize models catalog status
denoize models catalog update
denoize models catalog import catalog-v1.json catalog-v1.json.sig

# Inspect, rotate, or recover catalog trust. Rotations use a JSON bundle of
# detached signatures from the current and candidate root key sets.
denoize models catalog trust status
denoize models catalog trust import trust-root-v2.json trust-root-v2.signatures.json
denoize models catalog trust recover
# Only after correcting an accidental future system-clock jump:
denoize models catalog trust reset-time-floor

# Verify and import one release bundle without opening a network connection.
sha256sum --check denoize-models-v0.52.0.dmb.sha256
denoize models bundle inspect denoize-models-v0.52.0.dmb
denoize models bundle import denoize-models-v0.52.0.dmb

# Diagnose the whole cache without changing model data. Repair known packages,
# then preview and apply removal of stale denoize-owned state.
denoize models doctor
denoize models snapshot --pretty
denoize models repair all
denoize models prune --dry-run
denoize models prune

# Offline mode never opens a network connection.
denoize models install gtcrn-dns3 --offline

# Override the source and proxy for one model.
denoize models update gtcrn-dns3 \
  --url https://models.example.net/gtcrn.onnx \
  --proxy http://proxy.example.net:8080

# Read origin credentials from the environment, never a CLI secret value.
export MODEL_ACCESS_TOKEN='...'
denoize models install gtcrn-dns3 --bearer-token-env MODEL_ACCESS_TOKEN

export MODEL_BASIC_PASSWORD='...'
denoize models install gtcrn-dns3 \
  --basic-user buildbot --basic-password-env MODEL_BASIC_PASSWORD

# Ignore all proxy settings, or install an already transferred local file.
denoize models update gtcrn-dns3 --no-proxy
denoize models install gtcrn-dns3 --from /media/models/gtcrn_simple.onnx
```

The corresponding defaults are `DENOIZE_MODEL_OFFLINE`,
`DENOIZE_MODEL_URL`, `DENOIZE_MODEL_CATALOG_URL`, `DENOIZE_MODEL_PROXY`,
`DENOIZE_MODEL_BEARER_TOKEN`, `DENOIZE_MODEL_USERNAME`, and
`DENOIZE_MODEL_PASSWORD`. Standard `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`,
and `NO_PROXY` variables (including lowercase variants) are used when no
denoize-specific proxy override is active. `--proxy` selects an explicit
proxy; `--no-proxy` and an empty `DENOIZE_MODEL_PROXY` force a direct
connection.

Interrupted transfers are retained in a `.part` sidecar and resumed with HTTP
range requests. Saved `ETag` or `Last-Modified` validators and each
`Content-Range` are checked before appending; changed objects, malformed range
responses, or an unverified `416` response cause a clean restart. Every managed
model candidate must match both the catalog's exact byte length and SHA-256
before use: this includes fresh or resumed downloads, alternate `--url`
sources, `--from` imports, completed partials, and files already in the cache.
An update keeps the current verified model until its replacement is ready.
Each installation also receives content-addressed provenance that binds the
artifact to its catalog sequence, digest, and signing key, and records its
installation-time catalog origin and source. Existing verified caches are
migrated lazily; mismatched provenance fails verification. `denoize models info MODEL` reports these fields and the
pinned length as an unscaled decimal `size-bytes` value. This per-model
integrity bound is not an aggregate cache quota.

Official releases additionally provide a signed offline `.dmb` bundle for
closed networks. Its bounded, length-delimited format carries the exact catalog,
detached signature, trust root, every catalog model, upstream license text, and
source-provenance JSON. The manifest contains no extraction paths and the format
has no compression layer. `models bundle inspect` authenticates every byte
without changing catalog or cache state; `models bundle import` performs the
same full preflight before activating the catalog and atomically installing any
missing model. Both commands accept only a regular file and perform no network
I/O. A model installed this way records the bundle SHA-256 in local installation
provenance.

Catalog expiry and rollback rules still apply offline. The import may advance
the monotonic catalog floor before a later storage error; it rolls back model
artifacts created by that invocation, and retrying the same authenticated bundle
is safe. Existing valid models are retained. See the
[managed-model guide](docs/models.md#signed-offline-bundles) for the operator
layout used by `models bundle create` and the complete transaction contract.

Catalog sequence 1 is a compatibility exception because it predates signed
timestamps. The embedded v1 trust policy requires every later sequence to carry
`issued_at_unix_seconds` and `expires_at_unix_seconds`, with at most 180 days of
validity. The root itself expires at its displayed Unix time. denoize persists
the greatest trusted wall-clock value it has observed, so setting the system
clock backwards cannot reactivate expired authority. Expiry, a newly tightened
timestamp policy or key window, or `revoked_at_sequence` stops new installs,
updates, local imports, and artifact reacquisition; already installed bytes
remain usable and verifiable, and local provenance-only repair, diagnosis,
pruning, and removal remain available.

`models catalog trust recover` replaces corrupt or incomplete cached trust
metadata only with the root compiled into the running binary. It never lowers a
valid newer root or the catalog rollback floor; those cases require the missing
signed chain or a newer denoize binary. A newer embedded root is therefore the
independent emergency recovery channel. By default recovery also preserves the
greatest trusted time already observed. After correcting an accidental future
system-clock jump, the explicit `models catalog trust reset-time-floor` command
resets only that clock floor to the current system time while retaining the
active signed root and chain. It cannot lower either the trust-root version or
catalog rollback floor, and refuses the reset unless the active root is valid at
the corrected current time. Inspect the recorded value with `models catalog
trust status` before using the command.

`models doctor` inventories every active-catalog package plus cache sidecars
and orphan entries without changing artifacts, provenance, or download state.
An optional package that was never installed is reported as `missing` but does
not make a fresh cache unhealthy. Corrupt bytes, missing or mismatched
provenance, incomplete/stale downloads, unsafe entries, and catalog-orphaned
packages are reported separately. `models verify MODEL|all` remains the strict
package verification command.

`models repair MODEL|all` rebuilds provenance locally when verified bytes are
already present; otherwise it uses the same offline, source, proxy, and
authentication policy as install/update. Replacement bytes are staged and
verified before the old artifact is atomically replaced. `models prune
--dry-run` lists exact removable paths. Applying `models prune` deletes stale
sidecars, superseded provenance, and old package directories only when their
content-addressed provenance, artifact digest/size, and directory layout all
match denoize-managed state. Unknown data and symlinks, devices, or other
special entries are reported and retained.

`models snapshot [--json] [--pretty]` emits the stable
`denoize-automation-v1` document without network access. It binds the active
catalog and trust root to cache health, every expected model identity, validated
installation provenance, and the processing recipe ABI in one
generation-checked snapshot. Normal `--json` processing and batch NDJSON records use
`denoize-cli-output-v1` and expose each finite-file recipe digest. The desktop
model library exports the same automation snapshot atomically. See the
[stable JSON contract](docs/json.md) and its versioned schemas for field and
compatibility rules.

HTTPS model connections, including those tunneled through an HTTP CONNECT
proxy, use the operating system trust store. CLI Bearer tokens and Basic
passwords are accepted through environment variables, and diagnostics redact
credentials, query strings, and fragments. Signed `--url` values and proxy
credentials can still leak through process listings and shell history, so use
protected environment injection when that matters. See the
[managed-model guide](docs/models.md) for option combinations,
proxy precedence, and resume validation details.

### Long recordings with bounded memory

For long recordings, use the stateful streaming path. It keeps only a
fixed-size input block plus bounded decoder, backend overlap, recurrent,
resampler, and encoder state in memory instead of loading the whole file:

```sh
./target/release/denoize long-noisy.wav long-clean.wav --stream
```

`--stream` accepts content-detected WAV, FLAC, Ogg Vorbis, granule-aware Ogg
Opus, gapless MP3, frame-aware ADTS AAC, and edit-aware M4A AAC/ALAC input. It
writes WAV, FLAC, Ogg Opus, MP3, M4A AAC, or ADTS AAC with the compiled
Classical, RNNoise, DeepFilterNet, MossFormer2, and GTCRN backends, including
independent, linked-stereo, and mid/side channel modes. MossFormer2 and GTCRN
use an explicit `--onnx-model` or their installed managed model; DeepFilterNet
uses its embedded graph. A regular-file destination is staged, decoded
end-to-end to verify its codec, geometry, and presentation duration, then
published atomically. Supported metadata is retained unless `--no-metadata` is
selected. Bounded VAD preserves the presentation timeline across backend
latency. `--loudness` performs a fixed-memory analysis pass over an anonymous
PCM spool, then applies one constant gain during the verified encoding pass.
The default block size is 8192 frames; use `--stream-frames N` (1–1,048,576) to
trade latency and working memory for throughput. Noise profiling retains only
a bounded leading segment before output begins. Stream resource arithmetic is
checked from the input header, and the processor is constructed before an
output or temporary file is staged.

`-` selects stdin or stdout for `--stream`. Stdin is copied into an anonymous
regular-file spool before parsing, preserving one authoritative seekable input
without retaining all encoded bytes in RAM. Stdout accumulates bounded PCM and
encoded anonymous spools, applies metadata and optional two-pass loudness,
verifies the completed encoded stream, then copies it to the sink. Stdin and
stdout share the `--max-temp-space` allowance (1 GiB by default); supported
input metadata is preserved unless `--no-metadata` is selected. A sink error
can leave partial external bytes because a pipe has no atomic rename. `--resume`
intentionally rejects stdin/stdout because their anonymous spools cannot
survive process restart.

Library callers use `AudioStreamReader::from_reader_with_limits` for a plain
`Read` source and `SpooledAudioStreamWriter::new_with_limits` for a plain
`Write` sink. `StreamSpoolLimits` bounds encoded input bytes or, for output,
the simultaneous PCM spool, encoded spool, and codec auxiliary files. The
seekable `AudioStreamWriter` remains the lower-overhead choice when the caller
already owns a private `Write + Seek` transaction.

Add `--resume` to make a long stream restartable. The CLI periodically
synchronizes a private append-only checkpoint journal and an interleaved `f64`
PCM spool beside the destination. After interruption it deterministically
replays the same opened input to the last durable boundary, verifies the saved
PCM digest, restores backend state, and continues. The input bytes, effective
recipe, model bytes, source format, channel geometry, and block size are bound
to the checkpoint. A mismatch is preserved and rejected unless `--force`
explicitly discards it. Codec delay, Ogg granules, and M4A edit lists are
applied before presentation PCM reaches a durable boundary. The final encoded
output is staged, decoded for verification, and committed atomically; success
removes the journal and PCM spool while retaining the reusable lock file. The
journal records the exact verified staged-output fingerprint before
publication. If the process exits after the atomic commit but before receipt
publication or sidecar cleanup, the next identical resume verifies the
destination, reports a `skip/completed` plan when requested, can publish the
matching signed receipt, and removes the stale data sidecars without
reprocessing. A changed destination is preserved and rejected unless `--force`
resets the checkpoint. The PCM spool, staged encoded file, encoder auxiliary
data, and retained metadata are all charged to `--max-temp-space`.

Filesystem inputs are opened once per processing phase as validated regular
files. Size estimation, probing, decoding, and metadata reads within that
phase use the same opened filesystem object, so replacing the pathname cannot
silently mix bytes from two inputs. FIFOs, directories, and device files are
rejected before an audio parser or output staging step runs.

For the normal (decoded, non-streaming) path, `--max-memory MB` caps requested
denoize-owned decoded PCM capacities and explicitly accounted codec scratch
buffers, in addition to the conservative input-size preflight and final
decoded-working-set check. `--max-process-memory MB` adds weighted admission
across all active workers and retained model sessions. Batch preflight reserves
each decoder's complete configured allowance, rechecks it after model loading,
and starts a worker only when its RAM, temporary-output, CPU, and GPU request
fits atomically. Actual retained metadata is charged conservatively as well.

`--max-temp-space MB` bounds aggregate staged-output reservations (including a
restartable stream's PCM spool) and verifies the final staged length before
publication; it is not a filesystem quota.
`--max-gpu-jobs N` (default 1) serializes or bounds accelerated workers, while
`--max-gpu-memory MB` applies conservative model and transfer reservations.
Those GPU counters are not driver-reported VRAM usage. Internal allocations
made inside third-party codec or model runtimes can still fall outside the
cooperative counters, and allocator capacity rounding means they are not an
allocator-exact process RSS limit.

For an OS-enforced CLI boundary, add `--isolate`. Processing runs in a child;
on Unix, `--max-process-memory` becomes an `RLIMIT_AS` address-space ceiling,
and on Windows it becomes a Job Object process-memory ceiling. The parent
survives a decoder/model abort and publishes no staged output from a failed
child. Without a process-memory value, isolation still contains child failure
but does not invent a memory ceiling. Desktop file and batch jobs always use
the equivalent supervised child boundary; desktop live audio remains in
the application process under the cooperative governor.

FLAC and Ogg structure is also checked with finite block,
packet, page, stream, item, and aggregate metadata limits before a decoder can
materialize it—even with `--no-metadata`. When tags are preserved, their
retained payload budget is derived from the memory left after the decoded PCM
working set; the same limit is enforced again while writing the staged output.
The default limits remain finite when `--max-memory` is omitted.

The per-input limit applies to regular-file inputs/workers; stdin retains its
separate bounded WAV buffering path. Process admission makes `--jobs` an upper
bound rather than a promise that every worker can run simultaneously. Batch
probing, decode, model preparation, and metadata validation all finish before
the output directory or staging files are created. A streaming job stays
bounded by its block size, decoder allowance, and denoiser state, and metadata
uses a conservative share of the remaining budget:

```sh
denoize large.mp3 cleaned.wav --max-memory 1024
denoize recordings cleaned --batch --jobs 8 --max-memory 512 \
  --max-process-memory 2048 --max-temp-space 4096 --max-gpu-jobs 1
denoize input.m4a output.flac --max-process-memory 1024 --isolate
denoize long-noisy.wav long-clean.wav --stream --stream-frames 4096 --max-memory 64
denoize long-noisy.flac long-clean.wav --stream --resume \
  --stream-frames 4096 --max-memory 64 --max-temp-space 8192
```

## DAW plug-ins

The `denoize.clap` bundle exposes two effects with independent stable IDs. The
fixed-memory DSP effect is `org.penguin425.denoize`. It supports mono and stereo
ports, `f32` and `f64`
audio, in-place and out-of-place buffers, sample-accurate automation, bypass,
stereo linking, dry/wet mix, and output gain. Activation allocates all delay
and DSP state up front. The audio callback performs no allocation, locking,
filesystem or network I/O, or system calls.

The plug-in reports the fixed `fixed-10ms-v1` latency policy to the host. The
exact frame count is `ceil(sample_rate * 0.010)`: 441 frames at 44.1 kHz, 480
at 48 kHz, and 960 at 96 kHz. `denoize plugin latency` independently sends an
impulse through the bypassed `f64` processor and fails if the measured first
output frame differs from the reported value:

```sh
denoize plugin info --pretty
denoize plugin latency --sample-rate 48000 --pretty

denoize plugin preset create speech speech.json --name "Dialogue" --pretty
denoize plugin preset validate speech.json --json
denoize plugin session create speech.json session.json --stereo --pretty
denoize plugin session validate session.json --json
```

The model-backed effect is `denoize Neural`
(`org.penguin425.denoize.neural`). Install the pinned managed `gtcrn-dns3`
graph before a DAW activates it; the host process never downloads a model:

```sh
denoize models install gtcrn
denoize plugin neural info --sample-rate 48000 --pretty
denoize plugin neural latency --sample-rate 48000 --pretty
denoize plugin neural session create neural.json --stereo --fallback delayed-dry --pretty
denoize plugin neural session validate neural.json --json
```

Neural inference, graph preparation, resampling, and recurrent state run on one
permanent worker. The callback only copies through preallocated blocks and
bounded lock-free queues; it performs no model work, allocation, locks, waits,
I/O, network access, or logging. Its fixed policy is
`fixed-24x10ms-worker-v1`: `24 * ceil(sample_rate * 0.010)`, or 240 ms at
ordinary rates. Finite fractional CLAP sample rates are supported. A late,
invalid, or missing result uses latency-aligned dry audio by default; last-safe
gain and silence require an explicit parameter choice. The advertised
reference input is reserved for later typed target-speaker/AEC semantics.

Portable presets use
[`denoize-daw-preset-v1`](schemas/denoize-daw-preset-v1.schema.json). Complete
session state uses
[`denoize-daw-session-v1`](schemas/denoize-daw-session-v1.schema.json) and
binds the plug-in ID, latency policy, mono/stereo port configuration, and every
parameter. Both formats reject unknown fields and future versions, are bounded
to 64 KiB, read only regular non-symlink files, and publish atomically with
no-clobber as the default. CLAP host state serializes the same deterministic
session document, so standalone files and DAW restoration cannot drift into
separate state formats.

Neural state uses
[`denoize-neural-daw-session-v1`](schemas/denoize-neural-daw-session-v1.schema.json)
and additionally binds the exact model ID, graph SHA-256, overload fallback,
and scheduler policy. It is also path-free, closed, bounded, non-symlink, and
atomically no-clobber. See [Neural DAW plug-in](docs/neural-plugin.md) for the
scheduler, model trust boundary, state contract, evidence, and limitations.

From a repository checkout, build a local CLAP plug-in with
`cargo build --release -p denoize-clap`. On Linux,
copy `target/release/libdenoize_clap.so` to `~/.clap/denoize.clap`. Tagged
releases provide ready-to-copy archives for Linux x86-64, macOS Intel and Apple
Silicon, and Windows x86-64. macOS archives contain a complete
`denoize.clap` bundle. After copying it to a standard CLAP directory, restart
the DAW or run its plug-in rescan. CI and the tagged release workflow verify
both descriptors with the pinned official CLAP validator 0.4.1: 81 tests, 68
applicable passes, no failures or warnings, and 13 capability-based skips.

v0.79.0 adds one accessible native embedded editor for both CLAP descriptors.
Every visible control remains a host parameter, with keyboard navigation,
visible focus, deterministic software rendering, native AccessKit adapters,
bounded lock-free UI automation, and complete generic-host fallback when the
custom window API is unsupported or creation fails. The signed Linux X11 gate
opens and renders both editors in a real host, injects a bypass click, verifies
the exact three-event automation gesture, and exercises resize and lifecycle
rejection paths. See [Accessible plug-in editor](docs/plugin-editor.md) for the
supported window APIs, accessibility contract, evidence, and explicit limits.

VST3 3.8 bundles are available from v0.78.1. They statically adapt the same two
descriptors through exact pinned CLAP-wrapper, CLAP SDK, and VST3 SDK revisions,
so they cannot load a same-named external CLAP binary. The official 3.8.1
validator gate requires 94/94 passes. A pinned Ardour 8.4 headless real-host
gate also discovers, inserts, processes, saves, reloads in a fresh process, and
tears down both descriptors at 48 kHz. Release evidence includes a signed host
matrix and both bound logs. See [VST3 plug-in](docs/vst3-plugin.md) for
installation, reproducible build details, and the explicitly unclaimed f64,
custom-view, and proprietary-host cases. The v0.79.0 editor evidence is native
CLAP-only and does not silently widen those VST3 claims. AUv3 and LV2 remain
separate gates.

## Desktop app

The Tauri desktop app exposes single-file denoising, batch conversion, portable
project timelines, DAW plug-in state management, native degradation diagnosis,
no-reference before/after assessment, signed quality comparison, and model
management without sending audio off the computer. Its
default build includes every backend in the repository's `full` feature set;
FDK-AAC remains an explicit opt-in because of its separate licensing terms.
ONNX-based backends expose model-file, model-rate, and SGMSE quality controls
when selected; managed GTCRN weights are resolved automatically after install.
The resource panel applies aggregate RAM and staged-output admission, a
conservative GPU-memory reservation, and a GPU-worker concurrency limit to
single-file, batch, and live jobs. Final file, batch, and short preview work run
in supervised child processes so a decoder or model abort cannot directly take
down the UI. Unix workers disable core dumps and die with their parent; Windows
workers start behind a gate, enter a kill-on-close Job Object before processing,
and apply the configured process-memory ceiling. Without an explicit ceiling,
isolation still contains a worker crash but does not claim allocator-exact RSS.
Final workers exchange bounded nonce-authenticated progress records. A shared
commit/cancel fence prevents a cancelled or rejected worker from publishing a
later output.
The model manager shows signed-catalog identity and installed provenance and
can update the catalog or atomically export the stable automation JSON. Its
offline, alternate-source, proxy/direct,
authentication, and local-file controls are session-only. Bearer tokens and
Basic credentials are cleared after an operation starts, and none of these
download overrides are included in saved settings, named presets, or
CLI-compatible imports and exports.
Desktop batches accept files or folders, preserve relative paths, run with a
configurable worker count, continue after individual failures, and can resume
from the same `.denoize-state` journal used by the CLI in the output directory.
Single-file processing also provides a bounded non-destructive audition flow.
It renders at most 30 seconds and up to three candidate recipes, exposes
loudness-matched original, processed, and removed-signal audio, supports
keyboard seeking and looping, and includes a blind A/B choice. A selected
recipe is persisted locally, but applying it to a final job still requires the
same source fingerprint, effective backend, output format, and recipe. Restored
choices must be rendered again before use. The public
[`denoize-presentation-region-v1`](schemas/denoize-presentation-region-v1.schema.json)
locator stores exact presentation ticks rather than encoded packet time. A
cancelled or failed preview publishes no final output or restart state and its
private temporary directory is removed.
Desktop settings are restored automatically, can be stored as named presets,
and can be imported or exported as CLI-compatible TOML. Recent input files are
kept locally for quick reuse. The single-file and batch views also expose a
reproducibility mode that serializes processing and uses stable model seeds.
The single-file view can also run the bounded WAV, FLAC, Ogg Vorbis/Opus, MP3,
ADTS AAC, and M4A AAC/ALAC input path, choose any supported encoded output and
block size, preview the read-only v2 execution plan, publish a signed receipt,
and enable the same durable restart checkpoints as the CLI.
Audio files and folders can be dropped onto the single-file or batch input
zones; output folders have dedicated drop targets. Multiple audio files switch
the app to batch mode automatically.
The realtime page routes a selected capture device through a low-latency
backend to a playback device, with independent input/output sample rates,
adaptive clock correction, bounded playback priming, and explicit start/stop
controls. It reports queue depth, estimated capture-to-playback latency, drift
correction, underrun/overflow frames, dropped chunks, and reconnect attempts.
If a device disappears, the session retries the selected name (or the current
system default) for the configured finite recovery window. Live sessions
support the live-capable Classical, RNNoise, and GTCRN backends; other backends
are rejected before capture or playback starts. GTCRN requires its managed
model to be installed (or an explicit model path) and keeps one optimized graph
with independent recurrent state per processed channel. Headphones help
prevent acoustic feedback.

The DAW plug-in page displays the exact reported and measured latency for both
DSP and Neural effects at a selected sample rate. It also shows the pinned
neural model/install state, bounded queue and overload policy, loads the three
DSP factory presets, edits every stable DSP parameter, and imports or atomically
exports portable preset and deterministic session JSON. All validation and file
publication remain in Rust; the WebView does not implement a second parser or
state contract.

The Project page loads and validates source-bound manifests, selects a linear
sample-accurate timeline, previews or saves its exact execution plan, assembles
a verified float WAV, and optionally publishes a signed project receipt. It can
also create, inspect, and import offline project bundles. Source audio and model
package payloads are excluded by default and require explicit positive MiB
limits; project operations are serialized and existing destinations are never
replaced.

File and batch jobs keep owner-private recovery records while their exact
denoize staging files are live. After a crash, the desktop can retry the saved
request or discard the record and only those verified private stage files;
existing outputs and batch restart journals are never deleted by recovery.
Startup cleanup preserves previews owned by another running desktop instance.
The diagnostics export is bounded, owner-private, and no-clobber. It contains
only schema-defined capability, limit, recovery-count, and event-code fields;
paths, URLs, credentials, device names, free-form errors, and audio are not
recorded.

The desktop interface can switch between Japanese and English without a
restart and stores only that locale preference. Static interface text and
application-owned status messages are checked against the translation catalog
during the frontend build. Rust command failures cross the IPC boundary as a
stable code, bounded parameters, and a preserved technical detail; the WebView
localizes the code instead of treating backend prose as UI copy. Navigation,
preview candidates, seek controls, progress, and level meters expose keyboard
and ARIA semantics; visible focus, reduced-motion, and forced-colors modes are
also supported. CI starts the application in a real Linux WebKit WebView and
exercises control names, tab/panel state, keyboard navigation, the skip link,
locale changes, live-region semantics, and structured-error IPC.

```sh
cd apps/desktop
npm ci
npm run tauri -- dev

# Static UI contracts plus the real-WebKit accessibility run
npm run check:ui
npm run build
npm run test:a11y:webview

# Build a platform-native installer/package
npm run tauri -- build

# Optional FDK-AAC selector
npm run tauri -- build --features fdk-aac-encoder
```

Linux development requires the WebKitGTK 4.1 and GTK 3 development packages.
For Ubuntu 24.04 or later:

```sh
sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev patchelf xvfb
```

## Prebuilt binaries

Each [GitHub Release](https://github.com/penguin425/denoize/releases) contains
prebuilt `full`-feature binaries for:

- Linux x86-64
- macOS Intel and Apple Silicon
- Windows x86-64

The same release also includes CLAP and VST3 plug-in archives for those four target
architectures. Extract the matching `denoize-plugin-<tag>-<target>` archive and
copy its `denoize.clap` file or macOS bundle into a standard CLAP directory.
For VST3, extract `denoize-vst3-<tag>-<target>` and copy `denoize.vst3` into the
platform's standard VST3 directory.

Every archive has a matching `.sha256` checksum file. Releases also publish the
exact embedded model catalog, its detached signature, the exact embedded model
trust-root document, and `denoize-models-<tag>.dmb` plus its checksum. The signed
bundle contains the catalog models, upstream licenses, and provenance for
closed-network installation; the standalone catalog/root assets remain
available for independent audit and recovery tooling. Releases also publish
every versioned JSON Schema used by automation, monitoring, DAW state, and
verification clients.

Desktop releases also publish a signed recoverable-update manifest, exact
per-platform SBOMs, two authenticated `.dub` migration bundles per platform,
and a separate update-asset attestation. A bundle contains both the candidate
and its verified last-known-good installation, so startup-health or explicit
recovery does not require a network. See [recoverable application updates](docs/updates.md).

Every installable CLI archive, CLAP/VST3 archive, desktop package, crates.io archive,
and offline model bundle also has a per-artifact CycloneDX SBOM. The release
evidence archive binds those 22 artifacts and SBOMs to their sizes and SHA-256 digests,
while companion GitHub Sigstore/SLSA bundles prove the exact tag commit and
release workflow. See [release evidence and offline verification](docs/release-evidence.md)
for the trust model and an air-gapped verification procedure.

## Install with Cargo

The crates.io package provides the CLI and library with every
crates.io-compatible backend:

```sh
cargo install denoize --features full
```

For the embedded DeepFilterNet backend, use a prebuilt GitHub binary or build
this repository with its primary `Cargo.toml`.

### Publishing a release

1. Synchronize the root/crates.io manifests and lockfile, the desktop npm and
   Tauri manifests/configuration and lockfiles, and the generated CLI banner.
2. Run `bash scripts/verify-release-version.sh` to check all 13 version fields.
3. Commit and push the version change.
4. Create the tag from a commit on the default branch and push it:

```sh
git tag -a v0.1.0 -m "denoize v0.1.0"
git push origin v0.1.0
```

The `GitHub Release` workflow verifies that the tag is on the default branch and
matches every release version field, runs the full test suite, and builds all
CLI, CLAP, VST3, and desktop targets before publishing the crates.io package. It then checks
all archives, checksums, signatures, per-artifact SBOMs, build provenance,
recoverable update bundles, and updater metadata before publishing the draft release and generated notes. The
exact `.crate` archive is attested before publication and its checksum must
match the crates.io API afterward. When `docs/releases/vTAG.md` exists, its
curated notes are prepended to the generated notes. Desktop startup performs
only local update-health and managed-version repair; an online check, bounded
bundle download, and candidate activation each require an explicit user action.
The legacy `latest.json`
feed and the recoverable manifest use the same repository signing key. The updater private key is kept in the
`TAURI_SIGNING_PRIVATE_KEY` repository secret. A failed build leaves the release
as a draft and cannot publish the crate before every target and its evidence
have been verified.

## CLI highlights

### Portable projects and sample-accurate timelines

Create a source-bound project, review its exact presentation-timeline plan, and
assemble it without retaining whole-file PCM:

```sh
denoize project create project.json --root . --project-id interview \
  --source main=recording.wav \
  --selection intro=main,12.5,8.0,0+1 \
  --selection answer=main,45,20,0+1,0,0,0.25 --pretty
denoize project validate project.json --root . --pretty
denoize project plan create project.json assembled.wav \
  --root . --output assembly-plan.json --pretty
denoize project assemble project.json assembled.wav \
  --root . --plan assembly-plan.json --pretty
```

Selections use exact decoded-presentation ticks, explicit channel maps, silence
padding, and only adjacent unpadded crossfades. Manifests bind every source,
setting, preset, signed model package, plan, receipt, and license reference by
portable locator, byte length, and SHA-256. Changed sources, unsupported graph
shapes, unknown fields, path escapes, and output collisions fail before
publication. Missing sources can be relocated only to a complete fingerprint
and presentation-geometry match.

Offline `.dpb` bundles carry the manifest, settings, presets, verification
evidence, source licenses, model public keys, plans, and receipts. Source audio
and model package payloads remain references unless their include flag and a
positive aggregate byte ceiling are both supplied. Import authenticates the
complete bundle and publishes only to a new directory. Project batch and watch
automation invoke the same deterministic assembler; watch additionally signs
each successful output.

See [portable projects](docs/projects.md) for the CLI, Desktop, bundle, plan,
receipt, relocation, and safety contracts.

### Read-only plans and signed receipts

Preview exact file, batch, or bounded-stream work without creating output,
state, locks, or model-cache updates:

```sh
denoize plan noisy.wav clean.wav --pretty > plan.json
denoize plan input-dir output-dir --batch --resume --pretty > batch-plan.json
denoize plan long.mp3 clean.flac --stream --resume --pretty > stream-plan.json
```

Plans use portable relative locators rather than absolute paths and bind input
and model SHA-256 fingerprints, the resolved recipe/backend/accelerator, audio
geometry, publication decision, and conservative admitted resources. Planning
performs bounded decode, backend/model preparation, and encoder validation, so
it can fail before any execution side effect. A skipped batch item also binds
the exact fingerprint of the existing output whose journal evidence justified
that decision. Stream plans use additive v2 and inspect resumable checkpoint
sidecars without locking, repairing, truncating, or deleting them.

Generate an Ed25519 key and publish a receipt only after successful output:

```sh
denoize receipts keygen receipt-secret.json receipt-public.json
denoize noisy.wav clean.wav \
  --receipt clean.receipt.json --receipt-key receipt-secret.json
denoize receipts verify clean.receipt.json \
  --key receipt-public.json --plan plan.json --output-root . --pretty
```

The secret JSON is unencrypted and must remain private. denoize creates it
without clobbering, with Unix owner-only permissions or a protected Windows
DACL, and rejects keys with broader access. A separately supplied public key
or rotation/revocation policy is always required: a receipt never trusts an
embedded signer. Verification authenticates first and then independently
rehashes every rooted output. Batch receipts require the whole batch to finish
without failure or cancellation. Output and receipt are distinct atomic files,
so a final receipt-path race is reported after preserving already committed
audio rather than silently replacing either file. Streaming and stdin receipts
use additive v2 schemas. A captured stdout stream is verified with
`receipts verify --output CAPTURED_AUDIO`; its receipt is emitted only after
stdout accepts and flushes the complete verified bytes.

The desktop app exposes the same file, batch, and bounded-stream plan preview,
plan JSON export, optional receipt publication, owner-private key generation,
public-key export, trust-policy creation, and offline receipt/output
verification. Secret key paths are session-only UI values and are never stored
in desktop settings.

See the [stable JSON contracts](docs/json.md) for schema, privacy, verification,
and key-rotation details.

### Local authenticated IPC and durable jobs

`denoize ipc` exposes the same planner, resource admission, atomic publication,
checkpoint, and signed-receipt engine to trusted local automation. It is not a
network service: v1 binds only an ephemeral `127.0.0.1` TCP port, publishes that
endpoint in an owner-private discovery file, and requires an explicit bearer
capability on every framed JSON request. The transport is not encrypted; its
security boundary is the local OS account and the private state/grant files.

Initialize one state directory, then run the foreground server:

```sh
denoize ipc init --state-dir "$HOME/.local/state/denoize/ipc" \
  --admin-grant "$HOME/.config/denoize/ipc-admin.json" \
  --max-memory 1024 --max-temp-space 4096 --max-history 1024
denoize ipc serve --state-dir "$HOME/.local/state/denoize/ipc"
```

The initial administrator capability can manage grants and the server but
cannot submit audio. Create a least-privilege worker policy whose canonical
input/output roots are disjoint from secrets and state:

```json
{
  "label": "local-render-worker",
  "capabilities": ["plan", "submit", "read-own", "control-own"],
  "input_roots": ["/absolute/path/to/inbox"],
  "output_roots": ["/absolute/path/to/results"],
  "max_priority": 10,
  "expires_at_unix_millis": null
}
```

```sh
denoize ipc grant create worker-policy.json worker-grant.json \
  --discovery "$HOME/.local/state/denoize/ipc/discovery.json" \
  --grant "$HOME/.config/denoize/ipc-admin.json"

denoize ipc dry-run batch /absolute/path/to/inbox /absolute/path/to/results \
  --discovery "$HOME/.local/state/denoize/ipc/discovery.json" \
  --grant worker-grant.json -- --recursive --output-format flac
denoize ipc submit batch /absolute/path/to/inbox /absolute/path/to/results \
  --priority 5 --discovery "$HOME/.local/state/denoize/ipc/discovery.json" \
  --grant worker-grant.json -- --recursive --output-format flac
denoize ipc status JOB_ID --discovery DISCOVERY.json --grant worker-grant.json
denoize ipc pause JOB_ID --discovery DISCOVERY.json --grant worker-grant.json
denoize ipc resume JOB_ID --discovery DISCOVERY.json --grant worker-grant.json
denoize ipc history --limit 100 --discovery DISCOVERY.json --grant worker-grant.json
```

Arguments after `--` are ordinary processing options. The server rejects flags
that could redirect plans, receipts, resource governors, isolation, model
files, configuration, or publication outside its policy. Dry-run is mandatory
before admission and reports conservative RAM, temporary storage, CPU/GPU work,
destination create/replace/skip counts, overwrite policy, pause support, and an
exact execution-plan digest. V1 executes one job at a time; priority orders the
durable queue and is capped by the submitting capability.

Batch and durable stream jobs pause only after a verified checkpoint or atomic
publication boundary and resume by replanning the same request. A daemon crash
reclaims them through their lease and checkpoint. A file job has no safe
mid-file checkpoint, so an uncertain publication is reported and never retried
automatically. Cancellation preserves already published atomic outputs and
never emits a false success receipt. A valid signed receipt discovered during
recovery wins over an ambiguous process exit. Revocation blocks future requests
but does not silently delete already admitted work; use an explicit cancel
before revoking when queued/running jobs must stop.

Request/response sizes, request/planning/job timeouts, connections, queue,
history, concurrency, and optional memory/temp/GPU ceilings are finite and
published in `denoize-ipc-discovery-v1`. Terminal history is bounded and keeps
resource/destination summaries plus plan and receipt fingerprints, not input or
output paths; receipt artifacts are pruned when their history entries age out.
The desktop **IPC automation** page uses the same Rust client and keeps bearer
tokens outside the WebView. All eight IPC/job schemas ship with releases and
the crate; see the [stable JSON contracts](docs/json.md).

### Realtime audio

Build with the optional system-audio integration, list devices, then route a
microphone through a denoising backend to an output or virtual-audio device:

```sh
cargo build --release --features live,rnnoise,gtcrn
denoize live --list-devices
denoize live --backend rnnoise --input-device "Microphone" --output-device "Virtual Cable"
denoize live --backend gtcrn --live-latency 80 --max-drift-ppm 2500 \
  --reconnect-timeout 30000 --input-device "Microphone" --output-device "Virtual Cable"
```

Realtime processing runs outside the device callbacks and uses bounded queues.
The capture callback uses a non-waiting handoff, so an overloaded backend drops
stale chunks; the playback callback emits bounded silence instead of waiting
while the worker publishes a block. `--chunk-ms` controls the latency/throughput
trade-off and defaults to 100 ms. The low-latency Classical, RNNoise, and causal GTCRN backends are
live-capable; other backend selections are rejected before capture or playback
starts. Input and output devices may use different default sample rates: a
bounded asynchronous sinc converter maps capture frames to the playback clock,
while a PI controller keeps the playback queue near its target without an
abrupt timebase reset. `--live-latency 0` selects an automatic target of two
capture chunks with a 40 ms minimum; explicit targets are 20–5,000 ms.
`--max-drift-ppm` bounds clock correction (2,500 ppm by default, or zero to
disable drift correction while retaining nominal-rate conversion).

Device stream failures enter a finite exponential-backoff recovery loop.
`--reconnect-timeout` defaults to 30 seconds and zero disables recovery. Named
devices are reselected by an unambiguous exact name; duplicate exact names are
rejected rather than silently routing to a different device. An unspecified
device follows the current system default. Each recovered generation starts
from a cold backend/resampler
state and primes playback before sound resumes. Human-readable status is
written to stderr about once per second. `--json` instead emits
`denoize-cli-output-v1` NDJSON status records containing connection state,
sample rates, queue/latency measurements, drift correction, underruns,
overflows, dropped chunks, reconnects, generation, and accelerator selection.
Latency is an engineering estimate assembled from device callback timing,
capture chunking, algorithmic delay, processing time, and queued playback—not
a loopback measurement or an exact hardware guarantee.

Classical, RNNoise, and GTCRN sessions preserve denoiser, overlap, recurrent,
partial-frame, and sample-rate-converter state across consecutive capture
chunks. If an overloaded capture queue drops a chunk, the sequence gap clears
queued playback and cold resets processing state without reparsing a loaded
model before the next retained chunk; state is never shared between separate
live sessions. VAD-enabled Classical/RNNoise live processing keeps the legacy
chunk-compatible path and prints a one-time warning because causal VAD state
and delay alignment are not yet available. GTCRN rejects live VAD rather than
silently discarding its recurrent continuity.

### Batch processing

Process a directory tree concurrently while preserving its relative layout:

```sh
denoize recordings cleaned --batch --recursive --jobs 4 --output-format flac
```

Batch mode validates the complete input/output plan and each input's decoded
audio properties before creating the output directory or starting workers,
then continues after later per-file processing failures and reports a final
summary. Existing outputs remain protected unless `--force` is supplied, and
input/output directories must not overlap. Recursive discovery does not follow
directory symlinks, and planned destinations that resolve back into the input
tree are rejected.

Omit `--output-format` only when denoize can re-encode the same container and
codec (WAV, FLAC, Ogg Opus, MP3, AAC-in-MP4, or ADTS AAC). Decode-only
containers such as AIFF/AIFC, CAF, RF64/BWF, plus Ogg Vorbis and ALAC-in-MP4,
require an explicit output format. This prevents implicit Vorbis-to-Opus or
ALAC-to-AAC conversion; for example, use `--output-format flac` when that
conversion is intentional. AAC-in-MP4 and ADTS AAC also require a build with
the corresponding AAC encoder; unavailable outputs are rejected during
preflight.

With `--resume`, the v3 state journal binds each completion to the input bytes,
the actual selected backend and effective processing/codec/metadata settings,
any consumed model bytes, the destination identity, and the published output
bytes. A file is skipped only when all of those still match and the output is a
safe regular file with a single link. This also means `--backend auto` records
the backend it actually selected, not merely the word `auto`.
Execution-only controls such as memory/temporary/GPU limits, `--isolate`,
`--jobs`, and progress output do not change that audio recipe, although their
validation still applies to each run.

### Watch-folder automation

Run a durable local inbox that waits for complete files, processes them one at
a time, and signs every successful result:

```sh
denoize receipts keygen watch-secret.json watch-public.json
denoize watch incoming cleaned --recursive --output-format flac \
  --receipt-key watch-secret.json
```

Watch mode polls portably on Linux, macOS, and Windows. A candidate is opened
only when it is a supported regular audio file and its length, modification
stamp, filesystem identity, and SHA-256 content remain unchanged for the full
`--settle-ms` interval (2 seconds by default). Directory symlinks, FIFOs,
devices, and the separate output tree are never followed as inputs. Use
`--once` for a bounded settle-and-scan invocation suitable for schedulers; the
default command continues until Ctrl+C. `--poll-ms` controls the daemon scan
interval, while `--max-watch-files` bounds each traversal.

Each transition is atomically recorded in
`OUTPUT/.denoize-watch-state.json` while a sibling lock enforces one writer.
The state binds an opaque digest of the denoize version, processing template,
output format, receipt public-key identity, and any explicitly selected model
or model-key files. Reopening it with a different template fails without
touching prior outputs; choose a fresh `--watch-state` path to begin a deliberate
new generation.
An interrupted `processing` entry becomes a due retry on restart. If output
and receipt were both committed before the interruption, their signature,
settled input fingerprint, locator, and output bytes are verified and the job
is recovered without reprocessing. If both disappeared, the same job may be
recreated; a one-sided output/receipt pair is preserved as an operator-visible
failure rather than silently guessed or overwritten. Stable inputs already
recorded as complete are checked with filesystem metadata rather than hashed
on every poll.

Failures use bounded exponential delay from `--retry-initial-ms` through
`--retry-max-ms`. `--max-attempts` defaults to five. A permanent failure or an
exhausted retry budget is copied without clobbering to
`OUTPUT/.denoize-quarantine`, verified against the settled SHA-256, accompanied
by a `denoize-watch-quarantine-v1` JSON explanation, and only then removed from
the inbox. A failed copy leaves the source and a durable quarantine-pending
entry for the next cycle. Custom `--quarantine`, `--receipt-dir`, and
`--watch-state` paths must remain below the output root, which itself must not
overlap the input tree.

The unencrypted receipt key is mandatory and must remain outside both trees
and unchanged for the watcher lifetime. A missing or changed key or explicit
model artifact defers due jobs without consuming their attempt budgets or
quarantining their inputs; restart the watcher with a fresh state path to adopt
a deliberate processing-template change.

Outputs preserve relative layout and default to WAV; `--output-format` selects
another encoder. Existing unrelated destinations are never replaced. When a
later content generation would collide with a prior name, its full content
digest is inserted before the extension.
Watch mode is intentionally sequential and uses the normal per-input resource
governor; `--batch`, `--stream`, `--resume`, `--force`, `--report`, `--isolate`,
and `--jobs` are rejected.

The desktop **Watch folders** page uses the same state engine and isolated
per-file worker. Select an inbox, a separate output directory, and a receipt
secret key outside both trees; then choose the settle/retry policy and start
watching. The page displays observed, pending, successful, retrying,
quarantined, and superseded counts. **Stop** prevents another scan and cancels
the currently isolated item at its safe publication boundary. Watch paths and
the secret-key selection are session-only and are not stored in desktop
settings. Processing and resource settings come from the main denoise page,
while overwrite remains disabled.

### Automatic backend selection

Use `--backend auto` when the build contains multiple denoisers. Short and
quality-prioritized files use DeepFilterNet when available; long files use
RNNoise to bound processing cost. Realtime sessions prefer RNNoise. The
classical backend is the dependency-free fallback, and the selected backend is
reported before processing.

### Adaptive noise profiling

`--adaptive-noise` detects spectrally noise-like, low-speech-probability regions
throughout a recording and slowly refreshes the classical estimator's anchored
noise profile. This handles changing fans, air conditioning, and room tone
without assuming that the recording begins with silence. Tonal frames are
rejected to reduce the risk of learning sustained notes as noise.

### Voice activity detection

`--vad` detects speech with 20 ms energy frames, hangover, context padding, and
region merging. Long silent spans bypass expensive backend inference and are
strongly attenuated; enhanced speech retains a small dry-signal blend to protect
consonants and attacks. Output channel count and duration remain unchanged.

### Loudness delivery

Normalize denoised output to an EBU R128 integrated-loudness target while
respecting an oversampled true-peak ceiling:

```sh
denoize input.wav output.flac --loudness -16 --true-peak -1
```

The applied gain is reduced when necessary to satisfy the peak ceiling, so
peak safety takes precedence over reaching the requested LUFS exactly.

### Content modes

`--mode speech`, `--mode music`, and `--mode ambient` coordinate related DSP
controls instead of changing only one strength value. Speech mode enables VAD
and adaptive profiling; music mode prioritizes transients, stereo content, and
low suppression; ambient mode preserves environmental texture while tracking
slowly changing noise. Explicit options such as `--strength` still override the
mode defaults.

### Optional FDK-AAC encoder

Pure-Rust `oxideav-aac` remains the default. Source builders can opt into the
Fraunhofer encoder and select it per invocation:

```sh
cargo build --release --features fdk-aac-encoder
denoize input.wav output.m4a --aac-encoder fdk --m4a-bitrate 192
```

The FDK feature uses the third-party Rust port and is intentionally excluded
from `full` and official release binaries. Fraunhofer's codec source has its own
license and MPEG-AAC patent language; downstream distributors are responsible
for reviewing both. The project requires Rust 1.96 or newer.

### Raw ADTS AAC

`.aac` files are decoded and encoded directly as ADTS streams without an MP4
container or an ffmpeg conversion step. M4A and raw AAC share
`--m4a-bitrate`, which must be a positive kbps value that fits the encoder's
32-bit bps field; raw ADTS output currently uses the default oxideav encoder.

### Metadata preservation

File processing merges all readable input tags (for example ID3v2/ID3v1 and
APE tags) and remaps the complete set of recognized fields—title, artist,
album, track/disc numbers, dates, ReplayGain, lyrics, comments, and artwork—to
the destination container's tag type. Cover art bytes, MIME type, picture type,
and description are retained by formats that support embedded pictures.

For FLAC and Ogg outputs, arbitrary Vorbis Comment fields are copied verbatim,
including the standard `CHAPTER001`/`CHAPTER001NAME` chapter-comment convention.
Their native metadata is scanned and rewritten incrementally instead of
loading the complete audio file into memory. Oversized or malformed input
metadata fails before an output is staged or published.
ID3v2-prefixed FLAC/Ogg files are rejected by this bounded raw path; place
metadata in the container's native comment blocks instead.
When the source and destination use the same native container, format-specific
ID3v2 frames (including `CHAP`/`CTOC`) and MP4 atoms are retained as well. A
conversion to a different tag family keeps fields with a defined destination
mapping; container-specific fields without one cannot be represented there.
Use `--no-metadata` for a clean output.

### Quality comparison

```sh
denoize compare clean.wav noisy.wav enhanced.wav
denoize compare clean.wav noisy.wav enhanced.wav --json
denoize compare clean.wav noisy.wav enhanced.wav --html > report.html
denoize metrics clean.wav enhanced.wav --json | jq '.artifact_scores'
```

Quality metrics require sample-aligned PCM. Every input must have a non-zero
matching sample rate, the same channel count and frame count, and equal-length
channels within each file. Denoize rejects truncated or ragged inputs instead
of silently scoring only their common prefix.

The report shows noisy and enhanced SI-SDR, SI-SNR, SNR, segmental SNR, stereo
side SDR, inter-channel correlation error, STOI, PESQ, ViSQOL, and improvement
deltas. It also screens for musical noise, pumping, transient loss, and stereo
phase distortion. These artifact scores are deterministic
dependency-free indicators in `[0, 1]` (lower is better), not perceptual
listening-test replacements; phase distortion is reported only for stereo
inputs.

When a dB metric is undefined for a silent or otherwise degenerate reference,
the report uses a finite `-120 dB` floor so JSON output remains valid and
machine-readable.

STOI is calculated natively for sufficiently long reference/test pairs and is
reported in `[-1, 1]` (higher is better). ViSQOL MOS-LQO is available in the
pure-Rust build when the optional feature is enabled:

```sh
cargo install denoize --features visqol
denoize metrics clean.wav enhanced.wav --json | jq '{stoi, visqol, pesq}'
```

ViSQOL is a full-reference MOS estimate in `[1, 5]`. PESQ is intentionally
left as `null`: the ITU-T P.862 reference implementation and conformance
material require a separately licensed external adapter and are not bundled
with denoize. Inputs that are too short or a disabled optional implementation
are represented as `null` rather than preventing the rest of the report.

### Licensed-corpus release evaluation

`denoize evaluate` turns quality, output-integrity, and speed claims into one
reproducible, signed release gate. The same strict manifest and runner are
available in the CLI, library, and Desktop app:

```sh
denoize receipts keygen evaluation-secret.json evaluation-public.json
denoize evaluate validate corpus.json --corpus-root /datasets/release --pretty
denoize evaluate run corpus.json --corpus-root /datasets/release \
  --key evaluation-secret.json --output candidate.evaluation.json
denoize evaluate verify candidate.evaluation.json \
  --key evaluation-public.json --manifest corpus.json --pretty
denoize evaluate compare baseline.evaluation.json candidate.evaluation.json \
  --key evaluation-public.json --pretty
```

Each clean/noisy/model artifact pins a portable path, byte length, SHA-256,
SPDX license, source URI and immutable revision, and signal-preparation digest.
Audio stays below the caller-selected corpus root and is never embedded in the
result or release artifacts. Symlinks, escaping paths, changed files, missing
provenance, incomparable PCM geometry, unavailable metrics, and mismatched
hardware/runtime/model contexts fail closed.

Signed results include objective and perceptual metrics, duration/rate/channel
agreement, clipping, sample and true peak, DC offset, silence/dropout ratios,
integrated loudness, decode integrity, the canonical output fingerprint,
sorted performance samples, real-time factor, throughput, peak RSS when the OS
exposes it, and every threshold outcome. A required listening protocol cannot
be replaced by automation: `run` requires a matching, manifest-bound human
result. See [reproducible evaluation evidence](docs/evaluation.md) and the
[stable JSON contracts](docs/json.md).

### Configuration file

Reusable defaults can be stored in TOML and loaded with `--config`. TOML syntax
and enum names are checked while loading; explicit command-line numeric values
then override file defaults, and the final effective configuration is validated
before input decoding, output staging, or batch worker creation. For example,
FFT frames must be powers of two from 256 through 65,536, streaming blocks must
be from 1 through 1,048,576 frames, batch jobs from 1 through 32, and live
chunks from 10 through 2,000 ms. Non-finite effective floating-point settings
are rejected; live latency accepts zero for automatic or 20 through 5,000 ms,
drift correction accepts 0 through 10,000 ppm, and reconnect timeout accepts 0
through 300,000 ms. Loudness targets are limited to -70..0 LUFS and true-peak
ceilings to -20..0 dBTP.

```toml
backend = "auto"
accelerator = "cpu" # cpu|auto|gpu|metal|cuda
preset = "hifi"
mode = "speech"
strength = 0.45
adaptive_noise = true
vad = true
loudness_lufs = -16.0
true_peak_dbtp = -1.0
# deterministic = true  # serialize processing for reproducible output
# seed = 12345          # optional SGMSE sampler seed (implies deterministic)
# stream_frames = 8192
# chunk_ms = 100
# live_latency_ms = 0       # automatic; otherwise 20..5000
# max_drift_ppm = 2500     # 0..10000
# reconnect_timeout_ms = 30000 # 0 disables hotplug recovery
# max_memory_mb = 1024
# max_process_memory_mb = 2048
# max_temporary_mb = 4096
# max_gpu_memory_mb = 4096
# max_gpu_jobs = 1
# isolate = true # CLI only: run processing in a child boundary
```

```sh
denoize input.wav output.flac --config denoize.toml --strength 0.55
```

Use `--deterministic` when an audio result must be reproducible across runs.
The mode serializes channel/model and batch scheduling and uses a stable
stochastic-backend seed. `--seed N` selects an explicit SGMSE+ seed and implies
the mode. Diagnostic elapsed times and progress messages are intentionally not
part of the reproducibility guarantee.

### Batch progress and recovery

Batch runs show completed files, elapsed time, and ETA. `--resume` records v3
completion entries in `.denoize-state` under the output directory. The CLI and
desktop app use this same canonical state filename. State written by older v1
or v2 releases is read only as legacy evidence: it is never trusted for a skip.
Run once with `--force` to regenerate a replaceable legacy output and migrate it
to v3; the following identical run can then skip it.

The resume/force decision is deliberately conservative:

| Planned destination | Without `--force` | With `--force` |
|---|---|---|
| Exact safe v3 output | Skip | Skip |
| Output is missing | Process | Process |
| Legacy, untracked, changed, or unsafe existing output | Error and preserve it | Replace it when the path is safely replaceable |

“Changed” includes input content, the effective backend or recipe, model
content, and output content. Symlinks, multiply linked files, directories, and
special files are never accepted as completed outputs; directories and special
files remain non-replaceable even with `--force`.

The denoize package version participates in the v3 recipe hash. After a package
upgrade, `--resume` preserves an existing output and reports `recipeChanged`
unless `--force` is supplied. Regenerate it once with `--force` to migrate the
saved recipe; subsequent identical runs skip it normally.

Resumable ONNX-backed batches require a self-contained `.onnx` model. ONNX
models that declare external tensor sidecars remain usable in ordinary
non-resume batches, but `--resume` rejects them because a one-file model digest
cannot safely represent all consumed weights.

Every batch run creates the output directory only after complete input, codec,
and configuration preflight, then takes the shared `.denoize-batch.lock` before
reading resume state or deciding what to do with destinations. Another denoize
batch using that directory fails immediately while the lock is held. The lock,
canonical `.denoize-state`, and legacy desktop `.denoize-gui-state` migration
names are reserved from the planned output topology.

Ctrl+C stops work before publication where possible. Each output is encoded to
a randomly named private file in the destination directory. Publication then
serializes and synchronizes a journal prepare, the atomic output commit, and a
completion record, while rechecking the input and model. Cancellation before this gate
leaves the destination and state untouched; once an item enters the gate, its
publication is finished atomically. If a process exits between prepare and
completion, the next locked batch reconciles the prepared record against the
published output before making new decisions. A journal failure closes the gate
so later workers cannot publish unrecorded outputs.

Without `--force`, the destination is checked again at commit time to prevent
ordinary concurrent overwrites. On Unix, the batch output root must be owned by
the current user and must not be group/world writable; output paths through
non-sticky shared-writable or untrusted-owner ancestors are also rejected.
Extended ACLs that grant additional access are also rejected, as are network or
userspace filesystems whose ACLs cannot be verified safely. On Windows, atomic
private staging requires an ACL-capable filesystem such as NTFS; FAT and exFAT
output paths are rejected before encoding, newly created batch control files
receive a protected DACL, and the output root must not be writable by untrusted
accounts. Windows interprocess locking assumes that principals with write or
delete access to the output root or any pre-existing control/output entry
cooperate; denoize does not audit those DACLs for hostile-principal access. Use
`--no-progress` for quiet
operation or `--json` for NDJSON progress and summary records. On Unix,
`--force` also refuses to replace an existing file with an extended ACL or a
different owner, avoiding a silent loss or weakening of its access policy.
JSON summaries retain the `cancelled` boolean and also report
`cancelled_count`; together with succeeded, skipped, and failed, that count
partitions the batch total.

These checks define a non-adversarial local-filesystem, process-crash recovery
contract. The deterministic [resilience matrix](docs/resilience.md) exercises
every acknowledged journal/checkpoint publication prefix with abrupt child
process exits and simulates power loss at local synchronization boundaries. It
does not claim protection from a hostile process performing precisely timed ABA
path swaps, a lying drive cache, faulty hardware, remote-filesystem semantics,
or a kernel/filesystem that violates its documented durability behavior. Keep
independent backups for those failure classes.

```
-b, --backend <NAME>     classical|rnnoise|deepfilter
-a, --algorithm <NAME>    omlsa|logmmse|mmse|wiener|specsub|specsub-nl|specsub-geo
--window <NAME>          hann|hamming|sine|blackman|kaiser|flattop|dpss
--kaiser-beta <B>        Finite Kaiser β in 0..50 (default: 8.0)
--dpss-nw <NW>           Classical DPSS time-bandwidth product in (0, 8] (default: 3.0)
--multiband              Multiband spectral subtraction
--perceptual             Bark perceptual gain weighting
--postfilter             Musical-noise suppression post-filter
-p hifi                   Flagship preset (Kaiser + perceptual + postfilter)
--quality ultra           Maximum fidelity settings
--onnx-model <PATH>       Waveform ONNX model used by the onnx backend
--onnx-rate <HZ>          Model sample rate in 1..768000 Hz (default: 16000)
```

## Resilience testing

Every pull request runs fixed parser mutations and deterministic I/O-error and
crash-recovery matrices. A scheduled AddressSanitizer/libFuzzer workflow covers
all supported audio containers, execution documents, signed receipts and keys,
trust policies, and offline model bundles with finite input, RSS, and per-case
time limits. See [resilience testing](docs/resilience.md) for the exact commands,
resource-accounting scope, corpus-promotion rule, and debug-only fault protocol.

## Library API

```rust
use denoize::{denoise_file_with_backend, Backend, DenoiserConfig, Preset};

let cfg = Preset::HiFi.config(48000);
denoise_file_with_backend("noisy.wav", "clean.wav", cfg, Backend::Classical)?;

// With DeepFilterNet (GitHub/source build with --features full)
denoise_file_with_backend("noisy.wav", "clean.wav", cfg, Backend::DeepFilter)?;
```

For embedders that use `denoise_file_with_backend_config`, set
`BackendOptions { deterministic: true, ..Default::default() }` to serialize
model/channel work. Set `seed: Some(value)` to reproduce SGMSE+ sampling with
an explicit seed.

For reusable finite-file processing, prepare a common backend session once.
Batch workers and VAD regions use this same API, so fixed graphs are shared and
dynamic-shape adapters retain their most recent optimized graph:

```rust
use denoize::{Backend, BackendOptions, BackendSession, DenoiserConfig};

let session = BackendSession::prepare(Backend::Classical, BackendOptions::default())?;
let channels = vec![vec![0.0; 48_000]];
let enhanced = session.process(&channels, 48_000, &DenoiserConfig::default(48_000))?;
assert_eq!(enhanced[0].len(), channels[0].len());
```

With the `onnx` feature, embedders can also inspect and retain the generic
waveform model contract explicitly:

```rust
use denoize::{OnnxModelConfig, OnnxWaveformLayout, OnnxWaveformModel};

let model = OnnxWaveformModel::load(OnnxModelConfig {
    path: "model.onnx".into(),
    sample_rate: 16_000,
})?;
assert!(matches!(
    model.contract().layout(),
    OnnxWaveformLayout::BatchSamples | OnnxWaveformLayout::BatchChannelsSamples
));
let channels = vec![vec![0.0; 16_000]];
let enhanced = model.process(&channels, 16_000, true)?;
assert_eq!(enhanced[0].len(), channels[0].len());
```

## Roadmap status

| Priority | Technology | Status |
|----------|-----------|--------|
| 1 | DeepFilterNet v3 | ✅ `--features deepfilter` |
| 2 | RNNoise | ✅ `--features rnnoise` |
| 3 | Kaiser/Flat-top/DPSS windows | ✅ |
| 4 | Multiband / nonlinear SpecSub | ✅ |
| 5 | Perceptual weighting + musical-noise PF | ✅ |
| 6 | Pure-Rust external ONNX inference foundation | ✅ validated reusable waveform runtime (`--features onnx`) |
| 7 | BSRNN / MP-SENet / MossFormer2 adapters | ✅ implemented and quality-gated |
| 8 | SGMSE+ | ✅ 30-step PC sampler + score-model adapter |

See [ROADMAP.md](ROADMAP.md) for the implementation audit and the acceptance
criteria and numerical evidence for each named model.

## License

denoize-authored Rust code is MIT licensed. Bundled dependencies and reference
materials remain under their respective terms; see
[THIRD_PARTY.md](THIRD_PARTY.md) and [LICENSES](LICENSES) for notices,
corresponding-source information, and license texts.
