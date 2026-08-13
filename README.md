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
| `gtcrn` | `--features gtcrn` | Official 48K-parameter causal GTCRN; offline plus the stateful library stream API (not the `live` command) |

Build everything: `cargo build --release --features full`

The generic ONNX backend is the deployment foundation for future neural
models. It intentionally accepts only single-input/single-output waveform
models; spectral models and diffusion samplers require dedicated adapters.

> The prebuilt GitHub binaries include every backend. Because the DeepFilterNet
> Rust crate is not available from crates.io, the crates.io package's `full`
> feature currently includes RNNoise, generic ONNX, MP-SENet, BSRNN,
> MossFormer2, SGMSE+, and GTCRN, but not DeepFilterNet.

## Supported input formats

| Format | Decoder | Notes |
|--------|---------|-------|
| WAV/BWF | `hound` | 8–32 bit int / float; BWF metadata chunks are preserved for supported tags |
| RF64 | native RF64 reader | 64-bit-size PCM/WAVE, bounded chunk reads |
| AIFF/AIFC | `symphonia` | PCM and supported AIFC codecs |
| CAF | `symphonia` | PCM and ALAC/other supported CAF codecs |
| MP3 | `symphonia` + bounded `nanomp3` fallback (Pure Rust) | Xing/Info + LAME gapless trim, ID3v2, no resampling |
| M4A/AAC/ALAC | `oxideav-aac` + `symphonia` fallback | AAC-LC/ALAC decode with MP4 v0/v1 unity-rate edit-list timing; MP4 AAC-LC access units above 8,191 bytes are rejected (ALAC is unaffected) |
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

Model installs and updates support explicit network policy, authenticated
mirrors, resumable transfers, and air-gapped local files. Run
`denoize models --help` for the dedicated command reference.

```sh
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
`DENOIZE_MODEL_URL`, `DENOIZE_MODEL_PROXY`,
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
model candidate must match both the manifest's exact byte length and SHA-256
before use: this includes fresh or resumed downloads, alternate `--url`
sources, `--from` imports, completed partials, and files already in the cache.
An update keeps the current verified model until its replacement is ready.
`denoize models info MODEL` reports the pinned length as an unscaled decimal
`size-bytes` value. This per-model integrity bound is not an aggregate cache
quota.

HTTPS model connections, including those tunneled through an HTTP CONNECT
proxy, use the operating system trust store. CLI Bearer tokens and Basic
passwords are accepted through environment variables, and diagnostics redact
credentials, query strings, and fragments. Signed `--url` values and proxy
credentials can still leak through process listings and shell history, so use
protected environment injection when that matters. See the
[managed-model guide](docs/models.md) for option combinations,
proxy precedence, and resume validation details.

### Long recordings with bounded memory

For long WAV recordings, use the classical streaming path. It keeps only the
STFT overlap and a fixed-size input block in memory instead of loading the
whole file:

```sh
./target/release/denoize long-noisy.wav long-clean.wav --stream
```

`--stream` currently supports filesystem WAV-to-WAV processing with the
classical backend and independent channels. VAD, loudness normalization,
mid/side or linked stereo processing, and AI/encoded output require the normal
(non-streaming) path. The default block size is 8192 frames; use
`--stream-frames N` (1–1,048,576) to trade latency and working memory for
throughput. Noise profiling retains only a bounded leading segment before
output begins. Stream resource arithmetic is checked from the input header,
and the processor is constructed before an output or temporary file is staged.

Filesystem inputs are opened once per processing phase as validated regular
files. Size estimation, probing, decoding, and metadata reads within that
phase use the same opened filesystem object, so replacing the pathname cannot
silently mix bytes from two inputs. FIFOs, directories, and device files are
rejected before an audio parser or output staging step runs.

For the normal (decoded, non-streaming) path, `--max-memory MB` caps requested
denoize-owned decoded PCM capacities and explicitly accounted codec scratch
buffers, in addition to the conservative input-size preflight and final
decoded-working-set check. Internal allocations made inside third-party codec
libraries can fall outside this enforcement, and allocator capacity rounding
means the cap is not an allocator-exact process RSS limit. FLAC and Ogg
structure is also checked with finite block,
packet, page, stream, item, and aggregate metadata limits before a decoder can
materialize it—even with `--no-metadata`. When tags are preserved, their
retained payload budget is derived from the memory left after the decoded PCM
working set; the same limit is enforced again while writing the staged output.
The default limits remain finite when `--max-memory` is omitted.

The limit applies per regular-file input/worker; stdin retains its separate
bounded WAV buffering path. Batch jobs can use memory concurrently, so lower
`--jobs` when targeting a process-wide ceiling. Batch probing, decode, and
metadata validation all finish before the output directory or staging files
are created. A streaming WAV job stays bounded by its block size and denoiser
state, and metadata uses a conservative share of the remaining budget:

```sh
denoize large.mp3 cleaned.wav --max-memory 1024
denoize long-noisy.wav long-clean.wav --stream --stream-frames 4096 --max-memory 64
```

## Desktop app

The Tauri desktop app exposes single-file denoising, batch conversion, quality
comparison, and model management without sending audio off the computer. Its
default build includes every backend in the repository's `full` feature set;
FDK-AAC remains an explicit opt-in because of its separate licensing terms.
ONNX-based backends expose model-file, model-rate, and SGMSE quality controls
when selected; managed GTCRN weights are resolved automatically after install.
The model manager's offline, alternate-source, proxy/direct, authentication,
and local-file controls are session-only. Bearer tokens and Basic credentials are
cleared after an operation starts, and none of these download overrides are
included in saved settings, named presets, or CLI-compatible imports and
exports.
Desktop batches accept files or folders, preserve relative paths, run with a
configurable worker count, continue after individual failures, and can resume
from the same `.denoize-state` journal used by the CLI in the output directory.
Single-file processing also provides local waveform previews, RMS-matched
before/after switching, click-to-seek, and configurable section looping.
Desktop settings are restored automatically, can be stored as named presets,
and can be imported or exported as CLI-compatible TOML. Recent input files are
kept locally for quick reuse. The single-file and batch views also expose a
reproducibility mode that serializes processing and uses stable model seeds.
Audio files and folders can be dropped onto the single-file or batch input
zones; output folders have dedicated drop targets. Multiple audio files switch
the app to batch mode automatically.
The realtime page routes a selected capture device through a low-latency
backend to a playback device, with input/output meters, dropped-chunk counters,
and explicit start/stop controls. Live sessions support only the live-capable
Classical and RNNoise backends; other backends are rejected before capture or
playback starts. Headphones help prevent acoustic feedback.

```sh
cd apps/desktop
npm ci
npm run tauri -- dev

# Build a platform-native installer/package
npm run tauri -- build

# Optional FDK-AAC selector
npm run tauri -- build --features fdk-aac-encoder
```

Linux development requires the WebKitGTK 4.1 and GTK 3 development packages.
For Ubuntu 24.04 or later:

```sh
sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev patchelf
```

## Prebuilt binaries

Each [GitHub Release](https://github.com/penguin425/denoize/releases) contains
prebuilt `full`-feature binaries for:

- Linux x86-64
- macOS Intel and Apple Silicon
- Windows x86-64

Every archive has a matching `.sha256` checksum file.

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
2. Run `bash scripts/verify-release-version.sh` to check all 11 version fields.
3. Commit and push the version change.
4. Create the tag from a commit on the default branch and push it:

```sh
git tag -a v0.1.0 -m "denoize v0.1.0"
git push origin v0.1.0
```

The `GitHub Release` workflow verifies that the tag is on the default branch and
matches every release version field, runs the full test suite, and builds all
CLI and desktop targets before publishing the crates.io package. It then checks
all archives, checksums, signatures, and updater metadata before publishing the
draft release and generated notes. Installed desktop apps check the signed
`latest.json` feed on startup; updates are only installed after user
confirmation. The updater private key is kept in the
`TAURI_SIGNING_PRIVATE_KEY` repository secret. A failed build leaves the release
as a draft and cannot publish the crate before every target has built.

## CLI highlights

### Realtime audio

Build with the optional system-audio integration, list devices, then route a
microphone through a denoising backend to an output or virtual-audio device:

```sh
cargo build --release --features live,rnnoise
denoize live --list-devices
denoize live --backend rnnoise --input-device "Microphone" --output-device "Virtual Cable"
```

Realtime processing runs outside the device callbacks and uses bounded queues,
so an overloaded backend drops stale capture chunks instead of blocking the
audio thread. `--chunk-ms` controls the latency/throughput trade-off and defaults
to 100 ms. Only the low-latency Classical and RNNoise backends are live-capable;
other backend selections are rejected before capture or playback starts. Input
and output devices must currently share a default sample rate.

Classical and RNNoise sessions preserve denoiser, overlap, partial-frame, and
sample-rate-converter state across consecutive capture chunks. If an overloaded
capture queue drops a chunk, the sequence gap clears queued playback and cold
resets all processing state before the next retained chunk; state is never
shared between separate live sessions. VAD-enabled live processing keeps the
legacy chunk-compatible path and prints a one-time warning because causal VAD
state and delay alignment are not yet available.

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
Execution-only controls such as `--max-memory`, `--jobs`, and progress output
do not change that audio recipe, although their validation still applies to
each run.

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

### Configuration file

Reusable defaults can be stored in TOML and loaded with `--config`. TOML syntax
and enum names are checked while loading; explicit command-line numeric values
then override file defaults, and the final effective configuration is validated
before input decoding, output staging, or batch worker creation. For example,
FFT frames must be powers of two from 256 through 65,536, streaming blocks must
be from 1 through 1,048,576 frames, batch jobs from 1 through 32, and live
chunks from 10 through 2,000 ms. Non-finite effective floating-point settings
are rejected; loudness targets are limited to -70..0 LUFS and true-peak
ceilings to -20..0 dBTP.

```toml
backend = "auto"
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
# max_memory_mb = 1024
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
contract. They do not claim protection from a hostile process performing
precisely timed ABA path swaps, or from power loss and storage-level durability
failures; file synchronization and atomic rename reduce those risks but do not
extend this contract. Keep independent backups for those failure classes.

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

## Roadmap status

| Priority | Technology | Status |
|----------|-----------|--------|
| 1 | DeepFilterNet v3 | ✅ `--features deepfilter` |
| 2 | RNNoise | ✅ `--features rnnoise` |
| 3 | Kaiser/Flat-top/DPSS windows | ✅ |
| 4 | Multiband / nonlinear SpecSub | ✅ |
| 5 | Perceptual weighting + musical-noise PF | ✅ |
| 6 | Pure-Rust external ONNX inference foundation | 🟨 waveform contract implemented |
| 7 | BSRNN / MP-SENet / MossFormer2 adapters | ✅ implemented and quality-gated |
| 8 | SGMSE+ | ✅ 30-step PC sampler + score-model adapter |

See [ROADMAP.md](ROADMAP.md) for the implementation audit and the acceptance
criteria and numerical evidence for each named model.

## License

denoize-authored Rust code is MIT licensed. Bundled dependencies and reference
materials remain under their respective terms; see
[THIRD_PARTY.md](THIRD_PARTY.md) and [LICENSES](LICENSES) for notices,
corresponding-source information, and license texts.
