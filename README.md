# denoize

[日本語](README.ja.md) · [Releases](https://github.com/penguin425/denoize/releases/latest) · [Documentation](docs/README.md) · [docs.rs](https://docs.rs/denoize)

`denoize` is a local audio cleanup toolkit written in Rust. It provides a CLI,
a desktop app, DAW plug-ins, and embedding SDKs. Classical DSP is always
available; neural backends are optional.

## Install

### Prebuilt releases

The [latest release](https://github.com/penguin425/denoize/releases/latest)
contains:

- CLI binaries for Linux, macOS, and Windows
- desktop packages
- CLAP, VST3, AUv3, and LV2 plug-ins
- C, Web/WASM, Android, and iOS SDKs

Release assets include SHA-256 checksums and signed build evidence.

### Cargo

```sh
cargo install denoize --features full
```

The crates.io `full` feature includes every crates.io-compatible backend.
DeepFilterNet is available in prebuilt releases or when building this repository.

### Build from source

```sh
git clone https://github.com/penguin425/denoize.git
cd denoize
cargo build --release --features full
```

## Quick start

```sh
# Classical denoising
denoize noisy.wav clean.wav -p hifi

# Managed GTCRN model
denoize models install gtcrn
denoize noisy.wav clean.wav -b gtcrn

# Bounded-memory processing for long recordings
denoize long.wav clean.flac --stream --resume --max-memory 256

# Inspect a recording or apply deterministic repair
denoize diagnose damaged.wav
denoize restore damaged.wav restored.wav --report restoration.json
```

Run `denoize --help` for the command list and see the
[CLI reference](docs/cli.md) for complete options.

## What it includes

| Area | Highlights |
|---|---|
| Denoising | Classical DSP plus optional RNNoise, DeepFilterNet, GTCRN, and external ONNX backends |
| Analysis and repair | Diagnosis, no-reference assessment, deterministic restoration, and optional model-based repair |
| Production workflows | Bounded streaming, resume, batch processing, watch folders, projects, stable JSON, and signed receipts |
| Audio applications | Target-speaker and target-sound extraction, echo cancellation, microphone arrays, meeting tracks, and music restoration |
| Integrations | Desktop app, CLAP/VST3/AUv3/LV2 plug-ins, Rust library, and C/Web/mobile SDKs |

Specialized operations are explicit commands. Ordinary denoising never silently
becomes source separation, semantic removal, or generative restoration.

## Audio formats

Inputs include WAV/BWF/RF64, AIFF/AIFC, CAF, FLAC, Ogg Opus/Vorbis, MP3,
M4A/AAC, and ALAC. Outputs include WAV, FLAC, Ogg Opus, MP3, M4A, and AAC.

Channel, metadata, codec, and bounded-memory details are documented in the
[CLI reference](docs/cli.md).

## DAW plug-ins

Release archives provide two effects in each supported format:

- `denoize`: fixed-latency classical DSP
- `denoize Neural`: managed GTCRN inference with a fixed-latency safety fallback

The Neural plug-in never downloads a model from the host. Install the model
with `denoize models install gtcrn`, then reload or reactivate the effect.
Without the model, inference remains disabled while automation, accessible host
parameters, state, and the selected fallback continue to work.

See the [plug-in guides](docs/README.md#daw-plug-ins) for installation, latency,
state, accessibility, and host-validation details.

## Models and safety

- Audio processing is local. Network access is limited to explicit model,
  catalog, or update operations.
- Managed models and signed model packages are verified before use and fail
  closed when their identity or contract does not match.
- Commands do not overwrite output files unless explicitly requested.
- Quality scores and signal checks do not prove semantic, speaker, or artistic
  fidelity. The relevant guides state each operation's limits.

See [Models](docs/models.md), [Stable JSON contracts](docs/json.md), and
[Resilience testing](docs/resilience.md).

## Documentation

Start with the [documentation index](docs/README.md).

- [CLI reference](docs/cli.md)
- [Managed models](docs/models.md)
- [Desktop app](docs/desktop.md)
- [DAW plug-ins](docs/README.md#daw-plug-ins)
- [Embedding SDKs](docs/sdk.md)
- [Projects and automation](docs/projects.md)
- [Release evidence](docs/release-evidence.md)
- [Roadmap](ROADMAP.md)
- [Release process](RELEASING.md)

Machine-readable contracts are published in [`schemas/`](schemas/).

## Development

```sh
cargo test --locked
cargo test --locked --all-features
```

The minimum supported Rust version is 1.96.

## License

denoize-authored Rust code is MIT licensed. Third-party notices and license
texts are in [THIRD_PARTY.md](THIRD_PARTY.md) and [LICENSES](LICENSES).
