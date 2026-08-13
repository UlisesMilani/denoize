//! `denoize` command-line interface.

use denoize::audio::{
    ensure_memory_limit, estimate_audio_working_set_bytes, estimate_session_memory_bytes,
    estimate_stream_memory_bytes_checked, inspect_wav_session, read_audio,
    read_audio_from_session_with_limits, read_wav_bytes_with_limits, write_wav_bytes,
    write_wav_channel_mask_to_file, WavStreamReader, WavStreamWriter,
};
use denoize::batch_resume::{
    self, BatchSession, ConsumedModel, Digest, FileFingerprint, MetadataPolicy, ResumeDecision,
    ResumeExpectation, LEGACY_DESKTOP_STATE_FILE_NAME, LOCK_FILE_NAME, STATE_FILE_NAME,
};
use denoize::config::{MAX_SAMPLE_RATE, MAX_STREAM_BLOCK_FRAMES};
use denoize::decode::{
    probe_file_from_session_with_limits as probe_audio_session_with_limits, AudioCodec,
    AudioFormat, AudioProbe, DecodeLimits,
};
use denoize::denoiser::{DenoiserConfig, Preset, ProcessingMode};
use denoize::metadata::MetadataLimits;
use denoize::service::{self, BackendChoice, ProcessingOptions};
use denoize::window::MAX_DENOISER_DPSS_NW;
use denoize::AudioInputSession;
use denoize::{
    AacEncoder, Algorithm, AtomicOutput, Backend, BackendOptions, ChannelMode, CommitMode,
    DownmixMode, EncodeOptions, OnnxModelConfig, OutputFormat, SgmseProfile, StreamingDenoiser,
    WindowType,
};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const STREAM_BLOCK_FRAMES: usize = 8192;
const MIN_STREAM_BLOCK_FRAMES: usize = 1;
const MIN_LIVE_CHUNK_MS: u32 = 10;
const MAX_LIVE_CHUNK_MS: u32 = 2_000;
const MAX_BATCH_JOBS: usize = 32;
const VALIDATION_SAMPLE_RATE: u32 = 48_000;
const BYTES_PER_MIB: u64 = 1024 * 1024;
const INPUT_MEMORY_EXPANSION_FACTOR: u64 = 8;
const METADATA_REPRESENTATION_EXPANSION_FACTOR: u64 = 16;
const METADATA_DESCRIPTOR_BYTES: u64 = 256;
const STDIN_READ_CHUNK_BYTES: usize = 64 * 1024;
static CANCELLED: AtomicBool = AtomicBool::new(false);
static CANCEL_HANDLER: OnceLock<Result<(), String>> = OnceLock::new();

fn with_batch_publication_fence<T>(
    fence: &Mutex<()>,
    cancelled: &AtomicBool,
    publish: impl FnOnce() -> Result<T, String>,
) -> Result<Option<T>, String> {
    let _guard = fence
        .lock()
        .map_err(|_| "batch publication fence is poisoned".to_string())?;
    if cancelled.load(Ordering::SeqCst) {
        Ok(None)
    } else {
        publish().map(Some)
    }
}

#[derive(Serialize)]
struct ProcessResultJson<'a> {
    input: &'a str,
    output: &'a str,
    backend: &'a str,
    channels: usize,
    frames: usize,
    sample_rate: u32,
    elapsed_ms: f64,
}

#[derive(Serialize)]
struct StreamResultJson<'a> {
    input: &'a str,
    output: &'a str,
    backend: &'static str,
    channels: u16,
    frames: usize,
    sample_rate: u32,
    stream: bool,
}

#[derive(Serialize)]
#[serde(tag = "event", rename_all = "lowercase")]
enum BatchJson<'a> {
    Progress {
        status: &'a str,
        completed: usize,
        total: usize,
        elapsed_seconds: f64,
        eta_seconds: f64,
        input: &'a str,
    },
    Summary {
        total: usize,
        succeeded: usize,
        skipped: usize,
        failed: usize,
        cancelled_count: usize,
        cancelled: bool,
        output: &'a str,
    },
}

fn serialize_json_line<T: Serialize + ?Sized>(payload: &T) -> String {
    serde_json::to_string(payload).expect("fixed CLI JSON payload must serialize")
}

fn round_to_three_decimals(value: f64) -> f64 {
    format!("{value:.3}")
        .parse()
        .expect("formatted JSON number must parse")
}

fn process_result_json_line(
    input: &str,
    output: &str,
    backend: &str,
    channels: usize,
    frames: usize,
    sample_rate: u32,
    elapsed_ms: f64,
) -> String {
    serialize_json_line(&ProcessResultJson {
        input,
        output,
        backend,
        channels,
        frames,
        sample_rate,
        elapsed_ms: round_to_three_decimals(elapsed_ms),
    })
}

fn stream_result_json_line(
    input: &str,
    output: &str,
    channels: u16,
    frames: usize,
    sample_rate: u32,
) -> String {
    serialize_json_line(&StreamResultJson {
        input,
        output,
        backend: "classical",
        channels,
        frames,
        sample_rate,
        stream: true,
    })
}

fn batch_progress_json_line(
    status: &str,
    completed: usize,
    total: usize,
    elapsed_seconds: f64,
    eta_seconds: f64,
    input: &str,
) -> String {
    serialize_json_line(&BatchJson::Progress {
        status,
        completed,
        total,
        elapsed_seconds: round_to_three_decimals(elapsed_seconds),
        eta_seconds: round_to_three_decimals(eta_seconds),
        input,
    })
}

fn batch_summary_json_line(
    total: usize,
    succeeded: usize,
    skipped: usize,
    failed: usize,
    cancelled_count: usize,
    cancelled: bool,
    output: &str,
) -> String {
    serialize_json_line(&BatchJson::Summary {
        total,
        succeeded,
        skipped,
        failed,
        cancelled_count,
        cancelled,
        output,
    })
}

fn install_cancel_handler() -> Result<(), String> {
    CANCEL_HANDLER
        .get_or_init(|| {
            ctrlc::set_handler(|| CANCELLED.store(true, Ordering::SeqCst))
                .map_err(|error| format!("install Ctrl+C handler: {error}"))
        })
        .clone()
}

fn usage() -> String {
    let backends = Backend::available_names().join("|");
    format!(
        "\
denoize {VERSION} — pure-Rust audio denoiser engineered for the world's highest sound quality

Classical DSP + optional AI backends (RNNoise, DeepFilterNet v3, MP-SENet, BSRNN).
Input: WAV/BWF/RF64, AIFF, CAF, FLAC, Ogg Opus/Vorbis, MP3, M4A/ALAC, AAC (built in; no ffmpeg).
Output: WAV, FLAC, Ogg Opus, MP3, M4A, AAC.

USAGE:
    denoize <INPUT> <OUTPUT.wav|flac|opus|ogg|mp3|m4a|aac> [OPTIONS]
    denoize live [--input-device NAME] [--output-device NAME] [OPTIONS]
    denoize live --list-devices
    denoize models <COMMAND> [MODEL|all] [OPTIONS]  (run `denoize models --help`)
    denoize metrics <REFERENCE> <TEST> [--json|--markdown]
    denoize compare <CLEAN> <NOISY> <ENHANCED> [--json|--html]

LIVE:
    Low-latency live processing supports only classical and rnnoise; other
    backends are rejected before capture or playback starts.

OPTIONS:
        --config <PATH>      load TOML defaults (CLI options take precedence)
    -b, --backend <NAME>     auto|{backends}  (default: classical)
    -a, --algorithm <NAME>   omlsa|logmmse|mmse|wiener|specsub|specsub-nl|specsub-geo
    -p, --preset <NAME>      speech|music|aggressive|gentle|restore|hifi
        --mode <NAME>        speech|music|ambient processing intent
    -s, --strength <0..1>    denoising strength (default: 0.6)
        --profile <MS>       finite duration: <0 off, 0 auto, >0 up to 60000
        --no-profile         no profiling; rely on blind IMCRA bootstrap
        --no-adapt           freeze the noise estimate
        --adaptive-noise     learn noise from noise-only regions throughout the file
        --vad                speech-aware segmentation and silence suppression
        --frame <N>          FFT size: power of two in 256..65536 (default: 2048)
        --overlap <F>        overlap ratio 0.5..0.95 (default: 0.75)
        --window <NAME>      hann|hamming|sine|blackman|kaiser|flattop|dpss
        --kaiser-beta <B>    finite Kaiser beta in 0..50 (default: 8.0)
        --dpss-nw <NW>       classical DPSS time-bandwidth product in (0, {MAX_DENOISER_DPSS_NW}] (default: 3.0)
        --multiband          enable multiband spectral subtraction
        --perceptual         enable Bark-scale perceptual gain weighting
        --postfilter         enable musical-noise suppression post-filter
        --smoothing <0..1>   gain release smoothing (default: 0.6)
        --makeup <DB>        makeup gain in -120..120 dB (default: 0.0)
        --no-dc-block        disable DC-blocking pre-filter
        --quality <LEVEL>    high|ultra
        --no-transient       disable transient/onset protection
        --cepstral           enable cepstral gain smoothing
        --no-cepstral        disable cepstral smoothing
        --pre-emphasis       enable pre/de-emphasis
        --no-pre-emphasis    disable pre-emphasis
        --report             print settings report and exit
        --mp3-bitrate <KBPS> MP3 CBR bitrate (default: 192)
        --m4a-bitrate <KBPS> positive M4A/AAC CBR bitrate (default: 192)
        --aac-encoder <NAME> oxide|fdk (default: oxide)
        --downmix <MODE>     preserve|stereo (default: preserve; lossy outputs reject surround unless explicit)
        --loudness <LUFS>     finite normalization target in -70..0 LUFS
        --true-peak <DBTP>    finite ceiling in -20..0 dBTP with --loudness (default: -1)
        --onnx-model <PATH>   waveform ONNX model (required for -b onnx)
        --onnx-rate <HZ>      model sample rate in 1..768000 Hz (default: 16000)
        --channels <MODE>     independent|linked|mid-side (default: independent)
        --sgmse-profile <P>   fast|balanced|quality (default: balanced)
        --deterministic       serialize processing for reproducible audio output
        --seed <N>            SGMSE sampler seed (implies --deterministic)
        --batch               process files in INPUT directory into OUTPUT directory
        --stream              bounded-memory classical WAV-to-WAV processing
        --stream-frames <N>   block size in 1..1048576 frames (default: 8192)
        --max-memory <MB>     per-input denoize allocation/metadata cap in MiB (regular files; min: 1)
        --recursive           include subdirectories in batch mode
        --jobs <N>            workers in 1..32 (default: min(CPU count, 32))
        --output-format <EXT> convert all batch outputs (required when source codec cannot be preserved)
        --force               allow replacing existing output files
        --resume              verify and skip exact outputs recorded by v3 batch state
        --no-progress         suppress batch progress and ETA output
        --json                emit a machine-readable result
        --no-metadata         do not copy input tags/artwork/chapters to the output
        --input-device <NAME> live capture device (default: system default)
        --output-device <NAME> live playback device (default: system default)
        --chunk-ms <MS>       live chunk duration in 10..2000 ms (default: 100)
    -h, --help               show this help
    -V, --version            show version

BACKENDS (build with --features full for all):
    classical   Enhanced STFT/IMCRA/OMLSA pipeline (default)
    rnnoise     RNNoise via nnnoiseless (requires --features rnnoise)
    deepfilter  DeepFilterNet v3 (requires --features deepfilter)
    onnx        External waveform ONNX model (requires --features onnx)
    mpsenet     MP-SENet magnitude/phase model (requires --features mpsenet)
    bsrnn       ESPnet BSRNN spectral model (requires --features bsrnn)
    mossformer2 ClearerVoice MossFormer2 model (requires --features mossformer2)
    sgmse       SGMSE+ diffusion model (requires --features sgmse)
    gtcrn       Official causal GTCRN for files/library streams (not the live command)

PRESETS:
    hifi        Flagship transparency: OMLSA + protections + advanced DSP
    speech      Voice-optimised balance
    music       Instruments; enables perceptual + postfilter

CONFIGURATION:
    TOML syntax and enum names are checked when loaded. CLI values then override
    TOML numeric defaults, and the final effective configuration is validated
    before audio decoding, output staging, or batch worker creation.
"
    )
}

#[derive(Clone, Debug, Default)]
struct Overrides {
    backend: Option<Backend>,
    auto_backend: bool,
    algorithm: Option<Algorithm>,
    preset: Option<Preset>,
    mode: Option<ProcessingMode>,
    strength: Option<f64>,
    profile_ms: Option<f64>,
    no_profile: bool,
    no_adapt: bool,
    adaptive_noise: Option<bool>,
    vad: Option<bool>,
    frame_size: Option<usize>,
    overlap: Option<f64>,
    window: Option<WindowType>,
    kaiser_beta: Option<f64>,
    dpss_nw: Option<f64>,
    multiband: bool,
    perceptual: bool,
    postfilter: bool,
    smoothing: Option<f64>,
    makeup: Option<f64>,
    no_dc_block: bool,
    report: bool,
    quality: Option<String>,
    no_transient: bool,
    cepstral: bool,
    no_cepstral: bool,
    pre_emphasis: bool,
    no_pre_emphasis: bool,
    mp3_bitrate_kbps: Option<u32>,
    m4a_bitrate_kbps: Option<u32>,
    aac_encoder: Option<AacEncoder>,
    downmix: Option<DownmixMode>,
    loudness_lufs: Option<f64>,
    true_peak_dbtp: Option<f64>,
    onnx_model: Option<String>,
    onnx_sample_rate: Option<u32>,
    channel_mode: Option<ChannelMode>,
    sgmse_profile: Option<SgmseProfile>,
    deterministic: bool,
    seed: Option<u64>,
    batch: bool,
    stream: bool,
    stream_frames: Option<usize>,
    max_memory_mb: Option<usize>,
    recursive: bool,
    jobs: Option<usize>,
    output_format: Option<String>,
    force: bool,
    resume: bool,
    no_progress: bool,
    json: bool,
    no_metadata: bool,
    input_device: Option<String>,
    output_device: Option<String>,
    chunk_ms: Option<u32>,
    list_devices: bool,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileConfig {
    backend: Option<String>,
    algorithm: Option<String>,
    preset: Option<String>,
    mode: Option<String>,
    strength: Option<f64>,
    profile_ms: Option<f64>,
    adaptive_noise: Option<bool>,
    vad: Option<bool>,
    frame_size: Option<usize>,
    overlap: Option<f64>,
    window: Option<String>,
    kaiser_beta: Option<f64>,
    dpss_nw: Option<f64>,
    smoothing: Option<f64>,
    makeup_db: Option<f64>,
    quality: Option<String>,
    mp3_bitrate_kbps: Option<u32>,
    m4a_bitrate_kbps: Option<u32>,
    aac_encoder: Option<String>,
    loudness_lufs: Option<f64>,
    true_peak_dbtp: Option<f64>,
    onnx_model: Option<String>,
    onnx_rate: Option<u32>,
    channels: Option<String>,
    sgmse_profile: Option<String>,
    downmix: Option<String>,
    deterministic: bool,
    seed: Option<u64>,
    batch: bool,
    stream: bool,
    stream_frames: Option<usize>,
    max_memory_mb: Option<usize>,
    recursive: bool,
    jobs: Option<usize>,
    output_format: Option<String>,
    force: bool,
    resume: bool,
    progress: Option<bool>,
    preserve_metadata: Option<bool>,
    chunk_ms: Option<u32>,
}

fn load_config(path: &str) -> Result<Overrides, String> {
    let source = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read config {path}: {error}"))?;
    parse_config(&source, path)
}

fn parse_quality(value: &str, source: &str) -> Result<String, String> {
    match value.to_ascii_lowercase().as_str() {
        "high" => Ok("high".into()),
        // Preserve the long-standing aliases while exposing one canonical
        // effective value to backend selection and the quality preset logic.
        "ultra" | "max" | "highest" => Ok("ultra".into()),
        _ => Err(format!(
            "unknown quality{source}: {value} (expected high or ultra)"
        )),
    }
}

fn parse_config(source: &str, path: &str) -> Result<Overrides, String> {
    let config: FileConfig =
        toml::from_str(source).map_err(|error| format!("invalid config {path}: {error}"))?;
    let mut ov = Overrides::default();
    if let Some(name) = config.backend {
        if name.eq_ignore_ascii_case("auto") {
            ov.auto_backend = true;
        } else {
            ov.backend = Some(
                Backend::parse(&name)
                    .ok_or_else(|| format!("unknown backend in config: {name}"))?,
            );
        }
    }
    if let Some(name) = config.algorithm {
        ov.algorithm = Some(
            Algorithm::parse(&name)
                .ok_or_else(|| format!("unknown algorithm in config: {name}"))?,
        );
    }
    if let Some(name) = config.preset {
        ov.preset =
            Some(Preset::parse(&name).ok_or_else(|| format!("unknown preset in config: {name}"))?);
    }
    if let Some(name) = config.mode {
        ov.mode = Some(
            ProcessingMode::parse(&name)
                .ok_or_else(|| format!("unknown mode in config: {name}"))?,
        );
    }
    if let Some(name) = config.window {
        ov.window = Some(
            WindowType::parse(&name).ok_or_else(|| format!("unknown window in config: {name}"))?,
        );
    }
    if let Some(name) = config.channels {
        ov.channel_mode = Some(
            ChannelMode::parse(&name)
                .ok_or_else(|| format!("unknown channel mode in config: {name}"))?,
        );
    }
    if let Some(name) = config.downmix {
        ov.downmix = Some(DownmixMode::parse(&name).ok_or_else(|| {
            format!("unknown downmix mode in config: {name} (expected preserve or stereo)")
        })?);
    }
    if let Some(name) = config.aac_encoder {
        ov.aac_encoder = Some(AacEncoder::parse(&name).ok_or_else(|| {
            format!("unknown AAC encoder in config: {name} (expected oxide or fdk)")
        })?);
    }
    if let Some(profile) = config.sgmse_profile {
        ov.sgmse_profile = Some(SgmseProfile::parse(&profile).ok_or_else(|| {
            format!(
                "unknown SGMSE profile in config: {profile} (expected fast, balanced, or quality)"
            )
        })?);
    }
    ov.strength = config.strength;
    ov.profile_ms = config.profile_ms;
    ov.adaptive_noise = config.adaptive_noise;
    ov.vad = config.vad;
    ov.frame_size = config.frame_size;
    ov.overlap = config.overlap;
    ov.kaiser_beta = config.kaiser_beta;
    ov.dpss_nw = config.dpss_nw;
    ov.smoothing = config.smoothing;
    ov.makeup = config.makeup_db;
    ov.quality = config
        .quality
        .map(|value| parse_quality(&value, " in config"))
        .transpose()?;
    ov.mp3_bitrate_kbps = config.mp3_bitrate_kbps;
    ov.m4a_bitrate_kbps = config.m4a_bitrate_kbps;
    ov.loudness_lufs = config.loudness_lufs;
    ov.true_peak_dbtp = if config.loudness_lufs.is_none() && config.true_peak_dbtp == Some(-1.0) {
        None
    } else {
        config.true_peak_dbtp
    };
    ov.onnx_model = config.onnx_model;
    ov.onnx_sample_rate = config.onnx_rate;
    ov.deterministic = config.deterministic;
    ov.seed = config.seed;
    if ov.seed.is_some() {
        ov.deterministic = true;
    }
    ov.batch = config.batch;
    ov.stream = config.stream;
    ov.stream_frames = config.stream_frames;
    ov.max_memory_mb = config.max_memory_mb;
    ov.recursive = config.recursive;
    ov.jobs = config.jobs;
    ov.output_format = config
        .output_format
        .map(|value| {
            normalize_output_extension(&value)
                .map(|extension| extension.to_ascii_lowercase())
                .map_err(|error| format!("{error} in config"))
        })
        .transpose()?;
    ov.force = config.force;
    ov.resume = config.resume;
    ov.no_progress = config.progress == Some(false);
    ov.no_metadata = config.preserve_metadata == Some(false);
    ov.chunk_ms = config.chunk_ms;
    Ok(ov)
}

fn parse_value<T>(args: &[String], i: &mut usize, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    *i += 1;
    if *i >= args.len() {
        return Err(format!("missing value for {flag}"));
    }
    args[*i]
        .parse::<T>()
        .map_err(|e| format!("invalid value for {flag}: {e}"))
}

fn parse_args(args: &[String]) -> Result<(String, String, Overrides), String> {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let config_path = args
        .windows(2)
        .find(|pair| pair[0] == "--config")
        .map(|pair| pair[1].as_str());
    if args.last().map(String::as_str) == Some("--config") {
        return Err("missing value for --config".into());
    }
    let mut ov = match config_path {
        Some(path) => load_config(path)?,
        None => Overrides::default(),
    };

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--config" => {
                let _: String = parse_value(args, &mut i, a)?;
            }
            "-h" | "--help" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            "-V" | "--version" => {
                println!("denoize {VERSION}");
                std::process::exit(0);
            }
            "-b" | "--backend" => {
                let name: String = parse_value(args, &mut i, a)?;
                if name.eq_ignore_ascii_case("auto") {
                    ov.auto_backend = true;
                    ov.backend = None;
                    i += 1;
                    continue;
                }
                ov.auto_backend = false;
                ov.backend = Some(Backend::parse(&name).ok_or_else(|| {
                    format!(
                        "unknown backend: {name} (available: {:?})",
                        Backend::available_names()
                    )
                })?);
            }
            "-a" | "--algorithm" => {
                let name: String = parse_value(args, &mut i, a)?;
                ov.algorithm = Some(
                    Algorithm::parse(&name).ok_or_else(|| format!("unknown algorithm: {name}"))?,
                );
            }
            "-p" | "--preset" => {
                let name: String = parse_value(args, &mut i, a)?;
                ov.preset =
                    Some(Preset::parse(&name).ok_or_else(|| format!("unknown preset: {name}"))?);
            }
            "--mode" => {
                let name: String = parse_value(args, &mut i, a)?;
                ov.mode = Some(ProcessingMode::parse(&name).ok_or_else(|| {
                    format!("unknown mode: {name} (expected speech, music, or ambient)")
                })?);
            }
            "-s" | "--strength" => ov.strength = Some(parse_value(args, &mut i, a)?),
            "--profile" => ov.profile_ms = Some(parse_value(args, &mut i, a)?),
            "--no-profile" => ov.no_profile = true,
            "--no-adapt" => ov.no_adapt = true,
            "--adaptive-noise" => ov.adaptive_noise = Some(true),
            "--vad" => ov.vad = Some(true),
            "--frame" => ov.frame_size = Some(parse_value(args, &mut i, a)?),
            "--overlap" => ov.overlap = Some(parse_value(args, &mut i, a)?),
            "--window" => {
                let name: String = parse_value(args, &mut i, a)?;
                ov.window = Some(
                    WindowType::parse(&name).ok_or_else(|| format!("unknown window: {name}"))?,
                );
            }
            "--kaiser-beta" => ov.kaiser_beta = Some(parse_value(args, &mut i, a)?),
            "--dpss-nw" => ov.dpss_nw = Some(parse_value(args, &mut i, a)?),
            "--multiband" => ov.multiband = true,
            "--perceptual" => ov.perceptual = true,
            "--postfilter" => ov.postfilter = true,
            "--smoothing" => ov.smoothing = Some(parse_value(args, &mut i, a)?),
            "--makeup" => ov.makeup = Some(parse_value(args, &mut i, a)?),
            "--no-dc-block" => ov.no_dc_block = true,
            "--report" => ov.report = true,
            "--quality" => {
                let q: String = parse_value(args, &mut i, a)?;
                ov.quality = Some(parse_quality(&q, "")?);
            }
            "--no-transient" => ov.no_transient = true,
            "--cepstral" => ov.cepstral = true,
            "--no-cepstral" => ov.no_cepstral = true,
            "--pre-emphasis" => ov.pre_emphasis = true,
            "--no-pre-emphasis" => ov.no_pre_emphasis = true,
            "--mp3-bitrate" => ov.mp3_bitrate_kbps = Some(parse_value(args, &mut i, a)?),
            "--m4a-bitrate" => ov.m4a_bitrate_kbps = Some(parse_value(args, &mut i, a)?),
            "--aac-encoder" => {
                let name: String = parse_value(args, &mut i, a)?;
                ov.aac_encoder = Some(AacEncoder::parse(&name).ok_or_else(|| {
                    format!("unknown AAC encoder: {name} (expected oxide or fdk)")
                })?);
            }
            "--downmix" => {
                let mode: String = parse_value(args, &mut i, a)?;
                ov.downmix = Some(DownmixMode::parse(&mode).ok_or_else(|| {
                    format!("unknown downmix mode: {mode} (expected preserve or stereo)")
                })?);
            }
            "--loudness" => ov.loudness_lufs = Some(parse_value(args, &mut i, a)?),
            "--true-peak" => ov.true_peak_dbtp = Some(parse_value(args, &mut i, a)?),
            "--onnx-model" => ov.onnx_model = Some(parse_value(args, &mut i, a)?),
            "--onnx-rate" => ov.onnx_sample_rate = Some(parse_value(args, &mut i, a)?),
            "--channels" => {
                let mode: String = parse_value(args, &mut i, a)?;
                ov.channel_mode = Some(ChannelMode::parse(&mode).ok_or_else(|| {
                    format!(
                        "unknown channel mode: {mode} (expected independent, linked, or mid-side)"
                    )
                })?);
            }
            "--sgmse-profile" => {
                let profile: String = parse_value(args, &mut i, a)?;
                ov.sgmse_profile = Some(SgmseProfile::parse(&profile).ok_or_else(|| {
                    format!(
                        "unknown SGMSE profile: {profile} (expected fast, balanced, or quality)"
                    )
                })?);
            }
            "--deterministic" => ov.deterministic = true,
            "--seed" => {
                ov.seed = Some(parse_value(args, &mut i, a)?);
                ov.deterministic = true;
            }
            "--batch" => ov.batch = true,
            "--stream" => ov.stream = true,
            "--stream-frames" => ov.stream_frames = Some(parse_value(args, &mut i, a)?),
            "--max-memory" => ov.max_memory_mb = Some(parse_value(args, &mut i, a)?),
            "--recursive" => ov.recursive = true,
            "--jobs" => ov.jobs = Some(parse_value(args, &mut i, a)?),
            "--output-format" => {
                let value: String = parse_value(args, &mut i, a)?;
                ov.output_format = Some(normalize_output_extension(&value)?.to_ascii_lowercase());
            }
            "--force" => ov.force = true,
            "--resume" => ov.resume = true,
            "--no-progress" => ov.no_progress = true,
            "--json" => ov.json = true,
            "--no-metadata" => ov.no_metadata = true,
            "--input-device" => ov.input_device = Some(parse_value(args, &mut i, a)?),
            "--output-device" => ov.output_device = Some(parse_value(args, &mut i, a)?),
            "--chunk-ms" => ov.chunk_ms = Some(parse_value(args, &mut i, a)?),
            "--list-devices" => ov.list_devices = true,
            "-" => {
                if input.is_none() {
                    input = Some(a.clone());
                } else if output.is_none() {
                    output = Some(a.clone());
                } else {
                    return Err("unexpected extra argument: -".into());
                }
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown option: {other}"));
            }
            _ => {
                if input.is_none() {
                    input = Some(a.clone());
                } else if output.is_none() {
                    output = Some(a.clone());
                } else {
                    return Err(format!("unexpected extra argument: {a}"));
                }
            }
        }
        i += 1;
    }

    // Validate the fully merged, effective configuration before looking at
    // positional paths. This keeps configuration errors deterministic and
    // guarantees that invalid values cannot trigger input/output I/O.
    validate_effective_options(&ov, VALIDATION_SAMPLE_RATE)?;
    let input = input.ok_or("missing INPUT")?;
    let output = output.ok_or("missing OUTPUT audio path")?;
    Ok((input, output, ov))
}

fn checked_memory_limit_bytes(max_memory_mb: Option<usize>) -> Result<Option<u64>, String> {
    let Some(max_memory_mb) = max_memory_mb else {
        return Ok(None);
    };
    if max_memory_mb == 0 {
        return Err("--max-memory must be at least 1 MiB".into());
    }
    let max_memory_mb = u64::try_from(max_memory_mb)
        .map_err(|_| "--max-memory is too large to represent safely")?;
    max_memory_mb
        .checked_mul(BYTES_PER_MIB)
        .map(Some)
        .ok_or_else(|| "--max-memory is too large to represent safely".into())
}

/// Derive parser limits from bytes which are still available to metadata.
///
/// Metadata is represented more than once while it is translated between a
/// native container and Lofty's generic model. Reserving only one sixteenth
/// of the available working-set budget for payload keeps those copies and
/// allocator overhead conservative. Descriptor counts receive their own
/// finite bound so a stream of empty fields, pages, or blocks cannot evade the
/// byte limits.
fn metadata_limits_for_available_bytes(available: Option<u64>) -> MetadataLimits {
    let defaults = MetadataLimits::default();
    let Some(available) = available else {
        return defaults;
    };
    let payload_bytes = available / METADATA_REPRESENTATION_EXPANSION_FACTOR;
    let descriptor_count = payload_bytes / METADATA_DESCRIPTOR_BYTES;
    let payload = usize::try_from(payload_bytes).unwrap_or(usize::MAX);
    let descriptors = usize::try_from(descriptor_count).unwrap_or(usize::MAX);

    let mut limits = defaults;
    limits.max_total_bytes = defaults.max_total_bytes.min(payload);
    limits.max_item_bytes = defaults.max_item_bytes.min(payload);
    limits.max_items = defaults.max_items.min(descriptors);
    limits.max_flac_block_bytes = defaults.max_flac_block_bytes.min(payload);
    limits.max_flac_blocks = defaults.max_flac_blocks.min(descriptors);
    limits.max_ogg_packet_bytes = defaults.max_ogg_packet_bytes.min(payload);
    limits.max_ogg_pages = defaults.max_ogg_pages.min(descriptors);
    limits.max_ogg_streams = defaults.max_ogg_streams.min(descriptors);
    limits
}

fn decode_limits(max_memory_mb: Option<usize>) -> Result<DecodeLimits, String> {
    let max_working_set_bytes = checked_memory_limit_bytes(max_memory_mb)?;
    Ok(DecodeLimits::new(
        metadata_limits_for_available_bytes(max_working_set_bytes),
        max_working_set_bytes,
    ))
}

fn decode_metadata_limits(max_memory_mb: Option<usize>) -> Result<MetadataLimits, String> {
    Ok(decode_limits(max_memory_mb)?.metadata)
}

fn retained_metadata_limits(
    max_memory_mb: Option<usize>,
    retained_working_set_bytes: u64,
) -> Result<MetadataLimits, String> {
    let available = checked_memory_limit_bytes(max_memory_mb)?
        .map(|limit| limit.saturating_sub(retained_working_set_bytes));
    let mut retained = metadata_limits_for_available_bytes(available);

    // The decoder already proved the container structure against the full
    // per-file cap. Reuse those structural bounds when the optional metadata
    // payload has little or no remaining budget: a tagless FLAC still has a
    // mandatory 34-byte STREAMINFO block, and rejecting that block would make
    // `--max-memory 1` fail despite retaining no metadata at all.
    let structural = decode_metadata_limits(max_memory_mb)?;
    retained.max_flac_block_bytes = structural.max_flac_block_bytes;
    retained.max_flac_blocks = structural.max_flac_blocks;
    retained.max_ogg_packet_bytes = structural.max_ogg_packet_bytes;
    retained.max_ogg_pages = structural.max_ogg_pages;
    retained.max_ogg_streams = structural.max_ogg_streams;
    Ok(retained)
}

fn checked_m4a_bitrate_bps(kbps: u32) -> Result<u32, String> {
    kbps.checked_mul(1000).ok_or_else(|| {
        format!(
            "invalid --m4a-bitrate/m4a_bitrate_kbps value {kbps}: converting from kbps to bps exceeds the supported u32 representation (maximum {} kbps)",
            u32::MAX / 1000
        )
    })
}

fn build_encode_options(ov: &Overrides) -> Result<EncodeOptions, String> {
    let mut options = EncodeOptions::default();
    if let Some(kbps) = ov.mp3_bitrate_kbps {
        options.mp3_bitrate_kbps = kbps;
    }
    if let Some(kbps) = ov.m4a_bitrate_kbps {
        options.m4a_bitrate_bps = checked_m4a_bitrate_bps(kbps)?;
    }
    if let Some(encoder) = ov.aac_encoder {
        options.aac_encoder = encoder;
    }
    if let Some(downmix) = ov.downmix {
        options.downmix = downmix;
    }
    Ok(options)
}

fn validate_encode_preflight(
    options: EncodeOptions,
    formats: impl IntoIterator<Item = OutputFormat>,
) -> Result<(), String> {
    for format in formats {
        options.validate_options(format)?;
    }
    Ok(())
}

fn preflight_batch_items(
    items: &[BatchItem],
    ov: &Overrides,
    options: EncodeOptions,
    pre_resolved_backend_options: Option<&BackendOptions>,
) -> Result<Vec<PreparedBatchItem>, String> {
    let metadata_policy = if ov.no_metadata {
        MetadataPolicy::Drop
    } else {
        MetadataPolicy::Preserve
    };
    let mut model_fingerprints =
        std::collections::HashMap::<(std::path::PathBuf, u32), ConsumedModel>::new();
    let mut prepared = Vec::with_capacity(items.len());
    let decode_limits = decode_limits(ov.max_memory_mb)?;
    for item in items {
        let mut input_session = AudioInputSession::open(&item.input).map_err(|error| {
            format!(
                "open batch input {} during preflight: {error}",
                item.input.display()
            )
        })?;
        let current_probe = probe_audio_session_with_limits(&mut input_session, decode_limits)
            .map_err(|error| {
                format!(
                    "probe batch input {} during preflight: {error}",
                    item.input.display()
                )
            })?;
        if current_probe != item.probe {
            return Err(format!(
                "batch input codec/container changed after planning: {}",
                item.input.display()
            ));
        }
        let input_fingerprint = batch_resume::fingerprint_input_session(&mut input_session)
            .map_err(|error| {
                format!(
                    "fingerprint batch input {} during preflight: {error}",
                    item.input.display()
                )
            })?;
        let estimate = estimate_session_memory_bytes(&input_session);
        ensure_memory_limit(estimate, ov.max_memory_mb, "batch input preflight")?;
        let audio = read_audio_from_session_with_limits(&mut input_session, decode_limits)
            .map_err(|error| {
                format!(
                    "decode batch input {} during preflight: {error}",
                    item.input.display()
                )
            })?;
        let decoded_working_set = estimate_audio_working_set_bytes(&audio);
        ensure_memory_limit(
            decoded_working_set,
            ov.max_memory_mb,
            "batch decoded audio working set",
        )?;
        if !ov.no_metadata {
            let metadata_limits = retained_metadata_limits(ov.max_memory_mb, decoded_working_set)?;
            input_session
                .read_metadata_with_limits(metadata_limits)
                .map_err(|error| {
                    format!(
                        "read batch input metadata {} during preflight: {error}",
                        item.input.display()
                    )
                })?;
        }
        item.output_format
            .validate_config(&audio, &options)
            .map_err(|error| {
                format!(
                    "batch output preflight failed for {}: {error}",
                    item.input.display()
                )
            })?;
        let resolved_processing = service::resolve_processing_options(
            &audio,
            build_processing_options(
                ov,
                audio.sample_rate,
                pre_resolved_backend_options
                    .cloned()
                    .unwrap_or_else(|| build_backend_options(ov)),
            ),
        )
        .map_err(|error| {
            format!(
                "batch processing preflight failed for {}: {error}",
                item.input.display()
            )
        })?;
        let model = match batch_resume::consumed_model_config(&resolved_processing)? {
            Some(config) => {
                let key = (config.path.clone(), config.sample_rate);
                let model = match model_fingerprints.get(&key) {
                    Some(model) => model.clone(),
                    None => {
                        let model = if ov.resume {
                            batch_resume::fingerprint_resumable_model(config)
                        } else {
                            batch_resume::fingerprint_consumed_model(config)
                        }
                        .map_err(|error| {
                            format!(
                                "fingerprint selected backend model {}: {error}",
                                config.path.display()
                            )
                        })?;
                        model_fingerprints.insert(key, model.clone());
                        model
                    }
                };
                Some(model)
            }
            None => None,
        };
        let recipe = batch_resume::recipe_digest(
            &resolved_processing,
            audio.channels(),
            item.output_format,
            options,
            metadata_policy,
            model
                .as_ref()
                .map(|model| (&model.fingerprint, model.sample_rate)),
        )?;
        let input_identity = normalize_batch_path(&item.input)?;
        let item_id = batch_resume::item_identity(
            &input_identity,
            &item.input_relative,
            &item.destination_relative,
            item.output_format,
        );
        let expectation = ResumeExpectation::new(
            item_id,
            item.destination.clone(),
            item.input.clone(),
            input_fingerprint,
            model,
            recipe,
        );
        prepared.push(PreparedBatchItem {
            item: item.clone(),
            resolved_processing,
            expectation,
            recipe,
        });
    }
    // A cached model fingerprint is safe only if the model still matches once
    // the complete plan has been built. Inputs receive the same whole-plan
    // fence before the output directory or state can be touched.
    for item in &prepared {
        item.expectation.verify_sources()?;
    }
    debug_assert_eq!(prepared.len(), items.len());
    Ok(prepared)
}

fn effective_batch_jobs(ov: &Overrides) -> usize {
    ov.jobs.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(MAX_BATCH_JOBS)
    })
}

fn build_backend_options(ov: &Overrides) -> BackendOptions {
    BackendOptions {
        onnx: ov.onnx_model.as_ref().map(|path| OnnxModelConfig {
            path: path.into(),
            sample_rate: ov.onnx_sample_rate.unwrap_or(16_000),
        }),
        channel_mode: ov.channel_mode.unwrap_or_default(),
        sgmse_profile: ov.sgmse_profile.unwrap_or_default(),
        deterministic: ov.deterministic,
        seed: ov.seed,
    }
}

fn processing_backend_choice(ov: &Overrides) -> BackendChoice {
    if ov.auto_backend {
        BackendChoice::Auto
    } else {
        BackendChoice::Explicit(ov.backend.unwrap_or(Backend::Classical))
    }
}

fn build_processing_options(
    ov: &Overrides,
    sample_rate: u32,
    backend_options: BackendOptions,
) -> ProcessingOptions {
    ProcessingOptions {
        backend: processing_backend_choice(ov),
        quality: ov.quality.clone(),
        denoiser: build_config(ov, sample_rate),
        backend_options,
        loudness_lufs: ov.loudness_lufs,
        true_peak_dbtp: ov.true_peak_dbtp.unwrap_or(-1.0),
    }
}

fn resolve_explicit_backend_options(ov: &Overrides) -> Result<Option<BackendOptions>, String> {
    if ov.auto_backend {
        return Ok(None);
    }
    let backend = ov.backend.unwrap_or(Backend::Classical);
    service::resolve_backend_options(backend, build_backend_options(ov)).map(Some)
}

fn validate_effective_options(ov: &Overrides, sample_rate: u32) -> Result<(), String> {
    // `parse_config` deliberately postpones numeric checks until after CLI
    // overrides have been applied. Validate only this final effective config.
    build_config(ov, sample_rate)
        .validate_config()
        .map_err(|error| error.to_string())?;

    if let Some(loudness) = ov.loudness_lufs {
        if !loudness.is_finite() || !(-70.0..=0.0).contains(&loudness) {
            return Err(format!(
                "invalid --loudness/loudness_lufs value {loudness}: expected a finite value in [-70, 0] LUFS"
            ));
        }
    }
    if let Some(true_peak) = ov.true_peak_dbtp {
        if !true_peak.is_finite() || !(-20.0..=0.0).contains(&true_peak) {
            return Err(format!(
                "invalid --true-peak/true_peak_dbtp value {true_peak}: expected a finite value in [-20, 0] dBTP"
            ));
        }
        if ov.loudness_lufs.is_none() {
            return Err("--true-peak requires --loudness".into());
        }
    }
    if let Some(sample_rate) = ov.onnx_sample_rate {
        if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
            return Err(format!(
                "--onnx-rate/onnx_rate must be in 1..={MAX_SAMPLE_RATE} Hz"
            ));
        }
    }
    if !ov.auto_backend {
        build_backend_options(ov)
            .validate_config(ov.backend.unwrap_or(Backend::Classical))
            .map_err(|error| error.to_string())?;
    }
    if let Some(stream_frames) = ov.stream_frames {
        if !(MIN_STREAM_BLOCK_FRAMES..=MAX_STREAM_BLOCK_FRAMES).contains(&stream_frames) {
            return Err(format!(
                "--stream-frames/stream_frames must be in {MIN_STREAM_BLOCK_FRAMES}..={MAX_STREAM_BLOCK_FRAMES}"
            ));
        }
    }
    if let Some(chunk_ms) = ov.chunk_ms {
        if !(MIN_LIVE_CHUNK_MS..=MAX_LIVE_CHUNK_MS).contains(&chunk_ms) {
            return Err(format!(
                "--chunk-ms must be in {MIN_LIVE_CHUNK_MS}..={MAX_LIVE_CHUNK_MS}"
            ));
        }
    }
    if let Some(jobs) = ov.jobs {
        if !(1..=MAX_BATCH_JOBS).contains(&jobs) {
            return Err(format!("--jobs/jobs must be in 1..={MAX_BATCH_JOBS}"));
        }
    }
    let encode_options = build_encode_options(ov)?;
    if let Some(extension) = ov.output_format.as_deref() {
        let path = std::path::PathBuf::from(format!("output.{extension}"));
        validate_encode_preflight(encode_options, [OutputFormat::from_path(&path)?])?;
    }
    checked_memory_limit_bytes(ov.max_memory_mb)?;
    Ok(())
}

fn build_config(ov: &Overrides, sample_rate: u32) -> DenoiserConfig {
    let mut cfg = match ov.preset {
        Some(p) => p.config(sample_rate),
        None => DenoiserConfig::default(sample_rate),
    };
    if let Some(mode) = ov.mode {
        mode.apply(&mut cfg);
    }
    if let Some(a) = ov.algorithm {
        cfg.algorithm = a;
    }
    if let Some(s) = ov.strength {
        cfg.strength = s;
    }
    if ov.no_profile {
        cfg.profile_ms = -1.0;
    } else if let Some(ms) = ov.profile_ms {
        cfg.profile_ms = ms;
    }
    if ov.no_adapt {
        cfg.adapt = false;
    }
    if let Some(adaptive_noise) = ov.adaptive_noise {
        cfg.adaptive_noise = adaptive_noise;
    }
    if let Some(vad) = ov.vad {
        cfg.vad = vad;
    }
    if let Some(f) = ov.frame_size {
        cfg.frame_size = f;
    }
    if let Some(o) = ov.overlap {
        cfg.overlap = o;
    }
    if let Some(w) = ov.window {
        cfg.window = w;
    }
    if let Some(b) = ov.kaiser_beta {
        cfg.window_params.kaiser_beta = b;
    }
    if let Some(nw) = ov.dpss_nw {
        cfg.window_params.dpss_bandwidth = nw;
    }
    if ov.multiband {
        cfg.multiband = true;
    }
    if ov.perceptual {
        cfg.perceptual_weighting = true;
    }
    if ov.postfilter {
        cfg.musical_noise_postfilter = true;
    }
    if let Some(s) = ov.smoothing {
        cfg.smoothing = s;
    }
    if let Some(m) = ov.makeup {
        cfg.makeup_gain_db = m;
    }
    if ov.no_dc_block {
        cfg.dc_block = false;
    }

    if let Some(ref q) = ov.quality {
        match q.as_str() {
            "high" => {
                if cfg.frame_size < 2048 {
                    cfg.frame_size = 2048;
                }
                if cfg.overlap < 0.8 {
                    cfg.overlap = 0.8;
                }
                cfg.transient_protect = true;
                cfg.cepstral_smoothing = true;
                cfg.perceptual_weighting = true;
                cfg.musical_noise_postfilter = true;
                if !ov.no_pre_emphasis {
                    cfg.pre_emphasis = true;
                }
            }
            "ultra" | "max" | "highest" => {
                cfg.frame_size = cfg.frame_size.max(4096);
                cfg.overlap = 0.875;
                if ov.window.is_none() {
                    cfg.window = WindowType::Kaiser;
                }
                if ov.kaiser_beta.is_none() {
                    cfg.window_params.kaiser_beta = 10.0;
                }
                cfg.transient_protect = true;
                cfg.cepstral_smoothing = true;
                cfg.perceptual_weighting = true;
                cfg.musical_noise_postfilter = true;
                cfg.pre_emphasis = true;
                if ov.strength.is_none() && cfg.strength > 0.4 {
                    cfg.strength = 0.32;
                }
            }
            _ => {}
        }
    }

    if ov.no_transient {
        cfg.transient_protect = false;
    }
    if ov.cepstral {
        cfg.cepstral_smoothing = true;
    }
    if ov.no_cepstral {
        cfg.cepstral_smoothing = false;
    }
    if ov.pre_emphasis {
        cfg.pre_emphasis = true;
    }
    if ov.no_pre_emphasis {
        cfg.pre_emphasis = false;
    }

    cfg
}

fn print_report(
    input: &std::path::Path,
    audio: &denoize::Audio,
    cfg: &DenoiserConfig,
    backend: Backend,
) {
    let hop = (cfg.frame_size as f64 * (1.0 - cfg.overlap)).round() as usize;
    let g_min_db = -20.0 - 25.0 * cfg.strength;
    let dur = audio.frames() as f64 / audio.sample_rate as f64;
    println!("input      : {}", input.display());
    println!(
        "format     : {}ch, {:.2}s ({} frames), {} Hz, {}-bit {:?}",
        audio.channels(),
        dur,
        audio.frames(),
        audio.sample_rate,
        audio.bits_per_sample,
        audio.sample_format,
    );
    println!("layout     : {}", audio.channel_layout());
    if let Some(mask) = audio.channel_mask {
        println!("mask       : {mask}");
    }
    if let Some(pan) = audio.pan_info() {
        let positions = pan
            .iter()
            .enumerate()
            .map(|(index, info)| {
                format!(
                    "ch{}={:.0}°/{:.0}°",
                    index + 1,
                    info.azimuth_degrees,
                    info.elevation_degrees
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        println!("pan        : {positions}");
    }
    println!("backend    : {backend:?}");
    println!("algorithm  : {:?}", cfg.algorithm);
    println!(
        "strength   : {:.2}  (gain floor ~{:.0} dB)",
        cfg.strength, g_min_db
    );
    println!(
        "STFT       : frame={}, hop={}, overlap={:.0}%, window={:?}",
        cfg.frame_size,
        hop,
        cfg.overlap * 100.0,
        cfg.window,
    );
    println!(
        "advanced   : multiband={}, perceptual={}, postfilter={}",
        cfg.multiband, cfg.perceptual_weighting, cfg.musical_noise_postfilter
    );
    println!("smoothing  : {:.2}", cfg.smoothing);
    println!(
        "profile    : {}",
        if cfg.profile_ms < 0.0 {
            "disabled".to_string()
        } else if cfg.profile_ms == 0.0 {
            "auto (leading silence)".to_string()
        } else {
            format!("{:.0} ms", cfg.profile_ms)
        }
    );
    println!("adapt      : {}", cfg.adapt);
    println!("adaptive-profile: {}", cfg.adaptive_noise);
    println!("dc-block   : {}", cfg.dc_block);
    println!("makeup     : {:.1} dB", cfg.makeup_gain_db);
    println!(
        "hi-fi      : transient={}, cepstral={}, pre-emphasis={}",
        cfg.transient_protect, cfg.cepstral_smoothing, cfg.pre_emphasis
    );
}

fn run(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) == Some("live") {
        return run_live(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("models") {
        return run_models(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("metrics") {
        return run_metrics(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("compare") {
        return run_compare(&args[1..]);
    }
    let (input, output, ov) = parse_args(args)?;
    if ov.batch {
        if ov.stream {
            return Err("--stream cannot be combined with --batch".into());
        }
        return run_batch(&input, &output, &ov);
    }
    if ov.stream {
        return run_streaming_wav(&input, &output, ov);
    }
    run_one(&input, &output, ov)
}

#[cfg(feature = "live")]
fn run_live(args: &[String]) -> Result<(), String> {
    let mut parseable = vec!["-".to_string(), "-".to_string()];
    parseable.extend_from_slice(args);
    let (_, _, ov) = parse_args(&parseable)?;
    validate_effective_options(&ov, 48_000)?;
    if ov.list_devices {
        let (inputs, outputs) = denoize::live::device_names()?;
        println!("Input devices:");
        for device in inputs {
            println!("  {device}");
        }
        println!("Output devices:");
        for device in outputs {
            println!("  {device}");
        }
        return Ok(());
    }
    let backend = if ov.auto_backend {
        service::select_live_backend()
    } else {
        ov.backend.unwrap_or(Backend::Classical)
    };
    let sample_rate = 48_000;
    let denoiser = build_config(&ov, sample_rate);
    let backend_options = build_backend_options(&ov);
    denoize::live::run(denoize::live::LiveConfig {
        input_device: ov.input_device,
        output_device: ov.output_device,
        chunk_ms: ov.chunk_ms.unwrap_or(100),
        backend,
        backend_options,
        denoiser,
    })
}

#[cfg(not(feature = "live"))]
fn run_live(_args: &[String]) -> Result<(), String> {
    Err("live audio is unavailable in this build; rebuild with --features live".into())
}

fn ensure_output_available(path: &std::path::Path, force: bool) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if force && (metadata.is_file() || metadata.file_type().is_symlink()) => {
            Ok(())
        }
        Ok(_) if force => Err(format!(
            "output exists but is not a replaceable file or symlink: {}",
            path.display()
        )),
        Ok(_) => Err(format!(
            "output already exists: {} (use --force to replace it)",
            path.display()
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "inspect output destination {}: {error}",
            path.display()
        )),
    }
}

fn read_stdin_bytes(
    mut input: impl std::io::Read,
    max_memory_mb: Option<usize>,
) -> Result<Vec<u8>, String> {
    let max_encoded_bytes = checked_memory_limit_bytes(max_memory_mb)?
        .map(|limit| limit / INPUT_MEMORY_EXPANSION_FACTOR);
    let bounded_read_len = max_encoded_bytes
        .map(|limit| {
            limit
                .checked_add(1)
                .ok_or_else(|| "--max-memory stdin byte limit overflow".to_string())
                .and_then(|limit| {
                    usize::try_from(limit)
                        .map_err(|_| "--max-memory stdin byte limit is too large".to_string())
                })
        })
        .transpose()?;
    let initial_capacity = bounded_read_len
        .unwrap_or(STDIN_READ_CHUNK_BYTES)
        .min(STDIN_READ_CHUNK_BYTES);
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(initial_capacity)
        .map_err(|_| "unable to reserve memory for stdin input".to_string())?;

    let mut chunk = [0u8; STDIN_READ_CHUNK_BYTES];
    loop {
        let read_len = match bounded_read_len {
            Some(limit) => {
                let remaining = limit
                    .checked_sub(bytes.len())
                    .ok_or_else(|| "stdin byte limit accounting overflow".to_string())?;
                if remaining == 0 {
                    break;
                }
                remaining.min(chunk.len())
            }
            None => chunk.len(),
        };
        let read = input
            .read(&mut chunk[..read_len])
            .map_err(|error| format!("failed to read stdin: {error}"))?;
        if read == 0 {
            break;
        }
        bytes
            .try_reserve_exact(read)
            .map_err(|_| "unable to reserve memory for stdin input".to_string())?;
        bytes.extend_from_slice(&chunk[..read]);
    }

    let encoded_len = u64::try_from(bytes.len())
        .map_err(|_| "stdin input length is too large to represent safely".to_string())?;
    let estimate = encoded_len
        .checked_mul(INPUT_MEMORY_EXPANSION_FACTOR)
        .ok_or_else(|| "stdin input memory estimate overflow".to_string())?
        .max(BYTES_PER_MIB);
    ensure_memory_limit(estimate, max_memory_mb, "stdin input preflight")?;
    Ok(bytes)
}

fn run_one(input: &str, output: &str, ov: Overrides) -> Result<(), String> {
    run_one_with_output_format(
        std::path::Path::new(input),
        std::path::Path::new(output),
        ov,
        None,
        None,
    )
}

struct StagedProcessOutput {
    transaction: AtomicOutput,
    effective_recipe: Option<Digest>,
    backend: Backend,
    channels: usize,
    frames: usize,
    sample_rate: u32,
    elapsed_ms: f64,
}

fn run_one_with_output_format(
    input: &std::path::Path,
    output: &std::path::Path,
    ov: Overrides,
    planned_output_format: Option<OutputFormat>,
    pre_resolved_backend_options: Option<BackendOptions>,
) -> Result<(), String> {
    let commit_mode = if ov.force {
        CommitMode::Replace
    } else {
        CommitMode::NoClobber
    };
    let json = ov.json;
    let staged = process_one_to_staged_output(
        input,
        output,
        ov,
        planned_output_format,
        pre_resolved_backend_options,
        None,
        None,
        None,
        None,
        true,
    )?;
    let Some(staged) = staged else {
        return Ok(());
    };
    staged.transaction.commit(commit_mode)?;
    if json {
        let input = input.to_string_lossy();
        let output = output.to_string_lossy();
        println!(
            "{}",
            process_result_json_line(
                input.as_ref(),
                output.as_ref(),
                service::backend_name(staged.backend),
                staged.channels,
                staged.frames,
                staged.sample_rate,
                staged.elapsed_ms,
            )
        );
    }
    Ok(())
}

fn process_one_to_staged_output(
    input: &std::path::Path,
    output: &std::path::Path,
    ov: Overrides,
    planned_output_format: Option<OutputFormat>,
    pre_resolved_backend_options: Option<BackendOptions>,
    pre_resolved_processing: Option<service::ResolvedProcessingOptions>,
    recipe_metadata_policy: Option<MetadataPolicy>,
    expected_input_probe: Option<AudioProbe>,
    expected_input_fingerprint: Option<FileFingerprint>,
    inspect_destination: bool,
) -> Result<Option<StagedProcessOutput>, String> {
    validate_effective_options(&ov, VALIDATION_SAMPLE_RATE)?;
    let standard_input = input == std::path::Path::new("-");
    let standard_output = output == std::path::Path::new("-");
    let encode_options = build_encode_options(&ov)?;
    let output_format = if !ov.report && !standard_output {
        Some(match planned_output_format {
            Some(format) => format,
            None => OutputFormat::from_path(output)?,
        })
    } else {
        None
    };
    validate_encode_preflight(encode_options, output_format)?;

    let resolved_backend_options = match (
        pre_resolved_processing.as_ref(),
        pre_resolved_backend_options,
    ) {
        (Some(_), _) => None,
        (None, Some(options)) => Some(options),
        (None, None) => resolve_explicit_backend_options(&ov)?,
    };
    if inspect_destination && output_format.is_some() {
        ensure_output_available(output, ov.force)?;
    }
    let mut input_session = if standard_input {
        None
    } else {
        Some(AudioInputSession::open(input)?)
    };
    if let (Some(session), Some(expected)) = (&mut input_session, expected_input_probe) {
        let current = probe_audio_session_with_limits(session, decode_limits(ov.max_memory_mb)?)?;
        if current != expected {
            return Err(format!(
                "input codec/container changed after batch preflight: {}",
                input.display()
            ));
        }
    }
    if let (Some(session), Some(expected)) = (&mut input_session, expected_input_fingerprint) {
        let current = batch_resume::fingerprint_input_session(session)?;
        if current != expected {
            return Err(format!(
                "input bytes changed after batch preflight: {}",
                input.display()
            ));
        }
    }
    if let Some(session) = &input_session {
        let estimate = estimate_session_memory_bytes(session);
        ensure_memory_limit(estimate, ov.max_memory_mb, "input preflight")?;
    }
    let decode_limits = decode_limits(ov.max_memory_mb)?;
    let mut audio = if standard_input {
        let stdin = std::io::stdin();
        read_wav_bytes_with_limits(
            read_stdin_bytes(stdin.lock(), ov.max_memory_mb)?,
            decode_limits,
        )?
    } else {
        read_audio_from_session_with_limits(
            input_session
                .as_mut()
                .expect("filesystem input session was opened"),
            decode_limits,
        )?
    };
    let decoded_working_set = estimate_audio_working_set_bytes(&audio);
    ensure_memory_limit(
        decoded_working_set,
        ov.max_memory_mb,
        "decoded audio working set",
    )?;
    let metadata_limits = retained_metadata_limits(ov.max_memory_mb, decoded_working_set)?;
    let metadata = if !standard_input && !ov.no_metadata {
        input_session
            .as_mut()
            .expect("filesystem input session was opened")
            .read_metadata_with_limits(metadata_limits)?
    } else {
        None
    };
    validate_effective_options(&ov, audio.sample_rate)?;
    let resolved_processing = match pre_resolved_processing {
        Some(options) => options,
        None => service::resolve_processing_options(
            &audio,
            build_processing_options(
                &ov,
                audio.sample_rate,
                resolved_backend_options.unwrap_or_else(|| build_backend_options(&ov)),
            ),
        )?,
    };
    let backend = resolved_processing.backend;
    if ov.auto_backend && !ov.json {
        eprintln!(
            "denoize: auto-selected backend {}",
            service::backend_name(backend)
        );
    }

    if ov.report {
        print_report(input, &audio, &resolved_processing.denoiser, backend);
        return Ok(None);
    }

    if let Some(format) = output_format {
        format.validate_config(&audio, &encode_options)?;
    }

    let result = service::process_audio_resolved(&mut audio, &resolved_processing)?;
    if let Some(report) = result.loudness {
        if !ov.json {
            eprintln!(
                "denoize: loudness {:.2} -> {:.2} LUFS, true peak {:.2} dBTP, gain {:+.2} dB",
                report.input_lufs, report.output_lufs, report.true_peak_dbtp, report.gain_db
            );
        }
    } else if ov.true_peak_dbtp.is_some() {
        return Err("--true-peak requires --loudness".into());
    }
    if standard_output {
        let bytes = write_wav_bytes(&audio)?;
        std::io::Write::write_all(&mut std::io::stdout(), &bytes)
            .map_err(|error| format!("failed to write stdout: {error}"))?;
        Ok(None)
    } else {
        let output_format = output_format.expect("filesystem output was preflighted");
        let effective_recipe = recipe_metadata_policy
            .map(|metadata_policy| {
                let model = batch_resume::consumed_model(&resolved_processing)?;
                batch_resume::recipe_digest(
                    &resolved_processing,
                    audio.channels(),
                    output_format,
                    encode_options,
                    metadata_policy,
                    model
                        .as_ref()
                        .map(|model| (&model.fingerprint, model.sample_rate)),
                )
            })
            .transpose()?;
        let mut transaction = AtomicOutput::new(output)?;
        denoize::encode::write_audio_to_file(
            transaction.file_mut(),
            output_format,
            &audio,
            encode_options,
        )?;
        if let Some(metadata) = metadata {
            denoize::metadata::write_extended_to_file_with_limits(
                metadata,
                transaction.file_mut(),
                metadata_limits,
            )?;
        }
        Ok(Some(StagedProcessOutput {
            transaction,
            effective_recipe,
            backend: result.backend,
            channels: audio.channels(),
            frames: audio.frames(),
            sample_rate: audio.sample_rate,
            elapsed_ms: result.elapsed.as_secs_f64() * 1_000.0,
        }))
    }
}

fn run_streaming_wav(input: &str, output: &str, ov: Overrides) -> Result<(), String> {
    validate_effective_options(&ov, VALIDATION_SAMPLE_RATE)?;
    if input == "-" || output == "-" {
        return Err("--stream requires filesystem WAV input and output paths".into());
    }
    let input_path = std::path::Path::new(input);
    let output_path = std::path::Path::new(output);
    let is_wav = |path: &std::path::Path| {
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.eq_ignore_ascii_case("wav"))
            .unwrap_or(false)
    };
    if !is_wav(input_path) || !is_wav(output_path) {
        return Err("--stream currently supports WAV-to-WAV paths only".into());
    }
    if ov.auto_backend
        || ov
            .backend
            .is_some_and(|backend| backend != Backend::Classical)
    {
        return Err("--stream currently supports only the classical backend".into());
    }
    if ov
        .channel_mode
        .is_some_and(|mode| mode != ChannelMode::Independent)
    {
        return Err("--stream requires independent channels".into());
    }
    let preflight_cfg = build_config(&ov, VALIDATION_SAMPLE_RATE);
    if preflight_cfg.vad {
        return Err("--stream does not support VAD; omit --mode speech or --vad".into());
    }
    if ov.loudness_lufs.is_some() || ov.true_peak_dbtp.is_some() {
        return Err("--stream does not support loudness normalization".into());
    }
    ensure_output_available(output_path, ov.force)?;

    let mut input_session = AudioInputSession::open(input_path)?;
    let stream_info = inspect_wav_session(&mut input_session)?;
    let spec = stream_info.spec;
    let channel_mask = stream_info.channel_mask;
    validate_effective_options(&ov, spec.sample_rate)?;
    let cfg = build_config(&ov, spec.sample_rate);
    let block_frames = ov.stream_frames.unwrap_or(STREAM_BLOCK_FRAMES);
    let stream_working_set = estimate_stream_memory_bytes_checked(
        spec.channels as usize,
        block_frames,
        cfg.frame_size,
        spec.sample_rate,
        cfg.profile_ms,
    )
    .map_err(|error| error.to_string())?;
    ensure_memory_limit(
        stream_working_set,
        ov.max_memory_mb,
        "streaming working set",
    )?;
    let metadata_limits = retained_metadata_limits(ov.max_memory_mb, stream_working_set)?;
    if ov.report {
        println!(
            "input      : {input}\nformat     : {}ch, {} Hz, {}-bit {:?}\nbackend    : classical\nstream     : enabled ({} frames/block)",
            spec.channels, spec.sample_rate, spec.bits_per_sample, spec.sample_format, block_frames
        );
        return Ok(());
    }

    // Construct every allocation-sensitive processor before opening the
    // transactional output. Invalid or hostile resource plans therefore leave
    // neither a destination nor a temporary `.part` file behind.
    let mut processor = StreamingDenoiser::new(cfg, spec.channels as usize)?;
    let metadata = if !ov.no_metadata {
        input_session.read_metadata_with_limits(metadata_limits)?
    } else {
        None
    };
    let mut reader = WavStreamReader::from_session(input_session)?;
    let mut transaction = AtomicOutput::new(output_path)?;
    let frames = (|| -> Result<usize, String> {
        let sink = std::io::BufWriter::new(transaction.file_mut());
        let mut writer = WavStreamWriter::from_sink(sink, spec)?;
        let mut frames = 0usize;
        while let Some(block) = reader.next_block(block_frames)? {
            let block_frames = block.first().map(Vec::len).unwrap_or(0);
            let enhanced = processor.process_block(&block)?;
            writer.write_block(&enhanced)?;
            frames = frames.saturating_add(block_frames);
        }
        let tail = processor.finish()?;
        writer.write_block(&tail)?;
        writer.finalize()?;
        Ok(frames)
    })()?;
    write_wav_channel_mask_to_file(transaction.file_mut(), spec.channels as usize, channel_mask)?;
    if let Some(metadata) = metadata {
        denoize::metadata::write_extended_to_file_with_limits(
            metadata,
            transaction.file_mut(),
            metadata_limits,
        )?;
    }
    transaction.commit(if ov.force {
        CommitMode::Replace
    } else {
        CommitMode::NoClobber
    })?;
    if ov.json {
        println!(
            "{}",
            stream_result_json_line(input, output, spec.channels, frames, spec.sample_rate)
        );
    } else {
        eprintln!(
            "denoize: streaming classical WAV complete: {}ch x {} frames",
            spec.channels, frames
        );
    }
    Ok(())
}

enum BatchFileOutcome {
    Completed,
    Skipped,
    Failed(String),
    Cancelled,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct BatchCounts {
    succeeded: usize,
    skipped: usize,
    failed: usize,
    cancelled: usize,
}

fn count_batch_results(results: &[BatchFileOutcome]) -> BatchCounts {
    let mut counts = BatchCounts::default();
    for result in results {
        match result {
            BatchFileOutcome::Completed => counts.succeeded += 1,
            BatchFileOutcome::Skipped => counts.skipped += 1,
            BatchFileOutcome::Failed(_) => counts.failed += 1,
            BatchFileOutcome::Cancelled => counts.cancelled += 1,
        }
    }
    counts
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BatchItem {
    input: std::path::PathBuf,
    input_relative: std::path::PathBuf,
    destination: std::path::PathBuf,
    destination_relative: std::path::PathBuf,
    output_format: OutputFormat,
    probe: AudioProbe,
}

#[derive(Clone)]
struct PreparedBatchItem {
    item: BatchItem,
    resolved_processing: service::ResolvedProcessingOptions,
    expectation: ResumeExpectation,
    recipe: Digest,
}

#[derive(Clone)]
struct PlannedBatchItem {
    prepared: PreparedBatchItem,
    decision: ResumeDecision,
}

fn batch_probe_description(probe: &AudioProbe) -> &'static str {
    if probe.is_broadcast_wave {
        return "Broadcast Wave (BWF) PCM";
    }
    match (probe.format, probe.codec) {
        (AudioFormat::Wav, AudioCodec::Pcm) => "WAV PCM",
        (AudioFormat::Rf64, AudioCodec::Pcm) => "RF64 PCM",
        (AudioFormat::Aiff, AudioCodec::Pcm) => "AIFF/AIFC",
        (AudioFormat::Caf, AudioCodec::Pcm) => "CAF",
        (AudioFormat::Flac, AudioCodec::Flac) => "FLAC",
        (AudioFormat::OggOpus, AudioCodec::Opus) => "Ogg Opus",
        (AudioFormat::OggVorbis, AudioCodec::Vorbis) => "Ogg Vorbis",
        (AudioFormat::Mp3, AudioCodec::Mp3) => "MP3",
        (AudioFormat::M4a, AudioCodec::Aac) => "AAC-in-MP4",
        (AudioFormat::M4a, AudioCodec::Alac) => "ALAC-in-MP4",
        (AudioFormat::AacAdts, AudioCodec::Aac) => "ADTS AAC",
        _ => "unknown or ambiguous audio encoding",
    }
}

fn batch_can_preserve(probe: &AudioProbe, output_format: OutputFormat) -> bool {
    probe.audio_tracks == 1
        && !probe.has_non_audio_tracks
        && !probe.is_broadcast_wave
        && matches!(
            (probe.format, probe.codec, output_format),
            (AudioFormat::Wav, AudioCodec::Pcm, OutputFormat::Wav)
                | (AudioFormat::Flac, AudioCodec::Flac, OutputFormat::Flac)
                | (
                    AudioFormat::OggOpus,
                    AudioCodec::Opus,
                    OutputFormat::OggOpus
                )
                | (AudioFormat::Mp3, AudioCodec::Mp3, OutputFormat::Mp3)
                | (AudioFormat::M4a, AudioCodec::Aac, OutputFormat::M4a)
                | (AudioFormat::AacAdts, AudioCodec::Aac, OutputFormat::AacAdts)
        )
}

#[cfg(test)]
fn plan_batch_files(
    input_dir: &std::path::Path,
    output_dir: &std::path::Path,
    files: Vec<std::path::PathBuf>,
    output_extension: Option<&str>,
) -> Result<Vec<BatchItem>, String> {
    plan_batch_files_with_limits(
        input_dir,
        output_dir,
        files,
        output_extension,
        DecodeLimits::default(),
    )
}

fn plan_batch_files_with_limits(
    input_dir: &std::path::Path,
    output_dir: &std::path::Path,
    files: Vec<std::path::PathBuf>,
    output_extension: Option<&str>,
    decode_limits: DecodeLimits,
) -> Result<Vec<BatchItem>, String> {
    let mut items = Vec::with_capacity(files.len());
    for input in files {
        let relative = input
            .strip_prefix(input_dir)
            .map_err(|error| {
                format!(
                    "batch input {} is outside {}: {error}",
                    input.display(),
                    input_dir.display()
                )
            })?
            .to_path_buf();
        let mut destination = output_dir.join(&relative);
        if let Some(extension) = output_extension {
            destination.set_extension(extension);
        }

        let mut input_session = AudioInputSession::open(&input)
            .map_err(|error| format!("open batch input {}: {error}", input.display()))?;
        let probe = probe_audio_session_with_limits(&mut input_session, decode_limits)
            .map_err(|error| format!("probe batch input {}: {error}", input.display()))?;
        if probe.audio_tracks != 1 {
            return Err(format!(
                "batch input {} must contain exactly one supported audio track; found {}",
                input.display(),
                probe.audio_tracks
            ));
        }
        if probe.codec == AudioCodec::Unknown {
            return Err(format!(
                "batch input {} has no supported, unambiguous audio track",
                input.display()
            ));
        }
        let output_format = OutputFormat::from_path(&destination).map_err(|error| {
            if output_extension.is_none() {
                format!(
                    "batch cannot preserve {} ({}): {error}; specify --output-format wav, flac, opus, ogg, oga, mp3, m4a, or aac",
                    input.display(),
                    batch_probe_description(&probe)
                )
            } else {
                error
            }
        })?;
        if output_extension.is_none() && !batch_can_preserve(&probe, output_format) {
            let track_detail = if probe.audio_tracks != 1 || probe.has_non_audio_tracks {
                format!(
                    "; source contains {} audio track(s){}",
                    probe.audio_tracks,
                    if probe.has_non_audio_tracks {
                        " and non-audio tracks"
                    } else {
                        ""
                    }
                )
            } else {
                String::new()
            };
            return Err(format!(
                "batch cannot preserve {} ({}) without an explicit conversion{track_detail}; specify --output-format wav, flac, opus, ogg, oga, mp3, m4a, or aac",
                input.display(),
                batch_probe_description(&probe)
            ));
        }
        let destination_relative = destination
            .strip_prefix(output_dir)
            .map_err(|error| {
                format!(
                    "batch output {} is outside {}: {error}",
                    destination.display(),
                    output_dir.display()
                )
            })?
            .to_path_buf();
        items.push(BatchItem {
            input,
            input_relative: relative,
            destination,
            destination_relative,
            output_format,
            probe,
        });
    }
    validate_batch_destinations(input_dir, &items)?;
    Ok(items)
}

fn batch_collision_key(path: &std::path::Path) -> std::path::PathBuf {
    #[cfg(any(windows, target_os = "macos"))]
    {
        std::path::PathBuf::from(path.to_string_lossy().to_lowercase())
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        path.to_path_buf()
    }
}

fn validate_batch_destinations(
    input_dir: &std::path::Path,
    items: &[BatchItem],
) -> Result<(), String> {
    let input_root = normalize_batch_path(input_dir)?;
    let mut destinations = Vec::with_capacity(items.len());
    for item in items {
        let resolved = normalize_batch_path(&item.destination)?;
        if resolved.starts_with(&input_root) {
            return Err(format!(
                "batch output {} resolves inside the input directory; remove output symlinks or choose a separate output directory",
                item.destination.display()
            ));
        }
        destinations.push((batch_collision_key(&resolved), item));
    }
    destinations.sort_by(|left, right| left.0.cmp(&right.0));

    for pair in destinations.windows(2) {
        let (left_path, left) = &pair[0];
        let (right_path, right) = &pair[1];
        if right_path == left_path {
            return Err(format!(
                "multiple inputs map to the same batch output: {} and {} -> {}",
                left.input.display(),
                right.input.display(),
                right.destination.display()
            ));
        }
        if right_path.starts_with(left_path) {
            return Err(format!(
                "batch outputs conflict as a file and directory: {} -> {} and {} -> {}",
                left.input.display(),
                left.destination.display(),
                right.input.display(),
                right.destination.display()
            ));
        }
    }
    Ok(())
}

fn validate_batch_reserved_path(
    items: &[BatchItem],
    reserved: &std::path::Path,
    reserved_name: &str,
) -> Result<(), String> {
    let reserved = batch_collision_key(&normalize_batch_path(reserved)?);
    for item in items {
        let destination = batch_collision_key(&normalize_batch_path(&item.destination)?);
        if destination == reserved
            || destination.starts_with(&reserved)
            || reserved.starts_with(&destination)
        {
            return Err(format!(
                "batch output {} conflicts with reserved batch control path {reserved_name}",
                item.destination.display(),
            ));
        }
    }
    Ok(())
}

fn normalize_batch_path(path: &std::path::Path) -> Result<std::path::PathBuf, String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("resolve current directory: {error}"))?
            .join(path)
    };
    #[derive(Debug)]
    enum MissingComponent {
        Normal(std::ffi::OsString),
        Parent,
    }

    let mut ancestor = absolute.clone();
    let mut missing = Vec::new();
    loop {
        match std::fs::symlink_metadata(&ancestor) {
            Ok(_) => break,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "inspect batch path {}: {error}",
                    ancestor.display()
                ));
            }
        }
        let component = ancestor
            .components()
            .next_back()
            .ok_or_else(|| format!("cannot resolve batch path {}", absolute.display()))?;
        match component {
            std::path::Component::Normal(name) => {
                missing.push(MissingComponent::Normal(name.to_os_string()))
            }
            std::path::Component::ParentDir => missing.push(MissingComponent::Parent),
            std::path::Component::CurDir => {}
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                return Err(format!("cannot resolve batch path {}", absolute.display()));
            }
        }
        if !ancestor.pop() {
            return Err(format!("cannot resolve batch path {}", absolute.display()));
        }
    }
    let mut resolved = std::fs::canonicalize(&ancestor)
        .map_err(|error| format!("resolve {}: {error}", ancestor.display()))?;
    for component in missing.into_iter().rev() {
        match component {
            MissingComponent::Normal(name) => resolved.push(name),
            MissingComponent::Parent => {
                resolved.pop();
            }
        }
    }
    Ok(resolved)
}

fn validate_batch_directories(
    input_dir: &std::path::Path,
    output_dir: &std::path::Path,
) -> Result<(), String> {
    let input = normalize_batch_path(input_dir)?;
    let output = normalize_batch_path(output_dir)?;
    if input.starts_with(&output) || output.starts_with(&input) {
        return Err(format!(
            "batch input and output directories must not overlap: {} and {}",
            input_dir.display(),
            output_dir.display()
        ));
    }
    Ok(())
}

fn run_batch(input: &str, output: &str, ov: &Overrides) -> Result<(), String> {
    use rayon::prelude::*;

    validate_effective_options(ov, VALIDATION_SAMPLE_RATE)?;
    let encode_options = build_encode_options(ov)?;
    let resolved_backend_options = resolve_explicit_backend_options(ov)?;
    let jobs = effective_batch_jobs(ov);
    let input_dir = std::path::Path::new(input);
    let output_dir = std::path::Path::new(output);
    if !input_dir.is_dir() {
        return Err(format!("batch input is not a directory: {input}"));
    }
    validate_batch_directories(input_dir, output_dir)?;
    let output_extension = ov
        .output_format
        .as_deref()
        .map(normalize_output_extension)
        .transpose()?;
    let files = collect_batch_files(input_dir, ov.recursive)?;
    if files.is_empty() {
        return Err("batch input contains no supported audio files".into());
    }
    let items = plan_batch_files_with_limits(
        input_dir,
        output_dir,
        files,
        output_extension,
        decode_limits(ov.max_memory_mb)?,
    )?;
    let state_path = output_dir.join(STATE_FILE_NAME);
    let legacy_state_path = output_dir.join(LEGACY_DESKTOP_STATE_FILE_NAME);
    let lock_path = output_dir.join(LOCK_FILE_NAME);
    validate_batch_reserved_path(&items, &state_path, STATE_FILE_NAME)?;
    validate_batch_reserved_path(&items, &legacy_state_path, LEGACY_DESKTOP_STATE_FILE_NAME)?;
    validate_batch_reserved_path(&items, &lock_path, LOCK_FILE_NAME)?;
    validate_encode_preflight(encode_options, items.iter().map(|item| item.output_format))?;
    let prepared = preflight_batch_items(
        &items,
        ov,
        encode_options,
        resolved_backend_options.as_ref(),
    )?;

    std::fs::create_dir_all(output_dir).map_err(|e| format!("create batch output: {e}"))?;
    let session = Arc::new(BatchSession::acquire(output_dir, ov.resume)?);
    let planned = prepared
        .into_iter()
        .map(|prepared| {
            let decision = session.plan(&prepared.expectation, ov.force)?;
            Ok(PlannedBatchItem { prepared, decision })
        })
        .collect::<Result<Vec<_>, String>>()?;
    // Planning can be long for a large output set. Recheck every source after
    // the final decision and before activate performs the first state change.
    for item in &planned {
        item.prepared.expectation.verify_sources()?;
    }
    CANCELLED.store(false, Ordering::SeqCst);
    install_cancel_handler()?;
    let finished = AtomicUsize::new(0);
    let publication_fence = Mutex::new(());
    let started = Instant::now();
    let metadata_policy = if ov.no_metadata {
        MetadataPolicy::Drop
    } else {
        MetadataPolicy::Preserve
    };
    let process_item = |planned: &PlannedBatchItem| -> BatchFileOutcome {
        let item = &planned.prepared.item;
        let finish = |outcome, status| {
            report_batch_progress(&finished, items.len(), started, &item.input, status, ov);
            outcome
        };
        let commit_mode = match planned.decision {
            ResumeDecision::Skip { .. } => {
                return finish(BatchFileOutcome::Skipped, "skipped");
            }
            ResumeDecision::Process { commit_mode, .. } => commit_mode,
        };
        if CANCELLED.load(Ordering::SeqCst) {
            return finish(BatchFileOutcome::Cancelled, "cancelled");
        }
        if let Some(parent) = item.destination.parent() {
            if let Err(error) = std::fs::create_dir_all(parent) {
                return finish(
                    BatchFileOutcome::Failed(format!("create {}: {error}", parent.display())),
                    "failed",
                );
            }
        }
        let mut options = ov.clone();
        options.batch = false;
        options.json = false;
        let staged = match process_one_to_staged_output(
            &item.input,
            &item.destination,
            options,
            Some(item.output_format),
            None,
            Some(planned.prepared.resolved_processing.clone()),
            Some(metadata_policy),
            Some(item.probe),
            Some(planned.prepared.expectation.input_fingerprint()),
            false,
        ) {
            Ok(staged) => staged,
            Err(error) => return finish(BatchFileOutcome::Failed(error), "failed"),
        };
        let Some(staged) = staged else {
            // Batch --report intentionally retains its existing report-only
            // behavior and has no filesystem output to publish.
            return finish(BatchFileOutcome::Completed, "completed");
        };
        if staged.effective_recipe != Some(planned.prepared.recipe) {
            return finish(
                BatchFileOutcome::Failed(format!(
                    "effective batch recipe changed after preflight for {}",
                    item.input.display()
                )),
                "failed",
            );
        }
        match with_batch_publication_fence(&publication_fence, &CANCELLED, || {
            session.publish(
                &planned.prepared.expectation,
                staged.transaction,
                commit_mode,
            )
        }) {
            Ok(Some(_)) => finish(BatchFileOutcome::Completed, "completed"),
            Ok(None) => finish(BatchFileOutcome::Cancelled, "cancelled"),
            Err(error) => finish(BatchFileOutcome::Failed(error), "failed"),
        }
    };
    let results = if CANCELLED.load(Ordering::SeqCst) {
        // Do not activate (and therefore do not repair or create state) when
        // cancellation was observed before any item could publish. Running
        // the closure still gives every exact skip or cancelled item one
        // stable progress outcome.
        planned.iter().map(process_item).collect::<Vec<_>>()
    } else {
        session.activate()?;
        if ov.deterministic {
            planned.iter().map(process_item).collect::<Vec<_>>()
        } else {
            rayon::ThreadPoolBuilder::new()
                .num_threads(jobs)
                .build()
                .map_err(|e| format!("create batch worker pool: {e}"))?
                .install(|| planned.par_iter().map(process_item).collect::<Vec<_>>())
        }
    };
    let counts = count_batch_results(&results);
    let failures: Vec<_> = results
        .iter()
        .filter_map(|result| match result {
            BatchFileOutcome::Failed(error) => Some(error),
            _ => None,
        })
        .collect();
    debug_assert_eq!(counts.failed, failures.len());
    debug_assert_eq!(
        counts.succeeded + counts.skipped + counts.failed + counts.cancelled,
        items.len()
    );
    if ov.json {
        println!(
            "{}",
            batch_summary_json_line(
                items.len(),
                counts.succeeded,
                counts.skipped,
                counts.failed,
                counts.cancelled,
                counts.cancelled != 0,
                output,
            )
        );
    } else {
        eprintln!(
            "denoize: batch complete: {} succeeded, {} skipped, {} failed, {} cancelled",
            counts.succeeded, counts.skipped, counts.failed, counts.cancelled
        );
        for error in &failures {
            eprintln!("denoize: batch error: {error}");
        }
    }
    if failures.is_empty() && counts.cancelled == 0 {
        Ok(())
    } else {
        Err(format!(
            "{} batch file(s) failed and {} cancelled",
            failures.len(),
            counts.cancelled
        ))
    }
}

fn report_batch_progress(
    finished: &AtomicUsize,
    total: usize,
    started: Instant,
    path: &std::path::Path,
    status: &str,
    ov: &Overrides,
) {
    let count = finished.fetch_add(1, Ordering::Relaxed) + 1;
    let elapsed = started.elapsed().as_secs_f64();
    let eta = if count == 0 {
        0.0
    } else {
        elapsed / count as f64 * total.saturating_sub(count) as f64
    };
    if ov.json {
        let input = path.to_string_lossy();
        println!(
            "{}",
            batch_progress_json_line(status, count, total, elapsed, eta, input.as_ref())
        );
    } else if !ov.no_progress {
        eprintln!(
            "denoize: batch {count}/{total} {status} {} ({elapsed:.1}s elapsed, ETA {eta:.1}s)",
            path.display()
        );
    }
}

fn collect_batch_files(
    root: &std::path::Path,
    recursive: bool,
) -> Result<Vec<std::path::PathBuf>, String> {
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory)
            .map_err(|e| format!("read batch input {}: {e}", directory.display()))?
        {
            let entry = entry.map_err(|e| format!("read batch entry: {e}"))?;
            let file_type = entry
                .file_type()
                .map_err(|e| format!("read batch entry type: {e}"))?;
            let path = entry.path();
            if file_type.is_dir() && recursive {
                pending.push(path);
            } else if file_type.is_file() && is_supported_audio_path(&path) {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn is_supported_audio_path(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "wav"
                    | "rf64"
                    | "bwf"
                    | "aif"
                    | "aiff"
                    | "aifc"
                    | "caf"
                    | "mp3"
                    | "m4a"
                    | "mp4"
                    | "aac"
                    | "flac"
                    | "opus"
                    | "ogg"
                    | "oga"
                    | "vorbis"
            )
        })
        .unwrap_or(false)
}

fn normalize_output_extension(value: &str) -> Result<&str, String> {
    let extension = value.trim_start_matches('.');
    if matches!(
        extension.to_ascii_lowercase().as_str(),
        "wav" | "mp3" | "m4a" | "aac" | "flac" | "opus" | "ogg" | "oga"
    ) {
        Ok(extension)
    } else {
        Err(format!("unsupported --output-format: {value}"))
    }
}

#[cfg(test)]
mod json_output_tests {
    use super::*;
    use serde_json::Value;

    const SPECIAL_INPUT: &str = "input-cafe\u{301}-quote\"-slash\\-line\n-control\u{1}.wav";
    const SPECIAL_OUTPUT: &str = "output-cafe\u{301}-quote\"-slash\\-line\n-control\u{2}.wav";

    fn parse_json_line(line: &str) -> Value {
        assert!(
            !line.contains("\\u{"),
            "Rust escape leaked into JSON: {line}"
        );
        assert!(
            !line.contains('\n'),
            "serialized JSON line contains a physical newline"
        );
        serde_json::from_str(line).expect("CLI output must be valid JSON")
    }

    #[test]
    fn process_result_json_round_trips_special_paths() {
        let value = parse_json_line(&process_result_json_line(
            SPECIAL_INPUT,
            SPECIAL_OUTPUT,
            "classical",
            2,
            48_001,
            48_000,
            1.2345,
        ));

        assert_eq!(value.as_object().unwrap().len(), 7);
        assert_eq!(value["input"].as_str(), Some(SPECIAL_INPUT));
        assert_eq!(value["output"].as_str(), Some(SPECIAL_OUTPUT));
        assert_eq!(value["backend"].as_str(), Some("classical"));
        assert_eq!(value["channels"].as_u64(), Some(2));
        assert_eq!(value["frames"].as_u64(), Some(48_001));
        assert_eq!(value["sample_rate"].as_u64(), Some(48_000));
        assert_eq!(value["elapsed_ms"].as_f64(), Some(1.234));
    }

    #[test]
    fn stream_result_json_round_trips_special_paths() {
        let value = parse_json_line(&stream_result_json_line(
            SPECIAL_INPUT,
            SPECIAL_OUTPUT,
            2,
            8_193,
            44_100,
        ));

        assert_eq!(value.as_object().unwrap().len(), 7);
        assert_eq!(value["input"].as_str(), Some(SPECIAL_INPUT));
        assert_eq!(value["output"].as_str(), Some(SPECIAL_OUTPUT));
        assert_eq!(value["backend"].as_str(), Some("classical"));
        assert_eq!(value["channels"].as_u64(), Some(2));
        assert_eq!(value["frames"].as_u64(), Some(8_193));
        assert_eq!(value["sample_rate"].as_u64(), Some(44_100));
        assert_eq!(value["stream"].as_bool(), Some(true));
    }

    #[test]
    fn batch_progress_json_round_trips_special_paths() {
        let value = parse_json_line(&batch_progress_json_line(
            "completed",
            3,
            5,
            1.23456,
            0.45678,
            SPECIAL_INPUT,
        ));

        assert_eq!(value.as_object().unwrap().len(), 7);
        assert_eq!(value["event"].as_str(), Some("progress"));
        assert_eq!(value["status"].as_str(), Some("completed"));
        assert_eq!(value["completed"].as_u64(), Some(3));
        assert_eq!(value["total"].as_u64(), Some(5));
        assert_eq!(value["elapsed_seconds"].as_f64(), Some(1.235));
        assert_eq!(value["eta_seconds"].as_f64(), Some(0.457));
        assert_eq!(value["input"].as_str(), Some(SPECIAL_INPUT));
    }

    #[test]
    fn batch_summary_json_round_trips_special_paths() {
        let value = parse_json_line(&batch_summary_json_line(
            8,
            4,
            2,
            1,
            1,
            true,
            SPECIAL_OUTPUT,
        ));

        assert_eq!(value.as_object().unwrap().len(), 8);
        assert_eq!(value["event"].as_str(), Some("summary"));
        assert_eq!(value["total"].as_u64(), Some(8));
        assert_eq!(value["succeeded"].as_u64(), Some(4));
        assert_eq!(value["skipped"].as_u64(), Some(2));
        assert_eq!(value["failed"].as_u64(), Some(1));
        assert_eq!(value["cancelled_count"].as_u64(), Some(1));
        assert_eq!(value["cancelled"].as_bool(), Some(true));
        assert_eq!(value["output"].as_str(), Some(SPECIAL_OUTPUT));
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    // Item identities preserve each platform's raw OS path representation:
    // UTF-8 bytes on Unix-like targets and UTF-16LE code units on Windows.
    #[cfg(not(windows))]
    const FRONTEND_PARITY_ITEM_ID_HEX: &str =
        "795ada4ccf8186cdaa1d64cec4f53165bc5ca003d68e0964aee9a33a5f8105e8";
    #[cfg(windows)]
    const FRONTEND_PARITY_ITEM_ID_HEX: &str =
        "28a3a5bc0a5112777268b438a5357badea3c055ea91a1472a9cdba3c1a8522f0";
    // The package version is intentionally part of the v3 recipe ABI. Update
    // this value in both frontend tests when an intentional release bump lands.
    const FRONTEND_PARITY_RECIPE_HEX: &str =
        "c06b33c2d2c5808b54430fb4f83392b4efd5f958eced92e6b49a96fb67834125";

    #[test]
    fn cancellation_while_waiting_for_publication_fence_never_publishes() {
        let fence = Arc::new(Mutex::new(()));
        let cancelled = Arc::new(AtomicBool::new(false));
        let published = Arc::new(AtomicBool::new(false));
        let held = fence.lock().unwrap();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let worker_fence = Arc::clone(&fence);
        let worker_cancelled = Arc::clone(&cancelled);
        let worker_published = Arc::clone(&published);
        let worker = std::thread::spawn(move || {
            ready_tx.send(()).unwrap();
            with_batch_publication_fence(&worker_fence, &worker_cancelled, || {
                worker_published.store(true, Ordering::SeqCst);
                Ok(())
            })
        });

        ready_rx.recv().unwrap();
        cancelled.store(true, Ordering::SeqCst);
        drop(held);

        assert_eq!(worker.join().unwrap().unwrap(), None);
        assert!(!published.load(Ordering::SeqCst));
    }

    fn temporary_directory() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "denoize-batch-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    fn write_stereo_batch_wav(path: &std::path::Path) {
        let audio = denoize::Audio {
            sample_rate: 48_000,
            channels: vec![vec![0.0; 960], vec![0.0; 960]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };
        denoize::write_audio(path, &audio, EncodeOptions::default()).unwrap();
    }

    #[test]
    fn cli_batch_recipe_matches_the_frontend_parity_golden_vector() {
        let root = temporary_directory().join("frontend-parity-golden");
        let input = root.join("input");
        let output = root.join("output");
        std::fs::create_dir_all(&input).unwrap();
        let source = input.join("stereo.wav");
        write_stereo_batch_wav(&source);
        let options = parse_config(
            r#"
backend = "classical"
preset = "hifi"
mode = "speech"
strength = 0.37
adaptive_noise = false
vad = false
channels = "linked"
downmix = "preserve"
loudness_lufs = -16.0
true_peak_dbtp = -1.0
preserve_metadata = false
mp3_bitrate_kbps = 256
m4a_bitrate_kbps = 224
aac_encoder = "oxide"
output_format = "mp3"
batch = true
resume = true
max_memory_mb = 64
"#,
            "frontend-parity.toml",
        )
        .unwrap();
        let encode = build_encode_options(&options).unwrap();
        let items = plan_batch_files(
            &input,
            &output,
            vec![source.clone()],
            options.output_format.as_deref(),
        )
        .unwrap();
        let prepared = preflight_batch_items(
            &items,
            &options,
            encode,
            resolve_explicit_backend_options(&options).unwrap().as_ref(),
        )
        .unwrap();
        let prepared = &prepared[0];

        assert_eq!(prepared.resolved_processing.backend, Backend::Classical);
        assert!(!prepared.resolved_processing.denoiser.adaptive_noise);
        assert!(!prepared.resolved_processing.denoiser.vad);
        assert_eq!(
            prepared.resolved_processing.backend_options.channel_mode,
            ChannelMode::StereoLinked
        );
        assert_eq!(prepared.resolved_processing.loudness_lufs, Some(-16.0));
        assert_eq!(prepared.resolved_processing.true_peak_dbtp, -1.0);
        assert_eq!(prepared.item.output_format, OutputFormat::Mp3);
        assert_eq!(encode.mp3_bitrate_kbps, 256);
        assert!(options.no_metadata);
        assert_eq!(options.max_memory_mb, Some(64));
        assert!(prepared.expectation.model().is_none());
        assert_eq!(prepared.expectation.recipe(), prepared.recipe);
        assert_eq!(prepared.recipe.as_hex(), FRONTEND_PARITY_RECIPE_HEX);
        assert_eq!(
            prepared.expectation.item_id(),
            batch_resume::item_identity(
                &normalize_batch_path(&source).unwrap(),
                &prepared.item.input_relative,
                &prepared.item.destination_relative,
                OutputFormat::Mp3,
            )
        );

        let fixed_item_id = batch_resume::item_identity(
            std::path::Path::new("/denoize/frontend-parity/input/stereo.wav"),
            std::path::Path::new("stereo.wav"),
            std::path::Path::new("stereo.mp3"),
            OutputFormat::Mp3,
        );
        assert_eq!(fixed_item_id.as_hex(), FRONTEND_PARITY_ITEM_ID_HEX);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cli_treats_legacy_desktop_state_as_untrusted_then_preserves_it() {
        for (index, legacy) in [
            b"sample.wav\n".to_vec(),
            format!("v2:{}\n", "41".repeat(32)).into_bytes(),
        ]
        .into_iter()
        .enumerate()
        {
            let root = temporary_directory().join(format!("legacy-gui-state-{index}"));
            let input = root.join("input");
            let output = root.join("output");
            std::fs::create_dir_all(&input).unwrap();
            std::fs::create_dir_all(&output).unwrap();
            write_stereo_batch_wav(&input.join("sample.wav"));
            let destination = output.join("sample.wav");
            let original_output = b"legacy desktop output";
            std::fs::write(&destination, original_output).unwrap();
            let legacy_path = output.join(LEGACY_DESKTOP_STATE_FILE_NAME);
            std::fs::write(&legacy_path, &legacy).unwrap();
            let mut options = Overrides {
                batch: true,
                resume: true,
                no_progress: true,
                ..Overrides::default()
            };

            let error =
                run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap_err();
            assert!(error.contains("legacy"), "{error}");
            assert!(error.contains("--force"), "{error}");
            assert_eq!(std::fs::read(&destination).unwrap(), original_output);
            assert_eq!(std::fs::read(&legacy_path).unwrap(), legacy);
            assert!(!output.join(STATE_FILE_NAME).exists());

            options.force = true;
            run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap();
            let migrated_state = std::fs::read(output.join(STATE_FILE_NAME)).unwrap();
            let migrated_output = std::fs::read(&destination).unwrap();
            assert!(String::from_utf8_lossy(&migrated_state).contains("\"version\":3"));
            assert_eq!(std::fs::read(&legacy_path).unwrap(), legacy);

            options.force = false;
            run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap();
            assert_eq!(
                std::fs::read(output.join(STATE_FILE_NAME)).unwrap(),
                migrated_state
            );
            assert_eq!(std::fs::read(destination).unwrap(), migrated_output);
            assert_eq!(std::fs::read(&legacy_path).unwrap(), legacy);
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn batch_collection_is_recursive_and_sorted() {
        let root = temporary_directory();
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(root.join("b.wav"), []).unwrap();
        std::fs::write(root.join("ignore.txt"), []).unwrap();
        std::fs::write(nested.join("a.FLAC"), []).unwrap();

        assert_eq!(
            collect_batch_files(&root, false).unwrap(),
            vec![root.join("b.wav")]
        );
        assert_eq!(
            collect_batch_files(&root, true).unwrap(),
            vec![root.join("b.wav"), nested.join("a.FLAC")]
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn validates_batch_output_format() {
        assert_eq!(normalize_output_extension(".flac").unwrap(), "flac");
        assert_eq!(normalize_output_extension("aac").unwrap(), "aac");
        assert_eq!(normalize_output_extension("oga").unwrap(), "oga");
        assert!(normalize_output_extension("wma").is_err());
    }

    fn probe(format: AudioFormat, codec: AudioCodec) -> AudioProbe {
        AudioProbe {
            format,
            codec,
            audio_tracks: 1,
            has_non_audio_tracks: false,
            is_broadcast_wave: false,
        }
    }

    #[test]
    fn batch_preserve_policy_is_codec_and_container_exact() {
        for (source, output) in [
            (probe(AudioFormat::Wav, AudioCodec::Pcm), OutputFormat::Wav),
            (
                probe(AudioFormat::Flac, AudioCodec::Flac),
                OutputFormat::Flac,
            ),
            (
                probe(AudioFormat::OggOpus, AudioCodec::Opus),
                OutputFormat::OggOpus,
            ),
            (probe(AudioFormat::Mp3, AudioCodec::Mp3), OutputFormat::Mp3),
            (probe(AudioFormat::M4a, AudioCodec::Aac), OutputFormat::M4a),
            (
                probe(AudioFormat::AacAdts, AudioCodec::Aac),
                OutputFormat::AacAdts,
            ),
        ] {
            assert!(batch_can_preserve(&source, output));
        }

        for (source, output) in [
            (probe(AudioFormat::Rf64, AudioCodec::Pcm), OutputFormat::Wav),
            (probe(AudioFormat::Aiff, AudioCodec::Pcm), OutputFormat::Wav),
            (probe(AudioFormat::Caf, AudioCodec::Pcm), OutputFormat::Wav),
            (
                probe(AudioFormat::OggVorbis, AudioCodec::Vorbis),
                OutputFormat::OggOpus,
            ),
            (probe(AudioFormat::M4a, AudioCodec::Alac), OutputFormat::M4a),
        ] {
            assert!(!batch_can_preserve(&source, output));
        }

        let mut multi_track = probe(AudioFormat::M4a, AudioCodec::Aac);
        multi_track.audio_tracks = 2;
        assert!(!batch_can_preserve(&multi_track, OutputFormat::M4a));
        multi_track.audio_tracks = 1;
        multi_track.has_non_audio_tracks = true;
        assert!(!batch_can_preserve(&multi_track, OutputFormat::M4a));

        let mut broadcast_wave = probe(AudioFormat::Wav, AudioCodec::Pcm);
        broadcast_wave.is_broadcast_wave = true;
        assert!(!batch_can_preserve(&broadcast_wave, OutputFormat::Wav));
    }

    #[test]
    fn batch_resume_identity_includes_destination_and_codec() {
        let identity = std::path::Path::new("/input-a/voice.aiff");
        let input = std::path::Path::new("voice.aiff");
        let wav = batch_resume::item_identity(
            identity,
            input,
            std::path::Path::new("voice.wav"),
            OutputFormat::Wav,
        );
        let flac = batch_resume::item_identity(
            identity,
            input,
            std::path::Path::new("voice.flac"),
            OutputFormat::Flac,
        );
        let renamed = batch_resume::item_identity(
            identity,
            input,
            std::path::Path::new("nested/voice.wav"),
            OutputFormat::Wav,
        );

        assert_ne!(wav, flac);
        assert_ne!(wav, renamed);
        assert_ne!(
            wav,
            batch_resume::item_identity(
                std::path::Path::new("/input-b/voice.aiff"),
                input,
                std::path::Path::new("voice.wav"),
                OutputFormat::Wav,
            )
        );
        assert_eq!(wav.as_hex().len(), 64);
    }

    #[test]
    fn batch_plan_rejects_collisions_before_processing() {
        let root = temporary_directory();
        let input = root.join("input");
        let output = root.join("output");
        std::fs::create_dir_all(&input).unwrap();
        let aiff = input.join("clip.aiff");
        let caf = input.join("clip.caf");
        std::fs::write(&aiff, b"FORM\0\0\0\0AIFF").unwrap();
        std::fs::write(&caf, b"caff\0\x01\0\0\0\0\0\0").unwrap();

        let error = plan_batch_files(&input, &output, vec![aiff, caf], Some("flac")).unwrap_err();

        assert!(error.contains("multiple inputs map to the same batch output"));
        assert!(!output.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batch_plan_rejects_file_directory_prefix_collisions() {
        let root = temporary_directory();
        let input = root.join("input");
        let output = root.join("output");
        let nested = input.join("clip.wav");
        std::fs::create_dir_all(&nested).unwrap();
        let aiff = input.join("clip.aiff");
        let caf = nested.join("child.caf");
        std::fs::write(&aiff, b"FORM\0\0\0\0AIFF").unwrap();
        std::fs::write(&caf, b"caff\0\x01\0\0\0\0\0\0").unwrap();

        let error = plan_batch_files(&input, &output, vec![aiff, caf], Some("wav")).unwrap_err();

        assert!(error.contains("conflict as a file and directory"));
        assert!(!output.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn batch_collection_does_not_follow_directory_symlinks() {
        use std::os::unix::fs::symlink;

        let root = temporary_directory();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("voice.wav"), []).unwrap();
        symlink(&root, root.join("loop")).unwrap();

        assert_eq!(
            collect_batch_files(&root, true).unwrap(),
            vec![root.join("voice.wav")]
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn batch_plan_rejects_output_symlinks_into_the_input_tree() {
        use std::os::unix::fs::symlink;

        let root = temporary_directory();
        let input = root.join("input");
        let input_nested = input.join("nested");
        let output = root.join("output");
        std::fs::create_dir_all(&input_nested).unwrap();
        std::fs::create_dir_all(&output).unwrap();
        symlink(&input_nested, output.join("nested")).unwrap();
        let aiff = input_nested.join("voice.aiff");
        std::fs::write(&aiff, b"FORM\0\0\0\0AIFF").unwrap();

        let error = plan_batch_files(&input, &output, vec![aiff], Some("wav")).unwrap_err();

        assert!(error.contains("resolves inside the input directory"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batch_plan_accepts_explicit_conversion_for_decode_only_input() {
        let root = temporary_directory();
        let input = root.join("input");
        let output = root.join("output");
        std::fs::create_dir_all(&input).unwrap();
        let aiff = input.join("voice.aiff");
        std::fs::write(&aiff, b"FORM\0\0\0\0AIFF").unwrap();

        let items = plan_batch_files(&input, &output, vec![aiff.clone()], Some("wav")).unwrap();

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].input, aiff);
        assert_eq!(items[0].destination, output.join("voice.wav"));
        assert_eq!(items[0].input_relative, std::path::Path::new("voice.aiff"));
        assert_eq!(
            items[0].destination_relative,
            std::path::Path::new("voice.wav")
        );
        assert_eq!(items[0].output_format, OutputFormat::Wav);
        assert!(!output.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batch_preflight_has_no_output_side_effects() {
        let root = temporary_directory();
        let input = root.join("input");
        let output = root.join("output");
        std::fs::create_dir_all(&input).unwrap();
        let audio = denoize::Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 1_600]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };
        denoize::write_audio(input.join("a-valid.wav"), &audio, EncodeOptions::default()).unwrap();
        std::fs::write(input.join("b-decode-only.aiff"), b"FORM\0\0\0\0AIFF").unwrap();
        let options = Overrides {
            batch: true,
            no_progress: true,
            ..Overrides::default()
        };

        let error =
            run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap_err();

        assert!(error.contains("AIFF/AIFC"));
        assert!(error.contains("--output-format"));
        assert!(!output.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batch_rejects_overlapping_input_and_output_directories() {
        let root = temporary_directory();
        let nested = root.join("nested");
        std::fs::create_dir_all(&root).unwrap();

        assert!(validate_batch_directories(&root, &root).is_err());
        assert!(validate_batch_directories(&root, &nested).is_err());
        assert!(validate_batch_directories(&nested, &root).is_err());

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resume_state_path_is_reserved_before_processing() {
        let root = temporary_directory();
        let input = root.join("input");
        let output = root.join("output");
        let reserved = input.join(".denoize-state");
        std::fs::create_dir_all(&reserved).unwrap();
        let audio = denoize::Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 1_600]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };
        denoize::write_audio(reserved.join("voice.wav"), &audio, EncodeOptions::default()).unwrap();
        let options = Overrides {
            batch: true,
            recursive: true,
            resume: true,
            no_progress: true,
            ..Overrides::default()
        };

        let error =
            run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap_err();

        assert!(error.contains(STATE_FILE_NAME));
        assert!(!output.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batch_counts_distinguish_completed_skipped_and_failed_results() {
        let results = [
            BatchFileOutcome::Completed,
            BatchFileOutcome::Skipped,
            BatchFileOutcome::Failed("processing failed".into()),
            BatchFileOutcome::Cancelled,
        ];

        assert_eq!(
            count_batch_results(&results),
            BatchCounts {
                succeeded: 1,
                skipped: 1,
                failed: 1,
                cancelled: 1,
            }
        );
    }

    #[test]
    fn batch_processes_nested_audio_and_converts_format() {
        let root = temporary_directory();
        let input = root.join("input");
        let output = root.join("output");
        std::fs::create_dir_all(input.join("nested")).unwrap();
        let audio = denoize::Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 3_200]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };
        denoize::write_audio(
            input.join("nested/sample.wav"),
            &audio,
            EncodeOptions::default(),
        )
        .unwrap();
        let options = Overrides {
            batch: true,
            recursive: true,
            jobs: Some(2),
            output_format: Some("flac".into()),
            ..Overrides::default()
        };

        run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap();
        assert!(std::fs::symlink_metadata(output.join("nested/sample.flac"))
            .unwrap()
            .file_type()
            .is_file());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn deterministic_batch_is_byte_stable_even_with_multiple_requested_jobs() {
        let root = temporary_directory();
        let input = root.join("input");
        let output = root.join("output");
        std::fs::create_dir_all(&input).unwrap();
        for (name, frequency) in [("a.wav", 220.0), ("b.wav", 440.0)] {
            let audio = denoize::Audio {
                sample_rate: 16_000,
                channels: vec![(0..3_200)
                    .map(|index| {
                        (2.0 * std::f64::consts::PI * frequency * index as f64 / 16_000.0).sin()
                            * 0.2
                    })
                    .collect()],
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
                channel_mask: None,
            };
            denoize::write_audio(input.join(name), &audio, EncodeOptions::default()).unwrap();
        }
        let options = Overrides {
            batch: true,
            deterministic: true,
            force: true,
            jobs: Some(8),
            no_progress: true,
            ..Overrides::default()
        };

        run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap();
        let first_a = std::fs::read(output.join("a.wav")).unwrap();
        let first_b = std::fs::read(output.join("b.wav")).unwrap();
        run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap();
        assert_eq!(first_a, std::fs::read(output.join("a.wav")).unwrap());
        assert_eq!(first_b, std::fs::read(output.join("b.wav")).unwrap());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resume_skips_outputs_recorded_as_complete() {
        let root = temporary_directory();
        let input = root.join("input");
        let output = root.join("output");
        std::fs::create_dir_all(&input).unwrap();
        let audio = denoize::Audio {
            sample_rate: 16_000,
            channels: vec![vec![0.0; 1_600]],
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
            channel_mask: None,
        };
        denoize::write_audio(input.join("sample.wav"), &audio, EncodeOptions::default()).unwrap();
        let options = Overrides {
            batch: true,
            resume: true,
            no_progress: true,
            ..Overrides::default()
        };

        run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap();
        let first_modified = std::fs::metadata(output.join("sample.wav"))
            .unwrap()
            .modified()
            .unwrap();
        run_batch(input.to_str().unwrap(), output.to_str().unwrap(), &options).unwrap();
        let second_modified = std::fs::metadata(output.join("sample.wav"))
            .unwrap()
            .modified()
            .unwrap();
        assert_eq!(first_modified, second_modified);
        let state = std::fs::read_to_string(output.join(STATE_FILE_NAME)).unwrap();
        assert!(state.contains("\"version\":3"));
        assert!(state.contains("\"kind\":\"prepare\""));
        assert!(state.contains("\"kind\":\"complete\""));
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod auto_backend_tests {
    use super::*;

    #[test]
    fn parses_auto_backend() {
        let (_, _, options) = parse_args(&[
            "input.wav".into(),
            "output.wav".into(),
            "--backend".into(),
            "auto".into(),
        ])
        .unwrap();
        assert!(options.auto_backend);
        assert!(options.backend.is_none());
    }

    #[test]
    fn automatic_selection_uses_an_available_backend() {
        let selected = service::select_backend(BackendChoice::Auto, 30.0, None);
        assert!(Backend::available_names().contains(&service::backend_name(selected)));
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;

    #[test]
    fn parses_stream_option() {
        let (_, _, options) = parse_args(&[
            "input.wav".into(),
            "output.wav".into(),
            "--stream".into(),
            "--stream-frames".into(),
            "4096".into(),
            "--max-memory".into(),
            "64".into(),
        ])
        .unwrap();
        assert!(options.stream);
        assert_eq!(options.stream_frames, Some(4096));
        assert_eq!(options.max_memory_mb, Some(64));
    }

    #[test]
    fn rejects_out_of_range_resource_limits() {
        let error = validate_effective_options(
            &Overrides {
                max_memory_mb: Some(0),
                ..Overrides::default()
            },
            VALIDATION_SAMPLE_RATE,
        )
        .unwrap_err();
        assert!(error.contains("--max-memory"));
        let error = validate_effective_options(
            &Overrides {
                stream_frames: Some(MAX_STREAM_BLOCK_FRAMES + 1),
                ..Overrides::default()
            },
            VALIDATION_SAMPLE_RATE,
        )
        .unwrap_err();
        assert!(error.contains("--stream-frames"));
    }

    #[test]
    fn metadata_limits_reserve_payload_and_descriptor_overhead() {
        let limits = metadata_limits_for_available_bytes(Some(BYTES_PER_MIB));
        assert_eq!(limits.max_total_bytes, 64 * 1024);
        assert_eq!(limits.max_item_bytes, 64 * 1024);
        assert_eq!(limits.max_flac_block_bytes, 64 * 1024);
        assert_eq!(limits.max_ogg_packet_bytes, 64 * 1024);
        assert_eq!(limits.max_items, 256);
        assert_eq!(limits.max_flac_blocks, 256);
        assert_eq!(limits.max_ogg_pages, 256);
        assert_eq!(
            limits.max_ogg_streams,
            MetadataLimits::DEFAULT_MAX_OGG_STREAMS
        );

        let defaults = MetadataLimits::default();
        let uncapped = metadata_limits_for_available_bytes(None);
        assert_eq!(uncapped, defaults);
        let large = metadata_limits_for_available_bytes(Some(u64::MAX));
        assert_eq!(large, defaults);

        let exhausted = retained_metadata_limits(Some(1), BYTES_PER_MIB).unwrap();
        assert_eq!(exhausted.max_total_bytes, 0);
        assert_eq!(exhausted.max_items, 0);
        assert_eq!(exhausted.max_flac_block_bytes, 64 * 1024);
        assert_eq!(exhausted.max_flac_blocks, 256);
        assert_eq!(exhausted.max_ogg_packet_bytes, 64 * 1024);
        assert_eq!(exhausted.max_ogg_pages, 256);
    }

    #[test]
    fn streams_wav_without_loading_the_complete_audio() {
        let root = std::env::temp_dir().join(format!(
            "denoize-stream-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&root).unwrap();
        let input = root.join("input.wav");
        let output = root.join("output.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&input, spec).unwrap();
        for frame in 0..20_000 {
            let sample = (0.2
                * (2.0 * std::f64::consts::PI * 440.0 * frame as f64 / spec.sample_rate as f64)
                    .sin()
                * 32_767.0) as i16;
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();

        run_streaming_wav(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            Overrides {
                stream: true,
                stream_frames: Some(257),
                ..Overrides::default()
            },
        )
        .unwrap();
        let result = read_audio(&output).unwrap();
        assert_eq!(result.sample_rate, spec.sample_rate);
        assert_eq!(result.channels(), 1);
        assert_eq!(result.frames(), 20_000);
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod config_file_tests {
    use super::*;

    fn write_test_config(source: &str, label: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "denoize-{label}-{}-{}.toml",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, source).unwrap();
        path
    }

    fn cli_error(extra: &[&str]) -> String {
        let mut args = vec!["input.wav".into(), "output.wav".into()];
        args.extend(extra.iter().map(|value| (*value).to_string()));
        parse_args(&args).unwrap_err()
    }

    fn cli_ok(extra: &[&str]) {
        let mut args = vec!["input.wav".into(), "output.wav".into()];
        args.extend(extra.iter().map(|value| (*value).to_string()));
        parse_args(&args).unwrap();
    }

    #[test]
    fn parses_toml_defaults() {
        let options = parse_config(
            r#"
backend = "auto"
preset = "hifi"
mode = "speech"
strength = 0.42
dpss_nw = 2.5
kaiser_beta = 9.0
adaptive_noise = true
vad = true
preserve_metadata = false
downmix = "stereo"
deterministic = true
seed = 12345
stream_frames = 4096
max_memory_mb = 64
chunk_ms = 100
"#,
            "test.toml",
        )
        .unwrap();
        assert!(options.auto_backend);
        assert!(options.deterministic);
        assert_eq!(options.seed, Some(12345));
        assert_eq!(options.downmix, Some(DownmixMode::Stereo));
        assert_eq!(options.preset, Some(Preset::HiFi));
        assert_eq!(options.mode, Some(ProcessingMode::Speech));
        assert_eq!(options.strength, Some(0.42));
        assert_eq!(options.dpss_nw, Some(2.5));
        assert_eq!(options.kaiser_beta, Some(9.0));
        assert_eq!(options.adaptive_noise, Some(true));
        assert_eq!(options.vad, Some(true));
        assert!(options.no_metadata);
        assert_eq!(options.stream_frames, Some(4096));
        assert_eq!(options.max_memory_mb, Some(64));
        assert_eq!(options.chunk_ms, Some(100));
    }

    #[test]
    fn parses_desktop_exported_config() {
        let options = parse_config(
            r#"
backend = "auto"
preset = "hifi"
mode = "speech"
strength = 0.42
adaptive_noise = true
vad = true
channels = "linked"
downmix = "stereo"
loudness_lufs = -16.0
true_peak_dbtp = -1.0
preserve_metadata = false
force = true
mp3_bitrate_kbps = 256
m4a_bitrate_kbps = 224
aac_encoder = "oxide"
onnx_model = "model.onnx"
onnx_rate = 48000
sgmse_profile = "quality"
deterministic = true
"#,
            "desktop.toml",
        )
        .unwrap();

        assert!(options.auto_backend);
        assert_eq!(options.preset, Some(Preset::HiFi));
        assert_eq!(options.mode, Some(ProcessingMode::Speech));
        assert_eq!(options.strength, Some(0.42));
        assert_eq!(options.adaptive_noise, Some(true));
        assert_eq!(options.vad, Some(true));
        assert_eq!(options.channel_mode, Some(ChannelMode::StereoLinked));
        assert_eq!(options.downmix, Some(DownmixMode::Stereo));
        assert_eq!(options.loudness_lufs, Some(-16.0));
        assert_eq!(options.true_peak_dbtp, Some(-1.0));
        assert!(options.no_metadata);
        assert!(options.force);
        assert_eq!(options.mp3_bitrate_kbps, Some(256));
        assert_eq!(options.m4a_bitrate_kbps, Some(224));
        assert_eq!(options.aac_encoder, Some(AacEncoder::Oxide));
        assert_eq!(options.onnx_model.as_deref(), Some("model.onnx"));
        assert_eq!(options.onnx_sample_rate, Some(48_000));
        assert_eq!(options.sgmse_profile, Some(SgmseProfile::Quality));
        assert!(options.deterministic);
    }

    #[test]
    fn explicit_false_config_overrides_mode_boolean_defaults() {
        let options = parse_config(
            r#"
mode = "speech"
adaptive_noise = false
vad = false
"#,
            "explicit-false.toml",
        )
        .unwrap();

        let config = build_config(&options, 48_000);
        assert!(!config.adaptive_noise);
        assert!(!config.vad);
    }

    #[test]
    fn rejects_invalid_desktop_enum_values() {
        let error = parse_config("aac_encoder = \"invalid\"", "desktop.toml").unwrap_err();
        assert!(error.contains("unknown AAC encoder in config: invalid"));

        let error = parse_config("sgmse_profile = \"invalid\"", "desktop.toml").unwrap_err();
        assert!(error.contains("unknown SGMSE profile in config: invalid"));

        let error = parse_config("quality = \"impossible\"", "desktop.toml").unwrap_err();
        assert!(error.contains("unknown quality in config: impossible"));
    }

    #[test]
    fn accepts_legacy_desktop_true_peak_without_loudness() {
        let options = parse_config(
            "true_peak_dbtp = -1.0\nmp3_bitrate_kbps = 192\n",
            "legacy-desktop.toml",
        )
        .unwrap();
        assert_eq!(options.loudness_lufs, None);
        assert_eq!(options.true_peak_dbtp, None);

        let explicit = parse_config("true_peak_dbtp = -0.5", "manual.toml").unwrap();
        assert_eq!(explicit.true_peak_dbtp, Some(-0.5));
    }

    #[test]
    fn rejects_unknown_config_keys() {
        let error = parse_config("strenth = 0.5", "test.toml").unwrap_err();
        assert!(error.contains("unknown field"));
    }

    #[test]
    fn validates_configured_dpss_time_bandwidth_product() {
        for invalid in ["nan", "inf", "+inf", "-inf", "0.0", "-0.5", "8.000001"] {
            let options = parse_config(
                &format!("window = \"dpss\"\ndpss_nw = {invalid}"),
                "test.toml",
            )
            .unwrap();
            let error = validate_effective_options(&options, VALIDATION_SAMPLE_RATE).unwrap_err();
            assert!(
                error.contains("DPSS") || error.contains("dpss"),
                "unexpected error for {invalid}: {error}"
            );
        }

        let options = parse_config("window = \"dpss\"\ndpss_nw = 8.0", "test.toml").unwrap();
        validate_effective_options(&options, VALIDATION_SAMPLE_RATE).unwrap();
        assert_eq!(options.dpss_nw, Some(8.0));
    }

    #[test]
    fn invalid_active_dpss_nw_is_rejected_before_input_or_output_io() {
        let root = std::env::temp_dir().join(format!(
            "denoize-dpss-preflight-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let input = root.join("missing.wav");
        let output = root.join("output.wav");
        let error = run(&[
            input.to_string_lossy().into_owned(),
            output.to_string_lossy().into_owned(),
            "--window".into(),
            "dpss".into(),
            "--dpss-nw".into(),
            "9".into(),
        ])
        .unwrap_err();
        assert!(error.contains("dpss") || error.contains("DPSS"));
        assert!(!output.exists());
    }

    #[test]
    fn explicit_dpss_window_takes_precedence_over_ultra_quality() {
        let explicit = Overrides {
            window: Some(WindowType::Dpss),
            dpss_nw: Some(4.0),
            quality: Some("ultra".into()),
            ..Overrides::default()
        };
        let config = build_config(&explicit, 48_000);
        assert_eq!(config.window, WindowType::Dpss);
        assert_eq!(config.window_params.dpss_bandwidth, 4.0);

        let implicit = Overrides {
            quality: Some("ultra".into()),
            ..Overrides::default()
        };
        let config = build_config(&implicit, 48_000);
        assert_eq!(config.window, WindowType::Kaiser);
        assert_eq!(config.window_params.kaiser_beta, 10.0);
    }

    #[test]
    fn command_line_overrides_config_defaults() {
        let path = write_test_config(
            "backend = \"auto\"\nstrength = 0.25\ndpss_nw = 2.5\n",
            "config",
        );
        let args = vec![
            "input.wav".into(),
            "output.wav".into(),
            "--config".into(),
            path.to_string_lossy().into_owned(),
            "--backend".into(),
            "classical".into(),
            "--strength".into(),
            "0.75".into(),
            "--dpss-nw".into(),
            "4.0".into(),
        ];
        let (_, _, options) = parse_args(&args).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(options.backend, Some(Backend::Classical));
        assert!(!options.auto_backend);
        assert_eq!(options.strength, Some(0.75));
        assert_eq!(options.dpss_nw, Some(4.0));
    }

    #[test]
    fn numeric_cli_values_override_invalid_toml_defaults_before_validation() {
        let path = write_test_config(
            r#"
strength = nan
profile_ms = inf
frame_size = 131072
window = "kaiser"
kaiser_beta = nan
dpss_nw = 9.0
loudness_lufs = nan
true_peak_dbtp = -30.0
onnx_rate = 0
stream_frames = 0
max_memory_mb = 0
jobs = 33
chunk_ms = 2001
"#,
            "numeric-precedence",
        );
        let args = vec![
            "input.wav".into(),
            "output.wav".into(),
            "--config".into(),
            path.to_string_lossy().into_owned(),
            "--strength".into(),
            "0.5".into(),
            "--profile".into(),
            "-1".into(),
            "--frame".into(),
            "256".into(),
            "--dpss-nw".into(),
            "4".into(),
            "--kaiser-beta".into(),
            "8".into(),
            "--loudness".into(),
            "-16".into(),
            "--true-peak".into(),
            "-1".into(),
            "--onnx-rate".into(),
            "16000".into(),
            "--stream-frames".into(),
            "1".into(),
            "--max-memory".into(),
            "1".into(),
            "--jobs".into(),
            "1".into(),
            "--chunk-ms".into(),
            "100".into(),
        ];
        let (_, _, options) = parse_args(&args).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(options.strength, Some(0.5));
        assert_eq!(options.profile_ms, Some(-1.0));
        assert_eq!(options.frame_size, Some(256));
        assert_eq!(options.stream_frames, Some(1));
        assert_eq!(options.jobs, Some(1));
    }

    #[test]
    fn invalid_toml_enum_is_not_hidden_by_a_cli_override() {
        let path = write_test_config("quality = \"impossible\"\n", "enum-precedence");
        let error = parse_args(&[
            "input.wav".into(),
            "output.wav".into(),
            "--config".into(),
            path.to_string_lossy().into_owned(),
            "--quality".into(),
            "high".into(),
        ])
        .unwrap_err();
        std::fs::remove_file(path).unwrap();
        assert!(error.contains("unknown quality in config"));

        let path = write_test_config("output_format = \"wma\"\n", "format-precedence");
        let error = parse_args(&[
            "input".into(),
            "output".into(),
            "--config".into(),
            path.to_string_lossy().into_owned(),
            "--output-format".into(),
            "wav".into(),
        ])
        .unwrap_err();
        std::fs::remove_file(path).unwrap();
        assert!(error.contains("unsupported --output-format"));
    }

    #[test]
    fn rejects_unknown_cli_quality_and_normalizes_legacy_aliases() {
        assert!(cli_error(&["--quality", "impossible"]).contains("unknown quality"));
        let (_, _, options) = parse_args(&[
            "input.wav".into(),
            "output.wav".into(),
            "--quality".into(),
            "highest".into(),
        ])
        .unwrap();
        assert_eq!(options.quality.as_deref(), Some("ultra"));
    }

    #[test]
    fn rejects_non_finite_external_float_values() {
        for value in ["NaN", "inf", "-inf"] {
            for (flag, prefix) in [
                ("--strength", &[][..]),
                ("--profile", &[][..]),
                ("--overlap", &[][..]),
                ("--kaiser-beta", &["--window", "kaiser"][..]),
                ("--dpss-nw", &["--window", "dpss"][..]),
                ("--smoothing", &[][..]),
                ("--makeup", &[][..]),
                ("--loudness", &[][..]),
                ("--true-peak", &["--loudness", "-16"][..]),
            ] {
                let mut extra = prefix.to_vec();
                extra.extend([flag, value]);
                let error = cli_error(&extra);
                assert!(
                    error.contains("finite"),
                    "{flag}={value} produced unexpected error: {error}"
                );
            }
        }
    }

    #[test]
    fn validates_external_float_and_rate_boundaries() {
        for (flag, minimum, maximum, below, above, prefix) in [
            ("--strength", "0", "1", "-0.001", "1.001", &[][..]),
            ("--overlap", "0.5", "0.95", "0.499", "0.951", &[][..]),
            ("--smoothing", "0", "1", "-0.001", "1.001", &[][..]),
            ("--makeup", "-120", "120", "-120.001", "120.001", &[][..]),
            (
                "--kaiser-beta",
                "0",
                "50",
                "-0.001",
                "50.001",
                &["--window", "kaiser"][..],
            ),
            (
                "--dpss-nw",
                "0.001",
                "8",
                "0",
                "8.001",
                &["--window", "dpss"][..],
            ),
            ("--loudness", "-70", "0", "-70.001", "0.001", &[][..]),
            (
                "--true-peak",
                "-20",
                "0",
                "-20.001",
                "0.001",
                &["--loudness", "-16"][..],
            ),
        ] {
            for value in [minimum, maximum] {
                let mut extra = prefix.to_vec();
                extra.extend([flag, value]);
                cli_ok(&extra);
            }
            for value in [below, above] {
                let mut extra = prefix.to_vec();
                extra.extend([flag, value]);
                assert!(!cli_error(&extra).is_empty());
            }
        }

        cli_ok(&["--profile", "60000"]);
        assert!(cli_error(&["--profile", "60000.001"]).contains("profile"));
        for value in ["1", "768000"] {
            cli_ok(&["--onnx-rate", value]);
        }
        for value in ["0", "768001"] {
            assert!(cli_error(&["--onnx-rate", value]).contains("onnx-rate"));
        }
    }

    #[test]
    fn validates_frame_resource_and_live_boundaries() {
        for value in ["0", "255", "257", "65537", "131072"] {
            assert!(cli_error(&["--frame", value]).contains("frame"));
        }
        for value in ["256", "65536"] {
            parse_args(&[
                "input.wav".into(),
                "output.wav".into(),
                "--frame".into(),
                value.into(),
            ])
            .unwrap();
        }

        for value in ["0", "1048577"] {
            assert!(cli_error(&["--stream-frames", value]).contains("stream-frames"));
        }
        for value in ["1", "1048576"] {
            parse_args(&[
                "input.wav".into(),
                "output.wav".into(),
                "--stream-frames".into(),
                value.into(),
            ])
            .unwrap();
        }

        for value in ["0", "33"] {
            assert!(cli_error(&["--jobs", value]).contains("--jobs"));
        }
        for value in ["1", "32"] {
            parse_args(&[
                "input.wav".into(),
                "output.wav".into(),
                "--jobs".into(),
                value.into(),
            ])
            .unwrap();
        }

        for value in ["9", "2001"] {
            assert!(cli_error(&["--chunk-ms", value]).contains("--chunk-ms"));
        }
        for value in ["10", "2000"] {
            parse_args(&[
                "input.wav".into(),
                "output.wav".into(),
                "--chunk-ms".into(),
                value.into(),
            ])
            .unwrap();
        }
    }

    #[test]
    fn rejects_hostile_integer_values_without_arithmetic_overflow() {
        let usize_max = usize::MAX.to_string();
        for (flag, field) in [
            ("--frame", "frame_size"),
            ("--stream-frames", "stream-frames"),
            ("--jobs", "--jobs"),
            ("--max-memory", "--max-memory"),
        ] {
            let error = cli_error(&[flag, &usize_max]);
            assert!(error.contains(field), "{flag} produced: {error}");
        }
        assert!(cli_error(&["--chunk-ms", &u32::MAX.to_string()]).contains("--chunk-ms"));
    }

    #[test]
    fn invalid_configuration_precedes_missing_input() {
        let error = parse_args(&["--strength".into(), "NaN".into()]).unwrap_err();
        assert!(error.contains("strength"));
        assert!(!error.contains("missing INPUT"));

        let error = parse_args(&["--jobs".into(), "33".into()]).unwrap_err();
        assert!(error.contains("--jobs"));
        assert!(!error.contains("missing INPUT"));

        let error =
            parse_args(&["--batch".into(), "--output-format".into(), "wma".into()]).unwrap_err();
        assert!(error.contains("unsupported --output-format"));
        assert!(!error.contains("missing INPUT"));
    }

    #[test]
    fn preserves_profile_and_true_peak_sentinel_semantics() {
        for profile in ["-1000000", "-1", "0"] {
            parse_args(&[
                "input.wav".into(),
                "output.wav".into(),
                "--profile".into(),
                profile.into(),
            ])
            .unwrap();
        }
        parse_args(&[
            "input.wav".into(),
            "output.wav".into(),
            "--loudness".into(),
            "-16".into(),
            "--true-peak".into(),
            "-1".into(),
        ])
        .unwrap();
    }

    #[test]
    fn default_batch_worker_count_is_bounded() {
        let jobs = effective_batch_jobs(&Overrides::default());
        assert!((1..=MAX_BATCH_JOBS).contains(&jobs));
    }

    #[test]
    fn parses_explicit_downmix_mode() {
        let (_, _, options) = parse_args(&[
            "input.wav".into(),
            "output.mp3".into(),
            "--downmix".into(),
            "stereo".into(),
        ])
        .unwrap();
        assert_eq!(options.downmix, Some(DownmixMode::Stereo));
    }

    #[test]
    fn parses_deterministic_seed_and_implies_mode() {
        let (_, _, options) = parse_args(&[
            "input.wav".into(),
            "output.wav".into(),
            "--seed".into(),
            "42".into(),
        ])
        .unwrap();
        assert!(options.deterministic);
        assert_eq!(options.seed, Some(42));
    }
}

fn run_metrics(args: &[String]) -> Result<(), String> {
    let reference = args.first().ok_or("metrics requires REFERENCE and TEST")?;
    let test = args.get(1).ok_or("metrics requires REFERENCE and TEST")?;
    let report =
        denoize::benchmark::BenchmarkReport::compare(&read_audio(reference)?, &read_audio(test)?)?;
    if args.iter().any(|argument| argument == "--json") {
        println!("{}", report.json());
    } else {
        println!("{}", report.markdown());
    }
    Ok(())
}

fn run_compare(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err("compare requires CLEAN NOISY ENHANCED".into());
    }
    if args[3..]
        .iter()
        .any(|argument| argument != "--json" && argument != "--html")
    {
        return Err("compare accepts only --json or --html after the input files".into());
    }
    if args.iter().any(|argument| argument == "--json")
        && args.iter().any(|argument| argument == "--html")
    {
        return Err("compare accepts only one output format".into());
    }
    let clean = args
        .first()
        .ok_or("compare requires CLEAN NOISY ENHANCED")?;
    let noisy = args.get(1).ok_or("compare requires CLEAN NOISY ENHANCED")?;
    let enhanced = args.get(2).ok_or("compare requires CLEAN NOISY ENHANCED")?;
    let report = denoize::benchmark::ComparisonReport::compare(
        &read_audio(clean)?,
        &read_audio(noisy)?,
        &read_audio(enhanced)?,
    )?;
    if args.iter().any(|argument| argument == "--json") {
        println!("{}", report.json());
    } else if args.iter().any(|argument| argument == "--html") {
        println!("{}", report.html());
    } else {
        println!("{}", report.markdown());
    }
    Ok(())
}

fn models_usage() -> &'static str {
    "\
Manage verified external models.

USAGE:
    denoize models list
    denoize models info <MODEL|all>
    denoize models install <MODEL|all> [DOWNLOAD OPTIONS]
    denoize models install <MODEL> --from <PATH>
    denoize models update <MODEL|all> [DOWNLOAD OPTIONS]
    denoize models verify <MODEL|all>
    denoize models remove <MODEL|all>
    denoize models path <MODEL|all>
    denoize models cache-dir

DOWNLOAD OPTIONS:
        --offline                  never access the network; use only verified cached data
        --proxy <URL>              use this proxy instead of proxy environment variables
        --no-proxy                 connect directly and ignore proxy environment variables
        --url <URL>                download one MODEL from an alternate HTTP(S) URL
        --bearer-token-env <VAR>   read a bearer token from environment variable VAR
        --basic-user <USER>        username for HTTP Basic authentication
        --basic-password-env <VAR> read the Basic password from environment variable VAR
        --from <PATH>              install one MODEL from a local file (install only)

Bearer tokens and Basic passwords are read from environment variables instead
of literal secret flags. Basic authentication requires both --basic-user and
--basic-password-env. Signed --url values and proxy credentials can still be
visible in process arguments. Alternate sources, origin authentication, and
--from accept one model, not `all`; --url rejects userinfo credentials.

ENVIRONMENT:
    DENOIZE_MODEL_OFFLINE, DENOIZE_MODEL_URL, DENOIZE_MODEL_PROXY,
    DENOIZE_MODEL_BEARER_TOKEN, DENOIZE_MODEL_USERNAME, DENOIZE_MODEL_PASSWORD
    HTTPS_PROXY, HTTP_PROXY, ALL_PROXY, NO_PROXY (and lowercase variants)
"
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ModelCommand {
    Info,
    Install,
    Update,
    Verify,
    Remove,
    Path,
}

#[derive(Debug)]
enum ParsedModelsCommand {
    Help,
    List,
    CacheDir,
    Run {
        command: ModelCommand,
        target: String,
        download_options: Option<Box<denoize::models::ModelDownloadOptions>>,
        source_file: Option<std::path::PathBuf>,
    },
}

fn models_option_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    *index += 1;
    let value = args
        .get(*index)
        .ok_or_else(|| format!("missing value for {flag}"))?;
    if value.is_empty() {
        return Err(format!("empty value for {flag}"));
    }
    Ok(value.clone())
}

fn validate_model_source_url(value: &str) -> Result<(), String> {
    let source = url::Url::parse(value)
        .map_err(|_| "invalid value for --url: expected an HTTP(S) URL".to_string())?;
    if !matches!(source.scheme(), "http" | "https") {
        return Err("invalid value for --url: expected an HTTP(S) URL".into());
    }
    if !source.username().is_empty() || source.password().is_some() {
        return Err(
            "--url must not contain credentials; use --bearer-token-env or Basic authentication options"
                .into(),
        );
    }
    Ok(())
}

fn read_model_secret<F>(
    flag: &str,
    variable: &str,
    read_environment: &mut F,
) -> Result<String, String>
where
    F: FnMut(&str) -> Result<String, String>,
{
    if variable.trim().is_empty() {
        return Err(format!("empty environment variable name for {flag}"));
    }
    let secret = read_environment(variable).map_err(|error| {
        format!("failed to read environment variable {variable} for {flag}: {error}")
    })?;
    if secret.is_empty() {
        return Err(format!(
            "environment variable {variable} referenced by {flag} is empty"
        ));
    }
    Ok(secret)
}

fn parse_models_command<F>(
    args: &[String],
    mut download_options: denoize::models::ModelDownloadOptions,
    mut read_environment: F,
) -> Result<ParsedModelsCommand, String>
where
    F: FnMut(&str) -> Result<String, String>,
{
    if args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
        || args.first().map(String::as_str) == Some("help")
    {
        return Ok(ParsedModelsCommand::Help);
    }

    let command_name = args.first().map(String::as_str).unwrap_or("list");
    if matches!(command_name, "list" | "cache-dir") {
        if args.len() > 1 {
            return Err(format!("models {command_name} accepts no arguments"));
        }
        return Ok(if command_name == "list" {
            ParsedModelsCommand::List
        } else {
            ParsedModelsCommand::CacheDir
        });
    }

    let command = match command_name {
        "info" => ModelCommand::Info,
        "install" => ModelCommand::Install,
        "update" => ModelCommand::Update,
        "verify" => ModelCommand::Verify,
        "remove" => ModelCommand::Remove,
        "path" => ModelCommand::Path,
        _ => return Err(format!("unknown models command: {command_name}")),
    };
    let target = args
        .get(1)
        .filter(|target| !target.starts_with('-'))
        .ok_or_else(|| format!("models {command_name} requires MODEL|all"))?
        .clone();

    if !matches!(command, ModelCommand::Install | ModelCommand::Update) {
        if args.len() > 2 {
            return Err(format!(
                "models {command_name} does not accept options or extra arguments"
            ));
        }
        return Ok(ParsedModelsCommand::Run {
            command,
            target,
            download_options: None,
            source_file: None,
        });
    }

    let mut offline_seen = false;
    let mut proxy_flag: Option<&str> = None;
    let mut source_url_seen = false;
    let mut bearer_variable: Option<String> = None;
    let mut basic_user: Option<String> = None;
    let mut basic_password_variable: Option<String> = None;
    let mut source_file: Option<std::path::PathBuf> = None;
    let mut index = 2;
    while index < args.len() {
        let flag = args[index].as_str();
        match flag {
            "--offline" => {
                if offline_seen {
                    return Err("--offline specified more than once".into());
                }
                offline_seen = true;
                download_options.offline = true;
            }
            "--proxy" => {
                if let Some(previous) = proxy_flag {
                    return Err(format!("--proxy cannot be combined with {previous}"));
                }
                let value = models_option_value(args, &mut index, flag)?;
                proxy_flag = Some("--proxy");
                download_options.proxy = denoize::models::ModelProxy::Url(value);
            }
            "--no-proxy" => {
                if let Some(previous) = proxy_flag {
                    return Err(format!("--no-proxy cannot be combined with {previous}"));
                }
                proxy_flag = Some("--no-proxy");
                download_options.proxy = denoize::models::ModelProxy::Disabled;
            }
            "--url" => {
                if source_url_seen {
                    return Err("--url specified more than once".into());
                }
                let value = models_option_value(args, &mut index, flag)?;
                validate_model_source_url(&value)?;
                source_url_seen = true;
                download_options.source_url = Some(value);
            }
            "--bearer-token-env" => {
                if bearer_variable.is_some() {
                    return Err("--bearer-token-env specified more than once".into());
                }
                bearer_variable = Some(models_option_value(args, &mut index, flag)?);
            }
            "--basic-user" => {
                if basic_user.is_some() {
                    return Err("--basic-user specified more than once".into());
                }
                basic_user = Some(models_option_value(args, &mut index, flag)?);
            }
            "--basic-password-env" => {
                if basic_password_variable.is_some() {
                    return Err("--basic-password-env specified more than once".into());
                }
                basic_password_variable = Some(models_option_value(args, &mut index, flag)?);
            }
            "--from" => {
                if source_file.is_some() {
                    return Err("--from specified more than once".into());
                }
                source_file = Some(models_option_value(args, &mut index, flag)?.into());
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown models {command_name} option: {value}"));
            }
            value => {
                return Err(format!(
                    "unexpected argument for models {command_name}: {value}"
                ));
            }
        }
        index += 1;
    }

    if source_file.is_some() {
        if command != ModelCommand::Install {
            return Err("--from is supported only by `models install`".into());
        }
        if target == "all" {
            return Err("--from requires one MODEL and cannot be used with `all`".into());
        }
        if source_url_seen
            || proxy_flag.is_some()
            || bearer_variable.is_some()
            || basic_user.is_some()
            || basic_password_variable.is_some()
        {
            return Err("--from cannot be combined with network download options".into());
        }
        download_options = denoize::models::ModelDownloadOptions::default();
        download_options.offline = offline_seen;
    }

    if bearer_variable.is_some() && (basic_user.is_some() || basic_password_variable.is_some()) {
        return Err(
            "--bearer-token-env cannot be combined with Basic authentication options".into(),
        );
    }
    download_options.authentication = if let Some(variable) = bearer_variable {
        Some(denoize::models::ModelAuthentication::Bearer(
            read_model_secret("--bearer-token-env", &variable, &mut read_environment)?,
        ))
    } else {
        match (basic_user, basic_password_variable) {
            (Some(username), Some(variable)) => {
                let password =
                    read_model_secret("--basic-password-env", &variable, &mut read_environment)?;
                Some(denoize::models::ModelAuthentication::Basic { username, password })
            }
            (None, None) => download_options.authentication,
            _ => {
                return Err(
                    "--basic-user and --basic-password-env must be specified together".into(),
                )
            }
        }
    };

    if target == "all" && download_options.source_url.is_some() {
        return Err(
            "an alternate model URL requires one MODEL and cannot be used with `all`".into(),
        );
    }
    if target == "all" && download_options.authentication.is_some() {
        return Err("model authentication requires one MODEL and cannot be used with `all`".into());
    }

    Ok(ParsedModelsCommand::Run {
        command,
        target,
        download_options: Some(Box::new(download_options)),
        source_file,
    })
}

fn model_download_options_from_environment_with<F>(
    args: &[String],
    mut read_environment: F,
) -> Result<denoize::models::ModelDownloadOptions, String>
where
    F: FnMut(&str) -> Option<String>,
{
    if args.iter().any(|argument| argument == "--from") {
        return Ok(denoize::models::ModelDownloadOptions::default());
    }
    let overrides_offline = args.iter().any(|argument| argument == "--offline");
    let overrides_source = args.iter().any(|argument| argument == "--url");
    let overrides_proxy = args
        .iter()
        .any(|argument| matches!(argument.as_str(), "--proxy" | "--no-proxy"));
    let overrides_authentication = args.iter().any(|argument| {
        matches!(
            argument.as_str(),
            "--bearer-token-env" | "--basic-user" | "--basic-password-env"
        )
    });
    denoize::models::ModelDownloadOptions::from_env_with(|name| {
        let overridden = match name {
            "DENOIZE_MODEL_OFFLINE" => overrides_offline,
            "DENOIZE_MODEL_URL" => overrides_source,
            "DENOIZE_MODEL_PROXY" => overrides_proxy,
            "DENOIZE_MODEL_BEARER_TOKEN" | "DENOIZE_MODEL_USERNAME" | "DENOIZE_MODEL_PASSWORD" => {
                overrides_authentication
            }
            _ => false,
        };
        (!overridden).then(|| read_environment(name)).flatten()
    })
}

fn model_info_output(model: &denoize::models::ModelInfo, path: &std::path::Path) -> String {
    format!(
        "name: {}\nbackend: {}\nsample-rate: {}\nlicense: {}\nrevision: {}\nsize-bytes: {}\nsha256: {}\nurl: {}\npath: {}\n",
        model.name,
        model.backend,
        model.sample_rate,
        model.license,
        model.revision,
        model.size_bytes,
        model.sha256,
        denoize::models::redact_url(model.url),
        path.display(),
    )
}

fn run_models(args: &[String]) -> Result<(), String> {
    let help_requested = args
        .iter()
        .any(|argument| matches!(argument.as_str(), "-h" | "--help"))
        || args.first().map(String::as_str) == Some("help");
    let download_command = matches!(args.first().map(String::as_str), Some("install" | "update"));
    let download_options = if download_command && !help_requested {
        model_download_options_from_environment_with(args, |name| std::env::var(name).ok())?
    } else {
        denoize::models::ModelDownloadOptions::default()
    };
    let parsed = parse_models_command(args, download_options, |name| {
        std::env::var(name).map_err(|error| error.to_string())
    })?;

    let (command, target, download_options, source_file) = match parsed {
        ParsedModelsCommand::Help => {
            print!("{}", models_usage());
            return Ok(());
        }
        ParsedModelsCommand::List => {
            println!("NAME\tBACKEND\tRATE\tLICENSE\tSTATUS");
            for model in denoize::models::MODELS {
                let status = if denoize::models::verify(model).is_ok() {
                    "installed"
                } else {
                    "not-installed"
                };
                println!(
                    "{}\t{}\t{}\t{}\t{}",
                    model.name, model.backend, model.sample_rate, model.license, status
                );
            }
            return Ok(());
        }
        ParsedModelsCommand::CacheDir => {
            println!("{}", denoize::models::cache_dir()?.display());
            return Ok(());
        }
        ParsedModelsCommand::Run {
            command,
            target,
            download_options,
            source_file,
        } => (command, target, download_options, source_file),
    };

    let models: Vec<_> = if target == "all" {
        denoize::models::MODELS.iter().collect()
    } else {
        vec![denoize::models::find(&target)
            .ok_or_else(|| format!("unknown model: {target} (run `denoize models list`)"))?]
    };
    for model in models {
        match command {
            ModelCommand::Info => {
                let path = denoize::models::path(model)?;
                print!("{}", model_info_output(model, &path));
            }
            ModelCommand::Install => {
                let installed = if let Some(source) = source_file.as_ref() {
                    denoize::models::install_from_file(model, source)?
                } else {
                    denoize::models::install_with_options(
                        model,
                        download_options
                            .as_ref()
                            .expect("download options exist for install"),
                    )?
                };
                println!("{}", installed.display());
            }
            ModelCommand::Update => println!(
                "{}",
                denoize::models::update_with_options(
                    model,
                    download_options
                        .as_ref()
                        .expect("download options exist for update"),
                )?
                .display()
            ),
            ModelCommand::Verify => {
                println!("verified {}", denoize::models::verify(model)?.display())
            }
            ModelCommand::Remove => println!(
                "{} {}",
                if denoize::models::remove(model)? {
                    "removed"
                } else {
                    "not-installed"
                },
                model.name
            ),
            ModelCommand::Path => println!("{}", denoize::models::path(model)?.display()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod model_command_tests {
    use super::*;

    fn missing_secret(name: &str) -> Result<String, String> {
        Err(format!("{name} is not set"))
    }

    #[test]
    fn model_info_reports_exact_manifest_size_in_bytes() {
        let model = denoize::models::ModelInfo {
            name: "test-model",
            backend: "test-backend",
            filename: "model.onnx",
            url: "https://models.example/model.onnx",
            revision: "test-revision",
            size_bytes: 12_345_678,
            sha256: "0123456789abcdef",
            license: "MIT",
            sample_rate: 16_000,
        };

        let output = model_info_output(&model, std::path::Path::new("model.onnx"));

        assert_eq!(
            output,
            "name: test-model\nbackend: test-backend\nsample-rate: 16000\nlicense: MIT\nrevision: test-revision\nsize-bytes: 12345678\nsha256: 0123456789abcdef\nurl: https://models.example/model.onnx\npath: model.onnx\n"
        );
    }

    #[test]
    fn explicit_model_flags_override_invalid_environment_defaults() {
        let args = vec![
            "install".into(),
            "gtcrn-dns3".into(),
            "--offline".into(),
            "--url".into(),
            "https://models.example/model.onnx".into(),
            "--no-proxy".into(),
            "--bearer-token-env".into(),
            "MODEL_TOKEN".into(),
        ];
        let options = model_download_options_from_environment_with(&args, |name| {
            Some(
                match name {
                    "DENOIZE_MODEL_OFFLINE" => "not-a-boolean",
                    "DENOIZE_MODEL_URL" => "environment-url",
                    "DENOIZE_MODEL_PROXY" => "environment-proxy",
                    "DENOIZE_MODEL_BEARER_TOKEN" => "environment-bearer",
                    "DENOIZE_MODEL_USERNAME" => "environment-user",
                    "DENOIZE_MODEL_PASSWORD" => "environment-password",
                    _ => return None,
                }
                .into(),
            )
        })
        .unwrap();
        assert!(!options.offline);
        assert!(options.source_url.is_none());
        assert!(matches!(
            options.proxy,
            denoize::models::ModelProxy::Environment
        ));
        assert!(options.authentication.is_none());
    }

    #[test]
    fn local_model_install_does_not_validate_unrelated_environment_defaults() {
        let args = vec![
            "install".into(),
            "gtcrn-dns3".into(),
            "--from".into(),
            "model.onnx".into(),
        ];
        let options = model_download_options_from_environment_with(&args, |_| {
            panic!("local installs must not read model download environment variables")
        })
        .unwrap();
        assert!(!options.offline);
        assert!(options.source_url.is_none());
        assert!(options.authentication.is_none());
    }

    #[test]
    fn parses_model_download_overrides_without_reading_process_environment() {
        let mut base = denoize::models::ModelDownloadOptions::default();
        base.source_url = Some("https://environment.invalid/model".into());
        base.authentication = Some(denoize::models::ModelAuthentication::Basic {
            username: "environment-user".into(),
            password: "environment-secret".into(),
        });
        let args = vec![
            "update".into(),
            "gtcrn-dns3".into(),
            "--url".into(),
            "https://models.example/model.onnx".into(),
            "--no-proxy".into(),
            "--bearer-token-env".into(),
            "MODEL_TOKEN".into(),
        ];
        let parsed = parse_models_command(&args, base, |name| {
            assert_eq!(name, "MODEL_TOKEN");
            Ok("secret-token".into())
        })
        .unwrap();

        let ParsedModelsCommand::Run {
            command,
            target,
            download_options: Some(options),
            source_file,
        } = parsed
        else {
            panic!("expected an executable model command");
        };
        assert_eq!(command, ModelCommand::Update);
        assert_eq!(target, "gtcrn-dns3");
        assert!(source_file.is_none());
        assert_eq!(
            options.source_url.as_deref(),
            Some("https://models.example/model.onnx")
        );
        assert!(matches!(
            options.proxy,
            denoize::models::ModelProxy::Disabled
        ));
        assert!(matches!(
            options.authentication,
            Some(denoize::models::ModelAuthentication::Bearer(ref token)) if token == "secret-token"
        ));
    }

    #[test]
    fn parses_basic_authentication_and_local_install() {
        let basic = vec![
            "install".into(),
            "gtcrn-dns3".into(),
            "--basic-user".into(),
            "release-bot".into(),
            "--basic-password-env".into(),
            "MODEL_PASSWORD".into(),
        ];
        let parsed = parse_models_command(
            &basic,
            denoize::models::ModelDownloadOptions::default(),
            |_| Ok("password-from-environment".into()),
        )
        .unwrap();
        let ParsedModelsCommand::Run {
            download_options: Some(options),
            ..
        } = parsed
        else {
            panic!("expected download options");
        };
        assert!(matches!(
            options.authentication,
            Some(denoize::models::ModelAuthentication::Basic {
                ref username,
                ref password,
            }) if username == "release-bot" && password == "password-from-environment"
        ));

        let local = vec![
            "install".into(),
            "gtcrn-dns3".into(),
            "--offline".into(),
            "--from".into(),
            "model.onnx".into(),
        ];
        let parsed = parse_models_command(
            &local,
            denoize::models::ModelDownloadOptions::default(),
            missing_secret,
        )
        .unwrap();
        let ParsedModelsCommand::Run {
            command,
            source_file: Some(source),
            download_options: Some(options),
            ..
        } = parsed
        else {
            panic!("expected a local install");
        };
        assert_eq!(command, ModelCommand::Install);
        assert_eq!(source, std::path::PathBuf::from("model.onnx"));
        assert!(options.offline);
    }

    #[test]
    fn rejects_conflicting_or_incomplete_model_options() {
        let cases = [
            (
                vec![
                    "install".into(),
                    "gtcrn-dns3".into(),
                    "--proxy".into(),
                    "http://proxy.example".into(),
                    "--no-proxy".into(),
                ],
                "cannot be combined",
            ),
            (
                vec![
                    "install".into(),
                    "gtcrn-dns3".into(),
                    "--basic-user".into(),
                    "release-bot".into(),
                ],
                "must be specified together",
            ),
            (
                vec![
                    "install".into(),
                    "gtcrn-dns3".into(),
                    "--bearer-token-env".into(),
                    "TOKEN".into(),
                    "--basic-user".into(),
                    "release-bot".into(),
                    "--basic-password-env".into(),
                    "PASSWORD".into(),
                ],
                "cannot be combined",
            ),
            (
                vec![
                    "install".into(),
                    "gtcrn-dns3".into(),
                    "--from".into(),
                    "model.onnx".into(),
                    "--proxy".into(),
                    "http://proxy.example".into(),
                ],
                "network download options",
            ),
        ];
        for (args, expected) in cases {
            let error = parse_models_command(
                &args,
                denoize::models::ModelDownloadOptions::default(),
                missing_secret,
            )
            .unwrap_err();
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    #[test]
    fn rejects_options_outside_their_supported_target_or_command() {
        let cases = [
            (
                vec!["info".into(), "gtcrn-dns3".into(), "--offline".into()],
                "does not accept options",
            ),
            (
                vec![
                    "update".into(),
                    "gtcrn-dns3".into(),
                    "--from".into(),
                    "model.onnx".into(),
                ],
                "install",
            ),
            (
                vec![
                    "install".into(),
                    "all".into(),
                    "--from".into(),
                    "model.onnx".into(),
                ],
                "cannot be used with `all`",
            ),
            (
                vec![
                    "update".into(),
                    "all".into(),
                    "--url".into(),
                    "https://models.example/model.onnx".into(),
                ],
                "cannot be used with `all`",
            ),
            (
                vec![
                    "install".into(),
                    "gtcrn-dns3".into(),
                    "--url".into(),
                    "https://user:secret@models.example/model.onnx".into(),
                ],
                "must not contain credentials",
            ),
        ];
        for (args, expected) in cases {
            let error = parse_models_command(
                &args,
                denoize::models::ModelDownloadOptions::default(),
                missing_secret,
            )
            .unwrap_err();
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    #[test]
    fn rejects_environment_source_or_authentication_for_all_models() {
        let args = vec!["update".into(), "all".into()];
        let mut source = denoize::models::ModelDownloadOptions::default();
        source.source_url = Some("https://mirror.example/model.onnx".into());
        let source_error = parse_models_command(&args, source, missing_secret).unwrap_err();
        assert!(source_error.contains("cannot be used with `all`"));

        let mut authenticated = denoize::models::ModelDownloadOptions::default();
        authenticated.authentication = Some(denoize::models::ModelAuthentication::Bearer(
            "environment-token".into(),
        ));
        let authentication_error =
            parse_models_command(&args, authenticated, missing_secret).unwrap_err();
        assert!(authentication_error.contains("requires one MODEL"));
    }

    #[test]
    fn reports_missing_secret_environment_variables_without_exposing_values() {
        let args = vec![
            "install".into(),
            "gtcrn-dns3".into(),
            "--bearer-token-env".into(),
            "MISSING_TOKEN".into(),
        ];
        let error = parse_models_command(
            &args,
            denoize::models::ModelDownloadOptions::default(),
            missing_secret,
        )
        .unwrap_err();
        assert!(error.contains("MISSING_TOKEN"));
        assert!(error.contains("not set"));
    }

    #[test]
    fn exposes_dedicated_models_help() {
        let parsed = parse_models_command(
            &["--help".into()],
            denoize::models::ModelDownloadOptions::default(),
            missing_secret,
        )
        .unwrap();
        assert!(matches!(parsed, ParsedModelsCommand::Help));
        for flag in [
            "--offline",
            "--proxy",
            "--no-proxy",
            "--url",
            "--bearer-token-env",
            "--basic-user",
            "--basic-password-env",
            "--from",
        ] {
            assert!(models_usage().contains(flag));
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("denoize: error: {e}");
        eprintln!("run 'denoize --help' for usage.");
        std::process::exit(1);
    }
}

#[cfg(all(test, feature = "onnx"))]
mod tests {
    use super::*;

    #[test]
    fn parses_onnx_model_options() {
        let args = vec![
            "input.wav".into(),
            "output.wav".into(),
            "--backend".into(),
            "onnx".into(),
            "--onnx-model".into(),
            "model.onnx".into(),
            "--onnx-rate".into(),
            "48000".into(),
        ];
        let (_, _, options) = parse_args(&args).unwrap();
        assert_eq!(options.backend, Some(Backend::Onnx));
        assert_eq!(options.onnx_model.as_deref(), Some("model.onnx"));
        assert_eq!(options.onnx_sample_rate, Some(48_000));
    }

    #[test]
    fn selected_external_backend_requires_a_model_before_input_io() {
        let error = parse_args(&[
            "--backend".into(),
            "onnx".into(),
            "--onnx-rate".into(),
            "16000".into(),
        ])
        .unwrap_err();
        assert!(error.contains("backend_options.onnx"));
        assert!(!error.contains("missing INPUT"));
    }

    #[test]
    fn parses_live_device_options() {
        let args = vec![
            "-".into(),
            "-".into(),
            "--input-device".into(),
            "Mic".into(),
            "--output-device".into(),
            "Cable".into(),
            "--chunk-ms".into(),
            "40".into(),
        ];
        let (_, _, options) = parse_args(&args).unwrap();
        assert_eq!(options.input_device.as_deref(), Some("Mic"));
        assert_eq!(options.output_device.as_deref(), Some("Cable"));
        assert_eq!(options.chunk_ms, Some(40));
    }
}
