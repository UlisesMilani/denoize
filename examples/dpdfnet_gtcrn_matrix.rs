//! Reproducible multi-condition quality matrix for the DPDFNet issue PoC.

use denoize::backend::dpdfnet::{
    DPDFNET2_STATE_SIZE, DPDFNET8_STATE_SIZE, MODEL_LOOKAHEAD_SAMPLES, SAMPLE_RATE as DPDFNET_RATE,
};
use denoize::{
    read_audio, write_wav, Audio, ComparisonReport, DpdfnetModel, GtcrnModel, OnnxModelConfig,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

const GTCRN_RATE: u32 = 16_000;

#[derive(Debug)]
struct Args {
    manifest: PathBuf,
    dpdfnet2_model: PathBuf,
    dpdfnet8_model: PathBuf,
    gtcrn_model: PathBuf,
    json: PathBuf,
    audio_dir: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MatrixManifest {
    schema: String,
    fixture_fingerprint: String,
    cases: Vec<MatrixCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MatrixCase {
    id: String,
    kind: String,
    speaker: String,
    noise: Option<String>,
    requested_snr_db: Option<f64>,
    actual_snr_db: Option<f64>,
    clean: PathBuf,
    noisy: PathBuf,
    sample_rate: u32,
    #[serde(default)]
    write_audio: bool,
}

fn main() {
    if env::args()
        .skip(1)
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!("{}", usage());
        return;
    }
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let manifest_bytes = std::fs::read(&args.manifest)
        .map_err(|error| format!("read matrix manifest {}: {error}", args.manifest.display()))?;
    let manifest: MatrixManifest = serde_json::from_slice(&manifest_bytes).map_err(|error| {
        format!(
            "decode matrix manifest {}: {error}",
            args.manifest.display()
        )
    })?;
    if manifest.schema != "denoize-dpdfnet-evaluation-manifest-v1" {
        return Err(format!(
            "unsupported matrix manifest schema `{}`",
            manifest.schema
        ));
    }
    if manifest.cases.is_empty() {
        return Err("matrix manifest contains no cases".into());
    }
    let manifest_dir = args
        .manifest
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    let started = Instant::now();
    let dpdfnet2 = DpdfnetModel::load(&OnnxModelConfig {
        path: args.dpdfnet2_model.clone(),
        sample_rate: DPDFNET_RATE,
    })?;
    let dpdfnet2_load_ms = milliseconds(started.elapsed());
    require_state_geometry(&dpdfnet2, DPDFNET2_STATE_SIZE, "DPDFNet-2")?;

    let started = Instant::now();
    let dpdfnet8 = DpdfnetModel::load(&OnnxModelConfig {
        path: args.dpdfnet8_model.clone(),
        sample_rate: DPDFNET_RATE,
    })?;
    let dpdfnet8_load_ms = milliseconds(started.elapsed());
    require_state_geometry(&dpdfnet8, DPDFNET8_STATE_SIZE, "DPDFNet-8")?;

    let started = Instant::now();
    let gtcrn = GtcrnModel::load(&OnnxModelConfig {
        path: args.gtcrn_model.clone(),
        sample_rate: GTCRN_RATE,
    })?;
    let gtcrn_load_ms = milliseconds(started.elapsed());

    warm_up(
        &manifest_dir,
        &manifest.cases[0],
        &dpdfnet2,
        &dpdfnet8,
        &gtcrn,
    )?;

    let mut results = Vec::with_capacity(manifest.cases.len());
    for (index, case) in manifest.cases.iter().enumerate() {
        eprintln!("case {}/{}: {}", index + 1, manifest.cases.len(), case.id);
        results.push(run_case(
            &manifest_dir,
            case,
            &dpdfnet2,
            &dpdfnet8,
            &gtcrn,
            args.audio_dir.as_deref(),
        )?);
    }

    let result = json!({
        "schema": "denoize-dpdfnet-gtcrn-matrix-v1",
        "fixture_manifest": args.manifest,
        "fixture_fingerprint": manifest.fixture_fingerprint,
        "environment": {
            "os": env::consts::OS,
            "arch": env::consts::ARCH,
            "logical_parallelism": std::thread::available_parallelism().map(|value| value.get()).ok(),
            "visqol_enabled": cfg!(feature = "visqol"),
        },
        "models": {
            "dpdfnet2_48khz_hr": {
                "path": args.dpdfnet2_model,
                "state_size": dpdfnet2.metadata().state_size,
                "upstream_profile_metadata": dpdfnet2.metadata().profile,
                "load_ms": dpdfnet2_load_ms,
            },
            "dpdfnet8_48khz_hr": {
                "path": args.dpdfnet8_model,
                "state_size": dpdfnet8.metadata().state_size,
                "upstream_profile_metadata": dpdfnet8.metadata().profile,
                "load_ms": dpdfnet8_load_ms,
            },
            "gtcrn": {
                "path": args.gtcrn_model,
                "load_ms": gtcrn_load_ms,
            },
        },
        "cases": results,
    });
    let bytes = serde_json::to_vec_pretty(&result)
        .map_err(|error| format!("encode matrix result: {error}"))?;
    std::fs::write(&args.json, bytes)
        .map_err(|error| format!("write matrix result {}: {error}", args.json.display()))?;
    println!("matrix JSON result: {}", args.json.display());
    Ok(())
}

fn require_state_geometry(
    model: &DpdfnetModel,
    expected: usize,
    label: &str,
) -> Result<(), String> {
    if model.metadata().state_size == expected {
        Ok(())
    } else {
        Err(format!(
            "{label} model has {} state scalars, expected {expected}",
            model.metadata().state_size
        ))
    }
}

fn warm_up(
    root: &Path,
    case: &MatrixCase,
    dpdfnet2: &DpdfnetModel,
    dpdfnet8: &DpdfnetModel,
    gtcrn: &GtcrnModel,
) -> Result<(), String> {
    let mut audio = load_case_audio(root, &case.noisy, case.sample_rate)?;
    let warmup_frames = (case.sample_rate as usize).min(audio.frames());
    for channel in &mut audio.channels {
        channel.truncate(warmup_frames);
    }
    dpdfnet2.process(&audio.channels, audio.sample_rate)?;
    dpdfnet8.process(&audio.channels, audio.sample_rate)?;
    gtcrn.process(&audio.channels, audio.sample_rate)?;
    Ok(())
}

fn run_case(
    root: &Path,
    case: &MatrixCase,
    dpdfnet2: &DpdfnetModel,
    dpdfnet8: &DpdfnetModel,
    gtcrn: &GtcrnModel,
    audio_dir: Option<&Path>,
) -> Result<Value, String> {
    validate_case(case)?;
    let mut clean = load_case_audio(root, &case.clean, case.sample_rate)?;
    let mut noisy = load_case_audio(root, &case.noisy, case.sample_rate)?;
    equalize_geometry(&mut clean, &mut noisy)?;
    let audio_seconds = clean.frames() as f64 / clean.sample_rate as f64;

    let started = Instant::now();
    let dpdfnet2_output = with_channels(
        &noisy,
        dpdfnet2.process(&noisy.channels, noisy.sample_rate)?,
    )?;
    let dpdfnet2_ms = milliseconds(started.elapsed());

    let started = Instant::now();
    let dpdfnet8_output = with_channels(
        &noisy,
        dpdfnet8.process(&noisy.channels, noisy.sample_rate)?,
    )?;
    let dpdfnet8_ms = milliseconds(started.elapsed());

    let started = Instant::now();
    let gtcrn_output = with_channels(&noisy, gtcrn.process(&noisy.channels, noisy.sample_rate)?)?;
    let gtcrn_ms = milliseconds(started.elapsed());

    let alignment_samples = ((MODEL_LOOKAHEAD_SAMPLES as u64 * clean.sample_rate as u64)
        / DPDFNET_RATE as u64) as usize;
    let common_frames = clean
        .frames()
        .checked_sub(alignment_samples)
        .ok_or_else(|| format!("case `{}` is shorter than DPDFNet lookahead", case.id))?;
    let clean = crop(&clean, 0, common_frames)?;
    let noisy = crop(&noisy, 0, common_frames)?;
    let dpdfnet2_output = crop(&dpdfnet2_output, alignment_samples, common_frames)?;
    let dpdfnet8_output = crop(&dpdfnet8_output, alignment_samples, common_frames)?;
    let gtcrn_output = crop(&gtcrn_output, 0, common_frames)?;

    let dpdfnet2_quality = ComparisonReport::compare(&clean, &noisy, &dpdfnet2_output)?;
    let dpdfnet8_quality = ComparisonReport::compare(&clean, &noisy, &dpdfnet8_output)?;
    let gtcrn_quality = ComparisonReport::compare(&clean, &noisy, &gtcrn_output)?;

    if case.write_audio {
        let root = audio_dir.ok_or_else(|| {
            format!(
                "case `{}` requests audio but --audio-dir is absent",
                case.id
            )
        })?;
        let output = root.join(&case.id);
        std::fs::create_dir_all(&output)
            .map_err(|error| format!("create listening directory {}: {error}", output.display()))?;
        write_wav(output.join("clean.wav"), &clean)?;
        write_wav(output.join("noisy.wav"), &noisy)?;
        write_wav(output.join("dpdfnet2.wav"), &dpdfnet2_output)?;
        write_wav(output.join("dpdfnet8.wav"), &dpdfnet8_output)?;
        write_wav(output.join("gtcrn.wav"), &gtcrn_output)?;
    }

    Ok(json!({
        "id": case.id,
        "kind": case.kind,
        "speaker": case.speaker,
        "noise": case.noise,
        "requested_snr_db": case.requested_snr_db,
        "actual_snr_db": case.actual_snr_db,
        "sample_rate": clean.sample_rate,
        "input_duration_seconds": audio_seconds,
        "evaluation_frames": common_frames,
        "alignment_samples": alignment_samples,
        "dpdfnet2_48khz_hr": model_result(dpdfnet2_ms, audio_seconds, &dpdfnet2_quality)?,
        "dpdfnet8_48khz_hr": model_result(dpdfnet8_ms, audio_seconds, &dpdfnet8_quality)?,
        "gtcrn": model_result(gtcrn_ms, audio_seconds, &gtcrn_quality)?,
        "dpdfnet2_vs_dpdfnet8": waveform_difference(&dpdfnet2_output, &dpdfnet8_output),
    }))
}

fn validate_case(case: &MatrixCase) -> Result<(), String> {
    if case.id.is_empty()
        || !case
            .id
            .bytes()
            .all(|value| value.is_ascii_alphanumeric() || matches!(value, b'-' | b'_'))
    {
        return Err(format!("case id `{}` is not path-safe", case.id));
    }
    if case.sample_rate == 0 {
        return Err(format!("case `{}` has a zero sample rate", case.id));
    }
    for (label, value) in [
        ("requested SNR", case.requested_snr_db),
        ("actual SNR", case.actual_snr_db),
    ] {
        if value.is_some_and(|value| !value.is_finite()) {
            return Err(format!("case `{}` has a non-finite {label}", case.id));
        }
    }
    Ok(())
}

fn load_case_audio(root: &Path, path: &Path, sample_rate: u32) -> Result<Audio, String> {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let mut audio = read_audio(&path)?;
    if audio.channels() != 1 || audio.frames() == 0 {
        return Err(format!(
            "matrix fixture {} must be non-empty mono audio",
            path.display()
        ));
    }
    if audio.sample_rate != sample_rate {
        audio.channels =
            denoize::resample::resample_channels(&audio.channels, audio.sample_rate, sample_rate)?;
        audio.sample_rate = sample_rate;
    }
    Ok(audio)
}

fn equalize_geometry(clean: &mut Audio, noisy: &mut Audio) -> Result<(), String> {
    if clean.sample_rate != noisy.sample_rate || clean.channels() != noisy.channels() {
        return Err("clean/noisy sample rate or channel count differs".into());
    }
    let frames = clean.frames().min(noisy.frames());
    if frames == 0 {
        return Err("clean/noisy fixture is empty".into());
    }
    for channel in &mut clean.channels {
        channel.truncate(frames);
    }
    for channel in &mut noisy.channels {
        channel.truncate(frames);
    }
    Ok(())
}

fn with_channels(template: &Audio, channels: Vec<Vec<f64>>) -> Result<Audio, String> {
    if channels.len() != template.channels()
        || channels
            .iter()
            .any(|channel| channel.len() != template.frames())
    {
        return Err("model output geometry differs from its input".into());
    }
    let mut output = template.clone();
    output.channels = channels;
    Ok(output)
}

fn crop(audio: &Audio, start: usize, frames: usize) -> Result<Audio, String> {
    let end = start
        .checked_add(frames)
        .ok_or_else(|| "audio crop overflow".to_string())?;
    if audio.channels.iter().any(|channel| end > channel.len()) {
        return Err("audio crop exceeds a channel".into());
    }
    let mut output = audio.clone();
    output.channels = audio
        .channels
        .iter()
        .map(|channel| channel[start..end].to_vec())
        .collect();
    Ok(output)
}

fn model_result(
    process_ms: f64,
    audio_seconds: f64,
    quality: &ComparisonReport,
) -> Result<Value, String> {
    Ok(json!({
        "process_ms": process_ms,
        "rtf": process_ms / 1_000.0 / audio_seconds,
        "quality": serde_json::from_str::<Value>(&quality.json())
            .map_err(|error| format!("decode quality report: {error}"))?,
    }))
}

fn waveform_difference(left: &Audio, right: &Audio) -> Value {
    let pairs = left
        .channels
        .iter()
        .flatten()
        .zip(right.channels.iter().flatten());
    let mut count = 0usize;
    let mut squared_error = 0.0;
    let mut left_energy = 0.0;
    let mut right_energy = 0.0;
    let mut product = 0.0;
    for (left, right) in pairs {
        count += 1;
        squared_error += (left - right).powi(2);
        left_energy += left * left;
        right_energy += right * right;
        product += left * right;
    }
    let scale = 1.0 / count.max(1) as f64;
    json!({
        "rms_difference": (squared_error * scale).sqrt(),
        "cosine_similarity": product / (left_energy * right_energy).sqrt().max(1.0e-20),
    })
}

fn milliseconds(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn parse_args() -> Result<Args, String> {
    let mut arguments = env::args().skip(1);
    let mut manifest = None;
    let mut dpdfnet2_model = None;
    let mut dpdfnet8_model = None;
    let mut gtcrn_model = None;
    let mut json = None;
    let mut audio_dir = None;
    while let Some(argument) = arguments.next() {
        let value = |arguments: &mut std::iter::Skip<std::env::Args>| {
            arguments
                .next()
                .ok_or_else(|| format!("{argument} requires a value"))
        };
        match argument.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(value(&mut arguments)?)),
            "--dpdfnet2-model" => {
                dpdfnet2_model = Some(PathBuf::from(value(&mut arguments)?));
            }
            "--dpdfnet8-model" => {
                dpdfnet8_model = Some(PathBuf::from(value(&mut arguments)?));
            }
            "--gtcrn-model" => gtcrn_model = Some(PathBuf::from(value(&mut arguments)?)),
            "--json" => json = Some(PathBuf::from(value(&mut arguments)?)),
            "--audio-dir" => audio_dir = Some(PathBuf::from(value(&mut arguments)?)),
            _ => return Err(format!("unknown argument `{argument}`\n{}", usage())),
        }
    }
    Ok(Args {
        manifest: required(manifest, "--manifest")?,
        dpdfnet2_model: required(dpdfnet2_model, "--dpdfnet2-model")?,
        dpdfnet8_model: required(dpdfnet8_model, "--dpdfnet8-model")?,
        gtcrn_model: required(gtcrn_model, "--gtcrn-model")?,
        json: required(json, "--json")?,
        audio_dir,
    })
}

fn required(value: Option<PathBuf>, name: &str) -> Result<PathBuf, String> {
    value.ok_or_else(|| format!("missing {name}\n{}", usage()))
}

fn usage() -> &'static str {
    "usage: dpdfnet_gtcrn_matrix --manifest MANIFEST.json \\\n+  --dpdfnet2-model DPDFNET2.onnx --dpdfnet8-model DPDFNET8.onnx \\\n+  --gtcrn-model GTCRN.onnx --json RESULT.json [--audio-dir DIR]"
}
