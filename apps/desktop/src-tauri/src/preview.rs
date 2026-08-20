use super::{
    configured_backend, desktop_output_format_name, desktop_resource_governor,
    parsed_backend_options_for, parsed_encode_options, processing_config, validate_process_options,
    IsolatedChild, JobControl, ProcessOptions,
};
use denoize::audio::{estimate_audio_memory_bytes, estimate_audio_working_set_bytes};
use denoize::batch_resume::{self, MetadataPolicy};
use denoize::encode::write_audio_to_file;
use denoize::service::{self, BackendChoice, ProcessingOptions};
use denoize::{
    AtomicOutput, Audio, AudioFormat, AudioInputSession, AudioStreamReader, BackendSession,
    CommitMode, DecodeLimits, EncodeOptions, OutputFormat, PresentationRegion, ResourceRequest,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};

pub(crate) const PREVIEW_WORKER_ARGUMENT: &str = "--denoize-preview-worker";
const PREVIEW_WORKER_SCHEMA: &str = "denoize-desktop-preview-worker-v1";
const PREVIEW_RESULT_SCHEMA: &str = "denoize-desktop-preview-v1";
const PREVIEW_OWNER_SCHEMA: &str = "denoize-desktop-preview-owner-v1";
const PREVIEW_START_GATE_FILE: &str = "start.gate";
const PREVIEW_OWNER_FILE: &str = ".denoize-preview-owner-v1.json";
const PREVIEW_SCHEMA_VERSION: u32 = 1;
const MIN_PREVIEW_SECONDS: f64 = 0.4;
const MAX_PREVIEW_SECONDS: f64 = 30.0;
const MAX_PREVIEW_MEMORY_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_PREVIEW_TEMPORARY_BYTES: u64 = 256 * 1024 * 1024;
const MAX_WORKER_DOCUMENT_BYTES: u64 = 128 * 1024;
const PREVIEW_BLOCK_FRAMES: usize = 8_192;
const MAX_PREVIEW_WORKER_SECONDS: u64 = 600;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct PreviewRequest {
    pub input: String,
    pub output: String,
    pub start_seconds: f64,
    pub duration_seconds: f64,
    pub points: usize,
    pub options: ProcessOptions,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PreviewWorkerRequest {
    schema: String,
    schema_version: u32,
    nonce: String,
    parent_process_id: u32,
    output_directory: String,
    start_gate: String,
    preview: PreviewRequest,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PreviewWorkerEnvelope {
    schema: String,
    schema_version: u32,
    nonce: String,
    result: Option<PreviewResult>,
    error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PreviewOwner {
    schema: String,
    schema_version: u32,
    preview_id: String,
    process_id: u32,
}

impl PreviewOwner {
    fn new(preview_id: String) -> Self {
        Self {
            schema: PREVIEW_OWNER_SCHEMA.into(),
            schema_version: PREVIEW_SCHEMA_VERSION,
            preview_id,
            process_id: std::process::id(),
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema != PREVIEW_OWNER_SCHEMA
            || self.schema_version != PREVIEW_SCHEMA_VERSION
            || self.process_id == 0
            || self.preview_id.len() != 64
            || !self
                .preview_id
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err("プレビュー所有markerが不正です".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct PreviewArtifact {
    pub source: String,
    pub playable_path: String,
    pub duration_seconds: f64,
    pub loudness_lufs: Option<f64>,
    pub rms_db: f64,
    pub waveform: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct PreviewResult {
    pub schema: String,
    pub schema_version: u32,
    pub preview_id: String,
    pub locator: PresentationRegion,
    pub recipe: String,
    pub output_format: String,
    pub backend: String,
    pub accelerator: String,
    pub options: ProcessOptions,
    pub original: PreviewArtifact,
    pub processed: PreviewArtifact,
    pub removed: PreviewArtifact,
}

impl PreviewWorkerRequest {
    fn validate(&self) -> Result<(), String> {
        if self.schema != PREVIEW_WORKER_SCHEMA || self.schema_version != PREVIEW_SCHEMA_VERSION {
            return Err("unsupported desktop preview worker request schema".into());
        }
        if self.nonce.len() != 64 || !self.nonce.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err("desktop preview worker nonce is invalid".into());
        }
        if self.parent_process_id == 0 {
            return Err("desktop preview worker parent process is invalid".into());
        }
        validate_preview_request(&self.preview)
    }
}

pub(crate) fn validate_preview_request(request: &PreviewRequest) -> Result<(), String> {
    validate_process_options(&request.options)?;
    if request.input.trim().is_empty() {
        return Err("プレビューする入力ファイルを選択してください".into());
    }
    if request.output.trim().is_empty() {
        return Err("最終出力形式を決める保存先を選択してください".into());
    }
    if !request.start_seconds.is_finite() || request.start_seconds < 0.0 {
        return Err("プレビュー開始位置は0以上の有限秒で指定してください".into());
    }
    if !request.duration_seconds.is_finite()
        || !(MIN_PREVIEW_SECONDS..=MAX_PREVIEW_SECONDS).contains(&request.duration_seconds)
    {
        return Err(format!(
            "プレビュー区間は{MIN_PREVIEW_SECONDS:.1}〜{MAX_PREVIEW_SECONDS:.0}秒にしてください"
        ));
    }
    if !(32..=512).contains(&request.points) {
        return Err("プレビュー波形は32〜512点にしてください".into());
    }
    OutputFormat::from_path(Path::new(&request.output))?;
    Ok(())
}

fn effective_decode_limits(options: &ProcessOptions) -> DecodeLimits {
    let configured = options.max_process_memory_mb.map(|value| {
        u64::try_from(value)
            .unwrap_or(u64::MAX)
            .saturating_mul(1024 * 1024)
    });
    let maximum = Some(
        configured
            .unwrap_or(MAX_PREVIEW_MEMORY_BYTES)
            .min(MAX_PREVIEW_MEMORY_BYTES),
    );
    DecodeLimits::new(
        denoize::metadata_limits_for_available_memory(maximum),
        maximum,
    )
}

fn presentation_frames(request: &PreviewRequest, sample_rate: u32) -> Result<(u64, u64), String> {
    let start = request.start_seconds * f64::from(sample_rate);
    let duration = request.duration_seconds * f64::from(sample_rate);
    if start > u64::MAX as f64 || duration > u64::MAX as f64 {
        return Err("プレビュー区間が大きすぎます".into());
    }
    Ok((start.round() as u64, (duration.round() as u64).max(1)))
}

fn checked_region_bytes(channels: usize, frames: u64) -> Result<u64, String> {
    u64::try_from(channels)
        .ok()
        .and_then(|channels| channels.checked_mul(frames))
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<f64>() as u64))
        .ok_or_else(|| "プレビューPCMサイズが大きすぎます".to_string())
}

fn empty_region(channels: usize, frames: usize) -> Result<Vec<Vec<f64>>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(channels)
        .map_err(|error| format!("プレビューチャンネルを確保できません: {error}"))?;
    for _ in 0..channels {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(frames)
            .map_err(|error| format!("プレビューPCMを確保できません: {error}"))?;
        output.push(channel);
    }
    Ok(output)
}

fn stream_format(format: AudioFormat) -> bool {
    matches!(
        format,
        AudioFormat::Wav
            | AudioFormat::Flac
            | AudioFormat::OggOpus
            | AudioFormat::OggVorbis
            | AudioFormat::Mp3
            | AudioFormat::M4a
            | AudioFormat::AacAdts
    )
}

fn decode_stream_region(
    session: AudioInputSession,
    source_fingerprint: batch_resume::FileFingerprint,
    request: &PreviewRequest,
    limits: DecodeLimits,
) -> Result<(Audio, PresentationRegion), String> {
    let mut reader = AudioStreamReader::from_session(session, limits)?;
    let info = reader.info();
    let channels = info.channels();
    let (start, requested_duration) = presentation_frames(request, info.sample_rate())?;
    let duration = match info.total_frames {
        Some(total) if start >= total => {
            return Err(format!(
                "プレビュー開始位置が入力の長さ（{:.3}秒）以降です",
                total as f64 / f64::from(info.sample_rate())
            ));
        }
        Some(total) => requested_duration.min(total - start),
        None => requested_duration,
    };
    let end = start
        .checked_add(duration)
        .ok_or_else(|| "プレビュー区間の終端が大きすぎます".to_string())?;
    let region_bytes = checked_region_bytes(channels, duration)?;
    let peak = region_bytes
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(info.decoder_additional_bytes))
        .ok_or_else(|| "プレビューのメモリ予約量が大きすぎます".to_string())?;
    if limits
        .max_working_set_bytes
        .is_some_and(|maximum| peak.max(1024 * 1024) > maximum)
    {
        return Err(format!(
            "プレビューには{} bytes必要ですが、プロセスメモリ上限を超えます",
            peak.max(1024 * 1024)
        ));
    }
    let region_frames = usize::try_from(duration)
        .map_err(|_| "プレビュー区間がメモリに収まりません".to_string())?;
    let mut output = empty_region(channels, region_frames)?;
    let mut decoded = 0_u64;
    while decoded < end {
        let Some(block) = reader.next_block(PREVIEW_BLOCK_FRAMES)? else {
            break;
        };
        if block.len() != channels {
            return Err("プレビューデコーダのチャンネル数が途中で変化しました".into());
        }
        let block_frames = block.first().map(Vec::len).unwrap_or(0);
        if block_frames == 0 || block.iter().any(|channel| channel.len() != block_frames) {
            return Err("プレビューデコーダが不正なPCMブロックを返しました".into());
        }
        let block_frames_u64 = u64::try_from(block_frames)
            .map_err(|_| "プレビューPCMブロックが大きすぎます".to_string())?;
        let block_end = decoded
            .checked_add(block_frames_u64)
            .ok_or_else(|| "プレビューフレーム数が大きすぎます".to_string())?;
        let overlap_start = decoded.max(start);
        let overlap_end = block_end.min(end);
        if overlap_start < overlap_end {
            let first = usize::try_from(overlap_start - decoded)
                .map_err(|_| "プレビュー開始位置がメモリに収まりません".to_string())?;
            let last = usize::try_from(overlap_end - decoded)
                .map_err(|_| "プレビュー終了位置がメモリに収まりません".to_string())?;
            for (destination, source) in output.iter_mut().zip(&block) {
                destination.extend_from_slice(&source[first..last]);
            }
        }
        decoded = block_end;
    }
    let actual_frames = output.first().map(Vec::len).unwrap_or(0);
    if actual_frames == 0 || output.iter().any(|channel| channel.len() != actual_frames) {
        return Err("プレビュー開始位置が入力の終端以降です".into());
    }
    if reader.fingerprint_input()? != source_fingerprint {
        return Err("プレビュー入力がデコード中に変更されました".into());
    }
    let actual_duration =
        u64::try_from(actual_frames).map_err(|_| "プレビュー区間が大きすぎます".to_string())?;
    let locator = PresentationRegion::new(
        source_fingerprint,
        info.sample_rate(),
        start,
        actual_duration,
    )?;
    Ok((
        Audio {
            sample_rate: info.sample_rate(),
            channels: output,
            bits_per_sample: 32,
            sample_format: info.output_spec.sample_format,
            channel_mask: info.channel_mask,
        },
        locator,
    ))
}

fn decode_fallback_region(
    mut session: AudioInputSession,
    source_fingerprint: batch_resume::FileFingerprint,
    request: &PreviewRequest,
    limits: DecodeLimits,
) -> Result<(Audio, PresentationRegion), String> {
    let audio = denoize::read_audio_from_session_with_limits(&mut session, limits)?;
    let (start, requested_duration) = presentation_frames(request, audio.sample_rate)?;
    let total =
        u64::try_from(audio.frames()).map_err(|_| "入力フレーム数が大きすぎます".to_string())?;
    if start >= total {
        return Err(format!(
            "プレビュー開始位置が入力の長さ（{:.3}秒）以降です",
            total as f64 / f64::from(audio.sample_rate)
        ));
    }
    let duration = requested_duration.min(total - start);
    let end = start
        .checked_add(duration)
        .ok_or_else(|| "プレビュー区間の終端が大きすぎます".to_string())?;
    let locator = PresentationRegion::new(source_fingerprint, audio.sample_rate, start, duration)?;
    locator.validate_source(source_fingerprint, audio.sample_rate, total)?;
    let start = usize::try_from(start)
        .map_err(|_| "プレビュー開始位置がメモリに収まりません".to_string())?;
    let end =
        usize::try_from(end).map_err(|_| "プレビュー終了位置がメモリに収まりません".to_string())?;
    let mut region = empty_region(audio.channels(), end - start)?;
    for (destination, source) in region.iter_mut().zip(&audio.channels) {
        destination.extend_from_slice(&source[start..end]);
    }
    if batch_resume::fingerprint_input_session(&mut session)? != source_fingerprint {
        return Err("プレビュー入力がデコード中に変更されました".into());
    }
    Ok((
        Audio {
            sample_rate: audio.sample_rate,
            channels: region,
            bits_per_sample: audio.bits_per_sample,
            sample_format: audio.sample_format,
            channel_mask: audio.channel_mask,
        },
        locator,
    ))
}

fn decode_region(request: &PreviewRequest) -> Result<(Audio, PresentationRegion), String> {
    let input = Path::new(&request.input);
    let mut session = AudioInputSession::open(input)?;
    let limits = effective_decode_limits(&request.options);
    let probe = denoize::probe_file_from_session_with_limits(&mut session, limits)?;
    if probe.audio_tracks != 1 || probe.codec == denoize::AudioCodec::Unknown {
        return Err("プレビュー入力には対応する音声トラックが1つ必要です".into());
    }
    let fingerprint = batch_resume::fingerprint_input_session(&mut session)?;
    let (audio, locator) = if stream_format(probe.format) {
        decode_stream_region(session, fingerprint, request, limits)?
    } else {
        decode_fallback_region(session, fingerprint, request, limits)?
    };
    if batch_resume::fingerprint_file(input)? != fingerprint {
        return Err("プレビュー入力のパスがデコード中に置き換えられました".into());
    }
    Ok((audio, locator))
}

fn fallible_clone_audio(audio: &Audio) -> Result<Audio, String> {
    let mut channels = Vec::new();
    channels
        .try_reserve_exact(audio.channels())
        .map_err(|error| format!("処理済みプレビューチャンネルを確保できません: {error}"))?;
    for source in &audio.channels {
        let mut channel = Vec::new();
        channel
            .try_reserve_exact(source.len())
            .map_err(|error| format!("処理済みプレビューPCMを確保できません: {error}"))?;
        channel.extend_from_slice(source);
        channels.push(channel);
    }
    Ok(Audio {
        sample_rate: audio.sample_rate,
        channels,
        bits_per_sample: audio.bits_per_sample,
        sample_format: audio.sample_format,
        channel_mask: audio.channel_mask,
    })
}

fn resolved_preview_processing(
    options: &ProcessOptions,
    audio: &Audio,
) -> Result<service::ResolvedProcessingOptions, String> {
    let configured = configured_backend(&options.backend)?;
    let choice = configured
        .map(BackendChoice::Explicit)
        .unwrap_or(BackendChoice::Auto);
    let selected = service::select_backend(
        choice,
        audio.frames() as f64 / f64::from(audio.sample_rate.max(1)),
        None,
    );
    let processing = ProcessingOptions {
        // Persisting a preview fixes this effective backend, so a final job
        // cannot reinterpret `auto` from a different duration.
        backend: BackendChoice::Explicit(selected),
        quality: None,
        denoiser: processing_config(options, audio.sample_rate)?,
        backend_options: parsed_backend_options_for(selected, options)?,
        loudness_lufs: options.loudness_lufs,
        true_peak_dbtp: options.true_peak_dbtp,
    };
    service::resolve_processing_options(audio, processing)
}

fn artifact_metrics(audio: &Audio, points: usize) -> (Option<f64>, f64, Vec<f64>) {
    let loudness = denoize::loudness::measure(audio).ok().map(|value| value.0);
    let mut sum = 0.0;
    let mut samples = 0_u64;
    let mut waveform = vec![0.0_f64; points];
    for channel in &audio.channels {
        for (index, sample) in channel.iter().enumerate() {
            let bucket = (index.saturating_mul(points) / audio.frames().max(1)).min(points - 1);
            if let Some(peak) = waveform.get_mut(bucket) {
                *peak = peak.max(sample.abs());
            }
            sum += sample * sample;
            samples = samples.saturating_add(1);
        }
    }
    let peak = waveform.iter().copied().fold(0.0_f64, f64::max).max(1e-12);
    for value in &mut waveform {
        *value /= peak;
    }
    let rms = (sum / samples.max(1) as f64).sqrt();
    (loudness, 20.0 * rms.max(1e-10).log10(), waveform)
}

fn removed_audio(
    original: &Audio,
    processed: &Audio,
    original_loudness: Option<f64>,
    original_rms: f64,
    processed_loudness: Option<f64>,
    processed_rms: f64,
) -> Result<Audio, String> {
    if original.sample_rate != processed.sample_rate
        || original.channels() != processed.channels()
        || original.frames() != processed.frames()
    {
        return Err("除去音モニターの入出力形状が一致しません".into());
    }
    let (original_level, processed_level) = match (original_loudness, processed_loudness) {
        (Some(original), Some(processed)) => (original, processed),
        _ => (original_rms, processed_rms),
    };
    let target = original_level.min(processed_level);
    let original_gain = 10_f64.powf((target - original_level).min(0.0) / 20.0);
    let processed_gain = 10_f64.powf((target - processed_level).min(0.0) / 20.0);
    let mut removed = fallible_clone_audio(original)?;
    for ((destination, source), enhanced) in removed
        .channels
        .iter_mut()
        .zip(&original.channels)
        .zip(&processed.channels)
    {
        for ((destination, source), enhanced) in destination.iter_mut().zip(source).zip(enhanced) {
            *destination =
                denoize::sanitize_sample(source.mul_add(original_gain, -enhanced * processed_gain));
        }
    }
    Ok(removed)
}

fn write_preview_audio(path: &Path, audio: &Audio) -> Result<(), String> {
    let mut stage = AtomicOutput::new_private(path)?;
    write_audio_to_file(
        stage.file_mut(),
        OutputFormat::Wav,
        audio,
        EncodeOptions::default(),
    )?;
    stage.commit(CommitMode::NoClobber)
}

fn render_preview(worker: &PreviewWorkerRequest) -> Result<PreviewResult, String> {
    worker.validate()?;
    let output_directory = Path::new(&worker.output_directory);
    validate_worker_directory(output_directory)?;
    let (original, locator) = decode_region(&worker.preview)?;
    let pcm_bytes = estimate_audio_memory_bytes(&original);
    let memory_bytes = pcm_bytes
        .checked_mul(4)
        .map(|value| value.max(estimate_audio_working_set_bytes(&original)))
        .ok_or_else(|| "プレビューのメモリ予約量が大きすぎます".to_string())?;
    if memory_bytes > MAX_PREVIEW_MEMORY_BYTES {
        return Err(format!(
            "プレビューには{memory_bytes} bytes必要で、固定上限を超えます"
        ));
    }
    let temporary_bytes = pcm_bytes
        .checked_mul(3)
        .and_then(|value| value.checked_add(24 * 1024))
        .ok_or_else(|| "プレビューの一時領域予約量が大きすぎます".to_string())?;
    if temporary_bytes > MAX_PREVIEW_TEMPORARY_BYTES {
        return Err(format!(
            "プレビューには{temporary_bytes} bytesの一時領域が必要で、固定上限を超えます"
        ));
    }
    let processing = resolved_preview_processing(&worker.preview.options, &original)?;
    let mut resource_request = ResourceRequest::worker(memory_bytes, temporary_bytes);
    if processing.accelerator.effective() != denoize::AcceleratorRuntime::Cpu {
        resource_request = resource_request
            .with_gpu_jobs(1)
            .with_gpu_memory_bytes(denoize::estimate_gpu_worker_bytes(&original)?);
    }
    resource_request = resource_request.checked_add(denoize::estimate_backend_session_request(
        processing.backend,
        &processing.backend_options,
        processing.accelerator,
    )?)?;
    if resource_request.memory_bytes() > MAX_PREVIEW_MEMORY_BYTES {
        return Err(format!(
            "プレビューのworker全体には{} bytes必要で、固定memory上限を超えます",
            resource_request.memory_bytes()
        ));
    }
    if resource_request.temporary_bytes() > MAX_PREVIEW_TEMPORARY_BYTES {
        return Err(format!(
            "プレビューのworker全体には{} bytesの一時領域が必要で、固定上限を超えます",
            resource_request.temporary_bytes()
        ));
    }
    let governor = desktop_resource_governor(&worker.preview.options, 1)?;
    let _permit = governor.acquire(resource_request)?;
    let session = BackendSession::prepare_with_accelerator(
        processing.backend,
        processing.backend_options.clone(),
        processing.accelerator,
    )?;
    let model = batch_resume::consumed_model_config(&processing)?
        .map(batch_resume::fingerprint_consumed_model)
        .transpose()?;
    let mut processed = fallible_clone_audio(&original)?;
    service::process_audio_resolved_with_session(&mut processed, &processing, &session)?;
    if let Some(model) = &model {
        if batch_resume::fingerprint_file(&model.path)? != model.fingerprint {
            return Err("プレビュー処理中にモデルが変更されました".into());
        }
    }
    let final_format = OutputFormat::from_path(Path::new(&worker.preview.output))?;
    let encode = parsed_encode_options(&worker.preview.options)?;
    let metadata = if worker.preview.options.preserve_metadata {
        MetadataPolicy::Preserve
    } else {
        MetadataPolicy::Drop
    };
    let recipe = batch_resume::recipe_digest(
        &processing,
        original.channels(),
        final_format,
        encode,
        metadata,
        model
            .as_ref()
            .map(|model| (&model.fingerprint, model.sample_rate)),
    )?;
    let original_path = output_directory.join("original.wav");
    let processed_path = output_directory.join("processed.wav");
    let removed_path = output_directory.join("removed.wav");
    let (original_loudness, original_rms, original_waveform) =
        artifact_metrics(&original, worker.preview.points);
    let (processed_loudness, processed_rms, processed_waveform) =
        artifact_metrics(&processed, worker.preview.points);
    let removed = removed_audio(
        &original,
        &processed,
        original_loudness,
        original_rms,
        processed_loudness,
        processed_rms,
    )?;
    let (removed_loudness, removed_rms, removed_waveform) =
        artifact_metrics(&removed, worker.preview.points);
    write_preview_audio(&original_path, &original)?;
    write_preview_audio(&processed_path, &processed)?;
    write_preview_audio(&removed_path, &removed)?;
    let duration_seconds = original.frames() as f64 / f64::from(original.sample_rate.max(1));
    let mut effective_options = worker.preview.options.clone();
    effective_options.backend = service::backend_name(processing.backend).into();
    Ok(PreviewResult {
        schema: PREVIEW_RESULT_SCHEMA.into(),
        schema_version: PREVIEW_SCHEMA_VERSION,
        preview_id: worker.nonce.clone(),
        locator,
        recipe: recipe.as_hex(),
        output_format: desktop_output_format_name(final_format).into(),
        backend: service::backend_name(processing.backend).into(),
        accelerator: processing.accelerator.effective().name().into(),
        options: effective_options,
        original: PreviewArtifact {
            source: "original".into(),
            playable_path: original_path.to_string_lossy().into_owned(),
            duration_seconds,
            loudness_lufs: original_loudness,
            rms_db: original_rms,
            waveform: original_waveform,
        },
        processed: PreviewArtifact {
            source: "processed".into(),
            playable_path: processed_path.to_string_lossy().into_owned(),
            duration_seconds,
            loudness_lufs: processed_loudness,
            rms_db: processed_rms,
            waveform: processed_waveform,
        },
        removed: PreviewArtifact {
            source: "removed".into(),
            playable_path: removed_path.to_string_lossy().into_owned(),
            duration_seconds,
            loudness_lufs: removed_loudness,
            rms_db: removed_rms,
            waveform: removed_waveform,
        },
    })
}

fn validate_worker_directory(path: &Path) -> Result<(), String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("プレビューフォルダを確認できません: {error}"))?;
    if !metadata.file_type().is_dir() {
        return Err("プレビュー出力先は実ディレクトリでなければなりません".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err(
                "プレビューフォルダは現在のユーザーだけがアクセス可能でなければなりません".into(),
            );
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err("プレビューフォルダにWindows reparse pointは使用できません".into());
        }
    }
    Ok(())
}

fn preview_root() -> Result<PathBuf, String> {
    #[cfg(unix)]
    let name = format!("denoize-previews-{}", unsafe { libc::geteuid() });
    #[cfg(not(unix))]
    let name = "denoize-previews".to_string();
    let root = std::env::temp_dir().join(name);
    match std::fs::symlink_metadata(&root) {
        Ok(_) => validate_worker_directory(&root)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            #[cfg(unix)]
            let created = {
                use std::os::unix::fs::DirBuilderExt as _;
                let mut builder = std::fs::DirBuilder::new();
                builder.mode(0o700).create(&root)
            };
            #[cfg(not(unix))]
            let created = std::fs::create_dir(&root);
            if let Err(error) = created {
                if error.kind() != std::io::ErrorKind::AlreadyExists {
                    return Err(format!("プレビュー領域を作成できません: {error}"));
                }
            }
        }
        Err(error) => return Err(format!("プレビュー領域を確認できません: {error}")),
    }
    validate_worker_directory(&root)?;
    Ok(root)
}

fn nonce(request: &PreviewRequest, job_id: u64) -> Result<String, String> {
    let encoded = serde_json::to_vec(request)
        .map_err(|error| format!("プレビュー要求を符号化できません: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(b"denoize-desktop-preview-v1\0");
    hasher.update(std::process::id().to_le_bytes());
    hasher.update(job_id.to_le_bytes());
    hasher.update(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .to_le_bytes(),
    );
    hasher.update(encoded);
    Ok(format!("{:x}", hasher.finalize()))
}

fn write_private_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("プレビューJSONを符号化できません: {error}"))?;
    if bytes.len() as u64 > MAX_WORKER_DOCUMENT_BYTES {
        return Err("プレビューJSONが固定上限を超えました".into());
    }
    let mut stage = AtomicOutput::new_private(path)?;
    stage
        .file_mut()
        .write_all(&bytes)
        .map_err(|error| format!("プレビューJSONを書き込めません: {error}"))?;
    stage.commit(CommitMode::NoClobber)
}

fn create_private_marker(path: &Path) -> Result<(), String> {
    AtomicOutput::new_private(path)?.commit(CommitMode::NoClobber)
}

fn validate_private_worker_file(path: &Path, maximum: u64, name: &str) -> Result<(), String> {
    let metadata =
        std::fs::symlink_metadata(path).map_err(|error| format!("inspect {name}: {error}"))?;
    if !metadata.file_type().is_file() || metadata.len() > maximum {
        return Err(format!("{name} must be a bounded regular file"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.nlink() != 1
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err(format!("{name} has unsafe owner, link count, or mode"));
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(format!("{name} must not be a Windows reparse point"));
        }
    }
    Ok(())
}

fn read_preview_owner(directory: &Path) -> Result<PreviewOwner, String> {
    let path = directory.join(PREVIEW_OWNER_FILE);
    let metadata = std::fs::symlink_metadata(&path)
        .map_err(|error| format!("プレビュー所有markerを確認できません: {error}"))?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_WORKER_DOCUMENT_BYTES {
        return Err("プレビュー所有markerはbounded regular fileでなければなりません".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        if metadata.uid() != unsafe { libc::geteuid() }
            || metadata.nlink() != 1
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err("プレビュー所有markerのowner、link数、またはmodeが不正です".into());
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err("プレビュー所有markerにWindows reparse pointは使用できません".into());
        }
    }
    let bytes = std::fs::read(&path)
        .map_err(|error| format!("プレビュー所有markerを読み込めません: {error}"))?;
    let owner: PreviewOwner = serde_json::from_slice(&bytes)
        .map_err(|error| format!("プレビュー所有markerが不正です: {error}"))?;
    owner.validate()?;
    Ok(owner)
}

fn spawn_worker(request_path: &Path, directory: &Path) -> Result<IsolatedChild, String> {
    let executable = std::env::current_exe()
        .map_err(|error| format!("プレビューワーカー実行ファイルを確認できません: {error}"))?;
    let mut command = Command::new(executable);
    command
        .arg(PREVIEW_WORKER_ARGUMENT)
        .arg(request_path)
        .current_dir(directory)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        let expected_parent = std::process::id();
        unsafe {
            command.pre_exec(move || {
                let core = libc::rlimit {
                    rlim_cur: 0,
                    rlim_max: 0,
                };
                if libc::setrlimit(libc::RLIMIT_CORE, &core) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                #[cfg(target_os = "linux")]
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                #[cfg(target_os = "linux")]
                if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                #[cfg(target_os = "linux")]
                if libc::getppid() as u32 != expected_parent {
                    return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
                }
                Ok(())
            });
        }
    }
    let child = command
        .spawn()
        .map_err(|error| format!("隔離プレビューワーカーを開始できません: {error}"))?;
    IsolatedChild::new(child, None)
}

fn completed_worker_status(status: ExitStatus) -> Result<(), String> {
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "隔離プレビューワーカーが異常終了しました（{}）",
            status
                .code()
                .map_or_else(|| "signal".into(), |code| code.to_string())
        ))
    }
}

pub(crate) fn render_isolated(
    request: PreviewRequest,
    job_id: u64,
    control: &JobControl,
) -> Result<PreviewResult, String> {
    validate_preview_request(&request)?;
    let root = preview_root()?;
    let directory = tempfile::Builder::new()
        .prefix("preview-")
        .tempdir_in(&root)
        .map_err(|error| format!("プレビューフォルダを作成できません: {error}"))?;
    validate_worker_directory(directory.path())?;
    let nonce = nonce(&request, job_id)?;
    let start_gate = directory.path().join(PREVIEW_START_GATE_FILE);
    create_private_marker(&start_gate)?;
    write_private_json(
        &directory.path().join(PREVIEW_OWNER_FILE),
        &PreviewOwner::new(nonce.clone()),
    )?;
    let worker = PreviewWorkerRequest {
        schema: PREVIEW_WORKER_SCHEMA.into(),
        schema_version: PREVIEW_SCHEMA_VERSION,
        nonce: nonce.clone(),
        parent_process_id: std::process::id(),
        output_directory: directory.path().to_string_lossy().into_owned(),
        start_gate: start_gate.to_string_lossy().into_owned(),
        preview: request,
    };
    let request_path = directory.path().join("request.json");
    let response_path = directory.path().join("response.json");
    write_private_json(&request_path, &worker)?;
    if control.is_cancelled() {
        return Err("cancelled".into());
    }
    let child = spawn_worker(&request_path, directory.path())?;
    control.install_child(child)?;
    std::fs::remove_file(&start_gate)
        .map_err(|error| format!("プレビューワーカー開始gateを解除できません: {error}"))?;
    let status =
        control.wait_for_child(std::time::Duration::from_secs(MAX_PREVIEW_WORKER_SECONDS))?;
    if control.is_cancelled() {
        return Err("cancelled".into());
    }
    completed_worker_status(status)?;
    validate_private_worker_file(
        &response_path,
        MAX_WORKER_DOCUMENT_BYTES,
        "preview worker response",
    )?;
    let bytes = std::fs::read(&response_path)
        .map_err(|error| format!("隔離プレビュー結果を読み込めません: {error}"))?;
    let envelope: PreviewWorkerEnvelope = serde_json::from_slice(&bytes)
        .map_err(|error| format!("隔離プレビュー結果が不正です: {error}"))?;
    if envelope.schema != PREVIEW_WORKER_SCHEMA
        || envelope.schema_version != PREVIEW_SCHEMA_VERSION
        || envelope.nonce != nonce
    {
        return Err("隔離プレビュー結果のidentityが一致しません".into());
    }
    let result = match (envelope.result, envelope.error) {
        (Some(result), None) => result,
        (None, Some(error)) => return Err(error),
        _ => return Err("隔離プレビューワーカーの結果とerrorが排他的ではありません".into()),
    };
    let expected_original = directory.path().join("original.wav");
    let expected_processed = directory.path().join("processed.wav");
    let expected_removed = directory.path().join("removed.wav");
    if Path::new(&result.original.playable_path) != expected_original
        || Path::new(&result.processed.playable_path) != expected_processed
        || Path::new(&result.removed.playable_path) != expected_removed
        || result.preview_id != nonce
        || result.schema != PREVIEW_RESULT_SCHEMA
        || result.schema_version != PREVIEW_SCHEMA_VERSION
        || result.original.source != "original"
        || result.processed.source != "processed"
        || result.removed.source != "removed"
        || result.recipe.len() != 64
        || !result
            .recipe
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("隔離プレビュー結果の出力pathが一致しません".into());
    }
    for (path, artifact) in [
        (&expected_original, &result.original),
        (&expected_processed, &result.processed),
        (&expected_removed, &result.removed),
    ] {
        validate_private_worker_file(path, MAX_PREVIEW_MEMORY_BYTES, "preview audio")?;
        if !artifact.duration_seconds.is_finite()
            || artifact.duration_seconds <= 0.0
            || artifact.duration_seconds > MAX_PREVIEW_SECONDS
            || !artifact.rms_db.is_finite()
            || artifact
                .loudness_lufs
                .is_some_and(|value| !value.is_finite())
            || artifact.waveform.len() != worker.preview.points
            || artifact
                .waveform
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err("隔離プレビュー結果の音声指標が不正です".into());
        }
    }
    let marker_path = directory.path().join(format!("{nonce}.id"));
    let marker = AtomicOutput::new_private(&marker_path)?;
    marker.commit(CommitMode::NoClobber)?;
    std::fs::remove_file(&request_path)
        .map_err(|error| format!("プレビュー要求を消去できません: {error}"))?;
    std::fs::remove_file(&response_path)
        .map_err(|error| format!("プレビュー応答を消去できません: {error}"))?;
    let _kept = directory.keep();
    Ok(result)
}

pub(crate) fn release_preview(preview_id: &str) -> Result<(), String> {
    if preview_id.len() != 64 || !preview_id.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("プレビューIDが不正です".into());
    }
    release_preview_in(&preview_root()?, preview_id)
}

fn release_preview_in(root: &Path, preview_id: &str) -> Result<(), String> {
    for entry in std::fs::read_dir(&root)
        .map_err(|error| format!("プレビュー領域を読み取れません: {error}"))?
    {
        let entry = entry.map_err(|error| format!("プレビュー項目を読み取れません: {error}"))?;
        if !entry
            .file_type()
            .map_err(|error| format!("プレビュー項目を確認できません: {error}"))?
            .is_dir()
        {
            continue;
        }
        if validate_worker_directory(&entry.path()).is_err() {
            continue;
        }
        let Ok(owner) = read_preview_owner(&entry.path()) else {
            continue;
        };
        let marker = entry.path().join(format!("{preview_id}.id"));
        let exact_marker =
            std::fs::symlink_metadata(&marker).is_ok_and(|metadata| metadata.file_type().is_file());
        if owner.preview_id == preview_id && exact_marker {
            std::fs::remove_dir_all(entry.path())
                .map_err(|error| format!("プレビューを消去できません: {error}"))?;
            return Ok(());
        }
    }
    Ok(())
}

pub(crate) fn cleanup_preview_root() -> Result<u64, String> {
    cleanup_preview_root_at(&preview_root()?)
}

fn cleanup_preview_root_at(root: &Path) -> Result<u64, String> {
    let mut removed = 0_u64;
    for entry in std::fs::read_dir(&root)
        .map_err(|error| format!("プレビュー領域を読み取れません: {error}"))?
    {
        let entry = entry.map_err(|error| format!("プレビュー項目を読み取れません: {error}"))?;
        if !entry
            .file_type()
            .map_err(|error| format!("プレビュー項目を確認できません: {error}"))?
            .is_dir()
        {
            continue;
        }
        if validate_worker_directory(&entry.path()).is_err() {
            continue;
        }
        let Ok(owner) = read_preview_owner(&entry.path()) else {
            continue;
        };
        if super::recovery::process_is_alive(owner.process_id) {
            continue;
        }
        std::fs::remove_dir_all(entry.path())
            .map_err(|error| format!("古いプレビューを消去できません: {error}"))?;
        removed = removed.saturating_add(1);
    }
    Ok(removed)
}

fn worker_request(path: &Path) -> Result<PreviewWorkerRequest, String> {
    validate_private_worker_file(path, MAX_WORKER_DOCUMENT_BYTES, "preview worker request")?;
    let bytes =
        std::fs::read(path).map_err(|error| format!("read preview worker request: {error}"))?;
    let request: PreviewWorkerRequest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse preview worker request: {error}"))?;
    request.validate()?;
    let request_parent = path
        .parent()
        .ok_or_else(|| "preview worker request has no parent directory".to_string())?;
    let requested_output = Path::new(&request.output_directory);
    if std::fs::canonicalize(request_parent).ok() != std::fs::canonicalize(requested_output).ok() {
        return Err("preview worker output directory does not match the request directory".into());
    }
    let expected_start_gate = request_parent.join(PREVIEW_START_GATE_FILE);
    if Path::new(&request.start_gate) != expected_start_gate {
        return Err("preview worker start gate does not match the request directory".into());
    }
    Ok(request)
}

fn wait_for_start_gate(path: &Path) -> Result<(), String> {
    let started = std::time::Instant::now();
    loop {
        match std::fs::symlink_metadata(path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Ok(metadata) if metadata.file_type().is_file() => {}
            Ok(_) => return Err("preview worker start gate is not a regular file".into()),
            Err(error) => return Err(format!("inspect preview worker start gate: {error}")),
        }
        if started.elapsed() >= std::time::Duration::from_secs(5) {
            return Err("preview worker start gate timed out".into());
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[cfg(unix)]
pub(crate) fn install_worker_parent_watchdog(expected_parent: u32) -> Result<(), String> {
    if unsafe { libc::getppid() as u32 } != expected_parent {
        return Err("preview worker parent process does not match the request".into());
    }
    #[cfg(target_os = "linux")]
    {
        if unsafe { libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) } != 0 {
            return Err(format!(
                "install preview worker parent-death signal: {}",
                std::io::Error::last_os_error()
            ));
        }
        if unsafe { libc::getppid() as u32 } != expected_parent {
            return Err("preview worker parent exited while installing watchdog".into());
        }
    }
    #[cfg(not(target_os = "linux"))]
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(250));
        if unsafe { libc::getppid() as u32 } != expected_parent {
            unsafe { libc::_exit(4) };
        }
    });
    Ok(())
}

#[cfg(not(unix))]
pub(crate) fn install_worker_parent_watchdog(_expected_parent: u32) -> Result<(), String> {
    Ok(())
}

pub fn run_preview_worker(request_path: &Path) -> i32 {
    let request = match worker_request(request_path) {
        Ok(request) => request,
        Err(error) => {
            eprintln!("denoize preview worker: {error}");
            return 2;
        }
    };
    if let Err(error) = wait_for_start_gate(Path::new(&request.start_gate))
        .and_then(|()| install_worker_parent_watchdog(request.parent_process_id))
    {
        eprintln!("denoize preview worker: {error}");
        return 2;
    }
    let result = render_preview(&request);
    let envelope = match result {
        Ok(result) => PreviewWorkerEnvelope {
            schema: PREVIEW_WORKER_SCHEMA.into(),
            schema_version: PREVIEW_SCHEMA_VERSION,
            nonce: request.nonce.clone(),
            result: Some(result),
            error: None,
        },
        Err(error) => PreviewWorkerEnvelope {
            schema: PREVIEW_WORKER_SCHEMA.into(),
            schema_version: PREVIEW_SCHEMA_VERSION,
            nonce: request.nonce.clone(),
            result: None,
            error: Some(error),
        },
    };
    let response = Path::new(&request.output_directory).join("response.json");
    match write_private_json(&response, &envelope) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("denoize preview worker: {error}");
            3
        }
    }
}

fn preview_worker_request_from_arguments(
    mut arguments: impl Iterator<Item = std::ffi::OsString>,
) -> Result<Option<PathBuf>, String> {
    if arguments.next().as_deref() != Some(std::ffi::OsStr::new(PREVIEW_WORKER_ARGUMENT)) {
        return Ok(None);
    }
    let request = arguments
        .next()
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| "preview worker request path is missing".to_string())?;
    if arguments.next().is_some() {
        return Err("preview worker accepts exactly one request path".into());
    }
    Ok(Some(request))
}

pub fn preview_worker_request_from_args() -> Result<Option<PathBuf>, String> {
    preview_worker_request_from_arguments(std::env::args_os().skip(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> ProcessOptions {
        ProcessOptions {
            backend: "classical".into(),
            preset: Some("hifi".into()),
            mode: Some("music".into()),
            strength: 0.4,
            adaptive_noise: false,
            vad: false,
            channel_mode: "linked".into(),
            downmix: "preserve".into(),
            loudness_lufs: None,
            true_peak_dbtp: -1.0,
            preserve_metadata: false,
            force: false,
            mp3_bitrate_kbps: 192,
            aac_bitrate_kbps: 192,
            aac_encoder: "oxide".into(),
            onnx_model: None,
            onnx_sample_rate: 16_000,
            sgmse_profile: "balanced".into(),
            accelerator: "cpu".into(),
            deterministic: false,
            seed: None,
            max_process_memory_mb: None,
            max_temporary_mb: None,
            max_gpu_memory_mb: None,
            max_gpu_jobs: 1,
        }
    }

    fn write_test_wav(path: &Path) {
        let sample_rate = 16_000_u32;
        let mut samples = Vec::with_capacity(sample_rate as usize * 2);
        for index in 0..sample_rate {
            let sample = if index % 80 < 40 {
                1_000_i16
            } else {
                -1_000_i16
            };
            samples.extend(sample.to_le_bytes());
        }
        let mut wav = Vec::with_capacity(44 + samples.len());
        wav.extend(b"RIFF");
        wav.extend((36_u32 + samples.len() as u32).to_le_bytes());
        wav.extend(b"WAVEfmt ");
        wav.extend(16_u32.to_le_bytes());
        wav.extend(1_u16.to_le_bytes());
        wav.extend(1_u16.to_le_bytes());
        wav.extend(sample_rate.to_le_bytes());
        wav.extend((sample_rate * 2).to_le_bytes());
        wav.extend(2_u16.to_le_bytes());
        wav.extend(16_u16.to_le_bytes());
        wav.extend(b"data");
        wav.extend((samples.len() as u32).to_le_bytes());
        wav.extend(samples);
        std::fs::write(path, wav).unwrap();
    }

    fn private_preview_directory(
        root: &Path,
        name: &str,
        preview_id: &str,
        process_id: u32,
    ) -> PathBuf {
        let path = root.join(name);
        std::fs::create_dir(&path).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        let mut owner = PreviewOwner::new(preview_id.into());
        owner.process_id = process_id;
        write_private_json(&path.join(PREVIEW_OWNER_FILE), &owner).unwrap();
        path
    }

    fn write_private_marker(path: &Path) {
        let marker = AtomicOutput::new_private(path).unwrap();
        marker.commit(CommitMode::NoClobber).unwrap();
    }

    #[test]
    fn preview_request_rejects_unbounded_and_non_finite_intervals_before_io() {
        let options = options();
        let request = |start_seconds, duration_seconds| PreviewRequest {
            input: "/missing/input.wav".into(),
            output: "/missing/output.wav".into(),
            start_seconds,
            duration_seconds,
            points: 180,
            options: options.clone(),
        };
        assert!(validate_preview_request(&request(-1.0, 1.0)).is_err());
        assert!(validate_preview_request(&request(0.0, f64::NAN)).is_err());
        assert!(validate_preview_request(&request(0.0, MAX_PREVIEW_SECONDS + 0.1)).is_err());
    }

    #[test]
    fn preview_worker_arguments_fail_closed() {
        let parse = |values: &[&str]| {
            preview_worker_request_from_arguments(
                values.iter().map(|value| std::ffi::OsString::from(*value)),
            )
        };
        assert_eq!(parse(&["--ordinary"]).unwrap(), None);
        assert!(parse(&[PREVIEW_WORKER_ARGUMENT]).is_err());
        assert!(parse(&[PREVIEW_WORKER_ARGUMENT, "request.json", "extra"]).is_err());
        assert_eq!(
            parse(&[PREVIEW_WORKER_ARGUMENT, "request.json"]).unwrap(),
            Some(PathBuf::from("request.json"))
        );
    }

    #[test]
    fn private_preview_documents_are_bounded_before_staging() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("oversized.json");
        let oversized = "x".repeat(MAX_WORKER_DOCUMENT_BYTES as usize + 1);
        assert!(write_private_json(&path, &oversized).is_err());
        assert!(!path.exists());
    }

    #[test]
    fn startup_cleanup_preserves_live_and_unidentified_preview_directories() {
        let root = tempfile::tempdir().unwrap();
        let live_id = "a".repeat(64);
        let stale_id = "b".repeat(64);
        let live =
            private_preview_directory(root.path(), "preview-live", &live_id, std::process::id());
        let stale = private_preview_directory(root.path(), "preview-stale", &stale_id, u32::MAX);
        let unidentified = root.path().join("preview-unidentified");
        std::fs::create_dir(&unidentified).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&unidentified, std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }

        assert_eq!(cleanup_preview_root_at(root.path()).unwrap(), 1);
        assert!(live.is_dir());
        assert!(!stale.exists());
        assert!(unidentified.is_dir());
    }

    #[test]
    fn release_requires_matching_owner_and_exact_regular_marker() {
        let root = tempfile::tempdir().unwrap();
        let first_id = "c".repeat(64);
        let second_id = "d".repeat(64);
        let first =
            private_preview_directory(root.path(), "preview-first", &first_id, std::process::id());
        let second = private_preview_directory(
            root.path(),
            "preview-second",
            &second_id,
            std::process::id(),
        );
        write_private_marker(&first.join(format!("{first_id}.id")));
        write_private_marker(&second.join(format!("{first_id}.id")));
        write_private_marker(&second.join(format!("{second_id}.id")));

        release_preview_in(root.path(), &first_id).unwrap();

        assert!(!first.exists());
        assert!(second.is_dir());
        release_preview_in(root.path(), &second_id).unwrap();
        assert!(!second.exists());
    }

    #[test]
    fn preview_render_is_source_bound_bounded_and_non_destructive() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("input.wav");
        let final_output = directory.path().join("final.wav");
        let preview_output = directory.path().join("private-preview");
        std::fs::create_dir(&preview_output).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&preview_output, std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        write_test_wav(&input);
        let request = PreviewWorkerRequest {
            schema: PREVIEW_WORKER_SCHEMA.into(),
            schema_version: PREVIEW_SCHEMA_VERSION,
            nonce: "a".repeat(64),
            parent_process_id: std::process::id(),
            output_directory: preview_output.to_string_lossy().into_owned(),
            start_gate: preview_output
                .join(PREVIEW_START_GATE_FILE)
                .to_string_lossy()
                .into_owned(),
            preview: PreviewRequest {
                input: input.to_string_lossy().into_owned(),
                output: final_output.to_string_lossy().into_owned(),
                start_seconds: 0.1,
                duration_seconds: MIN_PREVIEW_SECONDS,
                points: 64,
                options: options(),
            },
        };

        let result = render_preview(&request).unwrap();

        assert_eq!(result.schema, PREVIEW_RESULT_SCHEMA);
        assert_eq!(
            result.locator.source,
            batch_resume::fingerprint_file(&input).unwrap()
        );
        assert_eq!(result.original.waveform.len(), 64);
        assert_eq!(result.processed.waveform.len(), 64);
        assert_eq!(result.removed.waveform.len(), 64);
        assert!(Path::new(&result.original.playable_path).is_file());
        assert!(Path::new(&result.processed.playable_path).is_file());
        assert!(Path::new(&result.removed.playable_path).is_file());
        assert!(!final_output.exists());
        let staged = std::fs::read_dir(directory.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with(".denoize-") && name.ends_with(".part"))
            })
            .count();
        assert_eq!(staged, 0);
    }

    #[test]
    fn preview_clamps_a_requested_region_to_a_short_input() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("input.wav");
        let final_output = directory.path().join("final.wav");
        write_test_wav(&input);
        let request = PreviewRequest {
            input: input.to_string_lossy().into_owned(),
            output: final_output.to_string_lossy().into_owned(),
            start_seconds: 0.75,
            duration_seconds: 8.0,
            points: 64,
            options: options(),
        };

        let (audio, locator) = decode_region(&request).unwrap();

        assert_eq!(audio.frames(), 4_000);
        assert_eq!(locator.start_tick, 12_000);
        assert_eq!(locator.duration_ticks, 4_000);
        assert_eq!(locator.timescale, 16_000);
        assert!(!final_output.exists());
        let (loudness, rms, _) = artifact_metrics(&audio, 32);
        let removed = removed_audio(&audio, &audio, loudness, rms, loudness, rms).unwrap();
        assert!(removed
            .channels
            .iter()
            .flatten()
            .all(|sample| sample.abs() <= f64::EPSILON));

        let mut after_end = request;
        after_end.start_seconds = 1.0;
        assert!(decode_region(&after_end).unwrap_err().contains("開始位置"));
    }
}
