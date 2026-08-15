use denoize::audio::{
    estimate_audio_working_set_bytes, estimate_stream_memory_bytes_checked, read_audio,
    read_audio_from_session_with_limits, write_audio, WavStreamWriter,
};
use denoize::batch_resume::{
    self, BatchSession, Digest, MetadataPolicy, ResumeDecision, ResumeExpectation,
};
use denoize::benchmark::{BenchmarkReport, ComparisonReport};
use denoize::denoiser::{DenoiserConfig, Preset, ProcessingMode};
use denoize::encode::write_audio_to_file;
use denoize::metadata::MetadataLimits;
use denoize::models::{ModelAuthentication, ModelDownloadOptions, ModelProxy};
use denoize::service::{self, BackendChoice, ProcessingOptions};
use denoize::{
    AacEncoder, AcceleratorPreference, AtomicOutput, AudioStreamInfo, AudioStreamReader, Backend,
    BackendOptions, BackendSession, ChannelMode, CommitMode, DecodeLimits, DownmixMode,
    EncodeOptions, OnnxModelConfig, OutputFormat, ResourceGovernor, ResourceLimits, ResourcePermit,
    ResourceRequest, SgmseProfile, StreamingBackendSession,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::{AppHandle, Emitter, Manager, State};

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

const VALIDATION_SAMPLE_RATE_HZ: u32 = 48_000;
const MAX_MODEL_SAMPLE_RATE_HZ: u32 = 768_000;
const MIN_LOUDNESS_LUFS: f64 = -70.0;
const MAX_LOUDNESS_LUFS: f64 = 0.0;
const MIN_TRUE_PEAK_DBTP: f64 = -20.0;
const MAX_TRUE_PEAK_DBTP: f64 = 0.0;
const DEFAULT_MODEL_SAMPLE_RATE_HZ: u32 = 16_000;
const BYTES_PER_MIB: u64 = 1024 * 1024;
const DEFAULT_STREAM_BLOCK_FRAMES: usize = 8_192;
const STREAM_CHECKPOINT_FRAMES: u64 = 1_048_576;

const fn default_stream_block_frames() -> usize {
    DEFAULT_STREAM_BLOCK_FRAMES
}

fn default_accelerator() -> String {
    "cpu".into()
}

const fn default_max_gpu_jobs() -> usize {
    1
}

#[derive(Default)]
struct AppState {
    jobs: Arc<Mutex<HashMap<u64, Arc<JobControl>>>>,
    live: Arc<Mutex<Option<Arc<AtomicBool>>>>,
}

#[derive(Default)]
struct JobControl {
    cancelled: AtomicBool,
    commit_gate: Mutex<()>,
}

impl JobControl {
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    fn cancel(&self) -> Result<(), String> {
        // Signal first so no waiter can enter a later commit fence while this
        // call waits for the publication that already owns the gate.
        self.cancelled.store(true, Ordering::SeqCst);
        let _commit_guard = self
            .commit_gate
            .lock()
            .map_err(|_| "出力確定状態を取得できません")?;
        Ok(())
    }

    fn commit(&self, transaction: AtomicOutput, mode: CommitMode) -> Result<(), String> {
        self.commit_fence(|| transaction.commit(mode))
    }

    fn commit_fence<T>(&self, publish: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
        let _commit_guard = self
            .commit_gate
            .lock()
            .map_err(|_| "出力確定状態を取得できません")?;
        check_cancelled(self)?;
        publish()
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProcessOptions {
    backend: String,
    preset: Option<String>,
    mode: Option<String>,
    strength: f64,
    adaptive_noise: bool,
    vad: bool,
    channel_mode: String,
    downmix: String,
    loudness_lufs: Option<f64>,
    true_peak_dbtp: f64,
    preserve_metadata: bool,
    force: bool,
    mp3_bitrate_kbps: u32,
    aac_bitrate_kbps: u32,
    aac_encoder: String,
    onnx_model: Option<String>,
    onnx_sample_rate: u32,
    sgmse_profile: String,
    #[serde(default = "default_accelerator")]
    accelerator: String,
    #[serde(default)]
    deterministic: bool,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    max_process_memory_mb: Option<usize>,
    #[serde(default)]
    max_temporary_mb: Option<usize>,
    #[serde(default)]
    max_gpu_memory_mb: Option<usize>,
    #[serde(default = "default_max_gpu_jobs")]
    max_gpu_jobs: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GuiConfig {
    backend: String,
    preset: String,
    mode: String,
    strength: f64,
    adaptive_noise: bool,
    vad: bool,
    channels: String,
    downmix: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    loudness_lufs: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    true_peak_dbtp: Option<f64>,
    preserve_metadata: bool,
    force: bool,
    mp3_bitrate_kbps: u32,
    m4a_bitrate_kbps: u32,
    aac_encoder: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    onnx_model: Option<String>,
    onnx_rate: u32,
    sgmse_profile: String,
    #[serde(default = "default_accelerator")]
    accelerator: String,
    deterministic: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_process_memory_mb: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_temporary_mb: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_gpu_memory_mb: Option<usize>,
    #[serde(default = "default_max_gpu_jobs")]
    max_gpu_jobs: usize,
}

/// A typed, partial desktop/CLI configuration import.
///
/// Every field is optional so existing reusable TOML snippets continue to
/// overlay the settings currently shown in the UI. Serde still rejects unknown
/// fields and wrong value types before anything is applied.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GuiConfigPatch {
    backend: Option<String>,
    preset: Option<String>,
    mode: Option<String>,
    strength: Option<f64>,
    adaptive_noise: Option<bool>,
    vad: Option<bool>,
    channels: Option<String>,
    downmix: Option<String>,
    loudness_lufs: Option<f64>,
    true_peak_dbtp: Option<f64>,
    preserve_metadata: Option<bool>,
    force: Option<bool>,
    mp3_bitrate_kbps: Option<u32>,
    m4a_bitrate_kbps: Option<u32>,
    aac_encoder: Option<String>,
    onnx_model: Option<String>,
    onnx_rate: Option<u32>,
    sgmse_profile: Option<String>,
    accelerator: Option<String>,
    deterministic: Option<bool>,
    max_process_memory_mb: Option<usize>,
    max_temporary_mb: Option<usize>,
    max_gpu_memory_mb: Option<usize>,
    max_gpu_jobs: Option<usize>,
}

impl GuiConfig {
    fn process_options(&self) -> ProcessOptions {
        ProcessOptions {
            backend: self.backend.clone(),
            preset: Some(self.preset.clone()),
            mode: Some(self.mode.clone()),
            strength: self.strength,
            adaptive_noise: self.adaptive_noise,
            vad: self.vad,
            channel_mode: self.channels.clone(),
            downmix: self.downmix.clone(),
            loudness_lufs: self.loudness_lufs,
            true_peak_dbtp: self.true_peak_dbtp.unwrap_or(-1.0),
            preserve_metadata: self.preserve_metadata,
            force: self.force,
            mp3_bitrate_kbps: self.mp3_bitrate_kbps,
            aac_bitrate_kbps: self.m4a_bitrate_kbps,
            aac_encoder: self.aac_encoder.clone(),
            onnx_model: self.onnx_model.clone(),
            onnx_sample_rate: self.onnx_rate,
            sgmse_profile: self.sgmse_profile.clone(),
            accelerator: self.accelerator.clone(),
            deterministic: self.deterministic,
            seed: None,
            max_process_memory_mb: self.max_process_memory_mb,
            max_temporary_mb: self.max_temporary_mb,
            max_gpu_memory_mb: self.max_gpu_memory_mb,
            max_gpu_jobs: self.max_gpu_jobs,
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.loudness_lufs.is_none()
            && self
                .true_peak_dbtp
                .is_some_and(|true_peak| true_peak != -1.0)
        {
            return Err("true_peak_dbtp は loudness_lufs と一緒に指定してください".into());
        }
        validate_process_options(&self.process_options())
    }

    fn normalized(mut self) -> Result<Self, String> {
        let backend = configured_backend(&self.backend)?;
        if !backend.is_some_and(service::requires_external_model) {
            self.onnx_model = None;
            self.onnx_rate = DEFAULT_MODEL_SAMPLE_RATE_HZ;
        }
        self.validate()?;
        Ok(self)
    }
}

impl GuiConfigPatch {
    fn merge(self, mut current: GuiConfig) -> Result<GuiConfig, String> {
        macro_rules! replace_present {
            ($field:ident) => {
                if let Some(value) = self.$field {
                    current.$field = value;
                }
            };
        }

        replace_present!(backend);
        replace_present!(preset);
        replace_present!(mode);
        replace_present!(strength);
        replace_present!(adaptive_noise);
        replace_present!(vad);
        replace_present!(channels);
        replace_present!(downmix);
        let explicit_loudness_clear = self.loudness_lufs.is_none()
            && self
                .true_peak_dbtp
                .is_some_and(|true_peak| true_peak == -1.0);
        if explicit_loudness_clear {
            current.loudness_lufs = None;
            current.true_peak_dbtp = None;
        } else if let Some(value) = self.loudness_lufs {
            current.loudness_lufs = Some(value);
            if let Some(true_peak) = self.true_peak_dbtp {
                current.true_peak_dbtp = Some(true_peak);
            }
        } else if let Some(value) = self.true_peak_dbtp {
            current.true_peak_dbtp = Some(value);
        }
        replace_present!(preserve_metadata);
        replace_present!(force);
        replace_present!(mp3_bitrate_kbps);
        replace_present!(m4a_bitrate_kbps);
        replace_present!(aac_encoder);
        if let Some(value) = self.onnx_model {
            current.onnx_model = Some(value);
        }
        replace_present!(onnx_rate);
        replace_present!(sgmse_profile);
        replace_present!(accelerator);
        replace_present!(deterministic);
        if let Some(value) = self.max_process_memory_mb {
            current.max_process_memory_mb = Some(value);
        }
        if let Some(value) = self.max_temporary_mb {
            current.max_temporary_mb = Some(value);
        }
        if let Some(value) = self.max_gpu_memory_mb {
            current.max_gpu_memory_mb = Some(value);
        }
        replace_present!(max_gpu_jobs);
        current.normalized()
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProcessRequest {
    input: String,
    output: String,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    resume: bool,
    #[serde(default = "default_stream_block_frames")]
    stream_frames: usize,
    options: ProcessOptions,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BatchRequest {
    inputs: Vec<String>,
    input_dir: Option<String>,
    output_dir: String,
    output_format: String,
    recursive: bool,
    jobs: usize,
    resume: bool,
    options: ProcessOptions,
}

#[derive(Clone, Debug)]
struct BatchItem {
    input: PathBuf,
    output: PathBuf,
    output_format: OutputFormat,
    item_id: Digest,
}

#[derive(Clone, Debug)]
struct PreparedBatchItem {
    item: BatchItem,
    input_channels: usize,
    encode: EncodeOptions,
    metadata_policy: MetadataPolicy,
    processing: service::ResolvedProcessingOptions,
    backend_session: Arc<BackendSession>,
    _backend_session_permit: Arc<ResourcePermit>,
    governor: ResourceGovernor,
    resource_request: ResourceRequest,
    decode_limits: DecodeLimits,
    metadata_limits: MetadataLimits,
    expectation: ResumeExpectation,
}

#[derive(Clone, Debug)]
struct PlannedBatchItem {
    prepared: PreparedBatchItem,
    decision: ResumeDecision,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(not(feature = "live"), allow(dead_code))]
struct LiveRequest {
    input_device: Option<String>,
    output_device: Option<String>,
    chunk_ms: u32,
    backend: String,
    options: ProcessOptions,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LiveDevices {
    inputs: Vec<String>,
    outputs: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
#[cfg(feature = "live")]
struct LiveEvent {
    status: &'static str,
    message: String,
    sample_rate: u32,
    input_channels: usize,
    output_channels: usize,
    chunk_frames: usize,
    input_level: f32,
    output_level: f32,
    processed_chunks: u64,
    dropped_chunks: u64,
    accelerator: Option<AcceleratorResult>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AcceleratorResult {
    requested: &'static str,
    effective: &'static str,
    fallback: Option<&'static str>,
}

fn accelerator_result(selection: denoize::AcceleratorSelection) -> AcceleratorResult {
    AcceleratorResult {
        requested: selection.requested().name(),
        effective: selection.effective().name(),
        fallback: selection.fallback().map(|fallback| fallback.name()),
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JobProgress {
    job_id: u64,
    kind: &'static str,
    status: &'static str,
    message: String,
    current: usize,
    total: usize,
    fraction: f64,
    elapsed_seconds: f64,
    output: Option<String>,
    error: Option<String>,
    eta_seconds: Option<f64>,
    item: Option<String>,
    item_status: Option<&'static str>,
    item_id: Option<String>,
    resume_reason: Option<&'static str>,
    accelerator: Option<AcceleratorResult>,
}

#[derive(Debug)]
struct ProcessFileResult {
    output: String,
    accelerator: denoize::AcceleratorSelection,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelProgress {
    job_id: u64,
    name: String,
    status: &'static str,
    message: String,
    downloaded: u64,
    total: Option<u64>,
    fraction: Option<f64>,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelActionOptions {
    #[serde(default)]
    offline: bool,
    source_url: Option<String>,
    proxy_url: Option<String>,
    #[serde(default)]
    direct: bool,
    bearer_token: Option<String>,
    basic_username: Option<String>,
    basic_password: Option<String>,
    source_path: Option<String>,
}

fn model_action_options(
    input: Option<ModelActionOptions>,
) -> Result<(ModelDownloadOptions, Option<PathBuf>), String> {
    model_action_options_with_environment(input, |name| std::env::var(name).ok())
}

fn catalog_action_options(
    input: Option<ModelActionOptions>,
) -> Result<ModelDownloadOptions, String> {
    catalog_action_options_with_environment(input, |name| std::env::var(name).ok())
}

fn catalog_action_options_with_environment<F>(
    input: Option<ModelActionOptions>,
    mut read_environment: F,
) -> Result<ModelDownloadOptions, String>
where
    F: FnMut(&str) -> Option<String>,
{
    let (options, source) = model_action_options_with_environment(input, |name| {
        let name = if name == "DENOIZE_MODEL_URL" {
            "DENOIZE_MODEL_CATALOG_URL"
        } else {
            name
        };
        read_environment(name)
    })?;
    if source.is_some() {
        return Err("ローカルカタログはCLIの models catalog import で導入してください".into());
    }
    Ok(options)
}

fn model_action_options_with_environment<F>(
    input: Option<ModelActionOptions>,
    mut read_environment: F,
) -> Result<(ModelDownloadOptions, Option<PathBuf>), String>
where
    F: FnMut(&str) -> Option<String>,
{
    let input = input.unwrap_or_default();
    let source_url = trimmed_value(input.source_url);
    let proxy_url = trimmed_value(input.proxy_url);
    let bearer_token = trimmed_value(input.bearer_token);
    let basic_username = trimmed_value(input.basic_username);
    let basic_password = input.basic_password.filter(|value| !value.is_empty());
    let source_path = input
        .source_path
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);

    if input.direct && proxy_url.is_some() {
        return Err("プロキシURLと直接接続は同時に指定できません".into());
    }
    if bearer_token.is_some() && (basic_username.is_some() || basic_password.is_some()) {
        return Err("Bearer認証とBasic認証は同時に指定できません".into());
    }
    let authentication = if let Some(token) = bearer_token {
        Some(ModelAuthentication::Bearer(token))
    } else {
        match (basic_username, basic_password) {
            (Some(username), Some(password)) => {
                Some(ModelAuthentication::Basic { username, password })
            }
            (None, None) => None,
            _ => return Err("Basic認証のユーザー名とパスワードは両方指定してください".into()),
        }
    };

    if source_path.is_some() {
        if input.offline
            || source_url.is_some()
            || proxy_url.is_some()
            || input.direct
            || authentication.is_some()
        {
            return Err(
                "ローカルファイルはネットワーク・認証オプションと同時に指定できません".into(),
            );
        }
        return Ok((ModelDownloadOptions::default(), source_path));
    }

    let overrides_authentication = authentication.is_some();
    let mut options = ModelDownloadOptions::from_env_with(|name| {
        let overridden = match name {
            "DENOIZE_MODEL_OFFLINE" => input.offline,
            "DENOIZE_MODEL_URL" => source_url.is_some(),
            "DENOIZE_MODEL_PROXY" => input.direct || proxy_url.is_some(),
            "DENOIZE_MODEL_BEARER_TOKEN" | "DENOIZE_MODEL_USERNAME" | "DENOIZE_MODEL_PASSWORD" => {
                overrides_authentication
            }
            _ => false,
        };
        (!overridden).then(|| read_environment(name)).flatten()
    })?;
    if input.offline {
        options.offline = true;
    }
    if source_url.is_some() {
        options.source_url = source_url;
    }
    if input.direct {
        options.proxy = ModelProxy::Disabled;
    } else if let Some(url) = proxy_url {
        options.proxy = ModelProxy::Url(url);
    }
    if authentication.is_some() {
        options.authentication = authentication;
    }
    Ok((options, source_path))
}

fn trimmed_value(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let value = value.trim();
        (!value.is_empty()).then(|| value.to_owned())
    })
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AppInfo {
    version: &'static str,
    backends: Vec<BackendInfo>,
    formats: Vec<&'static str>,
    fdk_available: bool,
    accelerators: Vec<AcceleratorInfo>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BackendInfo {
    name: &'static str,
    external_model: bool,
    managed_model: Option<&'static str>,
    sample_rate: Option<u32>,
    accelerated: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AcceleratorInfo {
    name: &'static str,
    compiled: bool,
    available: bool,
    device: Option<String>,
    memory_bytes: Option<u64>,
    compute_capability: Option<String>,
    detail: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ArtifactMetrics {
    musical_noise_score: f64,
    pumping_score: f64,
    transient_loss_score: f64,
    phase_distortion_score: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ComparisonMetrics {
    si_sdr_db: f64,
    si_snr_db: f64,
    snr_db: f64,
    segmental_snr_db: f64,
    stereo_side_sdr_db: Option<f64>,
    correlation_error: Option<f64>,
    stoi: Option<f64>,
    pesq: Option<f64>,
    visqol: Option<f64>,
    artifact_scores: ArtifactMetrics,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ComparisonMetricSet {
    noisy: ComparisonMetrics,
    enhanced: ComparisonMetrics,
    improvement: ComparisonMetrics,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ComparisonOutput {
    markdown: String,
    json: String,
    html: String,
    noisy_snr_db: f64,
    enhanced_snr_db: f64,
    improvement_db: f64,
    metrics: ComparisonMetricSet,
}

fn comparison_metrics(report: &BenchmarkReport) -> ComparisonMetrics {
    ComparisonMetrics {
        si_sdr_db: report.si_sdr_db,
        si_snr_db: report.si_snr_db,
        snr_db: report.snr_db,
        segmental_snr_db: report.segmental_snr_db,
        stereo_side_sdr_db: report.stereo_side_sdr_db,
        correlation_error: report.correlation_error,
        stoi: report.stoi,
        pesq: report.pesq,
        visqol: report.visqol,
        artifact_scores: ArtifactMetrics {
            musical_noise_score: report.artifact_scores.musical_noise_score,
            pumping_score: report.artifact_scores.pumping_score,
            transient_loss_score: report.artifact_scores.transient_loss_score,
            phase_distortion_score: report.artifact_scores.phase_distortion_score,
        },
    }
}

fn optional_metric_difference(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a - b),
        _ => None,
    }
}

fn comparison_metric_set(report: &ComparisonReport) -> ComparisonMetricSet {
    let noisy = comparison_metrics(&report.noisy);
    let enhanced = comparison_metrics(&report.enhanced);
    let improvement = ComparisonMetrics {
        si_sdr_db: report.enhanced.si_sdr_db - report.noisy.si_sdr_db,
        si_snr_db: report.enhanced.si_snr_db - report.noisy.si_snr_db,
        snr_db: report.enhanced.snr_db - report.noisy.snr_db,
        segmental_snr_db: report.enhanced.segmental_snr_db - report.noisy.segmental_snr_db,
        stereo_side_sdr_db: optional_metric_difference(
            report.enhanced.stereo_side_sdr_db,
            report.noisy.stereo_side_sdr_db,
        ),
        correlation_error: optional_metric_difference(
            report.noisy.correlation_error,
            report.enhanced.correlation_error,
        ),
        stoi: optional_metric_difference(report.enhanced.stoi, report.noisy.stoi),
        pesq: optional_metric_difference(report.enhanced.pesq, report.noisy.pesq),
        visqol: optional_metric_difference(report.enhanced.visqol, report.noisy.visqol),
        artifact_scores: ArtifactMetrics {
            musical_noise_score: report.noisy.artifact_scores.musical_noise_score
                - report.enhanced.artifact_scores.musical_noise_score,
            pumping_score: report.noisy.artifact_scores.pumping_score
                - report.enhanced.artifact_scores.pumping_score,
            transient_loss_score: report.noisy.artifact_scores.transient_loss_score
                - report.enhanced.artifact_scores.transient_loss_score,
            phase_distortion_score: optional_metric_difference(
                report.noisy.artifact_scores.phase_distortion_score,
                report.enhanced.artifact_scores.phase_distortion_score,
            ),
        },
    };
    ComparisonMetricSet {
        noisy,
        enhanced,
        improvement,
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelRow {
    name: String,
    backend: String,
    license: String,
    sample_rate: u32,
    revision: String,
    installed: bool,
    path: String,
    catalog_sequence: u64,
    catalog_sha256: String,
    catalog_signing_key: String,
    provenance_source: Option<String>,
    installed_at_unix_seconds: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelCatalogRow {
    sequence: u64,
    sha256: String,
    signing_key: String,
    origin: String,
    model_count: usize,
    highest_accepted_sequence: u64,
    cached_path: String,
    issued_at_unix_seconds: Option<u64>,
    expires_at_unix_seconds: Option<u64>,
    trust_root_version: u64,
    trust_root_sha256: String,
    trust_root_expires_at_unix_seconds: u64,
    trust_root_highest_observed_unix_seconds: Option<u64>,
    acquisition_allowed: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelCacheIssueRow {
    kind: String,
    path: String,
    model: Option<String>,
    detail: String,
    prunable: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelCacheHealthRow {
    name: String,
    path: String,
    status: String,
    issues: Vec<ModelCacheIssueRow>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelCacheReportRow {
    cache_dir: String,
    catalog_sequence: u64,
    catalog_sha256: String,
    clean: bool,
    models: Vec<ModelCacheHealthRow>,
    issues: Vec<ModelCacheIssueRow>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelLibraryRow {
    models: Vec<ModelRow>,
    health: ModelCacheReportRow,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelPruneReportRow {
    dry_run: bool,
    would_remove: Vec<String>,
    removed: Vec<String>,
    retained: Vec<ModelCacheIssueRow>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct OfflineBundleModelRow {
    name: String,
    backend: String,
    artifact_filename: String,
    artifact_sha256: String,
    artifact_size_bytes: u64,
    license_filename: String,
    license_sha256: String,
    license_size_bytes: u64,
    provenance_filename: String,
    provenance_sha256: String,
    provenance_size_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct OfflineBundleRow {
    format_version: u32,
    bundle_sha256: String,
    size_bytes: u64,
    catalog_sequence: u64,
    catalog_sha256: String,
    catalog_signing_key_id: String,
    catalog_issued_at_unix_seconds: Option<u64>,
    catalog_expires_at_unix_seconds: Option<u64>,
    trust_root_version: u64,
    trust_root_sha256: String,
    models: Vec<OfflineBundleModelRow>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct OfflineBundleImportRow {
    bundle: OfflineBundleRow,
    installed: Vec<String>,
    already_present: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PreviewData {
    source: String,
    playable_path: String,
    duration_seconds: f64,
    rms_db: f64,
    waveform: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DropSelection {
    audio_files: Vec<String>,
    directories: Vec<String>,
    ignored: Vec<String>,
}

#[tauri::command]
fn app_info() -> AppInfo {
    let hardware = denoize::hardware_capabilities();
    AppInfo {
        version: env!("CARGO_PKG_VERSION"),
        backends: Backend::available_names()
            .iter()
            .filter_map(|name| Backend::parse(name))
            .map(|backend| BackendInfo {
                name: service::backend_name(backend),
                external_model: service::requires_external_model(backend),
                managed_model: (service::backend_name(backend) == "gtcrn").then_some("gtcrn"),
                sample_rate: match service::backend_name(backend) {
                    "bsrnn" | "mossformer2" => Some(48_000),
                    "onnx" | "mpsenet" | "sgmse" | "gtcrn" => Some(16_000),
                    _ => None,
                },
                accelerated: denoize::backend_supports_acceleration(backend),
            })
            .collect(),
        formats: vec!["wav", "flac", "opus", "mp3", "m4a", "aac"],
        fdk_available: cfg!(feature = "fdk-aac-encoder"),
        accelerators: hardware
            .runtimes()
            .iter()
            .map(|runtime| AcceleratorInfo {
                name: runtime.runtime().name(),
                compiled: runtime.compiled(),
                available: runtime.available(),
                device: runtime.device().map(str::to_owned),
                memory_bytes: runtime.memory_bytes(),
                compute_capability: runtime.compute_capability().map(str::to_owned),
                detail: runtime.detail().map(str::to_owned),
            })
            .collect(),
    }
}

#[tauri::command]
fn start_process(
    app: AppHandle,
    state: State<'_, AppState>,
    request: ProcessRequest,
) -> Result<u64, String> {
    validate_request(&request)?;
    let (job_id, control) = register_job(&state)?;
    let jobs = Arc::clone(&state.jobs);
    std::thread::spawn(move || {
        let started = Instant::now();
        emit_progress(
            &app,
            job_id,
            "file",
            "running",
            "音声を読み込んでいます",
            0,
            4,
            started,
            None,
            None,
        );
        let result = process_file(&request, &control, |stage, message| {
            emit_progress(
                &app, job_id, "file", "running", message, stage, 4, started, None, None,
            );
        });
        match result {
            Ok(result) => emit_progress_with_accelerator(
                &app,
                job_id,
                "file",
                "completed",
                "処理が完了しました",
                4,
                4,
                started,
                Some(result.output),
                None,
                Some(result.accelerator),
            ),
            Err(error) if error == "cancelled" => emit_progress(
                &app,
                job_id,
                "file",
                "cancelled",
                "処理をキャンセルしました",
                0,
                4,
                started,
                None,
                None,
            ),
            Err(error) => emit_progress(
                &app,
                job_id,
                "file",
                "failed",
                "処理に失敗しました",
                0,
                4,
                started,
                None,
                Some(error),
            ),
        }
        if let Ok(mut jobs) = jobs.lock() {
            jobs.remove(&job_id);
        }
    });
    Ok(job_id)
}

#[tauri::command]
fn start_batch(
    app: AppHandle,
    state: State<'_, AppState>,
    request: BatchRequest,
) -> Result<u64, String> {
    let RegisteredBatch {
        job_id,
        control,
        pool,
        session,
        items,
    } = prepare_registered_batch(&state, &request)?;
    let jobs = Arc::clone(&state.jobs);
    std::thread::spawn(move || {
        let started = Instant::now();
        let total = items.len();
        let finished = AtomicUsize::new(0);
        let run_item = |batch_item: &PlannedBatchItem| -> BatchItemOutcome {
            let commit_mode =
                match batch_item_commit_mode(batch_item.decision, control.is_cancelled()) {
                    Ok(commit_mode) => commit_mode,
                    Err(outcome) => return outcome,
                };
            let prepared = &batch_item.prepared;
            let worker_permit = match prepared
                .governor
                .acquire_with_cancel(prepared.resource_request, || control.is_cancelled())
            {
                Ok(permit) => permit,
                Err(_) if control.is_cancelled() => return BatchItemOutcome::Cancelled,
                Err(error) => return BatchItemOutcome::Failed(error),
            };
            let result = stage_batch_output(
                &prepared.item.input,
                &prepared.item.output,
                prepared.item.output_format,
                prepared.encode,
                prepared.metadata_policy,
                &prepared.processing,
                &prepared.backend_session,
                prepared.decode_limits,
                prepared.metadata_limits,
                prepared.resource_request.temporary_bytes(),
                &control,
            )
            .and_then(|transaction| {
                verify_prepared_batch_recipe(prepared)?;
                control.commit_fence(|| {
                    session
                        .publish(&prepared.expectation, transaction, commit_mode)
                        .map(|_| ())
                })
            });
            drop(worker_permit);
            match result {
                Ok(_) => BatchItemOutcome::Completed,
                Err(error) if error == "cancelled" => BatchItemOutcome::Cancelled,
                Err(error) => BatchItemOutcome::Failed(error),
            }
        };
        let process_item = |batch_item: &PlannedBatchItem| {
            let outcome = run_item(batch_item);
            let current = finished.fetch_add(1, Ordering::SeqCst) + 1;
            emit_batch_item(
                &app,
                job_id,
                outcome.status(),
                &batch_item.prepared.item,
                Some(batch_item.decision.reason()),
                current,
                total,
                started,
                outcome.error(),
                batch_item.prepared.processing.accelerator,
            );
            outcome
        };
        let outcomes = if let Some(pool) = pool {
            pool.install(|| items.par_iter().map(process_item).collect::<Vec<_>>())
        } else {
            items.iter().map(process_item).collect::<Vec<_>>()
        };
        let counts = count_batch_outcomes(&outcomes);
        debug_assert_eq!(counts.total(), total);
        debug_assert_eq!(finished.load(Ordering::SeqCst), total);
        let BatchTerminalOutcome {
            status,
            message,
            error,
        } = batch_terminal_outcome(counts);
        emit_progress(
            &app,
            job_id,
            "batch",
            status,
            &message,
            total,
            total,
            started,
            Some(request.output_dir),
            error,
        );
        if let Ok(mut jobs) = jobs.lock() {
            jobs.remove(&job_id);
        }
    });
    Ok(job_id)
}

struct RegisteredBatch {
    job_id: u64,
    control: Arc<JobControl>,
    pool: Option<rayon::ThreadPool>,
    session: BatchSession,
    items: Vec<PlannedBatchItem>,
}

fn prepare_registered_batch(
    state: &AppState,
    request: &BatchRequest,
) -> Result<RegisteredBatch, String> {
    let (job_id, control) = register_job(state)?;
    let prepared = (|| {
        let prepared = prepare_batch_request(request)?;
        let pool = if request.options.deterministic {
            None
        } else {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(request.jobs)
                    .build()
                    .map_err(|error| format!("並列処理を準備できませんでした: {error}"))?,
            )
        };
        let session = BatchSession::acquire(Path::new(&request.output_dir), request.resume)?;
        let items = plan_batch_items(&session, prepared, request.options.force)?;
        session.activate()?;
        Ok(RegisteredBatch {
            job_id,
            control: Arc::clone(&control),
            pool,
            session,
            items,
        })
    })();
    if prepared.is_err() {
        if let Ok(mut jobs) = state.jobs.lock() {
            jobs.remove(&job_id);
        }
    }
    prepared
}

#[derive(Debug, PartialEq, Eq)]
struct BatchTerminalOutcome {
    status: &'static str,
    message: String,
    error: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BatchOutcomeCounts {
    completed: usize,
    skipped: usize,
    failed: usize,
    cancelled: usize,
}

impl BatchOutcomeCounts {
    fn total(self) -> usize {
        self.completed + self.skipped + self.failed + self.cancelled
    }
}

#[derive(Debug, PartialEq, Eq)]
enum BatchItemOutcome {
    Completed,
    Skipped,
    Failed(String),
    Cancelled,
}

fn batch_item_commit_mode(
    decision: ResumeDecision,
    cancelled: bool,
) -> Result<CommitMode, BatchItemOutcome> {
    match decision {
        ResumeDecision::Skip { .. } => Err(BatchItemOutcome::Skipped),
        ResumeDecision::Process { .. } if cancelled => Err(BatchItemOutcome::Cancelled),
        ResumeDecision::Process { commit_mode, .. } => Ok(commit_mode),
    }
}

impl BatchItemOutcome {
    fn status(&self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Skipped => "skipped",
            Self::Failed(_) => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    fn error(&self) -> Option<String> {
        match self {
            Self::Failed(error) => Some(error.clone()),
            _ => None,
        }
    }
}

fn count_batch_outcomes(outcomes: &[BatchItemOutcome]) -> BatchOutcomeCounts {
    let mut counts = BatchOutcomeCounts::default();
    for outcome in outcomes {
        match outcome {
            BatchItemOutcome::Completed => counts.completed += 1,
            BatchItemOutcome::Skipped => counts.skipped += 1,
            BatchItemOutcome::Failed(_) => counts.failed += 1,
            BatchItemOutcome::Cancelled => counts.cancelled += 1,
        }
    }
    counts
}

fn batch_terminal_outcome(counts: BatchOutcomeCounts) -> BatchTerminalOutcome {
    let message = format!(
        "完了 {} · スキップ {} · 失敗 {} · キャンセル {}",
        counts.completed, counts.skipped, counts.failed, counts.cancelled
    );
    if counts.cancelled > 0 {
        BatchTerminalOutcome {
            status: "cancelled",
            message,
            error: None,
        }
    } else if counts.failed == 0 {
        BatchTerminalOutcome {
            status: "completed",
            message,
            error: None,
        }
    } else {
        BatchTerminalOutcome {
            status: "failed",
            message,
            error: Some(format!(
                "{}件のファイルを処理できませんでした",
                counts.failed
            )),
        }
    }
}

fn plan_batch_items(
    session: &BatchSession,
    prepared: Vec<PreparedBatchItem>,
    force: bool,
) -> Result<Vec<PlannedBatchItem>, String> {
    let mut planned = Vec::with_capacity(prepared.len());
    for prepared in prepared {
        let decision = session
            .plan(&prepared.expectation, force)
            .map_err(actionable_batch_resume_error)?;
        planned.push(PlannedBatchItem { prepared, decision });
    }
    for item in &planned {
        item.prepared.expectation.verify_sources()?;
    }
    Ok(planned)
}

fn actionable_batch_resume_error(error: String) -> String {
    if error.contains("without --force") {
        format!(
            "{error}\n既存出力が古い・未追跡・旧形式・安全でない状態です。「既存を上書き」を有効にして再処理してください"
        )
    } else {
        error
    }
}

fn collect_batch_items(request: &BatchRequest, extension: &str) -> Result<Vec<BatchItem>, String> {
    let output_root = Path::new(&request.output_dir);
    let mut sources = request.inputs.iter().map(PathBuf::from).collect::<Vec<_>>();
    let input_root = request.input_dir.as_deref().map(Path::new);
    if let Some(root) = input_root {
        if !root.is_dir() {
            return Err("入力フォルダが存在しません".into());
        }
        validate_batch_directories(root, output_root)?;
        collect_audio_files(root, request.recursive, &mut sources)?;
        if output_root.starts_with(root) {
            sources.retain(|path| !path.starts_with(output_root));
        }
    }
    sources.sort();
    sources.dedup();
    let mut destinations = HashSet::new();
    let items = sources
        .into_iter()
        .map(|input| {
            if !input.is_file() {
                return Err(format!("入力ファイルが存在しません: {}", input.display()));
            }
            let relative = input_root
                .and_then(|root| input.strip_prefix(root).ok())
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(input.file_name().unwrap_or_default()));
            let mut output = output_root.join(&relative);
            output.set_extension(extension);
            if input.to_str().is_none() || output.to_str().is_none() {
                return Err(format!(
                    "GUIバッチではUTF-8で表現できないパスを処理できません: {}",
                    input.display()
                ));
            }
            if !destinations.insert(output.clone()) {
                return Err(format!(
                    "同じ出力先になるファイルがあります: {}",
                    output.display()
                ));
            }
            let output_relative = output.strip_prefix(output_root).map_err(|error| {
                format!(
                    "バッチ出力 {} が出力フォルダ外です: {error}",
                    output.display()
                )
            })?;
            let output_format = OutputFormat::from_path(&output)?;
            let input_identity = std::fs::canonicalize(&input).map_err(|error| {
                format!("バッチ入力 {} を解決できません: {error}", input.display())
            })?;
            let item_id = batch_resume::item_identity(
                &input_identity,
                &relative,
                output_relative,
                output_format,
            );
            Ok(BatchItem {
                input,
                output,
                output_format,
                item_id,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    validate_batch_destinations(input_root, &items)?;
    Ok(items)
}

fn collect_audio_files(
    dir: &Path,
    recursive: bool,
    files: &mut Vec<PathBuf>,
) -> Result<(), String> {
    for entry in
        std::fs::read_dir(dir).map_err(|error| format!("入力フォルダを読めません: {error}"))?
    {
        let entry = entry.map_err(|error| error.to_string())?;
        let file_type = entry.file_type().map_err(|error| error.to_string())?;
        let path = entry.path();
        if file_type.is_dir() && recursive {
            collect_audio_files(&path, true, files)?;
        } else if file_type.is_file() && is_audio_path(&path) {
            files.push(path);
        }
    }
    Ok(())
}

fn is_audio_path(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "wav" | "flac" | "opus" | "ogg" | "mp3" | "m4a" | "aac"
            )
        })
}

fn batch_collision_key_with_case(path: &Path, case_insensitive: bool) -> PathBuf {
    if case_insensitive {
        PathBuf::from(path.to_string_lossy().to_lowercase())
    } else {
        path.to_path_buf()
    }
}

fn batch_collision_key(path: &Path) -> PathBuf {
    batch_collision_key_with_case(path, cfg!(any(windows, target_os = "macos")))
}

fn validate_batch_destinations(
    input_root: Option<&Path>,
    items: &[BatchItem],
) -> Result<(), String> {
    let input_root = input_root.map(normalize_batch_path).transpose()?;
    let input_paths = items
        .iter()
        .map(|item| normalize_batch_path(&item.input).map(|path| batch_collision_key(&path)))
        .collect::<Result<HashSet<_>, _>>()?;
    let mut destinations = Vec::with_capacity(items.len());
    for item in items {
        let resolved = normalize_batch_path(&item.output)?;
        if input_root
            .as_deref()
            .is_some_and(|root| resolved.starts_with(root))
        {
            return Err(format!(
                "バッチ出力 {} が入力フォルダ内へ解決されます。出力先のシンボリックリンクを除くか、別の出力フォルダを選択してください",
                item.output.display()
            ));
        }
        let collision_key = batch_collision_key(&resolved);
        if input_paths.contains(&collision_key) {
            return Err(format!(
                "バッチ出力 {} が入力ファイルを上書きします",
                item.output.display()
            ));
        }
        destinations.push((collision_key, item));
    }
    destinations.sort_by(|left, right| left.0.cmp(&right.0));

    for pair in destinations.windows(2) {
        let (left_path, left) = &pair[0];
        let (right_path, right) = &pair[1];
        if right_path == left_path {
            return Err(format!(
                "複数の入力が同じバッチ出力になります: {} と {} -> {}",
                left.input.display(),
                right.input.display(),
                right.output.display()
            ));
        }
        if right_path.starts_with(left_path) {
            return Err(format!(
                "バッチ出力がファイルとディレクトリとして競合します: {} -> {} / {} -> {}",
                left.input.display(),
                left.output.display(),
                right.input.display(),
                right.output.display()
            ));
        }
    }
    Ok(())
}

fn normalize_batch_path(path: &Path) -> Result<PathBuf, String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("現在のフォルダを解決できません: {error}"))?
            .join(path)
    };
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
                    "バッチパス {} を確認できません: {error}",
                    ancestor.display()
                ));
            }
        }
        let component = ancestor
            .components()
            .next_back()
            .ok_or_else(|| format!("バッチパス {} を解決できません", absolute.display()))?;
        match component {
            std::path::Component::Normal(name) => {
                missing.push(MissingComponent::Normal(name.to_os_string()))
            }
            std::path::Component::ParentDir => missing.push(MissingComponent::Parent),
            std::path::Component::CurDir => {}
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                return Err(format!(
                    "バッチパス {} を解決できません",
                    absolute.display()
                ));
            }
        }
        if !ancestor.pop() {
            return Err(format!(
                "バッチパス {} を解決できません",
                absolute.display()
            ));
        }
    }
    let mut resolved = std::fs::canonicalize(&ancestor)
        .map_err(|error| format!("{} を解決できません: {error}", ancestor.display()))?;
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

fn validate_batch_directories(input_dir: &Path, output_dir: &Path) -> Result<(), String> {
    let input = normalize_batch_path(input_dir)?;
    let output = normalize_batch_path(output_dir)?;
    if input.starts_with(&output) || output.starts_with(&input) {
        return Err(format!(
            "入力フォルダと出力フォルダは重ならない場所を選択してください: {} / {}",
            input_dir.display(),
            output_dir.display()
        ));
    }
    Ok(())
}

fn validate_batch_control_paths(items: &[BatchItem], output_root: &Path) -> Result<(), String> {
    for name in [
        batch_resume::STATE_FILE_NAME,
        batch_resume::LEGACY_DESKTOP_STATE_FILE_NAME,
        batch_resume::LOCK_FILE_NAME,
    ] {
        let reserved = output_root.join(name);
        let reserved_key = batch_collision_key(&normalize_batch_path(&reserved)?);
        for item in items {
            let output = batch_collision_key(&normalize_batch_path(&item.output)?);
            if output == reserved_key
                || output.starts_with(&reserved_key)
                || reserved_key.starts_with(&output)
            {
                return Err(format!(
                    "バッチ出力 {} は予約済みの管理パス {name} と競合します",
                    item.output.display()
                ));
            }
        }
    }
    Ok(())
}

#[tauri::command]
fn cancel_job(state: State<'_, AppState>, job_id: u64) -> Result<(), String> {
    let jobs = state
        .jobs
        .lock()
        .map_err(|_| "ジョブ状態を取得できません")?;
    let control = jobs.get(&job_id).ok_or("実行中のジョブが見つかりません")?;
    control.cancel()
}

#[tauri::command]
#[cfg(feature = "live")]
async fn live_devices() -> Result<LiveDevices, String> {
    tauri::async_runtime::spawn_blocking(|| {
        let (inputs, outputs) = denoize::live::device_names()?;
        Ok(LiveDevices { inputs, outputs })
    })
    .await
    .map_err(|error| format!("デバイス一覧の取得に失敗しました: {error}"))?
}

#[tauri::command]
#[cfg(not(feature = "live"))]
async fn live_devices() -> Result<LiveDevices, String> {
    Err("このビルドではライブ処理を利用できません".into())
}

#[tauri::command]
#[cfg(feature = "live")]
fn start_live(
    app: AppHandle,
    state: State<'_, AppState>,
    request: LiveRequest,
) -> Result<(), String> {
    let backend = validate_live_request(&request)?;
    let backend_options = parsed_backend_options_for(backend, &request.options)?;
    let denoiser = processing_config(&request.options, 48_000)?;
    let governor = desktop_resource_governor(&request.options, 1)?;
    let prepared = denoize::live::PreparedLiveConfig::new(denoize::live::LiveConfig {
        input_device: request.input_device,
        output_device: request.output_device,
        chunk_ms: request.chunk_ms,
        backend,
        backend_options,
        denoiser,
    })?;
    let accelerator = prepared.accelerator();
    let running = Arc::new(AtomicBool::new(true));
    register_live_session(&state, Arc::clone(&running))?;
    let live_state = Arc::clone(&state.live);
    std::thread::spawn(move || {
        let result = denoize::live::run_prepared_with_status_and_governor(
            prepared,
            running,
            &governor,
            |status| {
                let _ = app.emit(
                    "live-status",
                    LiveEvent {
                        status: "running",
                        message: "ライブ処理中".into(),
                        sample_rate: status.sample_rate,
                        input_channels: status.input_channels,
                        output_channels: status.output_channels,
                        chunk_frames: status.chunk_frames,
                        input_level: status.input_level,
                        output_level: status.output_level,
                        processed_chunks: status.processed_chunks,
                        dropped_chunks: status.dropped_chunks,
                        accelerator: Some(accelerator_result(status.accelerator)),
                    },
                );
            },
        );
        let (status, message) = match result {
            Ok(()) => ("stopped", "ライブ処理を停止しました".into()),
            Err(error) => ("failed", error),
        };
        let _ = app.emit(
            "live-status",
            LiveEvent {
                status,
                message,
                sample_rate: 0,
                input_channels: 0,
                output_channels: 0,
                chunk_frames: 0,
                input_level: 0.0,
                output_level: 0.0,
                processed_chunks: 0,
                dropped_chunks: 0,
                accelerator: Some(accelerator_result(accelerator)),
            },
        );
        if let Ok(mut live) = live_state.lock() {
            *live = None;
        }
    });
    Ok(())
}

#[tauri::command]
#[cfg(not(feature = "live"))]
fn start_live(
    _app: AppHandle,
    _state: State<'_, AppState>,
    _request: LiveRequest,
) -> Result<(), String> {
    Err("このビルドではライブ処理を利用できません".into())
}

#[tauri::command]
#[cfg(feature = "live")]
fn stop_live(state: State<'_, AppState>) -> Result<(), String> {
    let live = state
        .live
        .lock()
        .map_err(|_| "ライブ状態を取得できません")?;
    let running = live.as_ref().ok_or("ライブ処理は実行されていません")?;
    running.store(false, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
#[cfg(not(feature = "live"))]
fn stop_live(_state: State<'_, AppState>) -> Result<(), String> {
    Err("このビルドではライブ処理を利用できません".into())
}

#[tauri::command]
async fn compare_audio(
    clean: String,
    noisy: String,
    enhanced: String,
) -> Result<ComparisonOutput, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let report = ComparisonReport::compare(
            &read_audio(clean)?,
            &read_audio(noisy)?,
            &read_audio(enhanced)?,
        )?;
        Ok(ComparisonOutput {
            markdown: report.markdown(),
            json: report.json(),
            html: report.html(),
            noisy_snr_db: report.noisy.snr_db,
            enhanced_snr_db: report.enhanced.snr_db,
            improvement_db: report.enhanced.snr_db - report.noisy.snr_db,
            metrics: comparison_metric_set(&report),
        })
    })
    .await
    .map_err(|error| format!("比較タスクに失敗しました: {error}"))?
}

#[tauri::command]
fn list_models() -> Result<ModelLibraryRow, String> {
    let catalog = denoize::models::active_catalog()?;
    let health = denoize::models::doctor_model_cache_for_catalog(&catalog)?;
    let models = catalog
        .models()
        .iter()
        .map(|model| {
            let path = denoize::models::path_for_catalog_model(model)?;
            let cache_model = health
                .models
                .iter()
                .find(|candidate| candidate.name == model.name())
                .ok_or_else(|| format!("モデル診断結果がありません: {}", model.name()))?;
            let installed = cache_model.status == denoize::models::ModelCacheModelStatus::Healthy;
            let provenance = cache_model.provenance.as_ref();
            Ok(ModelRow {
                name: model.name().to_string(),
                backend: model.backend().to_string(),
                license: model.license().to_string(),
                sample_rate: model.sample_rate(),
                revision: model.revision().to_string(),
                installed,
                path: path.to_string_lossy().into_owned(),
                catalog_sequence: model.catalog_sequence(),
                catalog_sha256: model.catalog_sha256().to_string(),
                catalog_signing_key: model.catalog_signing_key_id().to_string(),
                provenance_source: provenance
                    .as_ref()
                    .map(|provenance| model_installation_source(&provenance.installation_source)),
                installed_at_unix_seconds: provenance
                    .map(|provenance| provenance.installed_at_unix_seconds),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(ModelLibraryRow {
        models,
        health: model_cache_report_row(health),
    })
}

fn model_catalog_origin(origin: &denoize::models::CatalogOrigin) -> String {
    match origin {
        denoize::models::CatalogOrigin::Embedded => "embedded".into(),
        denoize::models::CatalogOrigin::Signed { source } if source == "local-import" => {
            "signed:local-import".into()
        }
        denoize::models::CatalogOrigin::Signed { source } => {
            format!("signed:{}", denoize::models::redact_url(source))
        }
        _ => "unknown".into(),
    }
}

fn model_installation_source(source: &denoize::models::ModelInstallationSource) -> String {
    match source {
        denoize::models::ModelInstallationSource::CatalogUrl { url } => {
            format!("catalog-url:{}", denoize::models::redact_url(url))
        }
        denoize::models::ModelInstallationSource::AlternateUrl { url } => {
            format!("alternate-url:{}", denoize::models::redact_url(url))
        }
        denoize::models::ModelInstallationSource::LocalFile => "local-file".into(),
        denoize::models::ModelInstallationSource::CompletedPartial => "completed-partial".into(),
        denoize::models::ModelInstallationSource::ExistingCacheMigration => {
            "existing-cache-migration".into()
        }
        denoize::models::ModelInstallationSource::OfflineBundle { bundle_sha256 } => {
            format!("offline-bundle:{bundle_sha256}")
        }
        _ => "unknown".into(),
    }
}

fn model_cache_status(status: denoize::models::ModelCacheModelStatus) -> String {
    match status {
        denoize::models::ModelCacheModelStatus::Missing => "missing",
        denoize::models::ModelCacheModelStatus::Healthy => "healthy",
        denoize::models::ModelCacheModelStatus::Corrupt => "corrupt",
        denoize::models::ModelCacheModelStatus::ProvenanceMissing => "provenance-missing",
        denoize::models::ModelCacheModelStatus::ProvenanceInvalid => "provenance-invalid",
        denoize::models::ModelCacheModelStatus::Unsafe => "unsafe",
        _ => "unknown",
    }
    .into()
}

fn model_cache_issue_kind(kind: denoize::models::ModelCacheIssueKind) -> String {
    match kind {
        denoize::models::ModelCacheIssueKind::MissingArtifact => "missing-artifact",
        denoize::models::ModelCacheIssueKind::CorruptArtifact => "corrupt-artifact",
        denoize::models::ModelCacheIssueKind::MissingProvenance => "missing-provenance",
        denoize::models::ModelCacheIssueKind::InvalidProvenance => "invalid-provenance",
        denoize::models::ModelCacheIssueKind::IncompleteDownload => "incomplete-download",
        denoize::models::ModelCacheIssueKind::StaleDownloadState => "stale-download-state",
        denoize::models::ModelCacheIssueKind::OrphanedEntry => "orphaned-entry",
        denoize::models::ModelCacheIssueKind::UnsafeEntry => "unsafe-entry",
        _ => "unknown",
    }
    .into()
}

fn model_cache_issue_row(issue: denoize::models::ModelCacheIssue) -> ModelCacheIssueRow {
    ModelCacheIssueRow {
        kind: model_cache_issue_kind(issue.kind),
        path: issue.path.to_string_lossy().into_owned(),
        model: issue.model,
        detail: issue.detail,
        prunable: issue.prunable,
    }
}

fn model_cache_report_row(report: denoize::models::ModelCacheReport) -> ModelCacheReportRow {
    let clean = report.is_clean();
    ModelCacheReportRow {
        cache_dir: report.cache_dir.to_string_lossy().into_owned(),
        catalog_sequence: report.catalog_sequence,
        catalog_sha256: report.catalog_sha256,
        clean,
        models: report
            .models
            .into_iter()
            .map(|model| ModelCacheHealthRow {
                name: model.name,
                path: model.path.to_string_lossy().into_owned(),
                status: model_cache_status(model.status),
                issues: model
                    .issues
                    .into_iter()
                    .map(model_cache_issue_row)
                    .collect(),
            })
            .collect(),
        issues: report
            .issues
            .into_iter()
            .map(model_cache_issue_row)
            .collect(),
    }
}

fn current_model_catalog_row() -> Result<ModelCatalogRow, String> {
    let status = denoize::models::catalog_status()?;
    Ok(ModelCatalogRow {
        sequence: status.sequence,
        sha256: status.sha256,
        signing_key: status.signing_key_id,
        origin: model_catalog_origin(&status.origin),
        model_count: status.model_count,
        highest_accepted_sequence: status.highest_accepted_sequence,
        cached_path: status.cached_catalog_path.to_string_lossy().into_owned(),
        issued_at_unix_seconds: status.issued_at_unix_seconds,
        expires_at_unix_seconds: status.expires_at_unix_seconds,
        trust_root_version: status.trust_root_version,
        trust_root_sha256: status.trust_root_sha256,
        trust_root_expires_at_unix_seconds: status.trust_root_expires_at_unix_seconds,
        trust_root_highest_observed_unix_seconds: status.trust_root_highest_observed_unix_seconds,
        acquisition_allowed: status.acquisition_allowed,
    })
}

fn offline_bundle_row(info: denoize::models::OfflineBundleInfo) -> OfflineBundleRow {
    OfflineBundleRow {
        format_version: info.format_version,
        bundle_sha256: info.bundle_sha256,
        size_bytes: info.size_bytes,
        catalog_sequence: info.catalog_sequence,
        catalog_sha256: info.catalog_sha256,
        catalog_signing_key_id: info.catalog_signing_key_id,
        catalog_issued_at_unix_seconds: info.catalog_issued_at_unix_seconds,
        catalog_expires_at_unix_seconds: info.catalog_expires_at_unix_seconds,
        trust_root_version: info.trust_root_version,
        trust_root_sha256: info.trust_root_sha256,
        models: info
            .models
            .into_iter()
            .map(|model| OfflineBundleModelRow {
                name: model.name,
                backend: model.backend,
                artifact_filename: model.artifact_filename,
                artifact_sha256: model.artifact_sha256,
                artifact_size_bytes: model.artifact_size_bytes,
                license_filename: model.license_filename,
                license_sha256: model.license_sha256,
                license_size_bytes: model.license_size_bytes,
                provenance_filename: model.provenance_filename,
                provenance_sha256: model.provenance_sha256,
                provenance_size_bytes: model.provenance_size_bytes,
            })
            .collect(),
    }
}

#[tauri::command]
fn model_catalog_status() -> Result<ModelCatalogRow, String> {
    current_model_catalog_row()
}

#[tauri::command]
async fn update_model_catalog(
    options: Option<ModelActionOptions>,
) -> Result<ModelCatalogRow, String> {
    let options = catalog_action_options(options)?;
    tauri::async_runtime::spawn_blocking(move || {
        denoize::models::update_catalog(&options)?;
        current_model_catalog_row()
    })
    .await
    .map_err(|error| format!("モデルカタログ更新タスクに失敗しました: {error}"))?
}

#[tauri::command]
async fn inspect_model_bundle(path: String) -> Result<OfflineBundleRow, String> {
    tauri::async_runtime::spawn_blocking(move || {
        denoize::models::inspect_offline_bundle(path).map(offline_bundle_row)
    })
    .await
    .map_err(|error| format!("オフラインモデルバンドル検証タスクに失敗しました: {error}"))?
}

#[tauri::command]
async fn import_model_bundle(
    path: String,
    expected_bundle_sha256: String,
) -> Result<OfflineBundleImportRow, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let report =
            denoize::models::import_offline_bundle_if_sha256(path, &expected_bundle_sha256)?;
        Ok(OfflineBundleImportRow {
            bundle: offline_bundle_row(report.bundle),
            installed: report
                .installed
                .into_iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect(),
            already_present: report
                .already_present
                .into_iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect(),
        })
    })
    .await
    .map_err(|error| format!("オフラインモデルバンドル導入タスクに失敗しました: {error}"))?
}

#[tauri::command]
async fn recover_model_trust_root() -> Result<ModelCatalogRow, String> {
    tauri::async_runtime::spawn_blocking(|| {
        denoize::models::recover_embedded_trust_root()?;
        current_model_catalog_row()
    })
    .await
    .map_err(|error| format!("モデル信頼ルート復旧タスクに失敗しました: {error}"))?
}

#[tauri::command]
async fn reset_model_trust_time_floor() -> Result<ModelCatalogRow, String> {
    tauri::async_runtime::spawn_blocking(|| {
        denoize::models::reset_trust_time_floor()?;
        current_model_catalog_row()
    })
    .await
    .map_err(|error| format!("モデル信頼時刻リセットタスクに失敗しました: {error}"))?
}

#[tauri::command]
async fn model_cache_doctor() -> Result<ModelCacheReportRow, String> {
    tauri::async_runtime::spawn_blocking(|| {
        denoize::models::doctor_model_cache().map(model_cache_report_row)
    })
    .await
    .map_err(|error| format!("モデルキャッシュ診断タスクに失敗しました: {error}"))?
}

fn write_automation_json(path: &Path, json: &str) -> Result<(), String> {
    let mut transaction = AtomicOutput::new(path)?;
    std::io::Write::write_all(transaction.file_mut(), json.as_bytes())
        .map_err(|error| format!("自動化JSONを書き込めません: {error}"))?;
    transaction.commit(CommitMode::Replace)
}

fn write_automation_snapshot(path: &Path) -> Result<(), String> {
    let snapshot = denoize::automation::capture_automation_snapshot()?;
    let mut json = snapshot.to_pretty_json()?;
    json.push('\n');
    write_automation_json(path, &json)
}

#[tauri::command]
async fn save_automation_snapshot(path: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || write_automation_snapshot(Path::new(&path)))
        .await
        .map_err(|error| format!("自動化JSONの書出タスクに失敗しました: {error}"))?
}

#[tauri::command]
async fn prune_model_cache(dry_run: bool) -> Result<ModelPruneReportRow, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let report = denoize::models::prune_model_cache(dry_run)?;
        Ok(ModelPruneReportRow {
            dry_run: report.dry_run,
            would_remove: report
                .would_remove
                .into_iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect(),
            removed: report
                .removed
                .into_iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect(),
            retained: report
                .retained
                .into_iter()
                .map(model_cache_issue_row)
                .collect(),
        })
    })
    .await
    .map_err(|error| format!("モデルキャッシュ整理タスクに失敗しました: {error}"))?
}

#[tauri::command]
fn model_action(
    app: AppHandle,
    state: State<'_, AppState>,
    name: String,
    action: String,
    options: Option<ModelActionOptions>,
) -> Result<u64, String> {
    let catalog = denoize::models::active_catalog()?;
    let model = catalog
        .find(&name)
        .cloned()
        .ok_or_else(|| format!("不明なモデル: {name}"))?;
    if !matches!(
        action.as_str(),
        "install" | "update" | "verify" | "repair" | "remove"
    ) {
        return Err(format!("不明な操作: {action}"));
    }
    let (download_options, source_path) =
        if matches!(action.as_str(), "install" | "update" | "repair") {
            model_action_options(options)?
        } else {
            (ModelDownloadOptions::default(), None)
        };
    if source_path.is_some() && action != "install" {
        return Err("ローカルファイルは導入操作でのみ使用できます".into());
    }
    let (job_id, cancelled) = register_job(&state)?;
    let jobs = Arc::clone(&state.jobs);
    std::thread::spawn(move || {
        emit_model_progress(&app, job_id, &name, "running", "準備しています", 0, None);
        let progress_message = if source_path.is_some() {
            "ローカルモデルを検証しています"
        } else {
            "モデルをダウンロードしています"
        };
        let progress = |downloaded, total| {
            emit_model_progress(
                &app,
                job_id,
                &name,
                "running",
                progress_message,
                downloaded,
                total,
            );
        };
        let result = match action.as_str() {
            "install" => match source_path {
                Some(source) => denoize::models::install_catalog_model_from_file_with_progress(
                    &model,
                    source,
                    || cancelled.is_cancelled(),
                    progress,
                ),
                None => denoize::models::install_catalog_model_with_options_and_progress(
                    &model,
                    &download_options,
                    || cancelled.is_cancelled(),
                    progress,
                ),
            }
            .map(|path| path.display().to_string()),
            "update" => denoize::models::update_catalog_model_with_options_and_progress(
                &model,
                &download_options,
                || cancelled.is_cancelled(),
                progress,
            )
            .map(|path| path.display().to_string()),
            "verify" => {
                denoize::models::verify_catalog_model(&model).map(|path| path.display().to_string())
            }
            "repair" => denoize::models::repair_catalog_model_with_options_and_progress(
                &model,
                &download_options,
                || cancelled.is_cancelled(),
                progress,
            )
            .map(|outcome| match outcome {
                denoize::models::ModelRepairOutcome::AlreadyHealthy => {
                    "正常なため修復は不要です".into()
                }
                denoize::models::ModelRepairOutcome::ProvenanceRebuilt => {
                    "provenanceを再構築しました".into()
                }
                denoize::models::ModelRepairOutcome::ArtifactInstalled => {
                    "モデルを再取得して修復しました".into()
                }
                _ => "モデルを修復しました".into(),
            }),
            "remove" => {
                denoize::models::remove_catalog_model(&model).map(|_| "削除しました".into())
            }
            _ => unreachable!(),
        };
        match result {
            Ok(message) => {
                emit_model_progress(&app, job_id, &name, "completed", &message, 1, Some(1))
            }
            Err(error) if error == "cancelled" => emit_model_progress(
                &app,
                job_id,
                &name,
                "cancelled",
                "モデル操作を中断しました",
                0,
                None,
            ),
            Err(error) => emit_model_progress(&app, job_id, &name, "failed", &error, 0, None),
        }
        if let Ok(mut jobs) = jobs.lock() {
            jobs.remove(&job_id);
        }
    });
    Ok(job_id)
}

fn emit_model_progress(
    app: &AppHandle,
    job_id: u64,
    name: &str,
    status: &'static str,
    message: &str,
    downloaded: u64,
    total: Option<u64>,
) {
    let _ = app.emit(
        "model-progress",
        ModelProgress {
            job_id,
            name: name.into(),
            status,
            message: message.into(),
            downloaded,
            total,
            fraction: total
                .filter(|total| *total > 0)
                .map(|total| downloaded as f64 / total as f64),
        },
    );
}

#[tauri::command]
async fn prepare_preview(path: String, points: Option<usize>) -> Result<PreviewData, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let source = Path::new(&path);
        if !source.is_file() {
            return Err("プレビューする音声ファイルが存在しません".into());
        }
        let audio = read_audio(source)?;
        let frames = audio.frames();
        let point_count = points.unwrap_or(180).clamp(32, 512);
        let mut waveform = vec![0.0f64; point_count];
        let mut sum_squares = 0.0;
        let mut sample_count = 0usize;
        for channel in &audio.channels {
            for (index, sample) in channel.iter().enumerate() {
                let bucket = index.saturating_mul(point_count) / frames.max(1);
                if let Some(peak) = waveform.get_mut(bucket.min(point_count - 1)) {
                    *peak = peak.max(sample.abs());
                }
                sum_squares += sample * sample;
                sample_count += 1;
            }
        }
        let peak = waveform.iter().copied().fold(0.0f64, f64::max).max(1e-9);
        for value in &mut waveform {
            *value /= peak;
        }
        let rms = (sum_squares / sample_count.max(1) as f64).sqrt();
        let mut hasher = DefaultHasher::new();
        path.hash(&mut hasher);
        source
            .metadata()
            .and_then(|metadata| metadata.modified())
            .ok()
            .hash(&mut hasher);
        let preview_dir = std::env::temp_dir().join("denoize-previews");
        std::fs::create_dir_all(&preview_dir)
            .map_err(|error| format!("プレビューフォルダを作成できません: {error}"))?;
        let playable = preview_dir.join(format!("{:016x}.wav", hasher.finish()));
        if !playable.is_file() {
            write_audio(&playable, &audio, EncodeOptions::default())?;
        }
        Ok(PreviewData {
            source: path,
            playable_path: playable.to_string_lossy().into_owned(),
            duration_seconds: frames as f64 / audio.sample_rate.max(1) as f64,
            rms_db: 20.0 * rms.max(1e-10).log10(),
            waveform,
        })
    })
    .await
    .map_err(|error| format!("プレビュー処理に失敗しました: {error}"))?
}

#[tauri::command]
fn load_gui_config(path: String, current: GuiConfig) -> Result<GuiConfig, String> {
    let source =
        std::fs::read_to_string(&path).map_err(|error| format!("{path} を読めません: {error}"))?;
    parse_gui_config(&source, current)
}

#[tauri::command]
fn save_gui_config(path: String, config: GuiConfig) -> Result<(), String> {
    let mut config = config.normalized()?;
    // `-1` is the CLI-compatible legacy sentinel for explicitly disabling
    // loudness/true-peak processing. Keeping it in exported TOML distinguishes
    // a full disabled config from an omitted field in a partial overlay.
    if config.loudness_lufs.is_none() {
        config.true_peak_dbtp = Some(-1.0);
    }
    let source = toml::to_string_pretty(&config)
        .map_err(|error| format!("設定をTOMLへ変換できません: {error}"))?;
    std::fs::write(&path, source).map_err(|error| format!("{path} を保存できません: {error}"))
}

fn parse_gui_config(source: &str, current: GuiConfig) -> Result<GuiConfig, String> {
    let patch: GuiConfigPatch =
        toml::from_str(source).map_err(|error| format!("TOML設定が不正です: {error}"))?;
    patch.merge(current)
}

#[tauri::command]
fn classify_dropped_paths(paths: Vec<String>) -> DropSelection {
    let mut selection = DropSelection {
        audio_files: Vec::new(),
        directories: Vec::new(),
        ignored: Vec::new(),
    };
    for value in paths {
        let path = Path::new(&value);
        if path.is_dir() {
            selection.directories.push(value);
        } else if path.is_file() && is_audio_path(path) {
            selection.audio_files.push(value);
        } else {
            selection.ignored.push(value);
        }
    }
    selection
}

#[tauri::command]
fn save_text_file(path: String, contents: String) -> Result<(), String> {
    std::fs::write(&path, contents).map_err(|error| format!("{path} を保存できません: {error}"))
}

fn register_job(state: &AppState) -> Result<(u64, Arc<JobControl>), String> {
    let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
    let control = Arc::new(JobControl::default());
    let mut jobs = state
        .jobs
        .lock()
        .map_err(|_| "ジョブ状態を更新できません")?;
    let live = state
        .live
        .lock()
        .map_err(|_| "ライブ状態を取得できません")?;
    if live.is_some() {
        return Err("ライブ処理を停止してから開始してください".into());
    }
    if !jobs.is_empty() {
        return Err("別の処理が実行中です。完了またはキャンセル後に再試行してください".into());
    }
    jobs.insert(job_id, Arc::clone(&control));
    Ok((job_id, control))
}

fn register_live_session(state: &AppState, running: Arc<AtomicBool>) -> Result<(), String> {
    // Use the same jobs-then-live lock order as `register_job`. Holding both
    // guards across validation and insertion makes the desktop's one-active-
    // operation contract atomic instead of allowing simultaneous file/live
    // registration to pass two independent observations.
    let jobs = state
        .jobs
        .lock()
        .map_err(|_| "ジョブ状態を取得できません")?;
    let mut live = state
        .live
        .lock()
        .map_err(|_| "ライブ状態を更新できません")?;
    if !jobs.is_empty() {
        return Err("ファイル処理の完了後に開始してください".into());
    }
    if live.is_some() {
        return Err("ライブ処理は既に実行中です".into());
    }
    *live = Some(running);
    Ok(())
}

fn checked_desktop_mib(value: Option<usize>, name: &str) -> Result<Option<u64>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value == 0 {
        return Err(format!("{name}は1 MiB以上にしてください"));
    }
    u64::try_from(value)
        .ok()
        .and_then(|value| value.checked_mul(BYTES_PER_MIB))
        .map(Some)
        .ok_or_else(|| format!("{name}が大きすぎます"))
}

fn desktop_resource_governor(
    options: &ProcessOptions,
    cpu_jobs: usize,
) -> Result<ResourceGovernor, String> {
    ResourceGovernor::new(
        ResourceLimits::new()
            .with_max_memory_bytes(checked_desktop_mib(
                options.max_process_memory_mb,
                "プロセスメモリ上限",
            )?)
            .with_max_temporary_bytes(checked_desktop_mib(
                options.max_temporary_mb,
                "一時領域上限",
            )?)
            .with_max_cpu_jobs(Some(cpu_jobs))
            .with_max_gpu_jobs(Some(options.max_gpu_jobs))
            .with_max_gpu_memory_bytes(checked_desktop_mib(
                options.max_gpu_memory_mb,
                "GPUメモリ上限",
            )?),
    )
}

fn desktop_decode_limits(options: &ProcessOptions) -> Result<DecodeLimits, String> {
    let maximum = checked_desktop_mib(options.max_process_memory_mb, "プロセスメモリ上限")?;
    Ok(DecodeLimits::new(
        denoize::metadata_limits_for_available_memory(maximum),
        maximum,
    ))
}

fn desktop_retained_metadata_limits(
    maximum: Option<u64>,
    retained_working_set_bytes: u64,
) -> MetadataLimits {
    denoize::metadata_limits_after_retained_memory(maximum, retained_working_set_bytes)
}

fn validate_process_options(options: &ProcessOptions) -> Result<(), String> {
    if !options.strength.is_finite() || !(0.0..=1.0).contains(&options.strength) {
        return Err("強度は0〜1の有限値で指定してください".into());
    }
    if let Some(target) = options.loudness_lufs {
        if !target.is_finite() || !(MIN_LOUDNESS_LUFS..=MAX_LOUDNESS_LUFS).contains(&target) {
            return Err("ラウドネスは-70〜0 LUFSの有限値で指定してください".into());
        }
    }
    if !options.true_peak_dbtp.is_finite()
        || !(MIN_TRUE_PEAK_DBTP..=MAX_TRUE_PEAK_DBTP).contains(&options.true_peak_dbtp)
    {
        return Err("True Peakは-20〜0 dBTPの有限値で指定してください".into());
    }
    if options.loudness_lufs.is_none() && options.true_peak_dbtp != -1.0 {
        return Err("True Peakはラウドネス正規化と一緒に指定してください".into());
    }
    checked_desktop_mib(options.max_process_memory_mb, "プロセスメモリ上限")?;
    checked_desktop_mib(options.max_temporary_mb, "一時領域上限")?;
    checked_desktop_mib(options.max_gpu_memory_mb, "GPUメモリ上限")?;
    if !(1..=32).contains(&options.max_gpu_jobs) {
        return Err("GPU並列数は1〜32にしてください".into());
    }
    if options.mp3_bitrate_kbps < 32 || options.aac_bitrate_kbps < 32 {
        return Err("ビットレートは32kbps以上にしてください".into());
    }
    checked_aac_bitrate_bps(options.aac_bitrate_kbps)?;
    let backend = configured_backend(&options.backend)?;
    if DownmixMode::parse(&options.downmix).is_none() {
        return Err("ダウンミックスは preserve または stereo を指定してください".into());
    }
    parse_aac_encoder(&options.aac_encoder)?;
    if backend.is_some_and(service::requires_external_model)
        && !(1..=MAX_MODEL_SAMPLE_RATE_HZ).contains(&options.onnx_sample_rate)
    {
        return Err(format!(
            "モデルのサンプルレートは1〜{MAX_MODEL_SAMPLE_RATE_HZ}Hzにしてください"
        ));
    }
    let mut backend_options = parsed_backend_options(options)?;
    if let Some(backend) = backend {
        if !service::requires_external_model(backend) {
            backend_options.onnx = None;
        }
        backend_options
            .validate_config(backend)
            .map_err(|error| error.to_string())?;
    }
    processing_config(options, VALIDATION_SAMPLE_RATE_HZ)?;
    Ok(())
}

fn validate_batch_request(request: &BatchRequest) -> Result<String, String> {
    validate_process_options(&request.options)?;
    if !(1..=32).contains(&request.jobs) {
        return Err("並列数は1〜32にしてください".into());
    }
    let extension = request
        .output_format
        .trim_start_matches('.')
        .to_ascii_lowercase();
    let probe = PathBuf::from(format!("output.{extension}"));
    let format = OutputFormat::from_path(&probe)?;
    parsed_encode_options(&request.options)?.validate_options(format)?;
    Ok(extension)
}

fn prepare_batch_request(request: &BatchRequest) -> Result<Vec<PreparedBatchItem>, String> {
    let extension = validate_batch_request(request)?;
    preflight_explicit_backend_resources(&request.options)?;
    if !Path::new(&request.output_dir).is_dir() {
        return Err("出力フォルダが存在しません".into());
    }
    let items = collect_batch_items(request, &extension)?;
    if items.is_empty() {
        return Err("対応する音声ファイルがありません".into());
    }
    validate_batch_control_paths(&items, Path::new(&request.output_dir))?;
    let governor = desktop_resource_governor(&request.options, request.jobs)?;
    preflight_batch_items(&request.options, request.resume, items, &governor)
}

fn desktop_preflight_decode_admission(
    options: &ProcessOptions,
    governor: &ResourceGovernor,
) -> Result<(DecodeLimits, Option<ResourcePermit>), String> {
    let Some(process_limit) =
        checked_desktop_mib(options.max_process_memory_mb, "プロセスメモリ上限")?
    else {
        return Ok((DecodeLimits::default(), None));
    };
    let available = process_limit
        .checked_sub(governor.usage()?.memory_bytes())
        .ok_or_else(|| "モデルセッションがプロセスメモリ上限を超えています".to_string())?;
    if available < BYTES_PER_MIB {
        return Err("モデル読込後の利用可能メモリが1 MiB未満です".into());
    }
    let permit = governor
        .try_acquire(ResourceRequest::new().with_memory_bytes(available))?
        .ok_or_else(|| "バッチ事前検査用メモリを予約できません".to_string())?;
    Ok((
        DecodeLimits::new(
            denoize::metadata_limits_for_available_memory(Some(available)),
            Some(available),
        ),
        Some(permit),
    ))
}

fn desktop_worker_decode_limit(
    options: &ProcessOptions,
    governor: &ResourceGovernor,
    transient_audio_bytes: u64,
) -> Result<Option<u64>, String> {
    let Some(process_limit) =
        checked_desktop_mib(options.max_process_memory_mb, "プロセスメモリ上限")?
    else {
        return Ok(None);
    };
    let available = process_limit
        .checked_sub(
            governor
                .usage()?
                .memory_bytes()
                .saturating_sub(transient_audio_bytes),
        )
        .ok_or_else(|| "モデルセッションがプロセスメモリ上限を超えています".to_string())?;
    if available < BYTES_PER_MIB {
        return Err("デコード用の利用可能メモリが1 MiB未満です".into());
    }
    Ok(Some(available))
}

fn preflight_batch_items(
    options: &ProcessOptions,
    resume_enabled: bool,
    items: Vec<BatchItem>,
    governor: &ResourceGovernor,
) -> Result<Vec<PreparedBatchItem>, String> {
    let encode = parsed_encode_options(options)?;
    let metadata_policy = if options.preserve_metadata {
        MetadataPolicy::Preserve
    } else {
        MetadataPolicy::Drop
    };
    let mut model_fingerprints = HashMap::<(PathBuf, u32), batch_resume::ConsumedModel>::new();
    let mut backend_sessions = Vec::<(
        Backend,
        BackendOptions,
        denoize::AcceleratorSelection,
        Arc<BackendSession>,
        Arc<ResourcePermit>,
    )>::new();
    let mut prepared = Vec::with_capacity(items.len());
    for item in items {
        let (decode_limits, preflight_decode_permit) =
            desktop_preflight_decode_admission(options, governor)?;
        let mut input_session = denoize::AudioInputSession::open(&item.input).map_err(|error| {
            format!("バッチ入力 {} を開けません: {error}", item.input.display())
        })?;
        let input_bytes = input_session.len();
        let input_fingerprint = batch_resume::fingerprint_input_session(&mut input_session)
            .map_err(|error| {
                format!(
                    "バッチ入力 {} の内容を確認できません: {error}",
                    item.input.display()
                )
            })?;
        let mut audio = read_audio_from_session_with_limits(&mut input_session, decode_limits)
            .map_err(|error| {
                format!(
                    "バッチ入力 {} を事前検査できません: {error}",
                    item.input.display()
                )
            })?;
        drop(preflight_decode_permit);
        let mut decoded_working_set = estimate_audio_working_set_bytes(&audio);
        let mut audio_permit = Some(
            governor
                .try_acquire(ResourceRequest::new().with_memory_bytes(decoded_working_set))?
                .ok_or_else(|| {
                    format!(
                        "バッチ入力 {} がモデルと同時にプロセスメモリ上限へ収まりません",
                        item.input.display()
                    )
                })?,
        );
        item.output_format
            .validate_config(&audio, &encode)
            .map_err(|error| {
                format!(
                    "バッチ出力 {} のcodec設定が不正です: {error}",
                    item.output.display()
                )
            })?;
        let processing = resolved_processing_options(options, &audio).map_err(|error| {
            format!(
                "バッチ入力 {} の処理設定が不正です: {error}",
                item.input.display()
            )
        })?;
        let model = match batch_resume::consumed_model_config(&processing)
            .map_err(|error| format!("使用するモデル設定を確認できません: {error}"))?
        {
            Some(config) => {
                let key = (config.path.clone(), config.sample_rate);
                let model = match model_fingerprints.get(&key) {
                    Some(model) => model.clone(),
                    None => {
                        let model = if resume_enabled {
                            batch_resume::fingerprint_resumable_model(config)
                        } else {
                            batch_resume::fingerprint_consumed_model(config)
                        }
                        .map_err(|error| {
                            format!("使用するモデルの内容を確認できません: {error}")
                        })?;
                        model_fingerprints.insert(key, model.clone());
                        model
                    }
                };
                Some(model)
            }
            None => None,
        };
        // Bind the prepared graph to the model bytes already captured by the
        // resume recipe. The whole-plan source fence below re-hashes the path
        // after graph preparation and rejects a persistent replacement.
        let (backend_session, backend_session_permit) = if let Some((_, _, _, session, permit)) =
            backend_sessions
                .iter()
                .find(|(backend, backend_options, accelerator, _, _)| {
                    *backend == processing.backend
                        && backend_options == &processing.backend_options
                        && *accelerator == processing.accelerator
                }) {
            (Arc::clone(session), Arc::clone(permit))
        } else {
            let permit = Arc::new(
                governor
                    .try_acquire(denoize::estimate_backend_session_request(
                        processing.backend,
                        &processing.backend_options,
                        processing.accelerator,
                    )?)?
                    .ok_or_else(|| {
                        format!(
                            "バッチ入力 {} のモデルがプロセス資源上限へ収まりません",
                            item.input.display()
                        )
                    })?,
            );
            let session = Arc::new(
                BackendSession::prepare_with_accelerator(
                    processing.backend,
                    processing.backend_options.clone(),
                    processing.accelerator,
                )
                .map_err(|error| {
                    format!(
                        "バッチ入力 {} のバックエンドを準備できません: {error}",
                        item.input.display()
                    )
                })?,
            );
            backend_sessions.push((
                processing.backend,
                processing.backend_options.clone(),
                processing.accelerator,
                Arc::clone(&session),
                Arc::clone(&permit),
            ));
            (session, permit)
        };
        let final_decode_limit =
            desktop_worker_decode_limit(options, governor, decoded_working_set)?;
        let must_redecode = match (decode_limits.max_working_set_bytes, final_decode_limit) {
            (Some(initial), Some(final_limit)) => final_limit < initial,
            (None, Some(_)) => true,
            _ => false,
        };
        if must_redecode {
            drop(audio_permit.take());
            drop(audio);
            let final_limit = final_decode_limit.expect("再デコードには有限の上限が必要です");
            let decode_permit = governor
                .try_acquire(ResourceRequest::new().with_memory_bytes(final_limit))?
                .ok_or_else(|| {
                    format!(
                        "バッチ入力 {} の最終デコード予算を予約できません",
                        item.input.display()
                    )
                })?;
            audio = read_audio_from_session_with_limits(
                &mut input_session,
                DecodeLimits::new(
                    denoize::metadata_limits_for_available_memory(final_decode_limit),
                    final_decode_limit,
                ),
            )
            .map_err(|error| {
                format!(
                    "モデル読込後にバッチ入力 {} をデコードできません: {error}",
                    item.input.display()
                )
            })?;
            drop(decode_permit);
            decoded_working_set = estimate_audio_working_set_bytes(&audio);
            audio_permit = Some(
                governor
                    .try_acquire(ResourceRequest::new().with_memory_bytes(decoded_working_set))?
                    .ok_or_else(|| {
                        format!(
                            "バッチ入力 {} のデコード済み音声を保持できません",
                            item.input.display()
                        )
                    })?,
            );
        }
        let final_decode_limits = DecodeLimits::new(
            denoize::metadata_limits_for_available_memory(final_decode_limit),
            final_decode_limit,
        );
        let metadata_limits =
            desktop_retained_metadata_limits(final_decode_limit, decoded_working_set);
        let metadata_bytes = if metadata_policy == MetadataPolicy::Preserve {
            input_session
                .read_metadata_with_limits(metadata_limits)?
                .as_ref()
                .map(denoize::metadata::Metadata::estimated_memory_bytes)
                .unwrap_or(0)
        } else {
            0
        };
        let resource_request = desktop_worker_request(
            input_bytes,
            &audio,
            metadata_bytes,
            final_decode_limit,
            &processing,
            true,
        )?;
        let recipe = batch_resume::recipe_digest(
            &processing,
            audio.channels(),
            item.output_format,
            encode,
            metadata_policy,
            model
                .as_ref()
                .map(|model| (&model.fingerprint, model.sample_rate)),
        )?;
        let expectation = ResumeExpectation::new(
            item.item_id,
            item.output.clone(),
            item.input.clone(),
            input_fingerprint,
            model,
            recipe,
        );
        drop(audio_permit);
        drop(governor.try_acquire(resource_request)?.ok_or_else(|| {
            format!(
                "バッチ入力 {} を設定された資源上限で実行できません",
                item.input.display()
            )
        })?);
        prepared.push(PreparedBatchItem {
            item,
            input_channels: audio.channels(),
            encode,
            metadata_policy,
            processing,
            backend_session,
            _backend_session_permit: backend_session_permit,
            governor: governor.clone(),
            resource_request,
            decode_limits: final_decode_limits,
            metadata_limits,
            expectation,
        });
    }
    for item in &prepared {
        item.expectation.verify_sources()?;
        drop(
            governor
                .try_acquire(item.resource_request)?
                .ok_or_else(|| {
                    format!(
                        "バッチ入力 {} は全モデル読込後のプロセス資源上限へ収まりません",
                        item.item.input.display()
                    )
                })?,
        );
    }
    Ok(prepared)
}

#[cfg(feature = "live")]
fn validate_live_request(request: &LiveRequest) -> Result<Backend, String> {
    if !(10..=2_000).contains(&request.chunk_ms) {
        return Err("チャンク長は10〜2000msにしてください".into());
    }
    validate_process_options(&request.options)?;
    let backend = if request.backend == "auto" {
        service::select_live_backend()
    } else {
        Backend::parse(&request.backend)
            .ok_or_else(|| format!("利用できないバックエンドです: {}", request.backend))?
    };
    if !denoize::live::backend_is_live_capable(backend) {
        return Err(format!(
            "ライブ処理に対応していないバックエンドです: {}",
            service::backend_name(backend)
        ));
    }
    parsed_backend_options_for(backend, &request.options)?
        .validate_config(backend)
        .map_err(|error| error.to_string())?;
    Ok(backend)
}

#[cfg(not(feature = "live"))]
#[allow(dead_code)]
fn validate_live_request(_request: &LiveRequest) -> Result<Backend, String> {
    Err("このビルドではライブ処理を利用できません".into())
}

fn parse_aac_encoder(value: &str) -> Result<AacEncoder, String> {
    match value {
        "oxide" => Ok(AacEncoder::Oxide),
        "fdk" => Ok(AacEncoder::Fdk),
        other => Err(format!("不明なAACエンコーダー: {other}")),
    }
}

fn checked_aac_bitrate_bps(bitrate_kbps: u32) -> Result<u32, String> {
    bitrate_kbps
        .checked_mul(1_000)
        .ok_or_else(|| "AACビットレートが大きすぎます".to_string())
}

fn parsed_encode_options(options: &ProcessOptions) -> Result<EncodeOptions, String> {
    Ok(EncodeOptions {
        mp3_bitrate_kbps: options.mp3_bitrate_kbps,
        m4a_bitrate_bps: checked_aac_bitrate_bps(options.aac_bitrate_kbps)?,
        aac_encoder: parse_aac_encoder(&options.aac_encoder)?,
        downmix: DownmixMode::parse(&options.downmix).ok_or_else(|| {
            "ダウンミックスは preserve または stereo を指定してください".to_string()
        })?,
    })
}

fn parsed_backend_options(options: &ProcessOptions) -> Result<BackendOptions, String> {
    Ok(BackendOptions {
        onnx: options.onnx_model.as_ref().map(|path| OnnxModelConfig {
            path: path.into(),
            sample_rate: options.onnx_sample_rate,
        }),
        channel_mode: ChannelMode::parse(&options.channel_mode)
            .ok_or_else(|| format!("不明なチャンネルモード: {}", options.channel_mode))?,
        sgmse_profile: SgmseProfile::parse(&options.sgmse_profile)
            .ok_or_else(|| format!("不明なSGMSEプロファイル: {}", options.sgmse_profile))?,
        accelerator: AcceleratorPreference::parse(&options.accelerator)
            .ok_or_else(|| format!("不明なアクセラレータ: {}", options.accelerator))?,
        deterministic: options.deterministic,
        seed: options.seed,
    })
}

fn configured_backend(value: &str) -> Result<Option<Backend>, String> {
    if value == "auto" {
        Ok(None)
    } else {
        Backend::parse(value)
            .map(Some)
            .ok_or_else(|| format!("このビルドでは利用できないバックエンドです: {value}"))
    }
}

fn parsed_backend_options_for(
    backend: Backend,
    options: &ProcessOptions,
) -> Result<BackendOptions, String> {
    let mut backend_options = parsed_backend_options(options)?;
    if !service::requires_external_model(backend) {
        backend_options.onnx = None;
    }
    Ok(backend_options)
}

fn resolve_gui_backend_options(
    backend: Backend,
    options: &ProcessOptions,
) -> Result<BackendOptions, String> {
    service::resolve_backend_options(backend, parsed_backend_options_for(backend, options)?)
}

fn preflight_explicit_backend_resources(options: &ProcessOptions) -> Result<(), String> {
    if let Some(backend) = configured_backend(&options.backend)? {
        let backend_options = resolve_gui_backend_options(backend, options)?;
        denoize::select_accelerator(
            backend,
            backend_options.accelerator,
            backend_options.deterministic,
        )?;
    }
    Ok(())
}

fn ensure_output_available(path: &Path, force: bool) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if force && (metadata.is_file() || metadata.file_type().is_symlink()) => {
            Ok(())
        }
        Ok(_) if force => Err(format!(
            "出力先は置換可能なファイルまたはシンボリックリンクではありません: {}",
            path.display()
        )),
        Ok(_) => Err("出力ファイルが既に存在します。「上書きを許可」を有効にしてください".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("出力先を確認できません: {error}")),
    }
}

fn validate_request(request: &ProcessRequest) -> Result<(), String> {
    validate_process_options(&request.options)?;
    let format = OutputFormat::from_path(Path::new(&request.output))?;
    if request.stream {
        if format != OutputFormat::Wav {
            return Err("長時間ストリームの出力形式はWAVにしてください".into());
        }
        if !(1..=denoize::config::MAX_STREAM_BLOCK_FRAMES).contains(&request.stream_frames) {
            return Err(format!(
                "ストリームブロックは1〜{} framesにしてください",
                denoize::config::MAX_STREAM_BLOCK_FRAMES
            ));
        }
        if request.options.vad {
            return Err("長時間ストリームではVADを無効にしてください".into());
        }
        if request.options.loudness_lufs.is_some() {
            return Err("長時間ストリームではラウドネス正規化を無効にしてください".into());
        }
        if request.options.downmix != "preserve" {
            return Err("長時間ストリームではチャンネルレイアウトを保持してください".into());
        }
        if let Some(backend) = configured_backend(&request.options.backend)? {
            if !StreamingBackendSession::supports(backend) {
                return Err(format!(
                    "バックエンド {} は長時間ストリームに対応していません",
                    service::backend_name(backend)
                ));
            }
        }
    } else {
        if request.resume {
            return Err("単一ファイルの再開には長時間ストリームを有効にしてください".into());
        }
        format.validate_encoder(parse_aac_encoder(&request.options.aac_encoder)?)?;
    }
    preflight_explicit_backend_resources(&request.options)?;
    if !Path::new(&request.input).is_file() {
        return Err("入力ファイルが存在しません".into());
    }
    if request.resume {
        Ok(())
    } else {
        ensure_output_available(Path::new(&request.output), request.options.force)
    }
}

fn validated_processing_options(
    options: &ProcessOptions,
    audio: &denoize::Audio,
) -> Result<ProcessingOptions, String> {
    let denoiser = processing_config(options, audio.sample_rate)?;
    let backend = match configured_backend(&options.backend)? {
        Some(backend) => BackendChoice::Explicit(backend),
        None => BackendChoice::Auto,
    };
    let selected_backend = service::select_backend(
        backend,
        audio.frames() as f64 / audio.sample_rate.max(1) as f64,
        None,
    );
    let processing = ProcessingOptions {
        backend,
        quality: None,
        denoiser,
        backend_options: parsed_backend_options_for(selected_backend, options)?,
        loudness_lufs: options.loudness_lufs,
        true_peak_dbtp: options.true_peak_dbtp,
    };
    processing
        .validate_config(audio)
        .map_err(|error| error.to_string())?;
    Ok(processing)
}

fn resolved_processing_options(
    options: &ProcessOptions,
    audio: &denoize::Audio,
) -> Result<service::ResolvedProcessingOptions, String> {
    service::resolve_processing_options(audio, validated_processing_options(options, audio)?)
}

fn desktop_worker_request(
    input_bytes: u64,
    audio: &denoize::Audio,
    metadata_bytes: u64,
    decode_reservation_bytes: Option<u64>,
    processing: &service::ResolvedProcessingOptions,
    writes_output: bool,
) -> Result<ResourceRequest, String> {
    let memory_bytes = estimate_audio_working_set_bytes(audio)
        .checked_add(metadata_bytes)
        .ok_or_else(|| "ワーカーメモリ予約量が大きすぎます".to_string())?
        .max(decode_reservation_bytes.unwrap_or(0));
    let mut request = ResourceRequest::worker(
        memory_bytes,
        if writes_output {
            denoize::estimate_temporary_bytes(input_bytes, audio)?
        } else {
            0
        },
    );
    if processing.accelerator.effective() != denoize::AcceleratorRuntime::Cpu {
        request = request
            .with_gpu_jobs(1)
            .with_gpu_memory_bytes(denoize::estimate_gpu_worker_bytes(audio)?);
    }
    Ok(request)
}

fn desktop_stream_temporary_bytes(
    info: AudioStreamInfo,
    configured_limit: Option<u64>,
    checkpointed: bool,
    metadata_allowance_bytes: u64,
) -> Result<u64, String> {
    const MAX_WAV_FILE_BYTES: u64 = u32::MAX as u64 + 8;
    let Some(frames) = info.total_frames else {
        if let Some(limit) = configured_limit {
            return Ok(limit);
        }
        if !checkpointed {
            return Ok(MAX_WAV_FILE_BYTES);
        }
        let data_limit = MAX_WAV_FILE_BYTES.saturating_sub(68);
        let output_sample_bytes = u64::from(info.output_spec.bits_per_sample / 8);
        let max_samples = data_limit / output_sample_bytes;
        let spool_bytes = max_samples
            .checked_mul(std::mem::size_of::<f64>() as u64)
            .ok_or_else(|| "ストリームチェックポイントの一時領域が大きすぎます".to_string())?;
        return MAX_WAV_FILE_BYTES
            .checked_add(spool_bytes)
            .ok_or_else(|| "ストリームチェックポイントの一時領域が大きすぎます".to_string());
    };
    let data_bytes = frames
        .checked_mul(u64::from(info.output_spec.channels))
        .and_then(|samples| samples.checked_mul(u64::from(info.output_spec.bits_per_sample / 8)))
        .ok_or_else(|| "ストリーム出力サイズが大きすぎます".to_string())?;
    let file_bytes = data_bytes
        .checked_add(68)
        .and_then(|bytes| bytes.checked_add(metadata_allowance_bytes))
        .ok_or_else(|| "ストリーム出力サイズが大きすぎます".to_string())?;
    if file_bytes > MAX_WAV_FILE_BYTES {
        return Err("ストリーム出力がRIFF WAVのサイズ上限を超えます".into());
    }
    if !checkpointed {
        return Ok(file_bytes);
    }
    let spool_bytes = frames
        .checked_mul(u64::from(info.output_spec.channels))
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<f64>() as u64))
        .ok_or_else(|| "ストリームチェックポイントのPCMサイズが大きすぎます".to_string())?;
    file_bytes
        .checked_add(spool_bytes)
        .ok_or_else(|| "ストリームチェックポイントの一時領域が大きすぎます".to_string())
}

fn replay_desktop_stream_checkpoint(
    reader: &mut AudioStreamReader,
    processor: &mut StreamingBackendSession,
    block_frames: usize,
    checkpoint: batch_resume::StreamCheckpoint,
    channels: usize,
    control: &JobControl,
) -> Result<u64, String> {
    let mut digest = batch_resume::StreamPcmDigest::new(channels)?;
    let mut input_frames = 0_u64;
    while input_frames < checkpoint.input_frames() {
        check_cancelled(control)?;
        let block = reader
            .next_block(block_frames)?
            .ok_or_else(|| "チェックポイントが入力音声の終端を超えています".to_string())?;
        let frames = block.first().map(Vec::len).unwrap_or(0) as u64;
        let next = input_frames
            .checked_add(frames)
            .ok_or_else(|| "ストリーム再生フレーム数が大きすぎます".to_string())?;
        if next > checkpoint.input_frames() {
            return Err("チェックポイントとデコーダーブロック境界が一致しません".into());
        }
        digest.update(&processor.process_block(&block)?)?;
        input_frames = next;
    }
    if digest.frames() != checkpoint.output_frames()
        || digest.len() != checkpoint.spool_len()
        || digest.digest() != checkpoint.spool_digest()
    {
        return Err(
            "再生したストリーム状態がチェックポイントと一致しません。上書きで再作成してください"
                .into(),
        );
    }
    Ok(input_frames)
}

fn process_stream_file(
    request: &ProcessRequest,
    control: &JobControl,
    progress: impl Fn(usize, &'static str),
) -> Result<ProcessFileResult, String> {
    check_cancelled(control)?;
    let input = Path::new(&request.input);
    let output = Path::new(&request.output);
    let maximum = checked_desktop_mib(request.options.max_process_memory_mb, "プロセスメモリ上限")?;
    let mut input_session = denoize::AudioInputSession::open(input)?;
    let initial_limits = desktop_decode_limits(&request.options)?;
    let initial_info = denoize::inspect_audio_stream_session(&mut input_session, initial_limits)?;
    let spec = initial_info.output_spec;
    let channel_mask = initial_info.channel_mask;
    let config = processing_config(&request.options, spec.sample_rate)?;
    if config.vad {
        return Err("長時間ストリームではVADを無効にしてください".into());
    }
    let backend =
        configured_backend(&request.options.backend)?.unwrap_or_else(service::select_live_backend);
    if !StreamingBackendSession::supports(backend) {
        return Err(format!(
            "バックエンド {} は長時間ストリームに対応していません",
            service::backend_name(backend)
        ));
    }
    let backend_options = resolve_gui_backend_options(backend, &request.options)?;
    let accelerator = denoize::select_accelerator(
        backend,
        backend_options.accelerator,
        backend_options.deterministic,
    )?;
    let base_working_set = estimate_stream_memory_bytes_checked(
        spec.channels as usize,
        request.stream_frames,
        config.frame_size,
        spec.sample_rate,
        config.profile_ms,
    )
    .map_err(|error| error.to_string())?;
    let backend_state = StreamingBackendSession::estimate_additional_bytes(
        backend,
        spec.sample_rate,
        spec.channels as usize,
        backend_options.channel_mode,
    )
    .map_err(|error| error.to_string())?;
    let checkpoint_scratch = if request.resume {
        batch_resume::STREAM_CHECKPOINT_SCRATCH_BYTES
    } else {
        0
    };
    let initial_working_set = base_working_set
        .checked_add(backend_state)
        .and_then(|bytes| bytes.checked_add(initial_info.decoder_additional_bytes))
        .and_then(|bytes| bytes.checked_add(checkpoint_scratch))
        .ok_or_else(|| "ストリームのメモリ予約量が大きすぎます".to_string())?;
    if maximum.is_some_and(|limit| initial_working_set > limit) {
        return Err(format!(
            "ストリームには{initial_working_set} bytes必要ですが、プロセスメモリ上限を超えます"
        ));
    }
    let metadata_limits = desktop_retained_metadata_limits(maximum, initial_working_set);
    let decode_limits = DecodeLimits::new(metadata_limits, maximum);
    let info = denoize::inspect_audio_stream_session(&mut input_session, decode_limits)?;
    if info.format != initial_info.format
        || info.codec != initial_info.codec
        || info.output_spec != initial_info.output_spec
        || info.channel_mask != initial_info.channel_mask
        || info.total_frames != initial_info.total_frames
        || info.max_decoder_frames != initial_info.max_decoder_frames
    {
        return Err("事前検査中にストリーム入力形状が変化しました".into());
    }
    let working_set = base_working_set
        .checked_add(backend_state)
        .and_then(|bytes| bytes.checked_add(info.decoder_additional_bytes))
        .and_then(|bytes| bytes.checked_add(checkpoint_scratch))
        .ok_or_else(|| "ストリームのメモリ予約量が大きすぎます".to_string())?;
    if maximum.is_some_and(|limit| working_set > limit) {
        return Err(format!(
            "ストリームには{working_set} bytes必要ですが、プロセスメモリ上限を超えます"
        ));
    }

    let resolved = service::ResolvedProcessingOptions {
        backend,
        denoiser: config.clone(),
        backend_options: backend_options.clone(),
        accelerator,
        loudness_lufs: None,
        true_peak_dbtp: -1.0,
    };
    resolved.validate_config()?;
    let resume_identity = if request.resume {
        let input_fingerprint = batch_resume::fingerprint_input_session(&mut input_session)?;
        let model = match batch_resume::consumed_model_config(&resolved)? {
            Some(config) => Some(batch_resume::fingerprint_resumable_model(config)?),
            None => None,
        };
        let base_recipe = batch_resume::recipe_digest(
            &resolved,
            spec.channels as usize,
            OutputFormat::Wav,
            EncodeOptions::default(),
            if request.options.preserve_metadata {
                MetadataPolicy::Preserve
            } else {
                MetadataPolicy::Drop
            },
            model
                .as_ref()
                .map(|model| (&model.fingerprint, model.sample_rate)),
        )?;
        let recipe = batch_resume::stream_recipe_digest(base_recipe, request.stream_frames, info)?;
        Some((input_fingerprint, recipe, model))
    } else {
        None
    };
    let metadata = if request.options.preserve_metadata {
        input_session.read_metadata_with_limits(metadata_limits)?
    } else {
        None
    };
    let metadata_bytes = metadata
        .as_ref()
        .map(denoize::metadata::Metadata::estimated_memory_bytes)
        .unwrap_or(0);
    let temporary_bytes = desktop_stream_temporary_bytes(
        info,
        checked_desktop_mib(request.options.max_temporary_mb, "一時領域上限")?,
        request.resume,
        metadata_bytes,
    )?;
    let governor = desktop_resource_governor(&request.options, 1)?;
    let mut worker_request = ResourceRequest::worker(
        working_set
            .checked_add(metadata_bytes)
            .ok_or_else(|| "ストリームのメモリ予約量が大きすぎます".to_string())?,
        temporary_bytes,
    );
    if accelerator.effective() != denoize::AcceleratorRuntime::Cpu {
        worker_request = worker_request.with_gpu_jobs(1).with_gpu_memory_bytes(
            working_set
                .checked_mul(2)
                .ok_or_else(|| "ストリームのGPU予約量が大きすぎます".to_string())?,
        );
    }
    let request_resources = worker_request.checked_add(
        denoize::estimate_backend_session_request(backend, &backend_options, accelerator)?,
    )?;
    let _permit = governor
        .acquire_with_cancel(request_resources, || control.is_cancelled())
        .map_err(|error| {
            if control.is_cancelled() {
                "cancelled".to_string()
            } else {
                error
            }
        })?;
    let mut processor = StreamingBackendSession::new_with_accelerator(
        backend,
        spec.sample_rate,
        spec.channels as usize,
        config,
        backend_options,
        accelerator,
    )?;
    let mut reader = AudioStreamReader::from_session(input_session, decode_limits)?;
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("出力フォルダを作成できません: {error}"))?;
    }
    progress(1, "ストリームを処理しています");
    let commit_mode = if request.options.force {
        CommitMode::Replace
    } else {
        CommitMode::NoClobber
    };

    if let Some((input_fingerprint, recipe, model)) = resume_identity {
        if let Some(model) = model.as_ref() {
            if batch_resume::fingerprint_file(&model.path)? != model.fingerprint {
                return Err(format!(
                    "準備中にストリームモデルが変更されました: {}",
                    model.path.display()
                ));
            }
        }
        let acquired = batch_resume::StreamCheckpointSession::acquire(
            output,
            input_fingerprint,
            recipe,
            spec,
            request.stream_frames,
            Some(temporary_bytes),
            request.options.force,
        )?;
        let (mut checkpoint, loaded) = match acquired {
            batch_resume::StreamCheckpointAcquire::Completed(_) => {
                progress(4, "既存の完了済み出力を確認しました");
                return Ok(ProcessFileResult {
                    output: output.to_string_lossy().into_owned(),
                    accelerator,
                });
            }
            batch_resume::StreamCheckpointAcquire::Active(checkpoint, loaded) => {
                (checkpoint, loaded)
            }
        };
        let mut input_frames = match loaded {
            Some(saved) => replay_desktop_stream_checkpoint(
                &mut reader,
                &mut processor,
                request.stream_frames,
                saved,
                spec.channels as usize,
                control,
            )?,
            None => 0,
        };
        let mut next_checkpoint = input_frames
            .checked_div(STREAM_CHECKPOINT_FRAMES)
            .and_then(|multiple| multiple.checked_add(1))
            .and_then(|multiple| multiple.checked_mul(STREAM_CHECKPOINT_FRAMES))
            .unwrap_or(u64::MAX);
        while let Some(block) = reader.next_block(request.stream_frames)? {
            check_cancelled(control)?;
            let frames = block.first().map(Vec::len).unwrap_or(0) as u64;
            checkpoint.append_block(&processor.process_block(&block)?)?;
            input_frames = input_frames
                .checked_add(frames)
                .ok_or_else(|| "ストリーム入力フレーム数が大きすぎます".to_string())?;
            if input_frames >= next_checkpoint {
                checkpoint.checkpoint(input_frames)?;
                next_checkpoint = input_frames
                    .checked_div(STREAM_CHECKPOINT_FRAMES)
                    .and_then(|multiple| multiple.checked_add(1))
                    .and_then(|multiple| multiple.checked_mul(STREAM_CHECKPOINT_FRAMES))
                    .unwrap_or(u64::MAX);
            }
        }
        checkpoint.append_block(&processor.finish()?)?;
        if reader.fingerprint_input()? != input_fingerprint {
            return Err("処理中にストリーム入力が変更されました".into());
        }
        check_cancelled(control)?;
        progress(3, "WAV出力を準備しています");
        checkpoint.prepare_spool_read()?;
        let mut transaction = AtomicOutput::new(output)?;
        {
            let mut writer =
                WavStreamWriter::from_sink(BufWriter::new(transaction.file_mut()), spec)?;
            while let Some(block) = checkpoint.next_spool_block(request.stream_frames)? {
                check_cancelled(control)?;
                writer.write_block(&block)?;
            }
            writer.finalize()?;
        }
        denoize::audio::write_wav_channel_mask_to_file(
            transaction.file_mut(),
            spec.channels as usize,
            channel_mask,
        )?;
        if let Some(metadata) = metadata {
            denoize::metadata::write_extended_to_file_with_limits(
                metadata,
                transaction.file_mut(),
                metadata_limits,
            )?;
        }
        let staged_bytes = transaction
            .file_mut()
            .metadata()
            .map_err(|error| format!("一時出力のサイズを確認できません: {error}"))?
            .len();
        let combined = staged_bytes
            .checked_add(checkpoint.spool_len())
            .ok_or_else(|| "ストリームの一時領域が大きすぎます".to_string())?;
        if combined > temporary_bytes {
            return Err(format!(
                "チェックポイントと一時出力が予約量を超えました: {combined} > {temporary_bytes} bytes"
            ));
        }
        let output_fingerprint =
            batch_resume::fingerprint_open_file_at(transaction.file_mut(), output)?;
        checkpoint.prepare_publish(input_frames, output_fingerprint)?;
        progress(4, "出力を確定しています");
        control.commit(transaction, commit_mode)?;
        if let Err(error) = checkpoint.cleanup() {
            eprintln!("denoize desktop: checkpoint cleanup failed after commit: {error}");
        }
    } else {
        let mut transaction = AtomicOutput::new(output)?;
        {
            let mut writer =
                WavStreamWriter::from_sink(BufWriter::new(transaction.file_mut()), spec)?;
            while let Some(block) = reader.next_block(request.stream_frames)? {
                check_cancelled(control)?;
                writer.write_block(&processor.process_block(&block)?)?;
            }
            writer.write_block(&processor.finish()?)?;
            writer.finalize()?;
        }
        denoize::audio::write_wav_channel_mask_to_file(
            transaction.file_mut(),
            spec.channels as usize,
            channel_mask,
        )?;
        if let Some(metadata) = metadata {
            denoize::metadata::write_extended_to_file_with_limits(
                metadata,
                transaction.file_mut(),
                metadata_limits,
            )?;
        }
        let staged_bytes = transaction
            .file_mut()
            .metadata()
            .map_err(|error| format!("一時出力のサイズを確認できません: {error}"))?
            .len();
        if staged_bytes > temporary_bytes {
            return Err(format!(
                "一時出力が予約量を超えました: {staged_bytes} > {temporary_bytes} bytes"
            ));
        }
        progress(4, "出力を確定しています");
        control.commit(transaction, commit_mode)?;
    }
    Ok(ProcessFileResult {
        output: output.to_string_lossy().into_owned(),
        accelerator,
    })
}

fn process_file(
    request: &ProcessRequest,
    control: &JobControl,
    progress: impl Fn(usize, &'static str),
) -> Result<ProcessFileResult, String> {
    if request.stream {
        return process_stream_file(request, control, progress);
    }
    check_cancelled(control)?;
    let input = Path::new(&request.input);
    let output = Path::new(&request.output);
    let mut input_session = denoize::AudioInputSession::open(input)?;
    let input_bytes = input_session.len();
    let decode_limits = desktop_decode_limits(&request.options)?;
    let mut audio = read_audio_from_session_with_limits(&mut input_session, decode_limits)?;
    let decoded_working_set = estimate_audio_working_set_bytes(&audio);
    let metadata_limits =
        desktop_retained_metadata_limits(decode_limits.max_working_set_bytes, decoded_working_set);
    let metadata = if request.options.preserve_metadata {
        input_session.read_metadata_with_limits(metadata_limits)?
    } else {
        None
    };
    let metadata_bytes = metadata
        .as_ref()
        .map(denoize::metadata::Metadata::estimated_memory_bytes)
        .unwrap_or(0);
    let encode = parsed_encode_options(&request.options)?;
    let format = OutputFormat::from_path(output)?;
    format.validate_config(&audio, &encode)?;
    progress(1, "ノイズ除去を実行しています");
    check_cancelled(control)?;
    let processing = resolved_processing_options(&request.options, &audio)?;
    let governor = desktop_resource_governor(&request.options, 1)?;
    let worker_request =
        desktop_worker_request(input_bytes, &audio, metadata_bytes, None, &processing, true)?;
    let resource_request =
        worker_request.checked_add(denoize::estimate_backend_session_request(
            processing.backend,
            &processing.backend_options,
            processing.accelerator,
        )?)?;
    let _permit = governor
        .acquire_with_cancel(resource_request, || control.is_cancelled())
        .map_err(|error| {
            if control.is_cancelled() {
                "cancelled".to_string()
            } else {
                error
            }
        })?;
    let backend_session = BackendSession::prepare_with_accelerator(
        processing.backend,
        processing.backend_options.clone(),
        processing.accelerator,
    )?;
    progress(2, "ラウドネスと出力を準備しています");
    let processing_result =
        service::process_audio_resolved_with_session(&mut audio, &processing, &backend_session)?;
    check_cancelled(control)?;
    progress(3, "ファイルを書き出しています");
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("出力フォルダを作成できません: {error}"))?;
    }
    let mut transaction = AtomicOutput::new(output)?;
    write_audio_to_file(transaction.file_mut(), format, &audio, encode)?;
    if let Some(metadata) = metadata {
        denoize::metadata::write_extended_to_file_with_limits(
            metadata,
            transaction.file_mut(),
            metadata_limits,
        )?;
    }
    let staged_bytes = transaction
        .file_mut()
        .metadata()
        .map_err(|error| format!("一時出力のサイズを確認できません: {error}"))?
        .len();
    if staged_bytes > worker_request.temporary_bytes() {
        return Err(format!(
            "一時出力が予約量を超えました: {staged_bytes} > {} bytes",
            worker_request.temporary_bytes()
        ));
    }
    progress(4, "出力を確定しています");
    let commit_mode = if request.options.force {
        CommitMode::Replace
    } else {
        CommitMode::NoClobber
    };
    control.commit(transaction, commit_mode)?;
    Ok(ProcessFileResult {
        output: output.to_string_lossy().into_owned(),
        accelerator: processing_result.accelerator,
    })
}

fn stage_batch_output(
    input: &Path,
    output: &Path,
    format: OutputFormat,
    encode: EncodeOptions,
    metadata_policy: MetadataPolicy,
    processing: &service::ResolvedProcessingOptions,
    backend_session: &BackendSession,
    decode_limits: DecodeLimits,
    metadata_limits: MetadataLimits,
    temporary_reservation_bytes: u64,
    control: &JobControl,
) -> Result<AtomicOutput, String> {
    check_cancelled(control)?;
    let mut input_session = denoize::AudioInputSession::open(input)?;
    let mut audio = read_audio_from_session_with_limits(&mut input_session, decode_limits)?;
    let metadata = if metadata_policy == MetadataPolicy::Preserve {
        input_session.read_metadata_with_limits(metadata_limits)?
    } else {
        None
    };
    format.validate_config(&audio, &encode)?;
    check_cancelled(control)?;
    service::process_audio_resolved_with_session(&mut audio, processing, backend_session)?;
    check_cancelled(control)?;
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("出力フォルダを作成できません: {error}"))?;
    }
    let mut transaction = AtomicOutput::new(output)?;
    write_audio_to_file(transaction.file_mut(), format, &audio, encode)?;
    if let Some(metadata) = metadata {
        denoize::metadata::write_extended_to_file_with_limits(
            metadata,
            transaction.file_mut(),
            metadata_limits,
        )?;
    }
    let staged_bytes = transaction
        .file_mut()
        .metadata()
        .map_err(|error| format!("一時出力のサイズを確認できません: {error}"))?
        .len();
    if staged_bytes > temporary_reservation_bytes {
        return Err(format!(
            "一時出力が予約量を超えました: {staged_bytes} > {temporary_reservation_bytes} bytes"
        ));
    }
    Ok(transaction)
}

fn verify_prepared_batch_recipe(item: &PreparedBatchItem) -> Result<(), String> {
    let recipe = batch_resume::recipe_digest(
        &item.processing,
        item.input_channels,
        item.item.output_format,
        item.encode,
        item.metadata_policy,
        item.expectation
            .model()
            .map(|model| (&model.fingerprint, model.sample_rate)),
    )?;
    if recipe != item.expectation.recipe() {
        return Err(format!(
            "バッチ出力 {} の有効な処理設定が事前検査後に変化しました",
            item.item.output.display()
        ));
    }
    Ok(())
}

fn processing_config(options: &ProcessOptions, sample_rate: u32) -> Result<DenoiserConfig, String> {
    let mut config = match options.preset.as_deref() {
        Some("") | None => DenoiserConfig::default(sample_rate),
        Some(value) => Preset::parse(value)
            .ok_or_else(|| format!("不明なプリセット: {value}"))?
            .config(sample_rate),
    };
    if let Some(mode) = options.mode.as_deref().filter(|value| !value.is_empty()) {
        ProcessingMode::parse(mode)
            .ok_or_else(|| format!("不明な処理モード: {mode}"))?
            .apply(&mut config);
    }
    config.strength = options.strength;
    config.adaptive_noise = options.adaptive_noise;
    config.vad = options.vad;
    config
        .validate_config()
        .map_err(|error| error.to_string())?;
    Ok(config)
}

fn check_cancelled(control: &JobControl) -> Result<(), String> {
    if control.is_cancelled() {
        Err("cancelled".into())
    } else {
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_progress(
    app: &AppHandle,
    job_id: u64,
    kind: &'static str,
    status: &'static str,
    message: &str,
    current: usize,
    total: usize,
    started: Instant,
    output: Option<String>,
    error: Option<String>,
) {
    emit_progress_with_accelerator(
        app, job_id, kind, status, message, current, total, started, output, error, None,
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_progress_with_accelerator(
    app: &AppHandle,
    job_id: u64,
    kind: &'static str,
    status: &'static str,
    message: &str,
    current: usize,
    total: usize,
    started: Instant,
    output: Option<String>,
    error: Option<String>,
    accelerator: Option<denoize::AcceleratorSelection>,
) {
    let _ = app.emit(
        "job-progress",
        JobProgress {
            job_id,
            kind,
            status,
            message: message.into(),
            current,
            total,
            fraction: current as f64 / total.max(1) as f64,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            output,
            error,
            eta_seconds: None,
            item: None,
            item_status: None,
            item_id: None,
            resume_reason: None,
            accelerator: accelerator.map(accelerator_result),
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_batch_item(
    app: &AppHandle,
    job_id: u64,
    item_status: &'static str,
    item: &BatchItem,
    resume_reason: Option<batch_resume::ResumeReason>,
    current: usize,
    total: usize,
    started: Instant,
    error: Option<String>,
    accelerator: denoize::AcceleratorSelection,
) {
    let elapsed = started.elapsed().as_secs_f64();
    let eta =
        (current > 0).then(|| elapsed / current as f64 * total.saturating_sub(current) as f64);
    let name = item
        .input
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("audio");
    let _ = app.emit(
        "job-progress",
        JobProgress {
            job_id,
            kind: "batch",
            status: "running",
            message: format!("{name}: {item_status}"),
            current,
            total,
            fraction: current as f64 / total.max(1) as f64,
            elapsed_seconds: elapsed,
            output: Some(item.output.to_string_lossy().into_owned()),
            error,
            eta_seconds: eta,
            item: Some(item.input.to_string_lossy().into_owned()),
            item_status: Some(item_status),
            item_id: Some(item.item_id.as_hex()),
            resume_reason: resume_reason.map(batch_resume::ResumeReason::as_str),
            accelerator: Some(accelerator_result(accelerator)),
        },
    );
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            app_info,
            start_process,
            start_batch,
            cancel_job,
            live_devices,
            start_live,
            stop_live,
            compare_audio,
            list_models,
            model_catalog_status,
            update_model_catalog,
            inspect_model_bundle,
            import_model_bundle,
            recover_model_trust_root,
            reset_model_trust_time_floor,
            model_cache_doctor,
            save_automation_snapshot,
            prune_model_cache,
            model_action,
            prepare_preview,
            load_gui_config,
            save_gui_config,
            classify_dropped_paths,
            save_text_file
        ])
        .setup(|app| {
            let preview_dir = std::env::temp_dir().join("denoize-previews");
            let _ = std::fs::remove_dir_all(&preview_dir);
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_title(&format!("denoize {}", env!("CARGO_PKG_VERSION")));
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("failed to run denoize desktop");
}

#[cfg(test)]
mod tests {
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
        "885397b802157648c4e085c30d81e0bb47568d2cb6cfc88c842d8dab9b0cd504";

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn create(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "denoize-gui-{label}-{}-{}",
                std::process::id(),
                NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self { path }
        }

        fn join(&self, name: &str) -> PathBuf {
            self.path.join(name)
        }

        fn assert_no_staged_outputs(&self) {
            let staged: Vec<_> = std::fs::read_dir(&self.path)
                .unwrap()
                .filter_map(Result::ok)
                .map(|entry| entry.file_name())
                .filter(|name| name.to_string_lossy().starts_with(".denoize-"))
                .collect();
            assert!(staged.is_empty(), "staged outputs remain: {staged:?}");
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn desktop_automation_json_export_is_one_atomic_replacement() {
        let directory = TestDirectory::create("automation-json");
        let output = directory.join("denoize-automation.json");
        std::fs::write(&output, b"old contents").unwrap();
        let json = "{\"schema\":\"denoize-automation-v1\",\"schema_version\":1}\n";

        write_automation_json(&output, json).unwrap();

        assert_eq!(std::fs::read_to_string(&output).unwrap(), json);
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn desktop_offline_bundle_commands_fail_closed_on_invalid_inputs() {
        let directory = TestDirectory::create("invalid-offline-bundle");
        let missing = directory.join("missing.dmb");
        let inspect_error = tauri::async_runtime::block_on(inspect_model_bundle(
            missing.to_string_lossy().into_owned(),
        ))
        .unwrap_err();
        assert!(inspect_error.contains("file not found"), "{inspect_error}");

        let import_error = tauri::async_runtime::block_on(import_model_bundle(
            missing.to_string_lossy().into_owned(),
            "not-a-sha256".into(),
        ))
        .unwrap_err();
        assert!(
            import_error.contains("expected offline bundle SHA-256"),
            "{import_error}"
        );
    }

    fn write_test_wav(path: &Path) {
        let mut samples = Vec::with_capacity(3_200);
        for index in 0..1_600 {
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
        wav.extend(16_000_u32.to_le_bytes());
        wav.extend(32_000_u32.to_le_bytes());
        wav.extend(2_u16.to_le_bytes());
        wav.extend(16_u16.to_le_bytes());
        wav.extend(b"data");
        wav.extend((samples.len() as u32).to_le_bytes());
        wav.extend(samples);
        std::fs::write(path, wav).unwrap();
    }

    fn write_test_stereo_wav(path: &Path) {
        let frames = 960_u32;
        let samples = vec![0_u8; frames as usize * 2 * 2];
        let mut wav = Vec::with_capacity(44 + samples.len());
        wav.extend(b"RIFF");
        wav.extend((36_u32 + samples.len() as u32).to_le_bytes());
        wav.extend(b"WAVEfmt ");
        wav.extend(16_u32.to_le_bytes());
        wav.extend(1_u16.to_le_bytes());
        wav.extend(2_u16.to_le_bytes());
        wav.extend(48_000_u32.to_le_bytes());
        wav.extend(192_000_u32.to_le_bytes());
        wav.extend(4_u16.to_le_bytes());
        wav.extend(16_u16.to_le_bytes());
        wav.extend(b"data");
        wav.extend((samples.len() as u32).to_le_bytes());
        wav.extend(samples);
        std::fs::write(path, wav).unwrap();
    }

    fn classical_options(force: bool) -> ProcessOptions {
        let mut options = options();
        options.backend = "classical".into();
        options.preserve_metadata = false;
        options.force = force;
        options
    }

    fn process_request(input: &Path, output: &Path, options: ProcessOptions) -> ProcessRequest {
        ProcessRequest {
            input: input.to_string_lossy().into_owned(),
            output: output.to_string_lossy().into_owned(),
            stream: false,
            resume: false,
            stream_frames: DEFAULT_STREAM_BLOCK_FRAMES,
            options,
        }
    }

    fn options() -> ProcessOptions {
        ProcessOptions {
            backend: "auto".into(),
            preset: Some("hifi".into()),
            mode: Some("music".into()),
            strength: 0.4,
            adaptive_noise: false,
            vad: false,
            channel_mode: "linked".into(),
            downmix: "preserve".into(),
            loudness_lufs: None,
            true_peak_dbtp: -1.0,
            preserve_metadata: true,
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

    fn gui_config() -> GuiConfig {
        GuiConfig {
            backend: "auto".into(),
            preset: "hifi".into(),
            mode: "music".into(),
            strength: 0.4,
            adaptive_noise: false,
            vad: false,
            channels: "linked".into(),
            downmix: "preserve".into(),
            loudness_lufs: None,
            true_peak_dbtp: None,
            preserve_metadata: true,
            force: false,
            mp3_bitrate_kbps: 192,
            m4a_bitrate_kbps: 192,
            aac_encoder: "oxide".into(),
            onnx_model: None,
            onnx_rate: 16_000,
            sgmse_profile: "balanced".into(),
            accelerator: "cpu".into(),
            deterministic: false,
            max_process_memory_mb: None,
            max_temporary_mb: None,
            max_gpu_memory_mb: None,
            max_gpu_jobs: 1,
        }
    }

    fn gui_config_source() -> String {
        toml::to_string_pretty(&gui_config()).unwrap()
    }

    fn batch_request() -> BatchRequest {
        BatchRequest {
            inputs: Vec::new(),
            input_dir: None,
            output_dir: "missing-output-directory".into(),
            output_format: "wav".into(),
            recursive: false,
            jobs: 1,
            resume: false,
            options: options(),
        }
    }

    fn test_batch_item(input: PathBuf, output: PathBuf, id: u8) -> BatchItem {
        BatchItem {
            input,
            output_format: OutputFormat::from_path(&output).unwrap_or(OutputFormat::Wav),
            output,
            item_id: Digest::from_bytes([id; 32]),
        }
    }

    fn desktop_batch_fixture(directory: &TestDirectory, resume: bool, force: bool) -> BatchRequest {
        let input = directory.join("input");
        let output = directory.join("output");
        std::fs::create_dir_all(&input).unwrap();
        std::fs::create_dir_all(&output).unwrap();
        let source = input.join("sample.wav");
        if !source.exists() {
            write_test_wav(&source);
        }
        BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_string_lossy().into_owned()),
            output_dir: output.to_string_lossy().into_owned(),
            output_format: "wav".into(),
            recursive: false,
            jobs: 1,
            resume,
            options: classical_options(force),
        }
    }

    #[test]
    fn desktop_batch_recipe_matches_the_frontend_parity_golden_vector() {
        let directory = TestDirectory::create("frontend-parity-golden");
        let input = directory.join("input");
        let output = directory.join("output");
        std::fs::create_dir(&input).unwrap();
        std::fs::create_dir(&output).unwrap();
        let source = input.join("stereo.wav");
        write_test_stereo_wav(&source);
        let config = parse_gui_config(
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
force = false
mp3_bitrate_kbps = 256
m4a_bitrate_kbps = 224
aac_encoder = "oxide"
sgmse_profile = "balanced"
deterministic = false
"#,
            gui_config(),
        )
        .unwrap();
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_string_lossy().into_owned()),
            output_dir: output.to_string_lossy().into_owned(),
            output_format: "mp3".into(),
            recursive: false,
            jobs: 1,
            resume: true,
            options: config.process_options(),
        };
        let prepared = prepare_batch_request(&request).unwrap();
        let prepared = &prepared[0];

        assert_eq!(prepared.processing.backend, Backend::Classical);
        assert!(!prepared.processing.denoiser.adaptive_noise);
        assert!(!prepared.processing.denoiser.vad);
        assert_eq!(
            prepared.processing.backend_options.channel_mode,
            ChannelMode::StereoLinked
        );
        assert_eq!(prepared.processing.loudness_lufs, Some(-16.0));
        assert_eq!(prepared.processing.true_peak_dbtp, -1.0);
        assert_eq!(prepared.input_channels, 2);
        assert_eq!(prepared.item.output_format, OutputFormat::Mp3);
        assert_eq!(prepared.encode.mp3_bitrate_kbps, 256);
        assert_eq!(prepared.metadata_policy, MetadataPolicy::Drop);
        assert!(prepared.expectation.model().is_none());
        assert_eq!(
            prepared.expectation.recipe().as_hex(),
            FRONTEND_PARITY_RECIPE_HEX
        );
        assert_eq!(prepared.expectation.item_id(), prepared.item.item_id);

        let fixed_item_id = batch_resume::item_identity(
            Path::new("/denoize/frontend-parity/input/stereo.wav"),
            Path::new("stereo.wav"),
            Path::new("stereo.mp3"),
            OutputFormat::Mp3,
        );
        assert_eq!(fixed_item_id.as_hex(), FRONTEND_PARITY_ITEM_ID_HEX);
    }

    fn publish_planned_item(session: &BatchSession, item: &PlannedBatchItem) -> Result<(), String> {
        let ResumeDecision::Process { commit_mode, .. } = item.decision else {
            return Err("test item unexpectedly planned as a skip".into());
        };
        let control = JobControl::default();
        let transaction = stage_batch_output(
            &item.prepared.item.input,
            &item.prepared.item.output,
            item.prepared.item.output_format,
            item.prepared.encode,
            item.prepared.metadata_policy,
            &item.prepared.processing,
            &item.prepared.backend_session,
            item.prepared.decode_limits,
            item.prepared.metadata_limits,
            item.prepared.resource_request.temporary_bytes(),
            &control,
        )?;
        verify_prepared_batch_recipe(&item.prepared)?;
        control.commit_fence(|| {
            session
                .publish(&item.prepared.expectation, transaction, commit_mode)
                .map(|_| ())
        })
    }

    fn complete_desktop_batch(request: &BatchRequest) {
        let prepared = prepare_batch_request(request).unwrap();
        let session =
            BatchSession::acquire(Path::new(&request.output_dir), request.resume).unwrap();
        let planned = plan_batch_items(&session, prepared, request.options.force).unwrap();
        session.activate().unwrap();
        for item in &planned {
            if matches!(item.decision, ResumeDecision::Process { .. }) {
                publish_planned_item(&session, item).unwrap();
            }
        }
    }

    fn live_request() -> LiveRequest {
        LiveRequest {
            input_device: None,
            output_device: None,
            chunk_ms: 20,
            backend: "auto".into(),
            options: options(),
        }
    }

    #[test]
    fn gui_options_build_a_valid_processing_configuration() {
        let config = processing_config(&options(), 48_000).unwrap();
        assert_eq!(config.strength, 0.4);
        assert!(config.transient_protect);
        let selected = service::select_backend(BackendChoice::Auto, 30.0, None);
        assert_eq!(
            Backend::parse(service::backend_name(selected)),
            Some(selected)
        );
    }

    #[test]
    fn invalid_backend_is_rejected() {
        assert!(Backend::parse("missing").is_none());
    }

    #[test]
    fn app_info_reports_named_backend_model_rates() {
        let info = app_info();
        assert_eq!(info.accelerators.len(), 3);
        assert_eq!(info.accelerators[0].name, "cpu");
        assert!(info.accelerators[0].compiled);
        assert!(info.accelerators[0].available);
        assert!(info.accelerators[0].device.is_none());
        assert!(info.accelerators[0].memory_bytes.is_none());
        assert!(info.accelerators[0].compute_capability.is_none());
        assert_eq!(info.accelerators[1].name, "metal");
        assert_eq!(info.accelerators[2].name, "cuda");
        for (name, expected_rate) in [
            ("mpsenet", 16_000),
            ("sgmse", 16_000),
            ("gtcrn", 16_000),
            ("bsrnn", 48_000),
            ("mossformer2", 48_000),
        ] {
            if let Some(backend) = info.backends.iter().find(|backend| backend.name == name) {
                assert_eq!(backend.sample_rate, Some(expected_rate), "{name}");
            }
        }
    }

    #[test]
    fn process_options_reject_non_finite_numbers() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut process = options();
            process.strength = value;
            assert!(validate_process_options(&process)
                .unwrap_err()
                .contains("強度"));

            let mut process = options();
            process.loudness_lufs = Some(value);
            assert!(validate_process_options(&process)
                .unwrap_err()
                .contains("ラウドネス"));

            let mut process = options();
            process.true_peak_dbtp = value;
            assert!(validate_process_options(&process)
                .unwrap_err()
                .contains("True Peak"));
        }
    }

    #[test]
    fn process_numeric_bounds_are_inclusive() {
        for strength in [0.0, 1.0] {
            let mut process = options();
            process.strength = strength;
            validate_process_options(&process).unwrap();
        }
        for target in [MIN_LOUDNESS_LUFS, MAX_LOUDNESS_LUFS] {
            let mut process = options();
            process.loudness_lufs = Some(target);
            validate_process_options(&process).unwrap();
        }
        for peak in [MIN_TRUE_PEAK_DBTP, MAX_TRUE_PEAK_DBTP] {
            let mut process = options();
            process.loudness_lufs = Some(-23.0);
            process.true_peak_dbtp = peak;
            validate_process_options(&process).unwrap();
        }
        if Backend::parse("onnx").is_some() {
            for sample_rate in [1, MAX_MODEL_SAMPLE_RATE_HZ] {
                let mut process = options();
                process.backend = "onnx".into();
                process.onnx_model = Some("model.onnx".into());
                process.onnx_sample_rate = sample_rate;
                validate_process_options(&process).unwrap();
            }
        }
    }

    #[test]
    fn process_numeric_values_outside_bounds_are_rejected() {
        for strength in [-f64::EPSILON, 1.0 + f64::EPSILON] {
            let mut process = options();
            process.strength = strength;
            assert!(validate_process_options(&process).is_err());
        }
        for target in [MIN_LOUDNESS_LUFS - 0.1, MAX_LOUDNESS_LUFS + 0.1] {
            let mut process = options();
            process.loudness_lufs = Some(target);
            assert!(validate_process_options(&process).is_err());
        }
        for peak in [MIN_TRUE_PEAK_DBTP - 0.1, MAX_TRUE_PEAK_DBTP + 0.1] {
            let mut process = options();
            process.loudness_lufs = Some(-23.0);
            process.true_peak_dbtp = peak;
            assert!(validate_process_options(&process).is_err());
        }
        if Backend::parse("onnx").is_some() {
            for sample_rate in [0, MAX_MODEL_SAMPLE_RATE_HZ + 1] {
                let mut process = options();
                process.backend = "onnx".into();
                process.onnx_model = Some("model.onnx".into());
                process.onnx_sample_rate = sample_rate;
                assert!(validate_process_options(&process)
                    .unwrap_err()
                    .contains("サンプルレート"));
            }
        }
        let mut process = options();
        process.aac_bitrate_kbps = u32::MAX;
        assert!(validate_process_options(&process)
            .unwrap_err()
            .contains("ビットレートが大きすぎます"));
    }

    #[test]
    fn true_peak_requires_loudness_normalization() {
        let mut process = options();
        process.true_peak_dbtp = -2.0;
        let error = validate_process_options(&process).unwrap_err();
        assert!(error.contains("ラウドネス"), "unexpected error: {error}");

        let mut config = gui_config();
        config.true_peak_dbtp = Some(-2.0);
        let error = config.validate().unwrap_err();
        assert!(error.contains("loudness_lufs"), "unexpected error: {error}");

        config.true_peak_dbtp = Some(-1.0);
        config.validate().unwrap();
    }

    #[test]
    fn selected_backend_contract_is_validated_without_opening_the_model() {
        if Backend::parse("mpsenet").is_some() {
            let mut process = options();
            process.backend = "mpsenet".into();
            process.onnx_model = Some("model-that-must-not-be-opened.onnx".into());
            process.onnx_sample_rate = 48_000;
            assert!(validate_process_options(&process)
                .unwrap_err()
                .contains("backend_options.onnx.sample_rate"));

            process.onnx_sample_rate = 16_000;
            validate_process_options(&process).unwrap();
        }

        if Backend::parse("onnx").is_some() {
            let mut process = options();
            process.backend = "onnx".into();
            process.onnx_model = None;
            assert!(validate_process_options(&process)
                .unwrap_err()
                .contains("backend_options.onnx"));
        }
    }

    #[test]
    fn managed_gtcrn_ignores_caller_model_configuration() {
        let Some(backend) = Backend::parse("gtcrn") else {
            return;
        };
        let mut process = options();
        process.backend = "gtcrn".into();
        process.onnx_model = Some("caller-model-must-not-be-used.onnx".into());
        process.onnx_sample_rate = 0;

        validate_process_options(&process).unwrap();
        assert!(parsed_backend_options_for(backend, &process)
            .unwrap()
            .onnx
            .is_none());
    }

    #[test]
    fn non_external_backends_ignore_hidden_model_configuration() {
        for name in Backend::available_names().iter().copied().filter(|name| {
            Backend::parse(name).is_some_and(|backend| !service::requires_external_model(backend))
        }) {
            let backend = Backend::parse(name).unwrap();
            let mut process = options();
            process.backend = name.into();
            process.onnx_model = Some("hidden-model-must-not-be-used.onnx".into());
            process.onnx_sample_rate = 0;

            validate_process_options(&process).unwrap();
            assert!(parsed_backend_options_for(backend, &process)
                .unwrap()
                .onnx
                .is_none());
        }
    }

    #[test]
    fn unknown_ipc_option_strings_are_rejected() {
        let mutations: &[fn(&mut ProcessOptions)] = &[
            |process| process.backend = "missing".into(),
            |process| process.preset = Some("missing".into()),
            |process| process.mode = Some("missing".into()),
            |process| process.channel_mode = "missing".into(),
            |process| process.downmix = "missing".into(),
            |process| process.aac_encoder = "missing".into(),
            |process| process.sgmse_profile = "missing".into(),
            |process| process.accelerator = "missing".into(),
        ];
        for mutate in mutations {
            let mut process = options();
            mutate(&mut process);
            assert!(validate_process_options(&process).is_err());
        }

        let mut batch = batch_request();
        batch.output_format = "missing".into();
        assert!(validate_batch_request(&batch).is_err());

        let mut live = live_request();
        live.backend = "missing".into();
        assert!(validate_live_request(&live).is_err());
    }

    #[test]
    fn batch_jobs_and_live_chunk_bounds_are_enforced() {
        for jobs in [1, 32] {
            let mut batch = batch_request();
            batch.jobs = jobs;
            validate_batch_request(&batch).unwrap();
        }
        for jobs in [0, 33] {
            let mut batch = batch_request();
            batch.jobs = jobs;
            assert!(validate_batch_request(&batch)
                .unwrap_err()
                .contains("並列数"));
        }

        #[cfg(feature = "live")]
        {
            for chunk_ms in [10, 2_000] {
                let mut live = live_request();
                live.chunk_ms = chunk_ms;
                validate_live_request(&live).unwrap();
            }
            for chunk_ms in [9, 2_001] {
                let mut live = live_request();
                live.chunk_ms = chunk_ms;
                assert!(validate_live_request(&live)
                    .unwrap_err()
                    .contains("チャンク長"));
            }
        }
    }

    #[cfg(feature = "live")]
    #[test]
    fn non_live_backends_are_rejected_before_starting_a_session() {
        let Some(name) = Backend::available_names()
            .iter()
            .copied()
            .find(|name| !matches!(*name, "classical" | "rnnoise" | "gtcrn"))
        else {
            return;
        };
        let mut live = live_request();
        live.backend = name.into();
        let error = validate_live_request(&live).unwrap_err();
        assert!(error.contains("ライブ処理"), "unexpected error: {error}");
    }

    #[test]
    fn invalid_ipc_options_precede_io_and_preserve_state_and_output() {
        let directory = TestDirectory::create("invalid-ipc");
        let missing_input = directory.join("missing.wav");
        let output = directory.join("output.wav");
        std::fs::write(&output, b"existing output").unwrap();
        let state = AppState::default();
        let mut process = classical_options(false);
        process.strength = f64::NAN;

        let request = process_request(&missing_input, &output, process);
        let error = validate_request(&request).unwrap_err();

        assert!(error.contains("強度"));
        assert_eq!(std::fs::read(&output).unwrap(), b"existing output");
        assert!(state.jobs.lock().unwrap().is_empty());
        assert!(state.live.lock().unwrap().is_none());
        directory.assert_no_staged_outputs();

        let mut batch = batch_request();
        batch.output_dir = directory
            .join("missing-output")
            .to_string_lossy()
            .into_owned();
        batch.jobs = 0;
        assert!(prepare_batch_request(&batch)
            .unwrap_err()
            .contains("並列数"));
        assert!(!Path::new(&batch.output_dir).exists());
        assert!(state.jobs.lock().unwrap().is_empty());
    }

    #[test]
    fn stream_request_validation_rejects_incompatible_options() {
        let directory = TestDirectory::create("stream-validation");
        let input = directory.join("input.wav");
        write_test_wav(&input);

        let mut request = process_request(
            &input,
            &directory.join("output.flac"),
            classical_options(false),
        );
        request.stream = true;
        assert!(validate_request(&request)
            .unwrap_err()
            .contains("出力形式はWAV"));

        request.output = directory.join("output.wav").to_string_lossy().into_owned();
        request.stream_frames = 0;
        assert!(validate_request(&request)
            .unwrap_err()
            .contains("ストリームブロック"));

        request.stream_frames = DEFAULT_STREAM_BLOCK_FRAMES;
        request.options.vad = true;
        assert!(validate_request(&request).unwrap_err().contains("VAD"));

        request.options.vad = false;
        request.stream = false;
        request.resume = true;
        assert!(validate_request(&request)
            .unwrap_err()
            .contains("長時間ストリーム"));
    }

    #[test]
    fn desktop_flac_stream_resume_writes_wav_and_cleans_data_sidecars() {
        let directory = TestDirectory::create("stream-flac-resume");
        let wav = directory.join("source.wav");
        let input = directory.join("input.flac");
        let output = directory.join("output.wav");
        write_test_wav(&wav);
        let original = read_audio(&wav).unwrap();
        write_audio(&input, &original, EncodeOptions::default()).unwrap();

        let mut request = process_request(&input, &output, classical_options(false));
        request.stream = true;
        request.resume = true;
        request.stream_frames = 113;
        validate_request(&request).unwrap();
        process_file(&request, &JobControl::default(), |_, _| {}).unwrap();

        let enhanced = read_audio(&output).unwrap();
        assert_eq!(enhanced.sample_rate, original.sample_rate);
        assert_eq!(enhanced.channels(), original.channels());
        assert_eq!(enhanced.frames(), original.frames());
        let (state, spool, lock) = batch_resume::stream_checkpoint_sidecar_paths(&output).unwrap();
        assert!(!state.exists());
        assert!(!spool.exists());
        assert!(lock.is_file());
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn file_and_live_registration_share_one_atomic_operation_slot() {
        let state = AppState::default();
        let running = Arc::new(AtomicBool::new(true));
        register_live_session(&state, Arc::clone(&running)).unwrap();
        let file_error = register_job(&state)
            .err()
            .expect("a live session must exclude a file job");
        assert!(file_error.contains("ライブ処理を停止"));

        *state.live.lock().unwrap() = None;
        let (job_id, _) = register_job(&state).unwrap();
        assert!(register_live_session(&state, running)
            .unwrap_err()
            .contains("ファイル処理の完了後"));
        state.jobs.lock().unwrap().remove(&job_id);
    }

    #[test]
    fn active_job_rejection_precedes_resume_journal_mutation() {
        let directory = TestDirectory::create("active-job-resume-order");
        let state_path = directory.join(batch_resume::STATE_FILE_NAME);
        std::fs::write(&state_path, b"legacy/item.wav\n{\"version\":3").unwrap();
        let before = std::fs::read(&state_path).unwrap();
        let state = AppState::default();
        state
            .jobs
            .lock()
            .unwrap()
            .insert(999, Arc::new(JobControl::default()));
        let mut request = batch_request();
        request.output_dir = directory.path.to_string_lossy().into_owned();

        let error = prepare_registered_batch(&state, &request)
            .err()
            .expect("an active job must reject the batch");

        assert!(error.contains("別の処理が実行中"));
        assert_eq!(std::fs::read(&state_path).unwrap(), before);
        assert_eq!(state.jobs.lock().unwrap().len(), 1);
    }

    #[test]
    fn model_action_options_deserialize_camel_case_and_build_policy() {
        let input: ModelActionOptions = serde_json::from_value(serde_json::json!({
            "offline": true,
            "sourceUrl": " https://models.example.test/model.onnx ",
            "proxyUrl": "http://proxy.example.test:8080",
            "basicUsername": "alice",
            "basicPassword": " secret "
        }))
        .unwrap();
        let (options, source) = model_action_options(Some(input)).unwrap();
        assert!(options.offline);
        assert_eq!(
            options.source_url.as_deref(),
            Some("https://models.example.test/model.onnx")
        );
        assert_eq!(
            options.proxy,
            ModelProxy::Url("http://proxy.example.test:8080".into())
        );
        match options.authentication {
            Some(ModelAuthentication::Basic { username, password }) => {
                assert_eq!(username, "alice");
                assert_eq!(password, " secret ");
            }
            _ => panic!("expected basic authentication"),
        }
        assert!(source.is_none());
    }

    #[test]
    fn model_action_options_inherit_download_environment_defaults() {
        let (options, source) = model_action_options_with_environment(None, |name| {
            Some(
                match name {
                    "DENOIZE_MODEL_OFFLINE" => "true",
                    "DENOIZE_MODEL_URL" => "https://mirror.example.test/model.onnx",
                    "DENOIZE_MODEL_PROXY" => "http://proxy.example.test:8080",
                    "DENOIZE_MODEL_BEARER_TOKEN" => "environment-token",
                    _ => return None,
                }
                .into(),
            )
        })
        .unwrap();
        assert!(options.offline);
        assert_eq!(
            options.source_url.as_deref(),
            Some("https://mirror.example.test/model.onnx")
        );
        assert_eq!(
            options.proxy,
            ModelProxy::Url("http://proxy.example.test:8080".into())
        );
        assert!(matches!(
            options.authentication,
            Some(ModelAuthentication::Bearer(ref token)) if token == "environment-token"
        ));
        assert!(source.is_none());
    }

    #[test]
    fn catalog_options_use_the_dedicated_catalog_environment_source() {
        let options = catalog_action_options_with_environment(None, |name| match name {
            "DENOIZE_MODEL_CATALOG_URL" => Some("https://catalog.example.test/catalog.json".into()),
            "DENOIZE_MODEL_URL" => Some("https://wrong.example.test/model.onnx".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(
            options.source_url.as_deref(),
            Some("https://catalog.example.test/catalog.json")
        );

        let input = ModelActionOptions {
            source_path: Some("catalog.json".into()),
            ..ModelActionOptions::default()
        };
        let error = catalog_action_options_with_environment(Some(input), |_| None).unwrap_err();
        assert!(error.contains("models catalog import"), "{error}");
    }

    #[test]
    fn model_action_options_reject_local_and_network_controls() {
        let input = ModelActionOptions {
            source_path: Some("/tmp/model.onnx".into()),
            proxy_url: Some("http://proxy.example.test:8080".into()),
            ..Default::default()
        };
        assert!(model_action_options_with_environment(Some(input), |_| {
            panic!("a local install must not read download environment variables")
        })
        .unwrap_err()
        .contains("同時に指定できません"));
    }

    #[test]
    fn model_action_options_reject_conflicting_authentication() {
        let input = ModelActionOptions {
            bearer_token: Some("token".into()),
            basic_username: Some("alice".into()),
            basic_password: Some("secret".into()),
            ..Default::default()
        };
        assert_eq!(
            model_action_options(Some(input)).unwrap_err(),
            "Bearer認証とBasic認証は同時に指定できません"
        );
    }

    #[test]
    fn model_action_options_reject_partial_basic_authentication() {
        let input = ModelActionOptions {
            basic_username: Some("alice".into()),
            ..Default::default()
        };
        assert_eq!(
            model_action_options(Some(input)).unwrap_err(),
            "Basic認証のユーザー名とパスワードは両方指定してください"
        );
    }

    #[test]
    fn model_action_options_support_direct_connections() {
        let input = ModelActionOptions {
            direct: true,
            ..Default::default()
        };
        let (options, _) = model_action_options(Some(input)).unwrap();
        assert_eq!(options.proxy, ModelProxy::Disabled);
    }

    #[test]
    fn model_action_options_reject_proxy_with_direct_connection() {
        let input = ModelActionOptions {
            proxy_url: Some("http://proxy.example.test:8080".into()),
            direct: true,
            ..Default::default()
        };
        assert_eq!(
            model_action_options(Some(input)).unwrap_err(),
            "プロキシURLと直接接続は同時に指定できません"
        );
    }

    #[test]
    fn model_cache_health_labels_are_stable() {
        assert_eq!(
            model_cache_status(denoize::models::ModelCacheModelStatus::ProvenanceInvalid),
            "provenance-invalid"
        );
        assert_eq!(
            model_cache_issue_kind(denoize::models::ModelCacheIssueKind::InvalidProvenance),
            "invalid-provenance"
        );
        assert_eq!(
            model_cache_issue_kind(denoize::models::ModelCacheIssueKind::OrphanedEntry),
            "orphaned-entry"
        );
    }

    #[test]
    fn no_force_rechecks_destination_when_committing() {
        let directory = TestDirectory::create("commit-race");
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        write_test_wav(&input);
        let request = process_request(&input, &output, classical_options(false));
        validate_request(&request).unwrap();

        let result = process_file(&request, &JobControl::default(), |stage, _| {
            if stage == 3 {
                std::fs::write(&output, b"racing writer").unwrap();
            }
        });

        assert!(result.unwrap_err().contains("output already exists"));
        assert_eq!(std::fs::read(&output).unwrap(), b"racing writer");
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn cancellation_before_commit_preserves_existing_output() {
        let directory = TestDirectory::create("cancel-commit");
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        write_test_wav(&input);
        std::fs::write(&output, b"existing output").unwrap();
        let request = process_request(&input, &output, classical_options(true));
        let control = Arc::new(JobControl::default());
        let worker_control = Arc::clone(&control);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            process_file(&request, &worker_control, |stage, _| {
                if stage == 4 {
                    ready_tx.send(()).unwrap();
                    resume_rx.recv().unwrap();
                }
            })
        });

        ready_rx.recv().unwrap();
        control.cancel().unwrap();
        resume_tx.send(()).unwrap();
        let result = worker.join().unwrap();

        assert_eq!(result.unwrap_err(), "cancelled");
        assert_eq!(std::fs::read(&output).unwrap(), b"existing output");
        directory.assert_no_staged_outputs();
    }

    #[test]
    fn cancellation_waits_for_an_active_commit_fence() {
        let control = Arc::new(JobControl::default());
        let worker_control = Arc::clone(&control);
        let (inside_tx, inside_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let published = Arc::new(AtomicBool::new(false));
        let worker_published = Arc::clone(&published);
        let worker = std::thread::spawn(move || {
            worker_control.commit_fence(|| {
                inside_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                worker_published.store(true, Ordering::SeqCst);
                Ok(())
            })
        });

        inside_rx.recv().unwrap();
        let cancel_control = Arc::clone(&control);
        let (cancelled_tx, cancelled_rx) = std::sync::mpsc::channel();
        let canceller = std::thread::spawn(move || {
            cancel_control.cancel().unwrap();
            cancelled_tx.send(()).unwrap();
        });
        assert!(cancelled_rx.try_recv().is_err());
        let wait_started = Instant::now();
        while !control.is_cancelled() && wait_started.elapsed() < std::time::Duration::from_secs(1)
        {
            std::thread::yield_now();
        }
        assert!(
            control.is_cancelled(),
            "cancellation must be visible while the active publication finishes"
        );

        release_tx.send(()).unwrap();
        worker.join().unwrap().unwrap();
        cancelled_rx.recv().unwrap();
        canceller.join().unwrap();

        assert!(published.load(Ordering::SeqCst));
        assert!(control.is_cancelled());
    }

    #[test]
    fn commit_fence_propagates_shared_publisher_failures() {
        let control = JobControl::default();
        let error = control
            .commit_fence::<()>(|| Err("injected journal failure".into()))
            .unwrap_err();
        assert_eq!(error, "injected journal failure");
        assert!(!control.is_cancelled());
    }

    #[test]
    fn invalid_codec_config_precedes_processing_and_output_staging() {
        let directory = TestDirectory::create("codec-preflight");
        let input = directory.join("input.wav");
        write_test_wav(&input);
        let mut wav = std::fs::read(&input).unwrap();
        wav[24..28].copy_from_slice(&12_345_u32.to_le_bytes());
        wav[28..32].copy_from_slice(&(12_345_u32 * 2).to_le_bytes());
        std::fs::write(&input, wav).unwrap();
        let output_dir = directory.join("new-output-directory");
        let output = output_dir.join("output.mp3");
        let request = process_request(&input, &output, classical_options(false));
        let stages = Mutex::new(Vec::new());

        let error = process_file(&request, &JobControl::default(), |stage, _| {
            stages.lock().unwrap().push(stage);
        })
        .unwrap_err();

        assert!(
            error.contains("unsupported sample rate"),
            "unexpected error: {error}"
        );
        assert!(stages.lock().unwrap().is_empty());
        assert!(!output_dir.exists());
        directory.assert_no_staged_outputs();
    }

    #[cfg(unix)]
    #[test]
    fn legacy_gui_stage_symlink_does_not_clobber_its_target() {
        let directory = TestDirectory::create("stage-symlink");
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        let victim = directory.join("victim.bin");
        let legacy_stage = directory.join(".denoize-gui-output.wav.wav");
        write_test_wav(&input);
        std::fs::write(&victim, b"victim").unwrap();
        std::os::unix::fs::symlink(&victim, &legacy_stage).unwrap();
        let request = process_request(&input, &output, classical_options(false));

        process_file(&request, &JobControl::default(), |_, _| {}).unwrap();

        assert_eq!(std::fs::read(&victim).unwrap(), b"victim");
        assert!(std::fs::symlink_metadata(&legacy_stage)
            .unwrap()
            .file_type()
            .is_symlink());
        assert!(read_audio(&output).is_ok());
    }

    #[test]
    fn comparison_metrics_include_quality_and_artifact_improvements() {
        let report = |snr_db: f64, stoi: f64, musical_noise_score: f64| BenchmarkReport {
            frames: 1,
            sample_rate: 48_000,
            channels: 1,
            si_sdr_db: snr_db,
            si_snr_db: snr_db + 1.0,
            snr_db,
            segmental_snr_db: snr_db - 1.0,
            stereo_side_sdr_db: None,
            correlation_error: None,
            artifact_scores: denoize::benchmark::ArtifactReport {
                musical_noise_score,
                pumping_score: musical_noise_score + 0.1,
                transient_loss_score: musical_noise_score + 0.2,
                phase_distortion_score: None,
            },
            stoi: Some(stoi),
            pesq: None,
            visqol: Some(stoi + 1.0),
            elapsed_ms: None,
            peak_rss_bytes: None,
        };
        let comparison = ComparisonReport {
            noisy: report(2.0, 0.5, 0.4),
            enhanced: report(5.0, 0.8, 0.1),
        };
        let metrics = comparison_metric_set(&comparison);
        assert_eq!(metrics.noisy.snr_db, 2.0);
        assert_eq!(metrics.enhanced.stoi, Some(0.8));
        assert_eq!(metrics.improvement.snr_db, 3.0);
        assert!((metrics.improvement.stoi.unwrap() - 0.3).abs() < 1e-10);
        assert!((metrics.improvement.artifact_scores.musical_noise_score - 0.3).abs() < 1e-10);
        assert!((metrics.improvement.visqol.unwrap() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn batch_preflights_every_codec_before_state_or_output_changes() {
        let directory = TestDirectory::create("batch-codec-preflight");
        let input = directory.join("input");
        let output = directory.join("output");
        std::fs::create_dir(&input).unwrap();
        std::fs::create_dir(&output).unwrap();
        write_test_wav(&input.join("a-valid.wav"));
        write_test_wav(&input.join("b-invalid-rate.wav"));
        let invalid = input.join("b-invalid-rate.wav");
        let mut wav = std::fs::read(&invalid).unwrap();
        wav[24..28].copy_from_slice(&12_345_u32.to_le_bytes());
        wav[28..32].copy_from_slice(&(12_345_u32 * 2).to_le_bytes());
        std::fs::write(&invalid, wav).unwrap();
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_string_lossy().into_owned()),
            output_dir: output.to_string_lossy().into_owned(),
            output_format: "mp3".into(),
            recursive: false,
            jobs: 2,
            resume: true,
            options: classical_options(false),
        };

        let error = prepare_batch_request(&request).unwrap_err();

        assert!(error.contains("unsupported sample rate"), "{error}");
        assert!(!output.join("a-valid.mp3").exists());
        assert!(!output.join("b-invalid-rate.mp3").exists());
        assert!(!output.join(".denoize-state").exists());
        assert!(!output.join(".denoize-batch.lock").exists());
    }

    #[test]
    fn batch_preflights_actual_sample_rate_processing_before_outputs() {
        let directory = TestDirectory::create("batch-processing-preflight");
        let input = directory.join("input");
        let output = directory.join("output");
        std::fs::create_dir(&input).unwrap();
        std::fs::create_dir(&output).unwrap();
        write_test_wav(&input.join("a-valid.wav"));
        write_test_wav(&input.join("b-invalid-processing-rate.wav"));
        let invalid = input.join("b-invalid-processing-rate.wav");
        let mut wav = std::fs::read(&invalid).unwrap();
        let sample_rate = MAX_MODEL_SAMPLE_RATE_HZ + 1;
        wav[24..28].copy_from_slice(&sample_rate.to_le_bytes());
        wav[28..32].copy_from_slice(&(sample_rate * 2).to_le_bytes());
        std::fs::write(&invalid, wav).unwrap();
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_string_lossy().into_owned()),
            output_dir: output.to_string_lossy().into_owned(),
            output_format: "wav".into(),
            recursive: false,
            jobs: 2,
            resume: true,
            options: classical_options(false),
        };

        let error = prepare_batch_request(&request).unwrap_err();

        assert!(error.contains("sample_rate"), "{error}");
        assert!(!output.join("a-valid.wav").exists());
        assert!(!output.join("b-invalid-processing-rate.wav").exists());
        assert!(!output.join(".denoize-state").exists());
        assert!(!output.join(".denoize-batch.lock").exists());
    }

    #[test]
    fn batch_preflights_all_destinations_and_replacement_types() {
        let directory = TestDirectory::create("batch-output-preflight");
        let input = directory.join("input");
        let output = directory.join("output");
        std::fs::create_dir(&input).unwrap();
        std::fs::create_dir(&output).unwrap();
        write_test_wav(&input.join("a.wav"));
        write_test_wav(&input.join("b.wav"));
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_string_lossy().into_owned()),
            output_dir: output.to_string_lossy().into_owned(),
            output_format: "wav".into(),
            recursive: false,
            jobs: 2,
            resume: false,
            options: classical_options(false),
        };
        let items = prepare_batch_request(&request).unwrap();
        std::fs::write(output.join("b.wav"), b"existing").unwrap();

        let session = BatchSession::acquire(&output, false).unwrap();
        let error = plan_batch_items(&session, items, false).unwrap_err();
        assert!(error.contains("--force"), "{error}");
        assert!(!output.join("a.wav").exists());
        assert_eq!(std::fs::read(output.join("b.wav")).unwrap(), b"existing");
        assert!(!output.join(batch_resume::STATE_FILE_NAME).exists());
        drop(session);

        std::fs::remove_file(output.join("b.wav")).unwrap();
        std::fs::create_dir(output.join("b.wav")).unwrap();
        let items = prepare_batch_request(&request).unwrap();
        let session = BatchSession::acquire(&output, false).unwrap();
        let error = plan_batch_items(&session, items, true).unwrap_err();
        assert!(error.contains("cannot be replaced"), "{error}");
        assert!(!output.join("a.wav").exists());
    }

    #[cfg(unix)]
    #[test]
    fn linked_batch_outputs_are_never_treated_as_resumable_files() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::create("batch-resume-output-symlink");
        let input_dir = directory.join("input");
        let output_dir = directory.join("output");
        std::fs::create_dir(&input_dir).unwrap();
        std::fs::create_dir(&output_dir).unwrap();
        write_test_wav(&input_dir.join("sample.wav"));
        let victim = directory.join("victim.wav");
        let output = output_dir.join("sample.wav");
        write_test_wav(&victim);
        symlink(&victim, &output).unwrap();
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input_dir.to_string_lossy().into_owned()),
            output_dir: output_dir.to_string_lossy().into_owned(),
            output_format: "wav".into(),
            recursive: false,
            jobs: 1,
            resume: true,
            options: classical_options(false),
        };
        let prepared = prepare_batch_request(&request).unwrap();
        let session = BatchSession::acquire(&output_dir, true).unwrap();

        assert!(
            output.is_file(),
            "the follow-link check would incorrectly skip"
        );
        let error = plan_batch_items(&session, prepared.clone(), false).unwrap_err();
        assert!(error.contains("unsafe"), "{error}");
        let planned = plan_batch_items(&session, prepared, true).unwrap();
        assert!(matches!(
            planned[0].decision,
            ResumeDecision::Process {
                reason: batch_resume::ResumeReason::Unsafe,
                ..
            }
        ));
        assert!(std::fs::symlink_metadata(&output)
            .unwrap()
            .file_type()
            .is_symlink());
    }

    #[cfg(unix)]
    #[test]
    fn hard_linked_batch_outputs_are_never_treated_as_resumable_files() {
        use std::os::unix::fs::MetadataExt as _;

        let directory = TestDirectory::create("batch-resume-output-hardlink");
        let input_dir = directory.join("input");
        let output_dir = directory.join("output");
        std::fs::create_dir(&input_dir).unwrap();
        std::fs::create_dir(&output_dir).unwrap();
        write_test_wav(&input_dir.join("sample.wav"));
        let victim = directory.join("victim.wav");
        let output = output_dir.join("sample.wav");
        write_test_wav(&victim);
        std::fs::hard_link(&victim, &output).unwrap();
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input_dir.to_string_lossy().into_owned()),
            output_dir: output_dir.to_string_lossy().into_owned(),
            output_format: "wav".into(),
            recursive: false,
            jobs: 1,
            resume: true,
            options: classical_options(false),
        };
        let prepared = prepare_batch_request(&request).unwrap();
        let session = BatchSession::acquire(&output_dir, true).unwrap();

        let error = plan_batch_items(&session, prepared.clone(), false).unwrap_err();
        assert!(error.contains("unsafe"), "{error}");
        let planned = plan_batch_items(&session, prepared, true).unwrap();
        assert!(matches!(
            planned[0].decision,
            ResumeDecision::Process {
                reason: batch_resume::ResumeReason::Unsafe,
                ..
            }
        ));
        let output_metadata = std::fs::metadata(&output).unwrap();
        let victim_metadata = std::fs::metadata(&victim).unwrap();
        assert_eq!(output_metadata.dev(), victim_metadata.dev());
        assert_eq!(output_metadata.ino(), victim_metadata.ino());
        assert!(output_metadata.nlink() > 1);
    }

    #[test]
    fn batch_destination_preflight_rejects_exact_and_file_directory_collisions() {
        let directory = TestDirectory::create("batch-destination-collisions");
        let input_a = directory.join("input-a.wav");
        let input_b = directory.join("input-b.wav");
        let output = directory.join("output");
        write_test_wav(&input_a);
        write_test_wav(&input_b);
        std::fs::create_dir(&output).unwrap();

        let exact = vec![
            test_batch_item(input_a.clone(), output.join("same.wav"), 1),
            test_batch_item(input_b.clone(), output.join("same.wav"), 2),
        ];
        assert!(validate_batch_destinations(None, &exact)
            .unwrap_err()
            .contains("同じバッチ出力"));

        let file_and_directory = vec![
            test_batch_item(input_a, output.join("foo.flac"), 1),
            test_batch_item(input_b, output.join("foo.flac/bar.flac"), 2),
        ];
        assert!(validate_batch_destinations(None, &file_and_directory)
            .unwrap_err()
            .contains("ファイルとディレクトリ"));
        assert!(std::fs::read_dir(&output).unwrap().next().is_none());
    }

    #[test]
    fn case_insensitive_batch_collision_keys_are_normalized() {
        assert_eq!(
            batch_collision_key_with_case(Path::new("Output/Voice.WAV"), true),
            batch_collision_key_with_case(Path::new("output/voice.wav"), true)
        );
        assert_ne!(
            batch_collision_key_with_case(Path::new("Output/Voice.WAV"), false),
            batch_collision_key_with_case(Path::new("output/voice.wav"), false)
        );
    }

    #[cfg(unix)]
    #[test]
    fn batch_destination_preflight_rejects_symlinks_back_into_input() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::create("batch-destination-input-symlink");
        let input = directory.join("input");
        let nested = input.join("nested");
        let output = directory.join("output");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::create_dir(&output).unwrap();
        write_test_wav(&nested.join("voice.wav"));
        symlink(&nested, output.join("nested")).unwrap();
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_str().unwrap().into()),
            output_dir: output.to_str().unwrap().into(),
            output_format: "flac".into(),
            recursive: true,
            jobs: 1,
            resume: false,
            options: classical_options(false),
        };

        let error = collect_batch_items(&request, "flac").unwrap_err();
        assert!(error.contains("入力フォルダ内"), "{error}");
        assert!(!nested.join("voice.flac").exists());
    }

    #[test]
    fn batch_folder_preserves_relative_paths() {
        let root = std::env::temp_dir().join(format!(
            "denoize-gui-batch-{}-{}",
            std::process::id(),
            NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let input = root.join("input");
        let nested = input.join("nested");
        let output = root.join("output");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::create_dir_all(&output).unwrap();
        std::fs::write(input.join("one.wav"), []).unwrap();
        std::fs::write(nested.join("two.flac"), []).unwrap();
        std::fs::write(nested.join("ignored.txt"), []).unwrap();
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_string_lossy().into_owned()),
            output_dir: output.to_string_lossy().into_owned(),
            output_format: "opus".into(),
            recursive: true,
            jobs: 2,
            resume: true,
            options: options(),
        };
        let items = collect_batch_items(&request, "opus").unwrap();
        assert_eq!(items.len(), 2);
        assert!(items
            .iter()
            .any(|item| item.output == output.join("one.opus")));
        assert!(items
            .iter()
            .any(|item| item.output == output.join("nested/two.opus")));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batch_item_ids_include_input_identity_destination_and_format() {
        let relative = Path::new("voice.wav");
        let output = Path::new("nested/voice.output");
        let wav = batch_resume::item_identity(
            Path::new("/input-a/voice.wav"),
            relative,
            output,
            OutputFormat::Wav,
        );
        assert_ne!(
            wav,
            batch_resume::item_identity(
                Path::new("/input-b/voice.wav"),
                relative,
                output,
                OutputFormat::Wav,
            )
        );
        assert_ne!(
            wav,
            batch_resume::item_identity(
                Path::new("/input-a/voice.wav"),
                relative,
                output,
                OutputFormat::Flac,
            )
        );
        assert_ne!(
            wav,
            batch_resume::item_identity(
                Path::new("/input-a/voice.wav"),
                relative,
                Path::new("other/voice.output"),
                OutputFormat::Wav,
            )
        );
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_batch_paths_do_not_collide_and_are_rejected_before_side_effects() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt as _;

        let first_relative = PathBuf::from(OsString::from_vec(b"voice-\x80.wav".to_vec()));
        let second_relative = PathBuf::from(OsString::from_vec(b"voice-\x81.wav".to_vec()));
        assert_eq!(
            first_relative.to_string_lossy(),
            second_relative.to_string_lossy()
        );
        assert_ne!(
            batch_resume::item_identity(
                Path::new("/input/voice.wav"),
                &first_relative,
                &first_relative,
                OutputFormat::Wav,
            ),
            batch_resume::item_identity(
                Path::new("/input/voice.wav"),
                &second_relative,
                &second_relative,
                OutputFormat::Wav,
            )
        );

        let directory = TestDirectory::create("batch-non-utf8");
        let input = directory.join("input");
        let output = directory.join("output");
        std::fs::create_dir(&input).unwrap();
        std::fs::create_dir(&output).unwrap();
        write_test_wav(&input.join(first_relative));
        let request = BatchRequest {
            inputs: Vec::new(),
            input_dir: Some(input.to_str().unwrap().into()),
            output_dir: output.to_str().unwrap().into(),
            output_format: "wav".into(),
            recursive: false,
            jobs: 1,
            resume: true,
            options: classical_options(false),
        };

        let error = collect_batch_items(&request, "wav").unwrap_err();
        assert!(error.contains("UTF-8"), "{error}");
        assert!(std::fs::read_dir(&output).unwrap().next().is_none());
        assert!(!output.join(".denoize-state").exists());
        assert!(!output.join(".denoize-batch.lock").exists());
    }

    #[test]
    fn desktop_resume_uses_canonical_state_and_exact_force_still_skips() {
        let directory = TestDirectory::create("batch-v3-exact");
        let request = desktop_batch_fixture(&directory, true, false);
        complete_desktop_batch(&request);
        let output = Path::new(&request.output_dir);
        assert!(output.join(batch_resume::STATE_FILE_NAME).is_file());
        assert!(output.join(batch_resume::LOCK_FILE_NAME).is_file());
        assert!(!output
            .join(batch_resume::LEGACY_DESKTOP_STATE_FILE_NAME)
            .exists());

        let prepared = prepare_batch_request(&request).unwrap();
        let session = BatchSession::acquire(output, true).unwrap();
        let planned = plan_batch_items(&session, prepared, true).unwrap();
        assert!(matches!(
            planned[0].decision,
            ResumeDecision::Skip {
                reason: batch_resume::ResumeReason::Exact
            }
        ));
    }

    #[test]
    fn desktop_resume_detects_input_recipe_and_output_changes() {
        for change in ["input", "recipe", "output"] {
            let directory = TestDirectory::create(&format!("batch-v3-{change}"));
            let mut request = desktop_batch_fixture(&directory, true, false);
            complete_desktop_batch(&request);
            let output = PathBuf::from(&request.output_dir);
            let expected_reason = match change {
                "input" => {
                    let input = Path::new(request.input_dir.as_deref().unwrap()).join("sample.wav");
                    let mut bytes = std::fs::read(&input).unwrap();
                    *bytes.last_mut().unwrap() ^= 0x7f;
                    std::fs::write(input, bytes).unwrap();
                    batch_resume::ResumeReason::InputChanged
                }
                "recipe" => {
                    request.options.strength = 0.8;
                    batch_resume::ResumeReason::RecipeChanged
                }
                "output" => {
                    std::fs::write(output.join("sample.wav"), b"changed output").unwrap();
                    batch_resume::ResumeReason::OutputChanged
                }
                _ => unreachable!(),
            };
            let prepared = prepare_batch_request(&request).unwrap();
            let session = BatchSession::acquire(&output, true).unwrap();
            let error = plan_batch_items(&session, prepared.clone(), false).unwrap_err();
            assert!(error.contains("上書き"), "{change}: {error}");
            let planned = plan_batch_items(&session, prepared, true).unwrap();
            assert!(matches!(
                planned[0].decision,
                ResumeDecision::Process { reason, .. } if reason == expected_reason
            ));
        }
    }

    #[test]
    fn desktop_resume_tracks_the_consumed_model_fingerprint() {
        use std::io::Write as _;

        let directory = TestDirectory::create("batch-v3-model");
        let input = directory.join("input.wav");
        let model = directory.join("model.onnx");
        let output_root = directory.join("output");
        let output = output_root.join("output.wav");
        std::fs::create_dir(&output_root).unwrap();
        write_test_wav(&input);
        std::fs::write(&model, b"model one").unwrap();
        let input_fingerprint = batch_resume::fingerprint_file(&input).unwrap();
        let recipe = Digest::from_bytes([9; 32]);
        let expectation = ResumeExpectation::new(
            Digest::from_bytes([7; 32]),
            output.clone(),
            input.clone(),
            input_fingerprint,
            Some(batch_resume::ConsumedModel {
                path: model.clone(),
                fingerprint: batch_resume::fingerprint_file(&model).unwrap(),
                sample_rate: 16_000,
            }),
            recipe,
        );
        let session = BatchSession::acquire(&output_root, true).unwrap();
        let ResumeDecision::Process { commit_mode, .. } =
            session.plan(&expectation, false).unwrap()
        else {
            panic!("missing output must be processed");
        };
        session.activate().unwrap();
        let mut transaction = AtomicOutput::new(&output).unwrap();
        transaction.file_mut().write_all(b"encoded output").unwrap();
        session
            .publish(&expectation, transaction, commit_mode)
            .unwrap();
        drop(session);

        std::fs::write(&model, b"model two").unwrap();
        let changed = ResumeExpectation::new(
            expectation.item_id(),
            output,
            input,
            input_fingerprint,
            Some(batch_resume::ConsumedModel {
                path: model.clone(),
                fingerprint: batch_resume::fingerprint_file(&model).unwrap(),
                sample_rate: 16_000,
            }),
            recipe,
        );
        let session = BatchSession::acquire(&output_root, true).unwrap();
        let decision = session.plan(&changed, true).unwrap();
        assert!(matches!(
            decision,
            ResumeDecision::Process {
                reason: batch_resume::ResumeReason::ModelChanged,
                ..
            }
        ));
    }

    #[test]
    fn legacy_desktop_state_is_untrusted_migrated_and_never_modified() {
        let directory = TestDirectory::create("batch-v3-legacy");
        let request = desktop_batch_fixture(&directory, true, false);
        let output = PathBuf::from(&request.output_dir);
        let destination = output.join("sample.wav");
        std::fs::write(&destination, b"legacy output").unwrap();
        let legacy_path = output.join(batch_resume::LEGACY_DESKTOP_STATE_FILE_NAME);
        let legacy = format!("v2:{}\n", "11".repeat(32));
        std::fs::write(&legacy_path, &legacy).unwrap();

        let prepared = prepare_batch_request(&request).unwrap();
        let session = BatchSession::acquire(&output, true).unwrap();
        let error = plan_batch_items(&session, prepared.clone(), false).unwrap_err();
        assert!(error.contains("上書き"), "{error}");
        assert_eq!(std::fs::read_to_string(&legacy_path).unwrap(), legacy);
        let planned = plan_batch_items(&session, prepared, true).unwrap();
        assert!(matches!(
            planned[0].decision,
            ResumeDecision::Process {
                reason: batch_resume::ResumeReason::Legacy,
                ..
            }
        ));
        session.activate().unwrap();
        publish_planned_item(&session, &planned[0]).unwrap();
        drop(session);
        assert_eq!(std::fs::read_to_string(&legacy_path).unwrap(), legacy);

        let prepared = prepare_batch_request(&request).unwrap();
        let session = BatchSession::acquire(&output, true).unwrap();
        let planned = plan_batch_items(&session, prepared, false).unwrap();
        assert!(matches!(planned[0].decision, ResumeDecision::Skip { .. }));
        assert_eq!(std::fs::read_to_string(legacy_path).unwrap(), legacy);
    }

    #[test]
    fn desktop_migrates_canonical_v1_and_v2_without_touching_legacy_gui_state() {
        for (index, canonical_legacy) in [
            b"sample.wav\n".to_vec(),
            format!("v2:{}\n", "52".repeat(32)).into_bytes(),
        ]
        .into_iter()
        .enumerate()
        {
            let directory = TestDirectory::create(&format!("canonical-legacy-{index}"));
            let mut request = desktop_batch_fixture(&directory, true, false);
            let output = PathBuf::from(&request.output_dir);
            let destination = output.join("sample.wav");
            let original_output = b"canonical legacy output";
            std::fs::write(&destination, original_output).unwrap();
            let canonical_path = output.join(batch_resume::STATE_FILE_NAME);
            std::fs::write(&canonical_path, &canonical_legacy).unwrap();

            let prepared = prepare_batch_request(&request).unwrap();
            let session = BatchSession::acquire(&output, true).unwrap();
            let error = plan_batch_items(&session, prepared, false).unwrap_err();
            assert!(error.contains("legacy"), "{error}");
            assert_eq!(std::fs::read(&destination).unwrap(), original_output);
            assert_eq!(std::fs::read(&canonical_path).unwrap(), canonical_legacy);
            drop(session);

            let legacy_gui_path = output.join(batch_resume::LEGACY_DESKTOP_STATE_FILE_NAME);
            let legacy_gui = format!("v2:{}\n", "63".repeat(32)).into_bytes();
            std::fs::write(&legacy_gui_path, &legacy_gui).unwrap();
            request.options.force = true;
            let prepared = prepare_batch_request(&request).unwrap();
            let session = BatchSession::acquire(&output, true).unwrap();
            let planned = plan_batch_items(&session, prepared, true).unwrap();
            assert!(matches!(
                planned[0].decision,
                ResumeDecision::Process {
                    reason: batch_resume::ResumeReason::Legacy,
                    ..
                }
            ));
            session.activate().unwrap();
            publish_planned_item(&session, &planned[0]).unwrap();
            drop(session);
            let migrated_state = std::fs::read(&canonical_path).unwrap();
            assert!(migrated_state.starts_with(&canonical_legacy));
            assert!(String::from_utf8_lossy(&migrated_state).contains("\"version\":3"));
            assert_eq!(std::fs::read(&legacy_gui_path).unwrap(), legacy_gui);

            request.options.force = false;
            let prepared = prepare_batch_request(&request).unwrap();
            let session = BatchSession::acquire(&output, true).unwrap();
            let planned = plan_batch_items(&session, prepared, false).unwrap();
            assert!(matches!(
                planned[0].decision,
                ResumeDecision::Skip {
                    reason: batch_resume::ResumeReason::Exact
                }
            ));
            session.activate().unwrap();
            assert_eq!(std::fs::read(&canonical_path).unwrap(), migrated_state);
            assert_eq!(std::fs::read(&legacy_gui_path).unwrap(), legacy_gui);
        }
    }

    #[test]
    fn desktop_batch_session_guard_lives_in_the_worker() {
        let directory = TestDirectory::create("batch-v3-lock-lifetime");
        let output = directory.join("output");
        std::fs::create_dir(&output).unwrap();
        let session = BatchSession::acquire(&output, false).unwrap();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            ready_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            drop(session);
        });
        ready_rx.recv().unwrap();
        assert!(BatchSession::acquire(&output, false).is_err());
        release_tx.send(()).unwrap();
        worker.join().unwrap();
        assert!(BatchSession::acquire(&output, false).is_ok());
    }

    #[test]
    fn batch_outputs_cannot_claim_control_paths() {
        let directory = TestDirectory::create("batch-state-reserved");
        for name in [
            batch_resume::STATE_FILE_NAME,
            batch_resume::LEGACY_DESKTOP_STATE_FILE_NAME,
            batch_resume::LOCK_FILE_NAME,
        ] {
            let items = vec![test_batch_item(
                directory.join("input.wav"),
                directory.join(name).join("nested.wav"),
                1,
            )];
            let error = validate_batch_control_paths(&items, &directory.path).unwrap_err();
            assert!(error.contains(name), "{error}");
        }
    }

    #[test]
    fn successful_batch_has_completed_terminal_outcome() {
        let counts = BatchOutcomeCounts {
            completed: 3,
            skipped: 1,
            ..Default::default()
        };
        let outcome = batch_terminal_outcome(counts);
        assert_eq!(outcome.status, "completed");
        assert_eq!(
            outcome.message,
            "完了 3 · スキップ 1 · 失敗 0 · キャンセル 0"
        );
        assert_eq!(outcome.error, None);
        assert_eq!(counts.total(), 4);
    }

    #[test]
    fn mixed_batch_has_failed_terminal_outcome() {
        let counts = BatchOutcomeCounts {
            completed: 2,
            skipped: 1,
            failed: 1,
            cancelled: 0,
        };
        let outcome = batch_terminal_outcome(counts);
        assert_eq!(outcome.status, "failed");
        assert_eq!(
            outcome.message,
            "完了 2 · スキップ 1 · 失敗 1 · キャンセル 0"
        );
        assert_eq!(
            outcome.error.as_deref(),
            Some("1件のファイルを処理できませんでした")
        );
        assert_eq!(counts.total(), 4);
    }

    #[test]
    fn all_failed_batch_has_failed_terminal_outcome() {
        let counts = BatchOutcomeCounts {
            failed: 3,
            ..Default::default()
        };
        let outcome = batch_terminal_outcome(counts);
        assert_eq!(outcome.status, "failed");
        assert_eq!(
            outcome.message,
            "完了 0 · スキップ 0 · 失敗 3 · キャンセル 0"
        );
        assert_eq!(
            outcome.error.as_deref(),
            Some("3件のファイルを処理できませんでした")
        );
        assert_eq!(counts.total(), 3);
    }

    #[test]
    fn cancelled_batch_has_one_total_partition() {
        assert_eq!(
            batch_item_commit_mode(
                ResumeDecision::Skip {
                    reason: batch_resume::ResumeReason::Exact,
                },
                true,
            ),
            Err(BatchItemOutcome::Skipped),
            "an exact resume skip remains skipped after cancellation"
        );
        assert_eq!(
            batch_item_commit_mode(
                ResumeDecision::Process {
                    commit_mode: CommitMode::NoClobber,
                    reason: batch_resume::ResumeReason::Missing,
                },
                true,
            ),
            Err(BatchItemOutcome::Cancelled),
            "only work that still needs processing is cancelled"
        );
        let outcomes = vec![
            BatchItemOutcome::Completed,
            BatchItemOutcome::Skipped,
            BatchItemOutcome::Failed("injected".into()),
            BatchItemOutcome::Cancelled,
            BatchItemOutcome::Cancelled,
        ];
        assert_eq!(
            outcomes
                .iter()
                .map(BatchItemOutcome::status)
                .collect::<Vec<_>>(),
            vec!["completed", "skipped", "failed", "cancelled", "cancelled"]
        );
        assert_eq!(outcomes[2].error().as_deref(), Some("injected"));
        let counts = count_batch_outcomes(&outcomes);
        let outcome = batch_terminal_outcome(counts);
        assert_eq!(outcome.status, "cancelled");
        assert_eq!(
            outcome.message,
            "完了 1 · スキップ 1 · 失敗 1 · キャンセル 2"
        );
        assert_eq!(outcome.error, None);
        assert_eq!(counts.total(), 5);
    }

    #[test]
    fn valid_gui_toml_config_round_trips_without_nulls() {
        let path = std::env::temp_dir().join(format!(
            "denoize-gui-config-{}-{}.toml",
            std::process::id(),
            NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let mut expected = gui_config();
        expected.strength = 0.42;
        save_gui_config(path.to_string_lossy().into_owned(), expected.clone()).unwrap();
        let loaded = load_gui_config(path.to_string_lossy().into_owned(), gui_config()).unwrap();
        assert_eq!(loaded, expected);
        let source = std::fs::read_to_string(&path).unwrap();
        assert!(!source.contains("onnx_model"));
        assert!(!source.contains("loudness_lufs"));
        assert!(source.contains("true_peak_dbtp = -1.0"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn exported_loudness_sentinel_clears_an_enabled_current_config() {
        let directory = TestDirectory::create("gui-loudness-clear");
        let path = directory.join("config.toml");
        save_gui_config(path.to_str().unwrap().into(), gui_config()).unwrap();
        let mut current = gui_config();
        current.loudness_lufs = Some(-16.0);
        current.true_peak_dbtp = Some(-1.0);

        let loaded = load_gui_config(path.to_str().unwrap().into(), current).unwrap();

        assert!(loaded.loudness_lufs.is_none());
        assert!(loaded.true_peak_dbtp.is_none());
    }

    #[test]
    fn gui_toml_partial_patch_preserves_current_settings() {
        let mut current = gui_config();
        current.mode = "ambient".into();
        current.force = true;
        let loaded = parse_gui_config(
            "backend = \"classical\"\nstrength = 0.73\n",
            current.clone(),
        )
        .unwrap();

        assert_eq!(loaded.backend, "classical");
        assert_eq!(loaded.strength, 0.73);
        assert_eq!(loaded.mode, current.mode);
        assert_eq!(loaded.force, current.force);
        assert_eq!(loaded.preset, current.preset);
    }

    #[test]
    fn gui_toml_discards_hidden_models_for_non_external_backends() {
        let loaded = parse_gui_config(
            "backend = \"classical\"\nonnx_model = \"stale.onnx\"\nonnx_rate = 0\n",
            gui_config(),
        )
        .unwrap();

        assert!(loaded.onnx_model.is_none());
        assert_eq!(loaded.onnx_rate, DEFAULT_MODEL_SAMPLE_RATE_HZ);
    }

    #[test]
    fn gui_toml_config_rejects_boolean_strings() {
        for field in ["force", "preserve_metadata"] {
            let source = gui_config_source()
                .replace(&format!("{field} = false"), &format!("{field} = \"false\""))
                .replace(&format!("{field} = true"), &format!("{field} = \"true\""));
            assert!(parse_gui_config(&source, gui_config()).is_err(), "{field}");
        }
    }

    #[test]
    fn gui_toml_config_rejects_unknown_fields() {
        let source = format!("{}unknown_option = true\n", gui_config_source());
        let error = parse_gui_config(&source, gui_config()).unwrap_err();
        assert!(error.contains("unknown field"), "unexpected error: {error}");
    }

    #[test]
    fn gui_toml_config_rejects_unknown_enums() {
        let mut config = gui_config();
        config.channels = "surround".into();
        let source = toml::to_string_pretty(&config).unwrap();
        let error = parse_gui_config(&source, gui_config()).unwrap_err();
        assert!(
            error.contains("チャンネルモード"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn gui_toml_config_rejects_out_of_range_values() {
        let mut config = gui_config();
        config.strength = 1.01;
        let source = toml::to_string_pretty(&config).unwrap();
        let error = parse_gui_config(&source, gui_config()).unwrap_err();
        assert!(error.contains("強度"), "unexpected error: {error}");
    }

    #[test]
    fn dropped_paths_are_classified_without_reading_contents() {
        let root = std::env::temp_dir().join(format!(
            "denoize-gui-drop-{}-{}",
            std::process::id(),
            NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let audio = root.join("voice.wav");
        let ignored = root.join("notes.txt");
        std::fs::write(&audio, []).unwrap();
        std::fs::write(&ignored, []).unwrap();
        let result = classify_dropped_paths(vec![
            root.to_string_lossy().into_owned(),
            audio.to_string_lossy().into_owned(),
            ignored.to_string_lossy().into_owned(),
        ]);
        assert_eq!(result.directories.len(), 1);
        assert_eq!(result.audio_files.len(), 1);
        assert_eq!(result.ignored.len(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }
}
