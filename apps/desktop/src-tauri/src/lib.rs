use denoize::audio::{read_audio, write_audio};
use denoize::benchmark::{BenchmarkReport, ComparisonReport};
use denoize::denoiser::{DenoiserConfig, Preset, ProcessingMode};
use denoize::encode::write_audio_to_file;
use denoize::models::{ModelAuthentication, ModelDownloadOptions, ModelProxy};
use denoize::service::{self, BackendChoice, ProcessingOptions};
use denoize::{
    AacEncoder, AtomicOutput, Backend, BackendOptions, ChannelMode, CommitMode, DownmixMode,
    EncodeOptions, OnnxModelConfig, OutputFormat, SgmseProfile,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Write;
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
        let _commit_guard = self
            .commit_gate
            .lock()
            .map_err(|_| "出力確定状態を取得できません")?;
        self.cancelled.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn commit(&self, transaction: AtomicOutput, mode: CommitMode) -> Result<(), String> {
        let _commit_guard = self
            .commit_gate
            .lock()
            .map_err(|_| "出力確定状態を取得できません")?;
        check_cancelled(self)?;
        transaction.commit(mode)
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
    #[serde(default)]
    deterministic: bool,
    #[serde(default)]
    seed: Option<u64>,
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
    deterministic: bool,
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
    deterministic: Option<bool>,
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
            deterministic: self.deterministic,
            seed: None,
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
        replace_present!(deterministic);
        current.normalized()
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProcessRequest {
    input: String,
    output: String,
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
    state_key: String,
}

fn hash_batch_path(hasher: &mut sha2::Sha256, path: &Path) {
    use sha2::Digest as _;

    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt as _;
        hasher.update(path.as_os_str().as_bytes());
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt as _;
        for unit in path.as_os_str().encode_wide() {
            hasher.update(unit.to_le_bytes());
        }
    }
    #[cfg(not(any(unix, windows)))]
    hasher.update(path.to_string_lossy().as_bytes());
}

fn batch_state_key(
    input_identity: &Path,
    input_relative: &Path,
    output_relative: &Path,
    output_format: OutputFormat,
) -> String {
    use sha2::Digest as _;
    use std::fmt::Write as _;

    let mut hasher = sha2::Sha256::new();
    hasher.update(b"denoize-batch-state-v2\0");
    hash_batch_path(&mut hasher, input_identity);
    hasher.update(b"\0");
    hash_batch_path(&mut hasher, input_relative);
    hasher.update(b"\0");
    hash_batch_path(&mut hasher, output_relative);
    hasher.update(b"\0");
    hasher.update(match output_format {
        OutputFormat::Wav => b"wav".as_slice(),
        OutputFormat::Flac => b"flac".as_slice(),
        OutputFormat::OggOpus => b"ogg-opus".as_slice(),
        OutputFormat::Mp3 => b"mp3".as_slice(),
        OutputFormat::M4a => b"m4a-aac".as_slice(),
        OutputFormat::AacAdts => b"adts-aac".as_slice(),
    });
    let mut key = String::from("v2:");
    for byte in hasher.finalize() {
        write!(&mut key, "{byte:02x}").unwrap();
    }
    key
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
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
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BackendInfo {
    name: &'static str,
    external_model: bool,
    managed_model: Option<&'static str>,
    sample_rate: Option<u32>,
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
    name: &'static str,
    backend: &'static str,
    license: &'static str,
    sample_rate: u32,
    revision: &'static str,
    installed: bool,
    path: String,
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
            })
            .collect(),
        formats: vec!["wav", "flac", "opus", "mp3", "m4a", "aac"],
        fdk_available: cfg!(feature = "fdk-aac-encoder"),
    }
}

#[tauri::command]
fn start_process(
    app: AppHandle,
    state: State<'_, AppState>,
    request: ProcessRequest,
) -> Result<u64, String> {
    validate_request(&request.input, &request.output, &request.options)?;
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
            Ok(output) => emit_progress(
                &app,
                job_id,
                "file",
                "completed",
                "処理が完了しました",
                4,
                4,
                started,
                Some(output),
                None,
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
    let items = prepare_batch_request(&request)?;
    let state_path = Path::new(&request.output_dir).join(".denoize-gui-state");
    if request.resume {
        validate_batch_reserved_path(&items, &state_path)?;
    }
    let completed = if request.resume {
        read_batch_state(&state_path)?
    } else {
        HashSet::new()
    };
    preflight_batch_outputs(&request, &items, &completed)?;
    let state_file = if request.resume {
        Some(Arc::new(Mutex::new(open_batch_state_for_append(
            &state_path,
        )?)))
    } else {
        None
    };
    let (job_id, control) = register_job(&state)?;
    let jobs = Arc::clone(&state.jobs);
    std::thread::spawn(move || {
        let started = Instant::now();
        let total = items.len();
        let finished = AtomicUsize::new(0);
        let succeeded = AtomicUsize::new(0);
        let skipped = AtomicUsize::new(0);
        let failures = Mutex::new(Vec::<String>::new());
        let process_item = |batch_item: &BatchItem| {
            if control.is_cancelled() {
                return;
            }
            let report_failure = |error: String| {
                if let Ok(mut list) = failures.lock() {
                    list.push(format!("{}: {error}", batch_item.input.display()));
                }
                let current = finished.fetch_add(1, Ordering::SeqCst) + 1;
                emit_batch_item(
                    &app,
                    job_id,
                    "failed",
                    batch_item,
                    current,
                    total,
                    started,
                    Some(error),
                );
            };
            if request.resume && completed.contains(&batch_item.state_key) {
                match batch_output_is_regular_file(&batch_item.output) {
                    Ok(true) => {
                        skipped.fetch_add(1, Ordering::Relaxed);
                        let current = finished.fetch_add(1, Ordering::SeqCst) + 1;
                        emit_batch_item(
                            &app, job_id, "skipped", batch_item, current, total, started, None,
                        );
                        return;
                    }
                    Ok(false) => {}
                    Err(error) => {
                        report_failure(error);
                        return;
                    }
                }
            }
            let result = (|| {
                let input = batch_item.input.to_str().ok_or_else(|| {
                    format!(
                        "GUIバッチではUTF-8で表現できない入力パスを処理できません: {}",
                        batch_item.input.display()
                    )
                })?;
                let output = batch_item.output.to_str().ok_or_else(|| {
                    format!(
                        "GUIバッチではUTF-8で表現できない出力パスを処理できません: {}",
                        batch_item.output.display()
                    )
                })?;
                let process_request = ProcessRequest {
                    input: input.into(),
                    output: output.into(),
                    options: request.options.clone(),
                };
                validate_request(
                    &process_request.input,
                    &process_request.output,
                    &process_request.options,
                )?;
                process_file(&process_request, &control, |_, _| {})
            })()
            .and_then(|output| {
                if let Some(file) = &state_file {
                    let mut file = file
                        .lock()
                        .map_err(|_| "再開状態のロックを取得できません".to_string())?;
                    append_batch_state_entry(&mut *file, &batch_item.state_key)?;
                }
                Ok(output)
            });
            match result {
                Ok(_) => {
                    succeeded.fetch_add(1, Ordering::Relaxed);
                    let current = finished.fetch_add(1, Ordering::SeqCst) + 1;
                    emit_batch_item(
                        &app,
                        job_id,
                        "completed",
                        batch_item,
                        current,
                        total,
                        started,
                        None,
                    );
                }
                Err(error) if error == "cancelled" => {}
                Err(error) => report_failure(error),
            }
        };
        let pool_error = if request.options.deterministic {
            items.iter().for_each(process_item);
            None
        } else {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(request.jobs)
                .build();
            match pool {
                Ok(pool) => {
                    pool.install(|| items.par_iter().for_each(process_item));
                    None
                }
                Err(error) => Some(error.to_string()),
            }
        };
        let current = finished.load(Ordering::SeqCst);
        if control.is_cancelled() {
            emit_progress(
                &app,
                job_id,
                "batch",
                "cancelled",
                "バッチをキャンセルしました",
                current,
                total,
                started,
                None,
                None,
            );
        } else if let Some(error) = pool_error {
            emit_progress(
                &app,
                job_id,
                "batch",
                "failed",
                "バッチを開始できませんでした",
                current,
                total,
                started,
                Some(request.output_dir.clone()),
                Some(format!("並列処理を開始できませんでした: {error}")),
            );
        } else {
            let failure_count = failures.lock().map(|list| list.len()).unwrap_or(0);
            let success_count = succeeded.load(Ordering::Relaxed);
            let skipped_count = skipped.load(Ordering::Relaxed);
            let BatchTerminalOutcome {
                status,
                message,
                error,
            } = batch_terminal_outcome(success_count, skipped_count, failure_count);
            emit_progress(
                &app,
                job_id,
                "batch",
                status,
                &message,
                current,
                total,
                started,
                Some(request.output_dir),
                error,
            );
        }
        if let Ok(mut jobs) = jobs.lock() {
            jobs.remove(&job_id);
        }
    });
    Ok(job_id)
}

#[derive(Debug, PartialEq, Eq)]
struct BatchTerminalOutcome {
    status: &'static str,
    message: String,
    error: Option<String>,
}

fn batch_terminal_outcome(
    success_count: usize,
    skipped_count: usize,
    failure_count: usize,
) -> BatchTerminalOutcome {
    let message = format!("完了 {success_count} · スキップ {skipped_count} · 失敗 {failure_count}");
    if failure_count == 0 {
        BatchTerminalOutcome {
            status: "completed",
            message,
            error: None,
        }
    } else {
        BatchTerminalOutcome {
            status: "failed",
            message,
            error: Some(format!("{failure_count}件のファイルを処理できませんでした")),
        }
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
            let state_key =
                batch_state_key(&input_identity, &relative, output_relative, output_format);
            Ok(BatchItem {
                input,
                output,
                state_key,
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

fn validate_batch_reserved_path(items: &[BatchItem], state_path: &Path) -> Result<(), String> {
    let state_path = batch_collision_key(&normalize_batch_path(state_path)?);
    for item in items {
        let output = batch_collision_key(&normalize_batch_path(&item.output)?);
        if output == state_path
            || output.starts_with(&state_path)
            || state_path.starts_with(&output)
        {
            return Err(format!(
                "バッチ出力 {} は再開状態 .denoize-gui-state と競合します",
                item.output.display()
            ));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn batch_state_nofollow_flag() -> i32 {
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_os = "solaris",
        target_os = "illumos"
    ))]
    {
        // O_NOFOLLOW on Linux-family and System V-family targets.
        0x0002_0000
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "solaris",
        target_os = "illumos"
    )))]
    {
        // O_NOFOLLOW on Darwin and the BSD desktop targets.
        0x0000_0100
    }
}

fn configure_batch_state_open(options: &mut std::fs::OpenOptions) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(batch_state_nofollow_flag());
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
}

fn validate_batch_state_metadata(metadata: &std::fs::Metadata, path: &Path) -> Result<(), String> {
    if !metadata.is_file() {
        return Err(format!(
            "再開状態はシンボリックリンク、ディレクトリ、特殊ファイルではなく通常ファイルである必要があります: {}",
            path.display()
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if metadata.nlink() != 1 {
            return Err(format!(
                "再開状態に複数のハードリンクは使用できません: {}",
                path.display()
            ));
        }
    }
    Ok(())
}

fn validate_batch_state_file(file: &std::fs::File, path: &Path) -> Result<(), String> {
    let metadata = file
        .metadata()
        .map_err(|error| format!("再開状態 {} を確認できません: {error}", path.display()))?;
    validate_batch_state_metadata(&metadata, path)?;

    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle as _;
        use windows_sys::Win32::Storage::FileSystem::{
            GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
        };

        let mut information = BY_HANDLE_FILE_INFORMATION::default();
        // SAFETY: `file` owns a live handle and `information` is valid writable storage.
        let succeeded =
            unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut information) };
        if succeeded == 0 {
            return Err(format!(
                "再開状態 {} のハードリンク数を確認できません: {}",
                path.display(),
                std::io::Error::last_os_error()
            ));
        }
        if information.nNumberOfLinks != 1 {
            return Err(format!(
                "再開状態に複数のハードリンクは使用できません: {}",
                path.display()
            ));
        }
    }

    Ok(())
}

fn validate_existing_batch_state_path(path: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => validate_batch_state_metadata(&metadata, path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "再開状態 {} を確認できません: {error}",
            path.display()
        )),
    }
}

fn open_batch_state_for_append(path: &Path) -> Result<std::fs::File, String> {
    validate_existing_batch_state_path(path)?;
    let mut options = std::fs::OpenOptions::new();
    options.create(true).append(true);
    configure_batch_state_open(&mut options);
    let file = options
        .open(path)
        .map_err(|error| format!("再開状態 {} を開けません: {error}", path.display()))?;
    validate_batch_state_file(&file, path)?;
    Ok(file)
}

fn read_batch_state(path: &Path) -> Result<HashSet<String>, String> {
    use std::io::Read as _;

    validate_existing_batch_state_path(path)?;
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    configure_batch_state_open(&mut options);
    let mut file = match options.open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(HashSet::new()),
        Err(error) => return Err(format!("再開状態 {} を開けません: {error}", path.display())),
    };
    validate_batch_state_file(&file, path)?;
    let mut source = String::new();
    file.read_to_string(&mut source)
        .map_err(|error| format!("再開状態 {} を読めません: {error}", path.display()))?;
    Ok(source
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect())
}

fn append_batch_state_entry(file: &mut impl Write, state_key: &str) -> Result<(), String> {
    writeln!(file, "{state_key}").map_err(|error| format!("再開状態を書き込めません: {error}"))?;
    file.flush()
        .map_err(|error| format!("再開状態を確定できません: {error}"))
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
async fn live_devices() -> Result<LiveDevices, String> {
    tauri::async_runtime::spawn_blocking(|| {
        let (inputs, outputs) = denoize::live::device_names()?;
        Ok(LiveDevices { inputs, outputs })
    })
    .await
    .map_err(|error| format!("デバイス一覧の取得に失敗しました: {error}"))?
}

#[tauri::command]
fn start_live(
    app: AppHandle,
    state: State<'_, AppState>,
    request: LiveRequest,
) -> Result<(), String> {
    let backend = validate_live_request(&request)?;
    let backend_options = resolve_gui_backend_options(backend, &request.options)?;
    if !state
        .jobs
        .lock()
        .map_err(|_| "ジョブ状態を取得できません")?
        .is_empty()
    {
        return Err("ファイル処理の完了後に開始してください".into());
    }
    let denoiser = processing_config(&request.options, 48_000)?;
    let running = Arc::new(AtomicBool::new(true));
    {
        let mut live = state
            .live
            .lock()
            .map_err(|_| "ライブ状態を更新できません")?;
        if live.is_some() {
            return Err("ライブ処理は既に実行中です".into());
        }
        *live = Some(Arc::clone(&running));
    }
    let live_state = Arc::clone(&state.live);
    std::thread::spawn(move || {
        let config = denoize::live::LiveConfig {
            input_device: request.input_device,
            output_device: request.output_device,
            chunk_ms: request.chunk_ms,
            backend,
            backend_options,
            denoiser,
        };
        let result = denoize::live::run_with_status(config, running, |status| {
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
                },
            );
        });
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
            },
        );
        if let Ok(mut live) = live_state.lock() {
            *live = None;
        }
    });
    Ok(())
}

#[tauri::command]
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
fn list_models() -> Result<Vec<ModelRow>, String> {
    denoize::models::MODELS
        .iter()
        .map(|model| {
            let path = denoize::models::path(model)?;
            Ok(ModelRow {
                name: model.name,
                backend: model.backend,
                license: model.license,
                sample_rate: model.sample_rate,
                revision: model.revision,
                installed: denoize::models::verify(model).is_ok(),
                path: path.to_string_lossy().into_owned(),
            })
        })
        .collect()
}

#[tauri::command]
fn model_action(
    app: AppHandle,
    state: State<'_, AppState>,
    name: String,
    action: String,
    options: Option<ModelActionOptions>,
) -> Result<u64, String> {
    let model = denoize::models::find(&name).ok_or_else(|| format!("不明なモデル: {name}"))?;
    if !matches!(action.as_str(), "install" | "update" | "verify" | "remove") {
        return Err(format!("不明な操作: {action}"));
    }
    let (download_options, source_path) = if matches!(action.as_str(), "install" | "update") {
        model_action_options(options)?
    } else {
        (ModelDownloadOptions::default(), None)
    };
    if source_path.is_some() && action == "update" {
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
                Some(source) => denoize::models::install_from_file_with_progress(
                    model,
                    source,
                    || cancelled.is_cancelled(),
                    progress,
                ),
                None => denoize::models::install_with_options_and_progress(
                    model,
                    &download_options,
                    || cancelled.is_cancelled(),
                    progress,
                ),
            }
            .map(|path| path.display().to_string()),
            "update" => denoize::models::update_with_options_and_progress(
                model,
                &download_options,
                || cancelled.is_cancelled(),
                progress,
            )
            .map(|path| path.display().to_string()),
            "verify" => denoize::models::verify(model).map(|path| path.display().to_string()),
            "remove" => denoize::models::remove(model).map(|_| "削除しました".into()),
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

fn register_job(state: &State<'_, AppState>) -> Result<(u64, Arc<JobControl>), String> {
    if state
        .live
        .lock()
        .map_err(|_| "ライブ状態を取得できません")?
        .is_some()
    {
        return Err("ライブ処理を停止してから開始してください".into());
    }
    let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
    let control = Arc::new(JobControl::default());
    let mut jobs = state
        .jobs
        .lock()
        .map_err(|_| "ジョブ状態を更新できません")?;
    if !jobs.is_empty() {
        return Err("別の処理が実行中です。完了またはキャンセル後に再試行してください".into());
    }
    jobs.insert(job_id, Arc::clone(&control));
    Ok((job_id, control))
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

fn prepare_batch_request(request: &BatchRequest) -> Result<Vec<BatchItem>, String> {
    let extension = validate_batch_request(request)?;
    preflight_explicit_backend_resources(&request.options)?;
    if !Path::new(&request.output_dir).is_dir() {
        return Err("出力フォルダが存在しません".into());
    }
    let items = collect_batch_items(request, &extension)?;
    if items.is_empty() {
        return Err("対応する音声ファイルがありません".into());
    }
    preflight_batch_codecs(&request.options, &items)?;
    Ok(items)
}

fn preflight_batch_codecs(options: &ProcessOptions, items: &[BatchItem]) -> Result<(), String> {
    let encode = parsed_encode_options(options)?;
    for item in items {
        let format = OutputFormat::from_path(&item.output)?;
        let audio = read_audio(&item.input).map_err(|error| {
            format!(
                "バッチ入力 {} を事前検査できません: {error}",
                item.input.display()
            )
        })?;
        format.validate_config(&audio, &encode).map_err(|error| {
            format!(
                "バッチ出力 {} のcodec設定が不正です: {error}",
                item.output.display()
            )
        })?;
        validated_processing_options(options, &audio).map_err(|error| {
            format!(
                "バッチ入力 {} の処理設定が不正です: {error}",
                item.input.display()
            )
        })?;
    }
    Ok(())
}

fn preflight_batch_outputs(
    request: &BatchRequest,
    items: &[BatchItem],
    completed: &HashSet<String>,
) -> Result<(), String> {
    for item in items {
        let resumable = request.resume
            && completed.contains(&item.state_key)
            && batch_output_is_regular_file(&item.output)?;
        if !resumable {
            ensure_output_available(&item.output, request.options.force)?;
        }
    }
    Ok(())
}

fn batch_output_is_regular_file(path: &Path) -> Result<bool, String> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => Ok(metadata.is_file()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!(
            "バッチ出力 {} を確認できません: {error}",
            path.display()
        )),
    }
}

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
        resolve_gui_backend_options(backend, options)?;
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

fn validate_request(input: &str, output: &str, options: &ProcessOptions) -> Result<(), String> {
    validate_process_options(options)?;
    let format = OutputFormat::from_path(Path::new(output))?;
    format.validate_encoder(parse_aac_encoder(&options.aac_encoder)?)?;
    preflight_explicit_backend_resources(options)?;
    if !Path::new(input).is_file() {
        return Err("入力ファイルが存在しません".into());
    }
    ensure_output_available(Path::new(output), options.force)
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

fn process_file(
    request: &ProcessRequest,
    control: &JobControl,
    progress: impl Fn(usize, &'static str),
) -> Result<String, String> {
    check_cancelled(control)?;
    let input = Path::new(&request.input);
    let output = Path::new(&request.output);
    let metadata = if request.options.preserve_metadata {
        denoize::metadata::read_extended(input)?
    } else {
        None
    };
    let mut audio = read_audio(input)?;
    let encode = parsed_encode_options(&request.options)?;
    let format = OutputFormat::from_path(output)?;
    format.validate_config(&audio, &encode)?;
    progress(1, "ノイズ除去を実行しています");
    check_cancelled(control)?;
    let processing = validated_processing_options(&request.options, &audio)?;
    progress(2, "ラウドネスと出力を準備しています");
    service::process_audio(&mut audio, processing)?;
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
        denoize::metadata::write_extended_to_file(metadata, transaction.file_mut())?;
    }
    progress(4, "出力を確定しています");
    let commit_mode = if request.options.force {
        CommitMode::Replace
    } else {
        CommitMode::NoClobber
    };
    control.commit(transaction, commit_mode)?;
    Ok(output.to_string_lossy().into_owned())
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
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_batch_item(
    app: &AppHandle,
    job_id: u64,
    item_status: &'static str,
    item: &BatchItem,
    current: usize,
    total: usize,
    started: Instant,
    error: Option<String>,
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

    fn classical_options(force: bool) -> ProcessOptions {
        let mut options = options();
        options.backend = "classical".into();
        options.preserve_metadata = false;
        options.force = force;
        options
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
            deterministic: false,
            seed: None,
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
            deterministic: false,
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

    #[test]
    fn non_live_backends_are_rejected_before_starting_a_session() {
        let Some(name) = Backend::available_names()
            .iter()
            .copied()
            .find(|name| !matches!(*name, "classical" | "rnnoise"))
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

        let error = validate_request(
            &missing_input.to_string_lossy(),
            &output.to_string_lossy(),
            &process,
        )
        .unwrap_err();

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
    fn no_force_rechecks_destination_when_committing() {
        let directory = TestDirectory::create("commit-race");
        let input = directory.join("input.wav");
        let output = directory.join("output.wav");
        write_test_wav(&input);
        let request = ProcessRequest {
            input: input.to_string_lossy().into_owned(),
            output: output.to_string_lossy().into_owned(),
            options: classical_options(false),
        };
        validate_request(&request.input, &request.output, &request.options).unwrap();

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
        let request = ProcessRequest {
            input: input.to_string_lossy().into_owned(),
            output: output.to_string_lossy().into_owned(),
            options: classical_options(true),
        };
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
        let request = ProcessRequest {
            input: input.to_string_lossy().into_owned(),
            output: output.to_string_lossy().into_owned(),
            options: classical_options(false),
        };
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
        let request = ProcessRequest {
            input: input.to_string_lossy().into_owned(),
            output: output.to_string_lossy().into_owned(),
            options: classical_options(false),
        };

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
        assert!(!output.join(".denoize-gui-state").exists());
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
        assert!(!output.join(".denoize-gui-state").exists());
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
        let mut request = BatchRequest {
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

        let error = preflight_batch_outputs(&request, &items, &HashSet::new()).unwrap_err();
        assert!(error.contains("既に存在"), "{error}");
        assert!(!output.join("a.wav").exists());
        assert_eq!(std::fs::read(output.join("b.wav")).unwrap(), b"existing");

        request.options.force = true;
        std::fs::remove_file(output.join("b.wav")).unwrap();
        std::fs::create_dir(output.join("b.wav")).unwrap();
        let error = preflight_batch_outputs(&request, &items, &HashSet::new()).unwrap_err();
        assert!(error.contains("置換可能"), "{error}");
        assert!(!output.join("a.wav").exists());
    }

    #[cfg(unix)]
    #[test]
    fn completed_batch_symlinks_are_never_treated_as_resumable_files() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::create("batch-resume-output-symlink");
        let victim = directory.join("victim.wav");
        let output = directory.join("output.wav");
        write_test_wav(&victim);
        symlink(&victim, &output).unwrap();
        let item = BatchItem {
            input: directory.join("input.wav"),
            output: output.clone(),
            state_key: "v2:completed".into(),
        };
        let mut request = batch_request();
        request.resume = true;
        request.options.force = true;
        let completed = HashSet::from([item.state_key.clone()]);

        assert!(
            output.is_file(),
            "the follow-link check would incorrectly skip"
        );
        assert!(!batch_output_is_regular_file(&output).unwrap());
        preflight_batch_outputs(&request, &[item], &completed).unwrap();
        assert!(std::fs::symlink_metadata(&output)
            .unwrap()
            .file_type()
            .is_symlink());
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
            BatchItem {
                input: input_a.clone(),
                output: output.join("same.wav"),
                state_key: "a".into(),
            },
            BatchItem {
                input: input_b.clone(),
                output: output.join("same.wav"),
                state_key: "b".into(),
            },
        ];
        assert!(validate_batch_destinations(None, &exact)
            .unwrap_err()
            .contains("同じバッチ出力"));

        let file_and_directory = vec![
            BatchItem {
                input: input_a,
                output: output.join("foo.flac"),
                state_key: "a".into(),
            },
            BatchItem {
                input: input_b,
                output: output.join("foo.flac/bar.flac"),
                state_key: "b".into(),
            },
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
    fn batch_state_keys_include_input_identity_destination_and_format() {
        let relative = Path::new("voice.wav");
        let output = Path::new("nested/voice.output");
        let wav = batch_state_key(
            Path::new("/input-a/voice.wav"),
            relative,
            output,
            OutputFormat::Wav,
        );

        assert!(wav.starts_with("v2:"));
        assert_ne!(
            wav,
            batch_state_key(
                Path::new("/input-b/voice.wav"),
                relative,
                output,
                OutputFormat::Wav,
            )
        );
        assert_ne!(
            wav,
            batch_state_key(
                Path::new("/input-a/voice.wav"),
                relative,
                output,
                OutputFormat::Flac,
            )
        );
        assert_ne!(
            wav,
            batch_state_key(
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
            batch_state_key(
                Path::new("/input/voice.wav"),
                &first_relative,
                &first_relative,
                OutputFormat::Wav,
            ),
            batch_state_key(
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
        assert!(!output.join(".denoize-gui-state").exists());
    }

    #[test]
    fn batch_state_missing_file_is_empty() {
        let path = std::env::temp_dir().join(format!(
            "denoize-missing-state-{}-{}",
            std::process::id(),
            NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(read_batch_state(&path).unwrap().is_empty());
    }

    #[test]
    fn batch_state_read_and_write_errors_are_not_ignored() {
        let directory = TestDirectory::create("batch-state-errors");
        let state = directory.join(".denoize-gui-state");
        std::fs::write(&state, [0xff]).unwrap();
        assert!(read_batch_state(&state).unwrap_err().contains("読めません"));

        struct FailingWriter;
        impl Write for FailingWriter {
            fn write(&mut self, _buffer: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("injected write failure"))
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let error = append_batch_state_entry(&mut FailingWriter, "input.wav").unwrap_err();
        assert!(error.contains("書き込めません"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn batch_state_rejects_symlinks_and_hard_links_without_touching_targets() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::create("batch-state-links");
        let victim = directory.join("victim.txt");
        let symlink_state = directory.join("symlink-state");
        let hardlink_state = directory.join("hardlink-state");
        std::fs::write(&victim, b"victim\n").unwrap();
        symlink(&victim, &symlink_state).unwrap();

        assert!(read_batch_state(&symlink_state).is_err());
        assert!(open_batch_state_for_append(&symlink_state).is_err());
        assert_eq!(std::fs::read(&victim).unwrap(), b"victim\n");

        std::fs::hard_link(&victim, &hardlink_state).unwrap();
        assert!(read_batch_state(&hardlink_state).is_err());
        assert!(open_batch_state_for_append(&hardlink_state).is_err());
        assert_eq!(std::fs::read(&victim).unwrap(), b"victim\n");
    }

    #[test]
    fn batch_outputs_cannot_claim_the_resume_state_path() {
        let directory = TestDirectory::create("batch-state-reserved");
        let state = directory.join(".denoize-gui-state");
        let items = vec![BatchItem {
            input: directory.join("input.wav"),
            output: state.join("nested.wav"),
            state_key: "input.wav".into(),
        }];
        assert!(validate_batch_reserved_path(&items, &state).is_err());
    }

    #[test]
    fn successful_batch_has_completed_terminal_outcome() {
        let outcome = batch_terminal_outcome(3, 1, 0);
        assert_eq!(outcome.status, "completed");
        assert_eq!(outcome.message, "完了 3 · スキップ 1 · 失敗 0");
        assert_eq!(outcome.error, None);
    }

    #[test]
    fn mixed_batch_has_failed_terminal_outcome() {
        let outcome = batch_terminal_outcome(2, 1, 1);
        assert_eq!(outcome.status, "failed");
        assert_eq!(outcome.message, "完了 2 · スキップ 1 · 失敗 1");
        assert_eq!(
            outcome.error.as_deref(),
            Some("1件のファイルを処理できませんでした")
        );
    }

    #[test]
    fn all_failed_batch_has_failed_terminal_outcome() {
        let outcome = batch_terminal_outcome(0, 0, 3);
        assert_eq!(outcome.status, "failed");
        assert_eq!(outcome.message, "完了 0 · スキップ 0 · 失敗 3");
        assert_eq!(
            outcome.error.as_deref(),
            Some("3件のファイルを処理できませんでした")
        );
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
